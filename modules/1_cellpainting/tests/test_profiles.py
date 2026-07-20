import json
from pathlib import Path

import pytest
from cellpainting.download import make_s3_client, write_manifest
from cellpainting.profiles import (
    DEFAULT_CONFIG_PATH,
    ProfilesConfig,
    fetch_and_filter_profile,
    fetch_profile_csv,
    filter_profile_rows,
    filtered_profile_path,
    load_profiles_config,
    profile_key,
    profile_manifest_path,
    read_filtered_csv,
    write_filtered_csv,
)

EXPECTED_METADATA_COLS = {
    "Metadata_broad_sample",
    "Metadata_pert_iname",
    "Metadata_InChIKey",
    "Metadata_smiles",
    "Metadata_gene",
    "Metadata_pert_type",
    "Metadata_control_type",
}


def _sample_config(tmp_path: Path) -> ProfilesConfig:
    return ProfilesConfig(
        bucket="cellpainting-gallery",
        dataset="cpg0000-jump-pilot",
        source_name="source_4",
        batch="2020_11_04_CPJUMP1",
        plate="BR00116991",
        wells=["A01", "A03"],
        raw_dir=tmp_path / "raw",
    )


def test_profile_key_matches_the_known_good_key_for_br00116991():
    assert profile_key("cpg0000-jump-pilot", "source_4", "2020_11_04_CPJUMP1", "BR00116991") == (
        "cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/"
        "BR00116991/BR00116991_normalized_feature_select_batch.csv.gz"
    )


def test_profiles_config_key_is_derived_not_stored(tmp_path: Path):
    """`key` is a computed property, not a field -- there is exactly one
    place (`profile_key()`) that knows this S3 path format."""
    cfg = _sample_config(tmp_path)
    assert cfg.key == profile_key(cfg.dataset, cfg.source_name, cfg.batch, cfg.plate)


def test_filtered_profile_path_strips_csv_gz_suffix(tmp_path: Path):
    cfg = _sample_config(tmp_path)
    path = filtered_profile_path(cfg)
    expected_name = "BR00116991_normalized_feature_select_batch.filtered.csv"
    assert path == tmp_path / "raw" / "profiles" / expected_name


def test_load_profiles_config_reads_the_real_module_config():
    cfg = load_profiles_config(DEFAULT_CONFIG_PATH)
    assert cfg.bucket == "cellpainting-gallery"
    assert cfg.plate == "BR00116991"
    assert cfg.key.endswith("BR00116991_normalized_feature_select_batch.csv.gz")
    assert len(cfg.wells) == 14
    assert "A01" in cfg.wells


def test_filter_profile_rows_keeps_only_wanted_wells():
    csv_text = (
        "Metadata_Well,Metadata_broad_sample,Feature_1\n"
        "A01,BRD-1,0.5\n"
        "A02,BRD-2,0.1\n"
        "A03,BRD-3,0.9\n"
    )

    header, rows = filter_profile_rows(csv_text, ["A01", "A03"])

    assert header == ["Metadata_Well", "Metadata_broad_sample", "Feature_1"]
    assert [row[0] for row in rows] == ["A01", "A03"]


def test_write_and_read_filtered_csv_round_trips(tmp_path: Path):
    header = ["Metadata_Well", "Feature_1"]
    rows = [["A01", "0.5"], ["A03", "0.9"]]
    dest = tmp_path / "nested" / "profile.csv"

    write_filtered_csv(header, rows, dest)
    read_header, read_rows = read_filtered_csv(dest)

    assert read_header == header
    assert read_rows == rows


@pytest.mark.slow
def test_fetch_and_filter_profile_dry_run_matches_live_bucket(tmp_path: Path):
    """Exercises the real, public S3 bucket (no credentials needed).

    Regression check that Broad's published profile for BR00116991 still has
    the expected metadata columns and still contains exactly our 14 wells, as
    verified manually on 2026-07-20 (see docs/build-log.md).
    """
    cfg = load_profiles_config(DEFAULT_CONFIG_PATH)
    cfg = ProfilesConfig(
        bucket=cfg.bucket,
        dataset=cfg.dataset,
        source_name=cfg.source_name,
        batch=cfg.batch,
        plate=cfg.plate,
        wells=cfg.wells,
        raw_dir=tmp_path / "raw",
    )

    manifest = fetch_and_filter_profile(cfg, dry_run=True)

    assert manifest["n_rows"] == 14
    assert set(manifest["wells"]) == set(cfg.wells)
    assert not (tmp_path / "raw").exists()  # dry run must not write anything


@pytest.mark.slow
def test_fetch_and_filter_profile_dry_run_has_expected_metadata_columns(tmp_path: Path):
    cfg = load_profiles_config(DEFAULT_CONFIG_PATH)
    s3 = make_s3_client()

    header, _ = filter_profile_rows(fetch_profile_csv(s3, cfg.bucket, cfg.key), cfg.wells)

    assert EXPECTED_METADATA_COLS.issubset(set(header))


@pytest.mark.slow
def test_fetch_and_filter_profile_writes_and_resumes_idempotently(tmp_path: Path):
    cfg = load_profiles_config(DEFAULT_CONFIG_PATH)
    cfg = ProfilesConfig(
        bucket=cfg.bucket,
        dataset=cfg.dataset,
        source_name=cfg.source_name,
        batch=cfg.batch,
        plate=cfg.plate,
        wells=cfg.wells,
        raw_dir=tmp_path / "raw",
    )

    first = fetch_and_filter_profile(cfg, dry_run=False)
    dest = filtered_profile_path(cfg)
    assert dest.exists()
    written_at = dest.stat().st_mtime

    second = fetch_and_filter_profile(cfg, dry_run=False)  # should skip re-fetching from S3

    assert second["n_rows"] == first["n_rows"] == 14
    assert dest.stat().st_mtime == written_at  # untouched, not re-written

    write_manifest(second, profile_manifest_path(cfg))
    assert json.loads(profile_manifest_path(cfg).read_text())["n_rows"] == 14
