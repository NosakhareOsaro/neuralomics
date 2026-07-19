import json
from pathlib import Path

import pytest
from cellpainting.download import (
    DEFAULT_CONFIG_PATH,
    Compound,
    DataConfig,
    download_plate_subset,
    load_config,
    well_image_prefix,
    well_to_row_col,
    write_manifest,
)


def test_well_to_row_col_parses_single_letter_rows():
    assert well_to_row_col("A01") == (1, 1)
    assert well_to_row_col("C05") == (3, 5)
    assert well_to_row_col("P24") == (16, 24)


def test_well_to_row_col_parses_two_letter_rows():
    assert well_to_row_col("AA01") == (27, 1)


def test_well_to_row_col_rejects_malformed_well():
    with pytest.raises(ValueError):
        well_to_row_col("not-a-well")


def _sample_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        bucket="cellpainting-gallery",
        dataset="cpg0000-jump-pilot",
        source_name="source_4",
        batch="2020_11_04_CPJUMP1",
        plate="BR00116991",
        measurement="2020-11-05T19_51_35-Measurement1",
        compounds=[
            Compound(
                well="A01",
                broad_sample="BRD-A86665761-001-01-1",
                pert_iname="gabapentin-enacarbil",
                targets="CACNA1A",
            )
        ],
        raw_dir=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.json",
    )


def test_well_image_prefix_matches_verified_bucket_layout(tmp_path: Path):
    cfg = _sample_config(tmp_path)
    assert well_image_prefix(cfg, "A01") == (
        "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
        "BR00116991__2020-11-05T19_51_35-Measurement1/Images/r01c01"
    )


def test_load_config_reads_the_real_module_config():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.bucket == "cellpainting-gallery"
    assert cfg.dataset == "cpg0000-jump-pilot"
    assert cfg.plate == "BR00116991"
    assert 10 <= len(cfg.compounds) <= 15
    assert len({c.broad_sample for c in cfg.compounds}) == len(cfg.compounds)


def test_write_manifest_round_trips_json(tmp_path: Path):
    manifest = {"plate": "BR00116991", "total_files": 1008}
    path = tmp_path / "nested" / "manifest.json"

    write_manifest(manifest, path)

    assert json.loads(path.read_text()) == manifest


@pytest.mark.slow
def test_download_plate_subset_dry_run_matches_live_bucket(tmp_path: Path):
    """Exercises the real, public S3 bucket (no credentials needed).

    Regression check for the JUMP-CP layout assumptions baked into
    download.py: 9 sites x 8 channels x 1 plane = 72 images per well, and 8
    illumination-correction files per plate, as verified manually against the
    live bucket on 2026-07-20 (see docs/build-log.md).
    """
    cfg = _sample_config(tmp_path)

    manifest = download_plate_subset(cfg, dry_run=True)

    assert manifest["wells"][0]["n_files"] == 72
    assert manifest["illum_files"] == 8
    assert not (tmp_path / "raw").exists()  # dry run must not write anything
