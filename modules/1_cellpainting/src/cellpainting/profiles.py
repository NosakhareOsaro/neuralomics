"""Fetch Broad's precomputed CellProfiler profile for Module 1's validation subset.

We do not run CellProfiler ourselves -- see docs/build-log.md and this
module's README (Feature extraction section) for why. This pulls Broad's
already-published, per-well profile for plate BR00116991
(`BR00116991_normalized_feature_select_batch.csv.gz`) and filters it down to
the 14 wells configured in configs/data.yaml. Used only for the tabular
baseline / UMAP-SHAP track; the CNN trains on raw images (see download.py).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cellpainting.download import DEFAULT_CONFIG_PATH, make_s3_client, write_manifest

# The pycytominer profile variant this project uses -- see the README's
# Feature extraction section for why. This is the one place that knows the
# S3 key format for a plate's profile; `profile_key()` below is reused by
# both this module (BR00116991) and baseline.py (BR00116991's replicates),
# rather than each independently hardcoding/reconstructing the path, which a
# code review flagged as a silent-drift risk when the two definitions could
# disagree unnoticed. See docs/build-log.md.
PROFILE_VARIANT = "normalized_feature_select_batch"


def profile_key(dataset: str, source_name: str, batch: str, plate: str) -> str:
    return (
        f"{dataset}/{source_name}/workspace/profiles/{batch}/"
        f"{plate}/{plate}_{PROFILE_VARIANT}.csv.gz"
    )


@dataclass(frozen=True)
class ProfilesConfig:
    bucket: str
    dataset: str
    source_name: str
    batch: str
    plate: str
    wells: list[str]
    raw_dir: Path

    @property
    def key(self) -> str:
        return profile_key(self.dataset, self.source_name, self.batch, self.plate)


def load_profiles_config(path: Path = DEFAULT_CONFIG_PATH) -> ProfilesConfig:
    raw = yaml.safe_load(path.read_text())
    src = raw["source"]
    return ProfilesConfig(
        bucket=raw["profiles"]["bucket"],
        dataset=src["dataset"],
        source_name=src["source_name"],
        batch=src["batch"],
        plate=src["plate"],
        wells=[c["well"] for c in raw["compounds"]],
        raw_dir=Path(raw["output"]["raw_dir"]),
    )


def filtered_profile_path(cfg: ProfilesConfig) -> Path:
    basename = Path(cfg.key).name
    stem = basename[: -len(".csv.gz")] if basename.endswith(".csv.gz") else Path(basename).stem
    return cfg.raw_dir / "profiles" / f"{stem}.filtered.csv"


def profile_manifest_path(cfg: ProfilesConfig) -> Path:
    return cfg.raw_dir / "profiles" / "manifest.json"


def fetch_profile_csv(s3: Any, bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return gzip.decompress(obj["Body"].read()).decode("utf-8")


def filter_profile_rows(csv_text: str, wells: list[str]) -> tuple[list[str], list[list[str]]]:
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    well_idx = header.index("Metadata_Well")
    wanted = set(wells)
    rows = [row for row in reader if row[well_idx] in wanted]
    return header, rows


def write_filtered_csv(header: list[str], rows: list[list[str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def read_filtered_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def fetch_and_filter_profile(cfg: ProfilesConfig, *, dry_run: bool = False) -> dict[str, Any]:
    dest = filtered_profile_path(cfg)

    if dry_run:
        s3 = make_s3_client()
        header, rows = filter_profile_rows(fetch_profile_csv(s3, cfg.bucket, cfg.key), cfg.wells)
    elif dest.exists():
        header, rows = read_filtered_csv(dest)  # idempotent resume
    else:
        s3 = make_s3_client()
        header, rows = filter_profile_rows(fetch_profile_csv(s3, cfg.bucket, cfg.key), cfg.wells)
        write_filtered_csv(header, rows, dest)

    metadata_cols = [c for c in header if c.startswith("Metadata_")]
    well_idx = header.index("Metadata_Well")
    return {
        "dry_run": dry_run,
        "source_bucket": cfg.bucket,
        "source_key": cfg.key,
        "output_path": str(dest),
        "n_rows": len(rows),
        "n_metadata_cols": len(metadata_cols),
        "n_feature_cols": len(header) - len(metadata_cols),
        "wells": sorted({row[well_idx] for row in rows}),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and filter the profile without writing anything to disk",
    )
    args = parser.parse_args(argv)

    cfg = load_profiles_config(args.config)
    manifest = fetch_and_filter_profile(cfg, dry_run=args.dry_run)
    if not args.dry_run:
        write_manifest(manifest, profile_manifest_path(cfg))
    print(
        f"{manifest['n_rows']} wells, {manifest['n_feature_cols']} features, "
        f"{manifest['n_metadata_cols']} metadata cols"
        + (" (dry run, nothing written)" if args.dry_run else f" -> {manifest['output_path']}")
    )


if __name__ == "__main__":
    main()
