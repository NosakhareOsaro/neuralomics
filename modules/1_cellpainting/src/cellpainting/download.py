"""Download the Module 1 validation subset from the Cell Painting Gallery.

Pulls raw images and illumination-correction files for the wells listed in
configs/data.yaml -- one plate, restricted to a small compound subset -- from
the public, no-sign-request `cellpainting-gallery` S3 bucket. See
docs/build-log.md for why this subset was chosen over the full ~116TB JUMP-CP
dataset, and DATA_SOURCES.md for dataset-level provenance.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"

_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


@dataclass(frozen=True)
class Compound:
    well: str
    broad_sample: str
    pert_iname: str
    targets: str


@dataclass(frozen=True)
class DataConfig:
    bucket: str
    dataset: str
    source_name: str
    batch: str
    plate: str
    measurement: str
    compounds: list[Compound]
    raw_dir: Path
    manifest_path: Path

    @property
    def image_prefix(self) -> str:
        return (
            f"{self.dataset}/{self.source_name}/images/{self.batch}/images/"
            f"{self.plate}__{self.measurement}/Images/"
        )

    @property
    def illum_prefix(self) -> str:
        return f"{self.dataset}/{self.source_name}/images/{self.batch}/illum/{self.plate}/"


def well_to_row_col(well: str) -> tuple[int, int]:
    """Parse a plate well ID like 'A01' into 1-indexed (row, col).

    Row letters use base-26 (A=1 ... Z=26, AA=27 ...) to also cover the
    two-letter row IDs used on 1536-well plates, even though the plate this
    module targets only needs single-letter rows.
    """
    match = _WELL_RE.match(well)
    if not match:
        raise ValueError(f"not a valid well id: {well!r}")
    letters, digits = match.groups()
    row = 0
    for char in letters.upper():
        row = row * 26 + (ord(char) - ord("A") + 1)
    return row, int(digits)


def well_image_prefix(cfg: DataConfig, well: str) -> str:
    row, col = well_to_row_col(well)
    return f"{cfg.image_prefix}r{row:02d}c{col:02d}"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> DataConfig:
    raw = yaml.safe_load(path.read_text())
    src = raw["source"]
    output = raw["output"]
    return DataConfig(
        bucket=src["bucket"],
        dataset=src["dataset"],
        source_name=src["source_name"],
        batch=src["batch"],
        plate=src["plate"],
        measurement=src["measurement"],
        compounds=[Compound(**c) for c in raw["compounds"]],
        raw_dir=Path(output["raw_dir"]),
        manifest_path=Path(output["manifest_path"]),
    )


def make_s3_client() -> Any:
    """An S3 client for the public, no-sign-request cellpainting-gallery bucket."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def list_keys(s3: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(page.get("Contents", []))
    return keys


def download_key(s3: Any, bucket: str, key: str, dest_root: Path) -> None:
    dest = dest_root / key
    if dest.exists():
        return  # idempotent resume: already downloaded in a prior run
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest))


def download_plate_subset(cfg: DataConfig, *, dry_run: bool = False) -> dict[str, Any]:
    s3 = make_s3_client()
    manifest: dict[str, Any] = {
        "downloaded_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "bucket": cfg.bucket,
        "dataset": cfg.dataset,
        "batch": cfg.batch,
        "plate": cfg.plate,
        "wells": [],
    }

    for compound in cfg.compounds:
        objects = list_keys(s3, cfg.bucket, well_image_prefix(cfg, compound.well))
        if not dry_run:
            for obj in objects:
                download_key(s3, cfg.bucket, obj["Key"], cfg.raw_dir)
        manifest["wells"].append(
            {
                "well": compound.well,
                "broad_sample": compound.broad_sample,
                "pert_iname": compound.pert_iname,
                "targets": compound.targets,
                "n_files": len(objects),
                "n_bytes": sum(obj["Size"] for obj in objects),
            }
        )

    illum_objects = list_keys(s3, cfg.bucket, cfg.illum_prefix)
    if not dry_run:
        for obj in illum_objects:
            download_key(s3, cfg.bucket, obj["Key"], cfg.raw_dir)
    manifest["illum_files"] = len(illum_objects)
    manifest["illum_bytes"] = sum(obj["Size"] for obj in illum_objects)

    manifest["total_files"] = sum(w["n_files"] for w in manifest["wells"]) + manifest["illum_files"]
    manifest["total_bytes"] = sum(w["n_bytes"] for w in manifest["wells"]) + manifest["illum_bytes"]
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be downloaded (file counts/sizes) without transferring anything",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    manifest = download_plate_subset(cfg, dry_run=args.dry_run)
    write_manifest(manifest, cfg.manifest_path)
    print(
        f"{manifest['total_files']} files, {manifest['total_bytes'] / 1e9:.2f} GB "
        f"across {len(manifest['wells'])} wells"
        + (" (dry run, nothing downloaded)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
