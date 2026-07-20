"""Tabular baseline: compound-identity classification on Broad's CellProfiler
profiles -- see this module's README (Feature extraction, Evaluation plan)
and docs/build-log.md for why the classification target is compound identity
(14-way, `Metadata_broad_sample`) rather than mechanism-of-action, and why
BR00116992-BR00116994 are usable as in-batch replicates of BR00116991.

Evaluated with leave-one-plate-out cross-validation across all four plates
(n=56 = 14 compounds x 4 plates: each plate contributes exactly one well per
compound). Significance is assessed with a permutation-based null test
(shuffle labels within each plate, rerun the identical CV pipeline, see where
the real score falls in that empirical null distribution) rather than a fixed
accuracy threshold, since chance-level accuracy for 14-way classification
isn't simply 1/14 once CV-fold structure is accounted for.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cellpainting.download import DEFAULT_CONFIG_PATH, make_s3_client
from common.seeding import DEFAULT_SEED, set_global_seed

PROFILE_VARIANT = "normalized_feature_select_batch"
LABEL_FIELD = "Metadata_broad_sample"


@dataclass(frozen=True)
class BaselineConfig:
    bucket: str
    dataset: str
    source_name: str
    batch: str
    plates: list[str]
    wells: list[str]
    raw_dir: Path


def load_baseline_config(path: Path = DEFAULT_CONFIG_PATH) -> BaselineConfig:
    raw = yaml.safe_load(path.read_text())
    src = raw["source"]
    plates = [src["plate"], *raw.get("replicate_plates", [])]
    return BaselineConfig(
        bucket=raw["profiles"]["bucket"],
        dataset=src["dataset"],
        source_name=src["source_name"],
        batch=src["batch"],
        plates=plates,
        wells=[c["well"] for c in raw["compounds"]],
        raw_dir=Path(raw["output"]["raw_dir"]),
    )


def profile_key(cfg: BaselineConfig, plate: str) -> str:
    return (
        f"{cfg.dataset}/{cfg.source_name}/workspace/profiles/{cfg.batch}/"
        f"{plate}/{plate}_{PROFILE_VARIANT}.csv.gz"
    )


def filtered_plate_path(cfg: BaselineConfig, plate: str) -> Path:
    return cfg.raw_dir / "profiles" / f"{plate}_{PROFILE_VARIANT}.filtered.csv"


def fetch_plate_profile(
    s3: Any, cfg: BaselineConfig, plate: str
) -> tuple[list[str], list[list[str]]]:
    obj = s3.get_object(Bucket=cfg.bucket, Key=profile_key(cfg, plate))
    text = gzip.decompress(obj["Body"].read()).decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    well_idx = header.index("Metadata_Well")
    wanted = set(cfg.wells)
    rows = [row for row in reader if row[well_idx] in wanted]
    return header, rows


def load_plate_profile(
    s3: Any, cfg: BaselineConfig, plate: str, *, refresh: bool = False
) -> tuple[list[str], list[list[str]]]:
    """Idempotent per-plate fetch: reuses a cached filtered CSV unless --refresh."""
    dest = filtered_plate_path(cfg, plate)
    if not refresh and dest.exists():
        with dest.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        return header, rows

    header, rows = fetch_plate_profile(s3, cfg, plate)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return header, rows


def assemble_dataset(
    cfg: BaselineConfig, s3: Any | None = None, *, refresh: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Pools all configured plates into one (X, y, groups) dataset.

    X: (n_samples, n_features) float array of CellProfiler feature columns.
    y: (n_samples,) compound-identity labels (Metadata_broad_sample).
    groups: (n_samples,) plate id per sample, for leave-one-plate-out CV.
    """
    s3 = s3 or make_s3_client()
    header_ref: list[str] | None = None
    feature_idx: list[int] = []
    label_idx = -1
    X_rows: list[list[float]] = []
    y: list[str] = []
    groups: list[str] = []

    for plate in cfg.plates:
        header, rows = load_plate_profile(s3, cfg, plate, refresh=refresh)
        if header_ref is None:
            header_ref = header
            feature_idx = [i for i, c in enumerate(header) if not c.startswith("Metadata_")]
            label_idx = header.index(LABEL_FIELD)
        elif header != header_ref:
            raise ValueError(
                f"plate {plate}'s profile schema doesn't match {cfg.plates[0]}'s -- "
                "refusing to silently pool mismatched feature columns"
            )
        if len(rows) != len(cfg.wells):
            raise ValueError(
                f"plate {plate}: expected {len(cfg.wells)} of our wells, got {len(rows)}"
            )
        for row in rows:
            X_rows.append([float(row[i]) for i in feature_idx])
            y.append(row[label_idx])
            groups.append(plate)

    assert header_ref is not None
    feature_names = [header_ref[i] for i in feature_idx]
    X = np.asarray(X_rows, dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(
            "non-finite feature values found -- verified clean on 2026-07-20, "
            "re-check the source data"
        )
    return X, np.asarray(y), np.asarray(groups), feature_names


def make_classifier(seed: int = DEFAULT_SEED) -> Any:
    """A deliberately simple (non-deep) multinomial classifier: the naive
    baseline this module's CNN is ultimately compared against."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=0.1, random_state=seed),
    )


def lopo_cv_predictions(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = DEFAULT_SEED
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = make_classifier(seed)
        clf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])
    return y_pred


def evaluate(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = DEFAULT_SEED
) -> dict[str, float]:
    y_pred = lopo_cv_predictions(X, y, groups, seed=seed)
    return {
        "accuracy": float((y_pred == y).mean()),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
    }


def permutation_null_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Permutation-based significance test for the LOPO-CV accuracy.

    Rather than comparing observed accuracy to a fixed threshold like 1/14,
    which assumes chance performance is exactly 1/n_classes -- not
    guaranteed once CV-fold structure is involved -- this uses
    sklearn's `permutation_test_score` with `groups` passed through: labels
    are shuffled *within each plate* (every permutation still has all 14
    compounds present exactly once per plate, matching the real design)
    rather than shuffled globally, and the identical CV pipeline is rerun on
    each shuffle to build an empirical null distribution.
    """
    clf = make_classifier(seed)
    cv = LeaveOneGroupOut()
    score, permutation_scores, p_value = permutation_test_score(
        clf,
        X,
        y,
        groups=groups,
        cv=cv,
        n_permutations=n_permutations,
        scoring="accuracy",
        random_state=seed,
    )
    return {
        "observed_accuracy": float(score),
        "n_permutations": n_permutations,
        "null_mean": float(permutation_scores.mean()),
        "null_std": float(permutation_scores.std()),
        "p_value": float(p_value),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch profiles from S3 even if cached locally"
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    set_global_seed(args.seed)
    cfg = load_baseline_config(args.config)
    X, y, groups, feature_names = assemble_dataset(cfg, refresh=args.refresh)

    metrics = evaluate(X, y, groups, seed=args.seed)
    perm = permutation_null_test(X, y, groups, n_permutations=args.n_permutations, seed=args.seed)

    result = {
        "n_samples": len(y),
        "n_classes": len(set(y)),
        "n_features": len(feature_names),
        "plates": cfg.plates,
        **metrics,
        "permutation_test": perm,
    }

    dest = cfg.raw_dir / "baseline" / "results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"LOPO CV over {result['n_samples']} samples / {result['n_classes']} classes: "
        f"accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f}\n"
        f"Permutation test (n={perm['n_permutations']}): null accuracy "
        f"{perm['null_mean']:.3f} +/- {perm['null_std']:.3f}, p={perm['p_value']:.4f}\n"
        f"-> {dest}"
    )


if __name__ == "__main__":
    main()
