import gzip
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cellpainting.baseline import (
    DEFAULT_CONFIG_PATH,
    BaselineConfig,
    assemble_dataset,
    evaluate,
    filtered_plate_path,
    load_baseline_config,
    lopo_cv_predictions,
    make_classifier,
    permutation_null_test,
    profile_key,
)


def _sample_config(tmp_path: Path, plates: list[str] | None = None) -> BaselineConfig:
    return BaselineConfig(
        bucket="cellpainting-gallery",
        dataset="cpg0000-jump-pilot",
        source_name="source_4",
        batch="2020_11_04_CPJUMP1",
        plates=plates or ["BR00116991", "BR00116992"],
        wells=["A01", "A03"],
        raw_dir=tmp_path / "raw",
    )


class FakeS3:
    """In-memory stand-in for the boto3 S3 client -- no network involved."""

    def __init__(self, objects: dict[str, str]):
        self._objects = objects  # key -> plain-text CSV (will be gzipped on "get")

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self._objects:
            raise KeyError(Key)
        body = gzip.compress(self._objects[Key].encode("utf-8"))
        return {"Body": io.BytesIO(body)}


def _plate_csv(rows: list[tuple[str, str, float, float]]) -> str:
    lines = ["Metadata_Well,Metadata_broad_sample,Feature_1,Feature_2"]
    for well, brd, f1, f2 in rows:
        lines.append(f"{well},{brd},{f1},{f2}")
    return "\n".join(lines) + "\n"


def test_load_baseline_config_reads_the_real_module_config():
    cfg = load_baseline_config(DEFAULT_CONFIG_PATH)
    assert cfg.plates[0] == "BR00116991"
    assert cfg.plates[1:] == ["BR00116992", "BR00116993", "BR00116994"]
    assert len(cfg.wells) == 14


def test_profile_key_matches_the_known_good_key_for_br00116991(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991"])
    assert profile_key(cfg, "BR00116991") == (
        "cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/"
        "BR00116991/BR00116991_normalized_feature_select_batch.csv.gz"
    )


def test_filtered_plate_path_is_scoped_per_plate(tmp_path: Path):
    cfg = _sample_config(tmp_path)
    p91 = filtered_plate_path(cfg, "BR00116991")
    p92 = filtered_plate_path(cfg, "BR00116992")
    assert p91 != p92
    assert p91.name == "BR00116991_normalized_feature_select_batch.filtered.csv"


def test_assemble_dataset_pools_plates_and_builds_groups(tmp_path: Path):
    cfg = _sample_config(tmp_path)
    objects = {
        profile_key(cfg, "BR00116991"): _plate_csv(
            [("A01", "BRD-1", 0.1, 0.2), ("A03", "BRD-3", 0.3, 0.4)]
        ),
        profile_key(cfg, "BR00116992"): _plate_csv(
            [("A01", "BRD-1", 1.1, 1.2), ("A03", "BRD-3", 1.3, 1.4)]
        ),
    }
    s3 = FakeS3(objects)

    X, y, groups, feature_names = assemble_dataset(cfg, s3=s3)

    assert X.shape == (4, 2)
    assert feature_names == ["Feature_1", "Feature_2"]
    assert sorted(set(y)) == ["BRD-1", "BRD-3"]
    assert sorted(set(groups)) == ["BR00116991", "BR00116992"]
    # idempotent local cache was written for both plates
    assert filtered_plate_path(cfg, "BR00116991").exists()
    assert filtered_plate_path(cfg, "BR00116992").exists()


def test_assemble_dataset_resumes_from_cache_without_refetching(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991"])
    objects = {
        profile_key(cfg, "BR00116991"): _plate_csv(
            [("A01", "BRD-1", 0.1, 0.2), ("A03", "BRD-3", 0.3, 0.4)]
        ),
    }
    s3 = FakeS3(objects)
    assemble_dataset(cfg, s3=s3)

    s3_that_would_fail = FakeS3({})  # no objects -- would raise if re-fetched
    X, y, groups, _ = assemble_dataset(cfg, s3=s3_that_would_fail)

    assert X.shape == (2, 2)


def test_assemble_dataset_rejects_mismatched_schema_across_plates(tmp_path: Path):
    cfg = _sample_config(tmp_path)
    objects = {
        profile_key(cfg, "BR00116991"): _plate_csv(
            [("A01", "BRD-1", 0.1, 0.2), ("A03", "BRD-3", 0.3, 0.4)]
        ),
        profile_key(cfg, "BR00116992"): (
            "Metadata_Well,Metadata_broad_sample,Feature_1\nA01,BRD-1,1.1\nA03,BRD-3,1.3\n"
        ),
    }
    s3 = FakeS3(objects)

    with pytest.raises(ValueError, match="schema"):
        assemble_dataset(cfg, s3=s3)


def test_assemble_dataset_rejects_wrong_well_count(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991"])
    objects = {
        profile_key(cfg, "BR00116991"): _plate_csv([("A01", "BRD-1", 0.1, 0.2)]),  # missing A03
    }
    s3 = FakeS3(objects)

    with pytest.raises(ValueError, match="expected 2"):
        assemble_dataset(cfg, s3=s3)


def _synthetic_lopo_dataset(
    seed: int = 0, n_classes: int = 4, n_plates: int = 4, informative: bool = True
):
    rng = np.random.default_rng(seed)
    classes = [f"BRD-{i}" for i in range(n_classes)]
    class_centers = rng.normal(size=(n_classes, 5)) * (5.0 if informative else 0.0)

    X, y, groups = [], [], []
    for plate_i in range(n_plates):
        plate = f"PLATE-{plate_i}"
        for c_i, label in enumerate(classes):
            noise = rng.normal(size=5)
            X.append(class_centers[c_i] + noise)
            y.append(label)
            groups.append(plate)
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def test_make_classifier_is_fittable_pipeline():
    X, y, _ = _synthetic_lopo_dataset()
    clf = make_classifier(seed=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)


def test_lopo_cv_predictions_recovers_informative_signal():
    X, y, groups = _synthetic_lopo_dataset(informative=True)
    y_pred = lopo_cv_predictions(X, y, groups, seed=0)
    accuracy = (y_pred == y).mean()
    assert accuracy > 0.9  # well-separated classes should be nearly perfectly recovered out-of-fold


def test_evaluate_reports_accuracy_and_macro_f1():
    X, y, groups = _synthetic_lopo_dataset(informative=True)
    metrics = evaluate(X, y, groups, seed=0)
    assert metrics["accuracy"] > 0.9
    assert metrics["macro_f1"] > 0.9


def test_permutation_null_test_rejects_null_for_informative_features():
    X, y, groups = _synthetic_lopo_dataset(informative=True)
    result = permutation_null_test(X, y, groups, n_permutations=200, seed=0)
    assert result["observed_accuracy"] > 0.9
    assert result["p_value"] < 0.05  # real signal should look nothing like the shuffled null


def test_permutation_null_test_does_not_reject_null_for_uninformative_features():
    X, y, groups = _synthetic_lopo_dataset(informative=False)
    result = permutation_null_test(X, y, groups, n_permutations=200, seed=0)
    assert (
        result["p_value"] > 0.05
    )  # pure noise should look like the shuffled null most of the time


@pytest.mark.slow
def test_assemble_dataset_matches_live_bucket_for_all_four_plates(tmp_path: Path):
    """Exercises the real, public S3 bucket (no credentials needed).

    Regression check that BR00116991-BR00116994 still pool into a clean
    n=56, 14-class, 838-feature dataset with a balanced 14-per-plate group
    structure, as verified manually on 2026-07-20 (see docs/build-log.md).
    """
    cfg = load_baseline_config(DEFAULT_CONFIG_PATH)
    cfg = BaselineConfig(
        bucket=cfg.bucket,
        dataset=cfg.dataset,
        source_name=cfg.source_name,
        batch=cfg.batch,
        plates=cfg.plates,
        wells=cfg.wells,
        raw_dir=tmp_path / "raw",
    )

    X, y, groups, feature_names = assemble_dataset(cfg)

    assert X.shape == (56, 838)
    assert len(set(y)) == 14
    assert sorted(set(groups)) == ["BR00116991", "BR00116992", "BR00116993", "BR00116994"]
    for plate in cfg.plates:
        assert (groups == plate).sum() == 14
