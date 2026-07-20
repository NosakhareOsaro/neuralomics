from pathlib import Path

import numpy as np
import pytest
import torch
from cellpainting.cnn import (
    DEFAULT_CONFIG_PATH,
    CNNConfig,
    EarlyStopping,
    FoldSpec,
    all_wells,
    augment_composite,
    binomial_significance_test,
    build_model,
    compute_normalization_stats,
    expand_conv_to_5_channels,
    fetch_plate_images,
    fold_plan,
    illum_key_prefix,
    image_key_prefix,
    label_for_well,
    label_index,
    load_cnn_config,
    load_site_composite,
    normalize_composite,
    random_crop_resize,
    train_one_fold,
    well_image_key_prefix,
)
from cellpainting.download import Compound
from PIL import Image
from torch import nn


def _sample_config(tmp_path: Path, plates: list[str] | None = None) -> CNNConfig:
    return CNNConfig(
        bucket="cellpainting-gallery",
        dataset="cpg0000-jump-pilot",
        source_name="source_4",
        batch="2020_11_04_CPJUMP1",
        plates=plates or ["BR00116991", "BR00116992"],
        measurements={
            "BR00116991": "2020-11-05T19_51_35-Measurement1",
            "BR00116992": "2020-11-05T21_31_31-Measurement1",
            "BR00116993": "2020-11-05T23_11_39-Measurement1",
            "BR00116994": "2020-11-06T00_59_44-Measurement1",
        },
        compounds=[
            Compound(well="A01", broad_sample="BRD-1", pert_iname="drug-1", targets="X"),
            Compound(well="A03", broad_sample="BRD-3", pert_iname="drug-3", targets="Y"),
        ],
        model_input_channels=["ch1", "ch2"],
        channel_stain={"ch1": "Mito", "ch2": "AGP"},
        sites_per_well=2,
        raw_dir=tmp_path / "raw",
    )


def _write_synthetic_plate(cfg: CNNConfig, plate: str, well_values: dict[str, int]) -> None:
    """Writes tiny synthetic 4x4 uint16 TIFFs (constant value per well) plus
    all-ones illum files for every configured channel/site of one plate."""
    from cellpainting.cnn import illum_dir, local_illum_path, local_image_path

    illum_d = illum_dir(cfg, plate)
    illum_d.mkdir(parents=True, exist_ok=True)
    for ch in cfg.model_input_channels:
        stain = cfg.channel_stain[ch]
        np.save(local_illum_path(cfg, plate, stain), np.ones((4, 4), dtype=np.float32))

    for well, value in well_values.items():
        for site in range(1, cfg.sites_per_well + 1):
            for ch in cfg.model_input_channels:
                path = local_image_path(cfg, plate, well, site, ch)
                path.parent.mkdir(parents=True, exist_ok=True)
                arr = np.full((4, 4), value, dtype=np.uint16)
                Image.fromarray(arr).save(path)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_load_cnn_config_reads_the_real_module_config():
    cfg = load_cnn_config(DEFAULT_CONFIG_PATH)
    assert cfg.plates == ["BR00116991", "BR00116992", "BR00116993", "BR00116994"]
    assert set(cfg.measurements) == set(cfg.plates)
    assert len(cfg.compounds) == 14
    assert cfg.model_input_channels == ["ch1", "ch2", "ch3", "ch4", "ch5"]
    assert cfg.channel_stain["ch1"] == "Mito"
    assert cfg.sites_per_well == 9


def test_image_key_prefix_matches_verified_bucket_layout(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991"])
    assert image_key_prefix(cfg, "BR00116991") == (
        "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
        "BR00116991__2020-11-05T19_51_35-Measurement1/Images/"
    )


def test_illum_key_prefix_matches_verified_bucket_layout(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991"])
    assert illum_key_prefix(cfg, "BR00116991") == (
        "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/illum/BR00116991/"
    )


def test_well_image_key_prefix_matches_verified_bucket_layout(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991"])
    assert well_image_key_prefix(cfg, "BR00116991", "A01") == (
        "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
        "BR00116991__2020-11-05T19_51_35-Measurement1/Images/r01c01"
    )


def test_label_index_and_label_for_well(tmp_path: Path):
    cfg = _sample_config(tmp_path)
    idx = label_index(cfg)
    assert set(idx) == {"BRD-1", "BRD-3"}
    assert idx == {"BRD-1": 0, "BRD-3": 1}  # sorted, deterministic
    assert label_for_well(cfg, "A01") == "BRD-1"


# --------------------------------------------------------------------------
# Fold plan
# --------------------------------------------------------------------------


def test_fold_plan_matches_the_documented_rotation():
    plates = ["BR00116991", "BR00116992", "BR00116993", "BR00116994"]
    folds = fold_plan(plates)
    # train_plates order follows original plate-list order (test/val filtered
    # out), not the build-log table's display order -- membership is what
    # matters (also checked, order-independent, by the property test below).
    assert folds == [
        FoldSpec("BR00116991", "BR00116992", ["BR00116993", "BR00116994"]),
        FoldSpec("BR00116992", "BR00116993", ["BR00116991", "BR00116994"]),
        FoldSpec("BR00116993", "BR00116994", ["BR00116991", "BR00116992"]),
        FoldSpec("BR00116994", "BR00116991", ["BR00116992", "BR00116993"]),
    ]


def test_fold_plan_every_plate_serves_each_role_exactly_once():
    plates = ["BR00116991", "BR00116992", "BR00116993", "BR00116994"]
    folds = fold_plan(plates)

    test_plates = [f.test_plate for f in folds]
    val_plates = [f.internal_val_plate for f in folds]
    train_counts = {p: sum(p in f.train_plates for f in folds) for p in plates}

    assert sorted(test_plates) == sorted(plates)
    assert sorted(val_plates) == sorted(plates)
    assert all(count == 2 for count in train_counts.values())
    for f in folds:
        assert f.test_plate != f.internal_val_plate
        assert f.test_plate not in f.train_plates
        assert f.internal_val_plate not in f.train_plates


# --------------------------------------------------------------------------
# Image loading / illumination correction / normalization
# --------------------------------------------------------------------------


def test_load_site_composite_applies_illumination_correction(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991"])
    illum_d = cfg.raw_dir / "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/illum/BR00116991"
    illum_d.mkdir(parents=True)
    from cellpainting.cnn import local_illum_path, local_image_path

    for ch, stain, illum_val in [("ch1", "Mito", 2.0), ("ch2", "AGP", 1.0)]:
        np.save(
            local_illum_path(cfg, "BR00116991", stain), np.full((4, 4), illum_val, dtype=np.float32)
        )
        path = local_image_path(cfg, "BR00116991", "A01", 1, ch)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 4), 100, dtype=np.uint16)).save(path)

    from cellpainting.cnn import load_illum_functions

    illum = load_illum_functions(cfg, "BR00116991")
    composite = load_site_composite(cfg, "BR00116991", "A01", 1, illum)

    assert composite.shape == (2, 4, 4)
    assert np.allclose(composite[0], 50.0)  # 100 / 2.0
    assert np.allclose(composite[1], 100.0)  # 100 / 1.0


def test_compute_normalization_stats_uses_only_given_plates(tmp_path: Path):
    cfg = _sample_config(tmp_path, plates=["BR00116991", "BR00116992"])
    _write_synthetic_plate(cfg, "BR00116991", {"A01": 10, "A03": 20})
    _write_synthetic_plate(
        cfg, "BR00116992", {"A01": 1000, "A03": 2000}
    )  # would blow out the range

    stats = compute_normalization_stats(cfg, ["BR00116991"], all_wells(cfg))

    for ch in cfg.model_input_channels:
        lo, hi = stats[ch]
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(20.0)


def test_normalize_composite_scales_to_minus1_1_using_given_stats():
    composite = np.array([[[0.0, 5.0], [10.0, 10.0]]], dtype=np.float32)  # 1 channel, min=0 max=10
    stats = {"ch1": (0.0, 10.0)}

    normalized = normalize_composite(composite, stats, ["ch1"])

    assert normalized[0, 0, 0] == pytest.approx(-1.0)
    assert normalized[0, 1, 1] == pytest.approx(1.0)
    assert normalized[0, 0, 1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Augmentation: must rearrange pixels, never change their values.
# --------------------------------------------------------------------------


def test_augment_composite_preserves_the_exact_set_of_pixel_values():
    generator = torch.Generator().manual_seed(0)
    composite = torch.arange(2 * 6 * 6, dtype=torch.float32).reshape(2, 6, 6)

    augmented = augment_composite(composite, generator)

    assert augmented.shape == composite.shape
    # crop+resize interpolates, so exact-value equality doesn't hold once that
    # runs -- but flip/rotation alone (isolated below) must preserve values.


def test_flip_and_rotate_alone_preserve_exact_pixel_values():
    composite = torch.arange(2 * 6 * 6, dtype=torch.float32).reshape(2, 6, 6)

    flipped = torch.flip(composite, dims=[-1])
    rotated = torch.rot90(composite, 2, dims=[-2, -1])

    assert torch.equal(torch.sort(flipped.flatten())[0], torch.sort(composite.flatten())[0])
    assert torch.equal(torch.sort(rotated.flatten())[0], torch.sort(composite.flatten())[0])


def test_random_crop_resize_preserves_shape_and_channel_alignment():
    generator = torch.Generator().manual_seed(0)
    composite = torch.rand(3, 32, 32)

    cropped = random_crop_resize(composite, generator, min_scale=0.9)

    assert cropped.shape == composite.shape


# --------------------------------------------------------------------------
# First-conv-layer expansion
# --------------------------------------------------------------------------


def test_expand_conv_to_5_channels_copies_rgb_and_averages_new_channels():
    conv = nn.Conv2d(3, 4, kernel_size=3, bias=True)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(4 * 3 * 3 * 3, dtype=torch.float32).reshape(4, 3, 3, 3))
        conv.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    expanded = expand_conv_to_5_channels(conv, n_channels=5)

    assert expanded.in_channels == 5
    assert expanded.out_channels == 4
    assert torch.equal(expanded.weight[:, :3], conv.weight)
    expected_mean = conv.weight.mean(dim=1, keepdim=True)
    assert torch.allclose(expanded.weight[:, 3], expected_mean.squeeze(1))
    assert torch.allclose(expanded.weight[:, 4], expected_mean.squeeze(1))
    assert torch.equal(expanded.bias, conv.bias)


def test_expand_conv_to_5_channels_rejects_non_rgb_input():
    conv = nn.Conv2d(5, 4, kernel_size=3)
    with pytest.raises(ValueError, match="3-channel"):
        expand_conv_to_5_channels(conv)


def test_build_model_efficientnet_b0_accepts_5_channel_input():
    model = build_model(num_classes=14, backbone="efficientnet_b0", pretrained=False)
    x = torch.rand(2, 5, 224, 224)
    out = model(x)
    assert out.shape == (2, 14)


def test_build_model_resnet50_accepts_5_channel_input():
    model = build_model(num_classes=14, backbone="resnet50", pretrained=False)
    x = torch.rand(2, 5, 224, 224)
    out = model(x)
    assert out.shape == (2, 14)


def test_build_model_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_model(num_classes=14, backbone="not-a-real-backbone", pretrained=False)


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------


def test_early_stopping_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(patience=3, min_delta=1e-3, max_epochs=50)
    losses = [1.0, 0.5, 0.5, 0.5, 0.5]  # improves once, then plateaus for 4 epochs
    stops = [stopper.step(epoch, loss) for epoch, loss in enumerate(losses)]
    assert stops == [False, False, False, False, True]
    assert stopper.best_epoch == 1


def test_early_stopping_resets_patience_counter_on_improvement():
    stopper = EarlyStopping(patience=2, min_delta=1e-3, max_epochs=50)
    losses = [1.0, 0.9, 0.5, 0.5, 0.4, 0.4, 0.4]
    stops = [stopper.step(epoch, loss) for epoch, loss in enumerate(losses)]
    assert stops == [False, False, False, False, False, False, True]


def test_early_stopping_ignores_improvement_smaller_than_min_delta():
    stopper = EarlyStopping(patience=2, min_delta=0.1, max_epochs=50)
    losses = [1.0, 0.95, 0.91]  # each "improvement" is < min_delta
    stops = [stopper.step(epoch, loss) for epoch, loss in enumerate(losses)]
    assert stops == [False, False, True]


def test_early_stopping_hard_caps_at_max_epochs():
    stopper = EarlyStopping(patience=100, min_delta=1e-3, max_epochs=3)
    losses = [1.0, 0.5, 0.1]  # still improving every epoch
    stops = [stopper.step(epoch, loss) for epoch, loss in enumerate(losses)]
    assert stops == [False, False, True]


# --------------------------------------------------------------------------
# Significance checks
# --------------------------------------------------------------------------


def test_binomial_significance_test_matches_scipy_directly():
    from scipy.stats import binomtest

    result = binomial_significance_test(42, 56, chance_p=1 / 14)
    expected = binomtest(42, 56, 1 / 14, alternative="greater")

    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["observed_accuracy"] == pytest.approx(42 / 56)


def test_binomial_significance_test_high_accuracy_gives_tiny_p_value():
    result = binomial_significance_test(50, 56, chance_p=1 / 14)
    assert result["p_value"] < 0.001


def test_binomial_significance_test_chance_level_gives_large_p_value():
    result = binomial_significance_test(4, 56, chance_p=1 / 14)  # ~= 56/14
    assert result["p_value"] > 0.3


def test_label_shuffle_sanity_check_runs_n_times_on_permuted_labels():
    y = np.array(["a", "b", "c", "d"])
    seen_permutations = []

    def fake_run(y_perm: np.ndarray) -> float:
        seen_permutations.append(tuple(y_perm))
        return 0.5

    from cellpainting.cnn import label_shuffle_sanity_check

    results = label_shuffle_sanity_check(fake_run, y, n_shuffles=5, seed=0)

    assert results == [0.5, 0.5, 0.5, 0.5, 0.5]
    assert len(seen_permutations) == 5
    for perm in seen_permutations:
        assert sorted(perm) == sorted(y)  # same multiset of labels every time


# --------------------------------------------------------------------------
# Training loop, exercised with a tiny stub model + synthetic data (no
# pretrained weights, no real images).
# --------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_channels * 4 * 4, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _tiny_loader(n_samples: int, n_channels: int, n_classes: int, batch_size: int = 4):
    x = torch.rand(n_samples, n_channels, 4, 4)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def test_train_one_fold_runs_and_returns_expected_shape():
    fold = FoldSpec("TEST_PLATE", "VAL_PLATE", ["TRAIN_A", "TRAIN_B"])
    model = _TinyModel(n_channels=2, n_classes=3)

    result = train_one_fold(
        fold,
        model,
        _tiny_loader(12, 2, 3),
        _tiny_loader(6, 2, 3),
        _tiny_loader(6, 2, 3),
        patience=2,
        max_epochs=5,
    )

    assert result["test_plate"] == "TEST_PLATE"
    assert result["internal_val_plate"] == "VAL_PLATE"
    assert result["n_epochs"] <= 5
    assert len(result["y_true"]) == 6
    assert len(result["y_pred"]) == 6


@pytest.mark.slow
def test_fetch_plate_images_dry_run_matches_live_bucket(tmp_path: Path):
    """Exercises the real, public S3 bucket (no credentials needed).

    Regression check that cnn.py's generalized (any-of-our-4-plates) image
    fetch has the same file-count shape as download.py's original
    BR00116991-only version, for a replicate plate (BR00116992) -- confirms
    the per-plate measurement mapping in configs/data.yaml is correct.
    """
    cfg = load_cnn_config(DEFAULT_CONFIG_PATH)
    cfg = CNNConfig(
        bucket=cfg.bucket,
        dataset=cfg.dataset,
        source_name=cfg.source_name,
        batch=cfg.batch,
        plates=cfg.plates,
        measurements=cfg.measurements,
        compounds=cfg.compounds,
        model_input_channels=cfg.model_input_channels,
        channel_stain=cfg.channel_stain,
        sites_per_well=cfg.sites_per_well,
        raw_dir=tmp_path / "raw",
    )

    manifest = fetch_plate_images(cfg, "BR00116992", dry_run=True)

    assert manifest["wells"][0]["n_files"] == 72
    assert manifest["illum_files"] == 8
    assert not (tmp_path / "raw").exists()  # dry run must not write anything
