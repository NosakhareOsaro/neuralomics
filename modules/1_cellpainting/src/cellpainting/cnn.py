"""CNN: compound-identity classification (14-way) on raw 5-channel Cell
Painting pixel composites -- see the module README's Model architecture and
Evaluation plan sections, and docs/build-log.md, for the decisions made
before this code was written: pretrained EfficientNet-B0/ResNet50 backbone
with a 5-channel first conv layer (not from-scratch, not per-channel
replication); 4-fold leave-one-plate-out CV across BR00116991-BR00116994
with a rotating internal-validation plate for early stopping; loss-based
early stopping (patience=5); geometry-only augmentation (flip/rotate/crop,
no intensity jitter); illumination-corrected, per-channel min-max
normalization fit on each fold's training plates only; a closed-form
binomial significance test plus a small label-shuffle sanity check in place
of the tabular baseline's 1000-permutation test (infeasible to retrain a CNN
that many times).

`python -m cellpainting.cnn` is a complete, runnable CLI (config check, optional
`--fetch-images`, 4-fold training, pooled metrics, binomial test, results.json
-- same conventions as download.py/profiles.py/baseline.py), but has NOT been
run for real in this repository's local dev environment: per the project's
original compute-target decision (docs/build-log.md, repository scaffold),
real CNN training happens on Colab Pro, not here, and images for
BR00116992-BR00116994 haven't been downloaded yet (only their CellProfiler
profiles were, for the tabular baseline). Every function here is exercised by
fast unit tests against synthetic data and tiny stub models instead, plus one
live-bucket (`slow`) regression test for the new multi-plate image fetch --
see test_cnn.py.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from torch import nn

from cellpainting.download import (
    DEFAULT_CONFIG_PATH,
    Compound,
    download_key,
    list_keys,
    make_s3_client,
    well_to_row_col,
)
from common.seeding import DEFAULT_SEED, set_global_seed

INPUT_SIZE = 224  # the pretrained backbones' expected input resolution


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CNNConfig:
    bucket: str
    dataset: str
    source_name: str
    batch: str
    plates: list[str]
    measurements: dict[str, str]
    compounds: list[Compound]
    model_input_channels: list[str]
    channel_stain: dict[str, str]
    sites_per_well: int
    raw_dir: Path


def load_cnn_config(path: Path = DEFAULT_CONFIG_PATH) -> CNNConfig:
    raw = yaml.safe_load(path.read_text())
    src = raw["source"]
    plates = [src["plate"], *raw.get("replicate_plates", [])]
    return CNNConfig(
        bucket=src["bucket"],
        dataset=src["dataset"],
        source_name=src["source_name"],
        batch=src["batch"],
        plates=plates,
        measurements=raw["measurements"],
        compounds=[Compound(**c) for c in raw["compounds"]],
        model_input_channels=raw["model_input_channels"],
        channel_stain={k: v["stain"] for k, v in raw["channels"].items()},
        sites_per_well=raw["sites_per_well"],
        raw_dir=Path(raw["output"]["raw_dir"]),
    )


# --------------------------------------------------------------------------
# Fold plan: 4-fold leave-one-plate-out, rotating internal-validation plate.
# See docs/build-log.md ("CNN train/validation split and significance-
# testing strategy") for why -- exact rotation pinned down there.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    test_plate: str
    internal_val_plate: str
    train_plates: list[str]


def fold_plan(plates: list[str]) -> list[FoldSpec]:
    n = len(plates)
    folds = []
    for i in range(n):
        test_plate = plates[i]
        internal_val_plate = plates[(i + 1) % n]
        train_plates = [p for p in plates if p not in (test_plate, internal_val_plate)]
        folds.append(FoldSpec(test_plate, internal_val_plate, train_plates))
    return folds


def all_wells(cfg: CNNConfig) -> list[str]:
    return [c.well for c in cfg.compounds]


def label_for_well(cfg: CNNConfig, well: str) -> str:
    return next(c.broad_sample for c in cfg.compounds if c.well == well)


def label_index(cfg: CNNConfig) -> dict[str, int]:
    """compound broad_sample -> a stable 0..13 class index, sorted for determinism."""
    return {label: i for i, label in enumerate(sorted({c.broad_sample for c in cfg.compounds}))}


# --------------------------------------------------------------------------
# Image loading: per-plate S3/local paths, illumination correction.
# --------------------------------------------------------------------------


def image_key_prefix(cfg: CNNConfig, plate: str) -> str:
    measurement = cfg.measurements[plate]
    return (
        f"{cfg.dataset}/{cfg.source_name}/images/{cfg.batch}/images/"
        f"{plate}__{measurement}/Images/"
    )


def illum_key_prefix(cfg: CNNConfig, plate: str) -> str:
    return f"{cfg.dataset}/{cfg.source_name}/images/{cfg.batch}/illum/{plate}/"


def well_image_key_prefix(cfg: CNNConfig, plate: str, well: str) -> str:
    row, col = well_to_row_col(well)
    return f"{image_key_prefix(cfg, plate)}r{row:02d}c{col:02d}"


def image_dir(cfg: CNNConfig, plate: str) -> Path:
    return cfg.raw_dir / image_key_prefix(cfg, plate)


def illum_dir(cfg: CNNConfig, plate: str) -> Path:
    return cfg.raw_dir / illum_key_prefix(cfg, plate)


def local_image_path(cfg: CNNConfig, plate: str, well: str, site: int, channel: str) -> Path:
    row, col = well_to_row_col(well)
    filename = f"r{row:02d}c{col:02d}f{site:02d}p01-{channel}sk1fk1fl1.tiff"
    return image_dir(cfg, plate) / filename


def local_illum_path(cfg: CNNConfig, plate: str, stain: str) -> Path:
    return illum_dir(cfg, plate) / f"{plate}_Illum{stain}.npy"


def load_illum_functions(cfg: CNNConfig, plate: str) -> dict[str, np.ndarray]:
    """{channel_key: illum_function} for cfg.model_input_channels on one plate."""
    return {
        ch: np.load(local_illum_path(cfg, plate, cfg.channel_stain[ch]))
        for ch in cfg.model_input_channels
    }


def fetch_plate_images(cfg: CNNConfig, plate: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Fetch one plate's site images + illumination-correction files for our
    14 wells. Reuses `download.py`'s S3 primitives, generalized to any of
    our 4 plates -- `download.py` itself only ever fetches `source.plate`
    (`BR00116991`). Idempotent (`download_key` skips files already on disk),
    same convention as `download.py`/`profiles.py`. Not called automatically
    by anything in this module -- fetching ~2.4GB/plate is a deliberate,
    explicit action, same as the original BR00116991 download."""
    s3 = make_s3_client()
    manifest: dict[str, Any] = {"plate": plate, "dry_run": dry_run, "wells": []}
    for well in all_wells(cfg):
        objects = list_keys(s3, cfg.bucket, well_image_key_prefix(cfg, plate, well))
        if not dry_run:
            for obj in objects:
                download_key(s3, cfg.bucket, obj["Key"], cfg.raw_dir)
        manifest["wells"].append({"well": well, "n_files": len(objects)})

    illum_objects = list_keys(s3, cfg.bucket, illum_key_prefix(cfg, plate))
    if not dry_run:
        for obj in illum_objects:
            download_key(s3, cfg.bucket, obj["Key"], cfg.raw_dir)
    manifest["illum_files"] = len(illum_objects)
    return manifest


def load_site_composite(
    cfg: CNNConfig, plate: str, well: str, site: int, illum: dict[str, np.ndarray]
) -> np.ndarray:
    """One site's illumination-corrected, NOT-yet-normalized 5-channel composite.

    Returns (len(model_input_channels), H, W) float32. Division by the
    per-plate illumination function is standard flatfield/vignetting
    correction (CellProfiler/JUMP convention): the illum function is a
    smooth, plate-specific estimate of the multiplicative, purely-optical
    brightness pattern across the field, unrelated to biology.
    """
    channels = []
    for ch in cfg.model_input_channels:
        raw = np.asarray(Image.open(local_image_path(cfg, plate, well, site, ch)), dtype=np.float32)
        channels.append(raw / illum[ch])
    return np.stack(channels, axis=0)


# --------------------------------------------------------------------------
# Normalization: per-channel min-max to [-1, 1], fit on training plates only.
# --------------------------------------------------------------------------


def compute_normalization_stats(
    cfg: CNNConfig, plates: list[str], wells: list[str]
) -> dict[str, tuple[float, float]]:
    """{channel: (min, max)} over every site composite in `plates` -- never
    the internal-val or test plate for the fold this is computed for (see
    docs/build-log.md: computing stats from held-out data would leak
    test-time information into preprocessing)."""
    n_channels = len(cfg.model_input_channels)
    mins = np.full(n_channels, np.inf, dtype=np.float64)
    maxs = np.full(n_channels, -np.inf, dtype=np.float64)
    for plate in plates:
        illum = load_illum_functions(cfg, plate)
        for well in wells:
            for site in range(1, cfg.sites_per_well + 1):
                composite = load_site_composite(cfg, plate, well, site, illum)
                mins = np.minimum(mins, composite.reshape(n_channels, -1).min(axis=1))
                maxs = np.maximum(maxs, composite.reshape(n_channels, -1).max(axis=1))
    return {ch: (float(mins[i]), float(maxs[i])) for i, ch in enumerate(cfg.model_input_channels)}


def normalize_composite(
    composite: np.ndarray, stats: dict[str, tuple[float, float]], channels: list[str]
) -> np.ndarray:
    """Per-channel min-max scale to [-1, 1] using pre-fit (train-only) stats."""
    out = np.empty_like(composite, dtype=np.float32)
    for i, ch in enumerate(channels):
        lo, hi = stats[ch]
        out[i] = 2.0 * (composite[i] - lo) / (hi - lo) - 1.0
    return out


# --------------------------------------------------------------------------
# Augmentation: geometry only (flip / 90-degree rotation / modest crop-
# resize), applied identically across all channels together. Deliberately
# no brightness/contrast/intensity jitter -- see docs/build-log.md for why.
# --------------------------------------------------------------------------


def augment_composite(composite: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """composite: (C, H, W) tensor. Returns a spatially-augmented copy with
    every per-pixel value unchanged in magnitude, only rearranged."""
    if torch.rand(1, generator=generator).item() < 0.5:
        composite = torch.flip(composite, dims=[-1])
    if torch.rand(1, generator=generator).item() < 0.5:
        composite = torch.flip(composite, dims=[-2])
    k = int(torch.randint(0, 4, (1,), generator=generator).item())
    composite = torch.rot90(composite, k, dims=[-2, -1])
    return random_crop_resize(composite, generator, min_scale=0.9)


def random_crop_resize(
    composite: torch.Tensor, generator: torch.Generator, min_scale: float = 0.9
) -> torch.Tensor:
    c, h, w = composite.shape
    scale = min_scale + (1.0 - min_scale) * torch.rand(1, generator=generator).item()
    crop_h, crop_w = max(1, int(h * scale)), max(1, int(w * scale))
    top = int(torch.randint(0, h - crop_h + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - crop_w + 1, (1,), generator=generator).item())
    cropped = composite[:, top : top + crop_h, left : left + crop_w]
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)


def resize_to_input_size(composite: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        composite.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


# --------------------------------------------------------------------------
# Dataset: ties image loading -> illumination correction -> normalization ->
# (train-only) augmentation -> resize into one unit for real DataLoaders.
# --------------------------------------------------------------------------


class SiteCompositeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: CNNConfig,
        plates: list[str],
        norm_stats: dict[str, tuple[float, float]],
        labels: dict[str, int],
        *,
        augment: bool = False,
        seed: int = DEFAULT_SEED,
    ):
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.labels = labels
        self.augment = augment
        self.generator = torch.Generator().manual_seed(seed)
        self.illum = {plate: load_illum_functions(cfg, plate) for plate in plates}
        self.index: list[tuple[str, str, int]] = [
            (plate, well, site)
            for plate in plates
            for well in all_wells(cfg)
            for site in range(1, cfg.sites_per_well + 1)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        plate, well, site = self.index[i]
        composite = load_site_composite(self.cfg, plate, well, site, self.illum[plate])
        composite = normalize_composite(composite, self.norm_stats, self.cfg.model_input_channels)
        tensor = torch.from_numpy(composite)
        if self.augment:
            tensor = augment_composite(tensor, self.generator)
        tensor = resize_to_input_size(tensor)
        label = self.labels[label_for_well(self.cfg, well)]
        return tensor, label


def build_fold_datasets(
    cfg: CNNConfig, fold: FoldSpec, labels: dict[str, int]
) -> tuple[SiteCompositeDataset, SiteCompositeDataset, SiteCompositeDataset]:
    """(train, internal_val, test) datasets for one fold. Normalization stats
    are fit once on the fold's training plates only, then reused (never
    refit) for internal-val and test -- see docs/build-log.md."""
    stats = compute_normalization_stats(cfg, fold.train_plates, all_wells(cfg))
    train_ds = SiteCompositeDataset(cfg, fold.train_plates, stats, labels, augment=True)
    val_ds = SiteCompositeDataset(cfg, [fold.internal_val_plate], stats, labels, augment=False)
    test_ds = SiteCompositeDataset(cfg, [fold.test_plate], stats, labels, augment=False)
    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------------
# Model: pretrained backbone, first conv layer expanded to 5 channels.
# --------------------------------------------------------------------------


def expand_conv_to_5_channels(conv: nn.Conv2d, n_channels: int = 5) -> nn.Conv2d:
    """Warm-started (not random) expansion of a pretrained 3-channel (RGB)
    first conv layer to `n_channels`. Copies the pretrained RGB kernels into
    the first 3 input-channel slots and fills the rest with their mean --
    see docs/build-log.md ("First-conv-layer adaptation")."""
    if conv.in_channels != 3:
        raise ValueError(
            f"expected a 3-channel (RGB) conv layer, got in_channels={conv.in_channels}"
        )
    new_conv = nn.Conv2d(
        n_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = conv.weight
        mean_kernel = conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:n_channels] = mean_kernel.repeat(1, n_channels - 3, 1, 1)
        if conv.bias is not None:
            assert new_conv.bias is not None
            new_conv.bias.copy_(conv.bias)
    return new_conv


def build_model(
    num_classes: int,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    n_channels: int = 5,
) -> nn.Module:
    import torchvision

    if backbone == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.efficientnet_b0(weights=weights)
        model.features[0][0] = expand_conv_to_5_channels(model.features[0][0], n_channels)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        model.conv1 = expand_conv_to_5_channels(model.conv1, n_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {backbone!r}")
    return model


# --------------------------------------------------------------------------
# Early stopping: monitor internal-val loss, not accuracy/macro-F1 (too
# noisy at n=126) -- see docs/build-log.md.
# --------------------------------------------------------------------------


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-3
    max_epochs: int = 50
    best_loss: float = float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    def step(self, epoch: int, val_loss: float) -> bool:
        """Record one epoch's val loss; returns True if training should stop now."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience or epoch + 1 >= self.max_epochs


# --------------------------------------------------------------------------
# Training loop. `model_factory`/`data_loader_factory` are injected so tests
# can exercise the loop with tiny stub models/data instead of the real
# pretrained backbone and real (currently only partially downloaded) images.
# --------------------------------------------------------------------------


def train_one_fold(
    fold: FoldSpec,
    model: nn.Module,
    train_loader: Any,
    internal_val_loader: Any,
    test_loader: Any,
    *,
    lr: float = 1e-4,
    patience: int = 5,
    min_delta: float = 1e-3,
    max_epochs: int = 50,
    device: str = "cpu",
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience, min_delta=min_delta, max_epochs=max_epochs)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    epoch = 0
    while True:
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in internal_val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(loss_fn(model(x), y).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        should_stop = stopper.step(epoch, val_loss)
        if stopper.best_epoch == epoch:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epoch += 1
        if should_stop:
            break

    model.load_state_dict(best_state)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    return {
        "test_plate": fold.test_plate,
        "internal_val_plate": fold.internal_val_plate,
        "train_plates": fold.train_plates,
        "n_epochs": epoch,
        "best_epoch": stopper.best_epoch,
        "best_internal_val_loss": stopper.best_loss,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_lopo_folds(
    cfg: CNNConfig,
    *,
    backbone: str = "efficientnet_b0",
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Runs all 4 folds (docs/build-log.md's rotation), pools every fold's
    test predictions (each of the 504 site composites is a test prediction
    exactly once), and reports accuracy/macro-F1 the same way
    `baseline.py`'s `evaluate()` does for the tabular baseline."""
    labels = label_index(cfg)
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    fold_results = []

    for fold in fold_plan(cfg.plates):
        train_ds, val_ds, test_ds = build_fold_datasets(cfg, fold, labels)
        model = build_model(num_classes=len(labels), backbone=backbone)
        result = train_one_fold(
            fold,
            model,
            torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(val_ds, batch_size=batch_size),
            torch.utils.data.DataLoader(test_ds, batch_size=batch_size),
            device=device,
        )
        fold_results.append(result)
        all_y_true.extend(result["y_true"])
        all_y_pred.extend(result["y_pred"])

    n_correct = sum(t == p for t, p in zip(all_y_true, all_y_pred, strict=True))
    n_total = len(all_y_true)
    return {
        "n_samples": n_total,
        "n_classes": len(labels),
        "accuracy": n_correct / n_total,
        "macro_f1": float(f1_score(all_y_true, all_y_pred, average="macro")),
        "folds": fold_results,
        "binomial_test": binomial_significance_test(n_correct, n_total),
    }


# --------------------------------------------------------------------------
# Significance: closed-form binomial test (primary) + label-shuffle sanity
# check (secondary) -- not a 1000-permutation test, see docs/build-log.md.
# --------------------------------------------------------------------------


def binomial_significance_test(
    n_correct: int, n_total: int, chance_p: float = 1 / 14
) -> dict[str, float]:
    result = binomtest(n_correct, n_total, chance_p, alternative="greater")
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "chance_p": chance_p,
        "observed_accuracy": n_correct / n_total,
        "p_value": float(result.pvalue),
    }


def label_shuffle_sanity_check(
    run_all_folds: Callable[[np.ndarray], float],
    y: np.ndarray,
    *,
    n_shuffles: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[float]:
    """Runs `run_all_folds` (the full pipeline, pooled-accuracy return value)
    `n_shuffles` times on label-permuted data. Not enough shuffles for a
    real p-value -- a pipeline-correctness sanity check, not a significance
    test. See docs/build-log.md."""
    rng = np.random.default_rng(seed)
    return [run_all_folds(rng.permutation(y)) for _ in range(n_shuffles)]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--backbone", choices=["efficientnet_b0", "resnet50"], default="efficientnet_b0"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fetch-images",
        action="store_true",
        help=(
            "fetch any missing plates' images (~2.4GB/plate) before training -- "
            "an explicit opt-in, not done by default"
        ),
    )
    args = parser.parse_args(argv)

    set_global_seed(args.seed)
    cfg = load_cnn_config(args.config)

    if args.fetch_images:
        for plate in cfg.plates:
            if not image_dir(cfg, plate).exists():
                manifest = fetch_plate_images(cfg, plate)
                print(f"fetched {plate}: {manifest}")

    missing = [p for p in cfg.plates if not image_dir(cfg, p).exists()]
    if missing:
        raise SystemExit(
            f"Images not downloaded locally for: {', '.join(missing)}. Re-run with "
            "--fetch-images to fetch them (~2.4GB/plate, an explicit opt-in), or note "
            "that this module's real training run is scoped to Colab Pro, not this "
            "local environment (see docs/build-log.md)."
        )

    result = run_lopo_folds(
        cfg, backbone=args.backbone, batch_size=args.batch_size, device=args.device, seed=args.seed
    )

    dest = cfg.raw_dir / "cnn" / "results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"4-fold LOPO over {result['n_samples']} samples / {result['n_classes']} classes: "
        f"accuracy={result['accuracy']:.3f}, macro-F1={result['macro_f1']:.3f}\n"
        f"Binomial test vs. chance ~= 1/14: p={result['binomial_test']['p_value']:.4f}\n"
        f"-> {dest}"
    )


if __name__ == "__main__":
    main()
