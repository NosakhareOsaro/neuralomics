# Module 1 — Cell Painting morphological profiling

Status: in progress (scaffold). No trained model or results yet.

## Research question

Within the project's overall thesis, this module asks: how well does a CNN trained to classify mechanism-of-action (MoA) from Cell Painting images generalize to an imaging batch it never saw during training?

## Data source

**Dataset**: JUMP-CP (JUMP Cell Painting Consortium), hosted on the [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) (AWS Open Data, `s3://cellpainting-gallery`, no-sign-request public bucket). License: CC0. Full provenance in [`../../DATA_SOURCES.md`](../../DATA_SOURCES.md).

**Subset used (validation pass, not the full dataset)**: one plate, 14 compounds — within the agreed 10–15 range. This is a deliberate compute-driven substitution — see `docs/build-log.md` — chosen to validate the download → CellProfiler-feature → model pipeline end-to-end before deciding whether to scale to the originally scoped ~30–50 compound / 1–2 plate subset.

Exact subset, verified directly against the live, public, no-sign-request bucket (not assumed from documentation):

| Field | Value |
|---|---|
| Batch | `2020_11_04_CPJUMP1` |
| Plate | `BR00116991` (A549, parental cell line, 24h) |
| Platemap | `JUMP-Target-1_compound_platemap` |
| Compounds | 14 (row A, wells A01, A03–A08, A10–A16 — the platemap's non-control wells in that row) |
| Images per well | 72 (9 sites × 8 channels × 1 focal plane) |
| Total images | 1,008 |
| Total image size | ~2.4 GB |
| Illumination-correction files | 8 (one per channel, ~37 MB total) |

The full compound list (well, Broad compound ID, name, and known gene targets) is in [`configs/data.yaml`](configs/data.yaml), copied from [JUMP-Target](https://github.com/jump-cellpainting/JUMP-Target) and cross-referenced against this specific plate's platemap. Run the download script to actually pull this subset:

```bash
python -m cellpainting.download            # downloads images + illum files to data/raw/cellpainting/
python -m cellpainting.download --dry-run   # lists file counts/sizes without downloading anything
```

This plate acquired 8 channels per site, not the "standard" 5-channel Cell Painting stain set — a consequence of this batch's WGA-stain variant (see the CPJUMP1 experiment metadata). The channel-index → stain-name mapping is not assumed anywhere in this codebase; it is read from this plate's own `Images/Index.idx.xml` before feature extraction, since channel order is an acquisition detail rather than a fixed convention across plates.

**Compute target**: Colab Pro / notebook execution, not local GPU. Data volume and training scripts for this module are sized to fit a single Colab session (see `docs/build-log.md`).

## Model architecture

Planned: a CNN trained on Cell Painting composite images to classify compound mechanism-of-action (MoA), most likely on the 5 fluorescent stain channels (DNA/ER/RNA/AGP/Mito) of this plate's 8 acquired channels, once `Index.idx.xml` confirms which channel indices those are — the 3 remaining channels (brightfield/autofocus) are not part of the standard Cell Painting assay and are not expected to be model input. Architecture choice (custom small CNN vs. a pretrained backbone fine-tuned on multi-channel input) is not yet finalized and will be recorded in `docs/build-log.md` when decided, not silently assumed here.

## Evaluation plan

Per the project-wide evaluation philosophy ([`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)), this module reports two numbers:

1. **In-distribution**: MoA classification accuracy / macro-F1 on a held-out split from the same imaging batch as training.
2. **Transfer**: MoA classification accuracy / macro-F1 on a different imaging batch than any training example (cross-batch transfer).

The gap between the two, compared against a naive baseline (majority-class or non-deep tabular classifier on the precomputed CellProfiler features), is the headline result — see [`../../docs/impact.md`](../../docs/impact.md).

## Results

None yet. This section is filled in only once a real experiment run is logged in MLflow, per [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Layout

```
modules/1_cellpainting/
├── README.md       # this file
├── configs/        # YAML configs (data subset selection, training hyperparameters)
├── src/cellpainting/  # module source, importable as its own package
├── tests/          # pytest: data-processing unit tests + one metric regression test
└── Dockerfile       # training/eval runtime, cellpainting extra only
```

## Setup

```bash
pip install -e ".[dev,cellpainting]"
```
