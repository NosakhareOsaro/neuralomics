# Module 1 — Cell Painting morphological profiling

Status: in progress (scaffold). No trained model or results yet.

## Research question

Within the project's overall thesis, this module asks: how well does a CNN trained to classify mechanism-of-action (MoA) from Cell Painting images generalize to an imaging batch it never saw during training?

## Data source

**Dataset**: JUMP-CP (JUMP Cell Painting Consortium), hosted on the [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) (AWS Open Data, `s3://cellpainting-gallery`, no-sign-request public bucket). License: CC0. Full provenance in [`../../DATA_SOURCES.md`](../../DATA_SOURCES.md).

**Subset used (validation pass, not the full dataset)**: one plate from one imaging batch, restricted to **10–15 compounds**. This is a deliberate compute-driven substitution — see `docs/build-log.md` — chosen to validate the download → CellProfiler-feature → model pipeline end-to-end before deciding whether to scale to the originally scoped ~30–50 compound / 1–2 plate subset. The exact plate ID and compound list are not yet chosen; once the download script (`src/cellpainting/download.py`, not yet written) has run, they will be recorded here verbatim, along with resulting image and profile counts, rather than left as this placeholder.

**Compute target**: Colab Pro / notebook execution, not local GPU. Data volume and training scripts for this module are sized to fit a single Colab session (see `docs/build-log.md`).

## Model architecture

Planned: a CNN trained on 5-channel Cell Painting composite images to classify compound mechanism-of-action (MoA). Architecture choice (custom small CNN vs. a pretrained backbone fine-tuned on 5-channel input) is not yet finalized and will be recorded in `docs/build-log.md` when decided, not silently assumed here.

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
