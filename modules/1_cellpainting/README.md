# Module 1 — Cell Painting morphological profiling

Status: in progress (scaffold). No trained model or results yet.

## Research question

Within the project's overall thesis, this module asks: how well does a CNN trained to classify *compound identity* from Cell Painting images generalize to an imaging batch it never saw during training?

**Classification target: compound identity, not mechanism-of-action (MoA).** An earlier version of this module framed the task as MoA classification. That was never a well-defined task for this subset: `JUMP-Target-1_compound_metadata.tsv` (the source of the compound metadata below) has no MoA field, only per-compound gene-target lists — and grouping 14 compounds into MoA classes from those raw target lists would mean inventing ground truth, not using it. The task is instead a 14-way classification over compound identity (`Metadata_broad_sample` / `Metadata_pert_iname`), which uses labels that already exist and are verified below. See `docs/build-log.md` for the full reasoning. This applies to both the CNN (this section) and the tabular baseline (see Feature extraction / Evaluation plan below).

## Data source

**Dataset**: JUMP-CP (JUMP Cell Painting Consortium), hosted on the [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) (AWS Open Data, `s3://cellpainting-gallery`, no-sign-request public bucket). License: CC0. Full provenance in [`../../DATA_SOURCES.md`](../../DATA_SOURCES.md).

**Subset used (validation pass, not the full dataset)**: one plate, 14 compounds — within the agreed 10–15 range. This is a deliberate compute-driven substitution — see `docs/build-log.md` — chosen to validate the download → feature-extraction → model pipeline end-to-end before deciding whether to scale to the originally scoped ~30–50 compound / 1–2 plate subset. ("Feature extraction" here means fetching Broad's own precomputed CellProfiler profiles for this plate, not running CellProfiler ourselves — see the Feature extraction section below.)

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

This plate acquired 8 channels per site: the standard 5 fluorescent Cell Painting stains plus 3 non-fluorescent brightfield/z-stack channels that this CPJUMP1 batch collected in addition (not, as an earlier version of this doc guessed, something specific to the plate's "WGA" anomaly flag — that flag turned out to refer to dye-amount differences, not a channel-schema change; see `docs/build-log.md`). The channel-index → stain mapping was confirmed, not assumed, by cross-referencing three independent sources for this exact plate: the raw `Images/Index.idx.xml` acquisition metadata, the CPJUMP1 batch's own published README, and this plate's `load_data.csv` (Broad's CellProfiler input file, which maps each stain directly to a `chN` filename). All three agree:

| Channel | Stain | Fluorophore | Ex/Em (nm) |
|---|---|---|---|
| ch1 | Mito | Alexa 647 | 640 / 706 |
| ch2 | AGP | Alexa 568 | 561 / 599 |
| ch3 | RNA | 488 long | 488 / 599 |
| ch4 | ER | Alexa 488 | 488 / 522 |
| ch5 | DNA | Hoechst 33342 | 405 / 456 |
| ch6 | HighZBF (brightfield) | — | 740 / — |
| ch7 | LowZBF (brightfield) | — | 740 / — |
| ch8 | Brightfield | — | 740 / — |

Recorded in [`configs/data.yaml`](configs/data.yaml) (`channels:` / `model_input_channels:`). Model input will be ch1–ch5, the 5 fluorescent Cell Painting stains; ch6–ch8 are brightfield/z-stack channels outside the standard assay and are excluded.

**Compute target**: Colab Pro / notebook execution, not local GPU. Data volume and training scripts for this module are sized to fit a single Colab session (see `docs/build-log.md`).

## Feature extraction

**We do not run CellProfiler ourselves for this module.** Broad has already computed and published real, per-well CellProfiler profiles for this exact plate (`BR00116991`), via their own validated JUMP pipeline. Confirmed present in `s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/`:

- `pipelines/2020_11_04_CPJUMP1/CPJUMP1_analysis_without_batchfile.cppipe` — the actual CellProfiler pipeline Broad ran (kept for provenance/reference; not executed by this codebase).
- `profiles/2020_11_04_CPJUMP1/BR00116991/` — several pycytominer-processed, well-level profile variants derived from that pipeline's output.
- `backend/2020_11_04_CPJUMP1/BR00116991/BR00116991.sqlite` — the full per-cell measurement database (23.5GB). Exists, but is unnecessary at well-level granularity and far too large for this validation subset.

The base profile (`BR00116991_augmented.csv.gz`) has 5,792 real CellProfiler feature columns across all 384 wells, covering the standard measurement categories plus more: Texture (3,744), Correlation (672), Granularity (384), Intensity (360), RadialDistribution (288), AreaShape (162), Location (152). This codebase uses the smaller, already-normalized-and-feature-selected variant instead:

| File | Features | Metadata cols | Rows | Our 14 wells present |
|---|---|---|---|---|
| `BR00116991_normalized_feature_select_batch.csv.gz` | 838 | 13 | 384 (all wells) | confirmed, all 14 |

This is pycytominer's standard "ready for downstream ML" output: per-plate normalized, with redundant/blocklisted CellProfiler features already dropped. Verified directly by downloading and parsing it — not assumed from the filename. Metadata columns (`Metadata_broad_sample`, `Metadata_pert_iname`, `Metadata_InChIKey`, `Metadata_smiles`, `Metadata_gene`, `Metadata_pert_type`, `Metadata_control_type`, ...) match and exceed the compound metadata already in `configs/data.yaml`.

Run the profile-fetching script to actually pull and filter it, same conventions as `download.py` (config-driven, idempotent, `--dry-run` supported):

```bash
python -m cellpainting.profiles            # fetches + filters to our 14 wells, writes data/raw/cellpainting/profiles/
python -m cellpainting.profiles --dry-run   # fetches + filters in memory, prints counts, writes nothing
```

**Why not run CellProfiler locally**: headless CellProfiler needs a bioformats/Java dependency plus wxPython — a heavy, fragile install that fits poorly with the agreed Colab Pro compute target — and would require correctly wiring segmentation (nuclei/cell/cytoplasm) and per-object measurement modules ourselves, substantial infrastructure to re-derive something Broad has already computed and published for this exact plate.

**Why not a custom Python substitute** (e.g. scikit-image intensity/texture/shape stats): it would only approximate a subset of these categories and wouldn't be numerically comparable to the literature-standard JUMP profiles. Since the real thing is a free, already-verified ~1MB download for our wells, a substitute would be strictly worse here — more engineering risk for less rigor, not less cost.

**Scope**: this decision covers the tabular/naive-baseline and UMAP/SHAP track only. It does not change the CNN's input — see Model architecture below.

## Model architecture

Planned: a CNN trained directly on raw 5-channel Cell Painting pixel composites (ch1–ch5: Mito/AGP/RNA/ER/DNA, per the confirmed channel mapping above) to classify compound identity (14-way, not mechanism-of-action — see Research question above) — not on CellProfiler features. The CNN never runs CellProfiler or consumes its output; Broad's precomputed profiles (see Feature extraction above) feed only the naive/tabular baseline and the UMAP/SHAP exploratory analysis this module is compared against. Architecture choice (custom small CNN vs. a pretrained backbone fine-tuned on 5-channel input) is not yet finalized and will be recorded in `docs/build-log.md` when decided, not silently assumed here.

## Evaluation plan

Per the project-wide evaluation philosophy ([`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)), this module reports two numbers:

1. **In-distribution**: compound-identity classification accuracy / macro-F1 (14-way) on a held-out split from the same imaging batch as training.
2. **Transfer**: compound-identity classification accuracy / macro-F1 (14-way) on a different imaging batch than any training example (cross-batch transfer).

The gap between the two, compared against a naive baseline (majority-class or non-deep tabular classifier on Broad's precomputed CellProfiler features — see Feature extraction above), is the headline result — see [`../../docs/impact.md`](../../docs/impact.md).

**Tabular baseline's own robustness check**: `BR00116992`–`BR00116994` are three more plates in this same batch using the same platemap, confirmed (not assumed) to have identical well→compound assignment to `BR00116991` for our 14 wells — see `docs/build-log.md`. This gives the tabular baseline n=56 (14 compounds × 4 plates) instead of n=14, enabling real leave-one-plate-out cross-validation (train on 3 plates, test on the held-out 4th) rather than an unheld-out 14-point evaluation. This is a within-batch check on the baseline itself, separate from the CNN's cross-batch transfer evaluation above. Significance is assessed with a permutation-based null (label-shuffling), not a fixed accuracy threshold, since chance-level accuracy for 14-way classification isn't simply 1/14 once class-conditional feature structure and CV-fold correlation are accounted for.

Run the baseline (fetches/caches all four plates' profiles, runs leave-one-plate-out CV, then the permutation null test):

```bash
python -m cellpainting.baseline                        # 1000 permutations by default
python -m cellpainting.baseline --n-permutations 200    # fewer permutations, faster
python -m cellpainting.baseline --refresh               # re-fetch profiles from S3 instead of using the local cache
```

Writes results to `data/raw/cellpainting/baseline/results.json`.

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
