# PR notes: `module/1-cellpainting` → `main`

**Status: work in progress, not ready to merge.** This PR is opened early to get the pipeline groundwork reviewed incrementally rather than as one large end-of-module diff. The module's own README already states this: no CNN, no trained model, no logged results yet.

## Data / model / eval summary

- **Data subset**: one plate (`BR00116991`, batch `2020_11_04_CPJUMP1`, `JUMP-Target-1_compound_platemap`), 14 compounds, picked by querying the live `cellpainting-gallery` S3 bucket directly rather than trusting scraped docs. Images (1,008 files, ~2.4GB) downloaded; channel-to-stain mapping (ch1–ch5 fluorescent, ch6–ch8 brightfield) confirmed against three independent sources.
- **Feature extraction**: uses Broad's own precomputed CellProfiler profiles (`_normalized_feature_select_batch.csv.gz`, 838 features) rather than running CellProfiler locally or writing a custom substitute — see `modules/1_cellpainting/README.md`'s Feature extraction section for the full rationale.
- **Classification target**: reframed mid-branch from mechanism-of-action (MoA) to compound identity (14-way, `Metadata_broad_sample`/`Metadata_pert_iname`). `JUMP-Target-1_compound_metadata.tsv` has no MoA field for this compound set — grouping the 14 compounds into MoA classes from raw gene-target lists would have meant inventing ground truth. Applies to both the planned CNN and the tabular baseline.
- **Tabular baseline (built)**: leave-one-plate-out CV (logistic regression, `StandardScaler` + `LogisticRegression(C=0.1)`) over `BR00116991`'s three confirmed in-batch replicate plates (`BR00116992`–`BR00116994`), n=56, 14 classes. Real run: **accuracy 0.750, macro-F1 0.743**, permutation null 0.072 ± 0.036 (≈ 1/14, sanity-checks correctly), **p = 0.000999** (beat all 1000 permutations). Not yet logged to MLflow, so not yet in `docs/impact.md` per that page's own convention (MLflow-logged results only).
- **CNN**: not started. Architecture choice not yet finalized.
- **Cross-batch transfer evaluation** (the module's actual headline comparison): not started — needs a second batch, not yet selected.

## What was tested

- Every data-processing function in `download.py`, `profiles.py`, and `baseline.py` has a fast, pure-logic pytest test (config parsing, well-prefix construction, CSV filtering/round-tripping, an in-memory `FakeS3` for the baseline's fetch/cache/pool logic, schema-mismatch and wrong-well-count guards).
- `baseline.py`'s CV and permutation-test logic is exercised against synthetic informative/uninformative datasets with known expected outcomes (near-perfect recovery vs. p > 0.05 for pure noise).
- Live-bucket regression tests (marked `slow`, run only in the weekly smoke test) confirm the JUMP-CP layout assumptions this code depends on: file counts per well, profile schema (838 features, all 384 wells), and — added this round — that all four plates pool into a clean 56×838/14-class dataset with a balanced 14-per-plate group structure.
- `make test` (ruff, black, mypy, full fast suite) passes.

## Deliberately left out of scope (this PR)

- CNN implementation and training.
- Cross-batch transfer evaluation (needs a second batch to be selected and downloaded).
- MLflow run logging for the baseline (so its real numbers, while genuine, aren't yet reflected in `docs/impact.md`).
- UMAP/SHAP exploratory analysis track mentioned in the module README.
- Scaling past the 14-compound / 1-plate-plus-replicates validation subset.
