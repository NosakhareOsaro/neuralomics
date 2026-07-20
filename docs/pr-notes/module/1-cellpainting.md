# PR notes: `module/1-cellpainting` → `main`

**Status: this slice is complete, tested, and ready to merge as-is.**

**Scope of this PR**: the Module 1 data pipeline (download + replicate-plate verification) and the compound-identity tabular baseline, end to end. The CNN and the batch-effect / cross-batch transfer experiment — the module's eventual headline comparison — are explicitly **not** part of this PR. They're tracked as follow-up work on a new branch, `module/1-cellpainting-cnn`, so the scope of what's being merged here is unambiguous.

## Data / model / eval summary

- **Data subset**: one plate (`BR00116991`, batch `2020_11_04_CPJUMP1`, `JUMP-Target-1_compound_platemap`), 14 compounds, picked by querying the live `cellpainting-gallery` S3 bucket directly rather than trusting scraped docs. Images (1,008 files, ~2.4GB) downloaded; channel-to-stain mapping (ch1–ch5 fluorescent, ch6–ch8 brightfield) confirmed against three independent sources.
- **Feature extraction**: uses Broad's own precomputed CellProfiler profiles (`_normalized_feature_select_batch.csv.gz`, 838 features) rather than running CellProfiler locally or writing a custom substitute — see `modules/1_cellpainting/README.md`'s Feature extraction section for the full rationale.
- **Replicate-plate verification**: confirmed `BR00116992`–`BR00116994` have identical well→compound assignment to `BR00116991` for all 14 target wells, by diffing each plate's own profile against `BR00116991`'s directly on the live bucket (0 mismatches), not by trusting the shared platemap name alone.
- **Classification target**: reframed mid-branch from mechanism-of-action (MoA) to compound identity (14-way, `Metadata_broad_sample`/`Metadata_pert_iname`). `JUMP-Target-1_compound_metadata.tsv` has no MoA field for this compound set — grouping the 14 compounds into MoA classes from raw gene-target lists would have meant inventing ground truth. Applies to both the (not-yet-built) CNN's task definition and the tabular baseline built in this PR.
- **Tabular baseline**: leave-one-plate-out CV (logistic regression, `StandardScaler` + `LogisticRegression(C=0.1)`) over the four confirmed plates, n=56, 14 classes. Real run: **accuracy 0.750, macro-F1 0.743**, permutation null 0.072 ± 0.036 (≈ 1/14, sanity-checks correctly), **p = 0.000999** (beat all 1000 permutations). Not yet logged to MLflow, so not yet in `docs/impact.md` per that page's own convention (MLflow-logged results only) — that's the one piece of follow-up this PR itself still implies, independent of the CNN work.

## What was tested

- Every data-processing function in `download.py`, `profiles.py`, and `baseline.py` has a fast, pure-logic pytest test (config parsing, well-prefix construction, CSV filtering/round-tripping, an in-memory `FakeS3` for the baseline's fetch/cache/pool logic, schema-mismatch and wrong-well-count guards).
- `baseline.py`'s CV and permutation-test logic is exercised against synthetic informative/uninformative datasets with known expected outcomes (near-perfect recovery vs. p > 0.05 for pure noise).
- Live-bucket regression tests (marked `slow`, run only in the weekly smoke test) confirm the JUMP-CP layout assumptions this code depends on: file counts per well, profile schema (838 features, all 384 wells), and that all four plates pool into a clean 56×838/14-class dataset with a balanced 14-per-plate group structure.
- `make test` (ruff, black, mypy, full fast suite) passes.
- The baseline was run end to end against real data (not just tested in the abstract); see the real numbers above.

## Explicitly not part of this PR (tracked on `module/1-cellpainting-cnn`)

- CNN implementation and training.
- Cross-batch transfer evaluation (needs a second batch to be selected and downloaded) — the module's actual headline result.
- UMAP/SHAP exploratory analysis track mentioned in the module README.
- Scaling past the 14-compound / 1-plate-plus-replicates validation subset.

## Left as known follow-up (not blocking, not CNN-branch scope)

- MLflow run logging for the tabular baseline, so its real numbers can be added to `docs/impact.md` per that page's MLflow-only convention.
