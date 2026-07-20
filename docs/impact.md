# Impact

Concrete, verifiable technical claims from this project, intended as material a reference letter could cite directly. Every number here is computed from a real experiment run logged in MLflow — nothing on this page is projected, estimated, or aspirational. Until a module has real results, its section says so explicitly rather than being left implying a result exists.

## Module 1 — Cell Painting morphological profiling

Status: in progress. No results yet.

Planned claim shape (to be replaced with real numbers once the batch-transfer experiment runs): in-distribution compound-identity classification accuracy/macro-F1 (14-way — no MoA annotation exists for this pilot's compound set, see `docs/build-log.md`) vs. cross-batch transfer accuracy/macro-F1, and the point-drop between them, compared against a naive (e.g. majority-class or non-deep tabular) baseline, with the baseline's own significance assessed via a permutation-based null rather than a fixed accuracy threshold.

## Module 2 — GNN for scRNA-seq cell-type classification

Status: not started.

## Module 3 — Genomic foundation model fine-tuning

Status: not started.

## Module 4 — GNN drug-target interaction prediction

Status: not started.

## Cross-module synthesis

Once at least two modules have real transfer results, this section will compare how the transfer gap behaves across representation strategies (image-based vs. graph-based vs. foundation-model-based) — that comparison is the project's actual thesis, not any single module's number in isolation.
