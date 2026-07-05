# NeuralOmics

**A deep-learning-for-genomics research toolkit studying how representation-learning strategies transfer across unseen biological contexts.**

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)
<!-- CI badge: added once a GitHub remote exists, pointing at .github/workflows/ci.yml. -->
<!-- DOI badge: added once the repository is connected to Zenodo for the v1.0.0 release. -->

> Status: early development. This README is updated as each module lands; see [CHANGELOG.md](CHANGELOG.md) and [docs/build-log.md](docs/build-log.md) for what's actually done versus planned.

## Research question

Deep representation-learning methods for genomics and cell biology are usually benchmarked in-distribution: same batch, same tissue, same cell type as training. That number is a weak proxy for whether the learned representation captures anything biologically general. This project asks a single question across four otherwise-independent modeling problems:

**How well do image-based, graph-based, and foundation-model-based representation-learning strategies transfer across unseen biological contexts — different batches, tissues, cell types, or protein families than they were trained on?**

Each module below pairs a standard in-distribution result with a genuine transfer/generalization experiment, and reports both, including the gap between them.

| # | Module | Representation strategy | Transfer axis | Status |
|---|--------|--------------------------|----------------|--------|
| 1 | [Cell Painting morphological profiling](modules/1_cellpainting/) | Image-based (CNN on Cell Painting composites) | Cross-batch (train batch A, test batch B) | In progress |
| 2 | [GNN for scRNA-seq cell-type classification](modules/2_gnn_scrna/) | Graph-based (GCN/GAT/GraphSAGE on kNN cell graphs) | Cross-tissue (PBMC → pancreas) | Planned |
| 3 | [Genomic foundation model fine-tuning](modules/3_genomic_fm/) | Foundation-model-based (Nucleotide Transformer) | Held-out chromosomes | Planned |
| 4 | [GNN drug-target interaction prediction](modules/4_dti_gnn/) | Graph-based (bipartite drug-target GNN) | Held-out protein family | Planned |

A cross-module synthesis of these results (not just per-module numbers) is maintained in [docs/impact.md](docs/impact.md) as they land — no numbers are published there until they come from a real experiment run.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, including the shared `common/` utilities, per-module layout, and the MLOps serving layer (FastAPI + Kubernetes + Prometheus) that sits on top of modules 2 and 4.

## Repository structure

```
NeuralOmics/
├── modules/            # four independent, self-contained modules (see table above)
├── common/             # shared utilities: seeding policy, MLflow helpers, I/O
├── mlops/              # FastAPI serving, Helm chart, kind manifests, Prometheus config
├── docs/               # MkDocs site source, build-log, impact evidence, PR notes
└── .github/workflows/  # CI (lint/type-check/test) and a weekly smoke-test pipeline
```

## Reproducing this work

Each module pins its own dependencies as an optional extra in the root `pyproject.toml` so you only install what you need:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cellpainting]"   # swap the extra for the module you're working on
make test                              # runs lint, type-check, and pytest
```

Module-specific setup, data download instructions, and expected runtime/compute are documented in each module's own README. Where a public dataset is too large to use in full, the module README says so explicitly and states the exact subset used — see [DATA_SOURCES.md](DATA_SOURCES.md) for provenance of every dataset referenced in this repository.

## Experiment tracking

All training runs are logged with MLflow. A summary table of logged experiments per module will be added here as runs accumulate (currently: none yet — Module 1 is in progress).

## Random-seed policy

Documented in `common/seeding.py` and referenced by every module's training entry point; see that module's docstring for the exact policy.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Chosen over MIT for the explicit patent grant, which is more common practice for production-oriented ML tooling of this kind; see the note in `LICENSE` for the copyright holder.

## Citation

See [CITATION.cff](CITATION.cff). A Zenodo DOI will be attached at the v1.0.0 release; until then, cite the repository URL and commit hash.

## Data sources

Every dataset used in this repository is public. Full provenance, licenses, and links are tracked in [DATA_SOURCES.md](DATA_SOURCES.md).
