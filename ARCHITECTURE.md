# Architecture

## Overview

NeuralOmics is four independent modeling problems that share a common infrastructure layer and a common evaluation philosophy (in-distribution result + explicit transfer/generalization result). Modules do not depend on each other's code; they depend on `common/` for cross-cutting concerns, and modules 2 and 4 additionally feed the MLOps serving layer.

```
                         ┌─────────────────────────┐
                         │        common/          │
                         │  seeding · mlflow io ·   │
                         │  shared data utilities   │
                         └───────────┬─────────────┘
                                     │
       ┌──────────────┬─────────────┼─────────────┬──────────────┐
       │              │             │             │              │
 ┌───────────┐  ┌────────────┐ ┌────────────┐ ┌────────────┐
 │ Module 1  │  │  Module 2  │ │  Module 3  │ │  Module 4  │
 │Cell       │  │GNN scRNA-  │ │Genomic     │ │GNN drug-   │
 │Painting   │  │seq cell-   │ │foundation  │ │target      │
 │morphology │  │type class. │ │model FT    │ │interaction │
 └───────────┘  └─────┬──────┘ └────────────┘ └─────┬──────┘
                       │                             │
                       └─────────────┬───────────────┘
                                     │
                         ┌─────────────────────────┐
                         │       mlops/            │
                         │ FastAPI serving (2 & 4)  │
                         │ Docker → kind → Helm     │
                         │ Prometheus metrics       │
                         └─────────────────────────┘
```

Only modules 2 and 4 are exposed as served endpoints: they produce a discrete prediction (cell type; binding affinity) that maps naturally onto a request/response API. Modules 1 and 3 are analysis pipelines (profiling, fine-tuning + benchmarking) whose primary output is a trained model artifact and a results report, not a live inference service.

## Shared infrastructure (`common/`)

- **Seeding policy** (`common/seeding.py`): a single `set_global_seed()` entry point used by every module's training script, so runs are reproducible and the policy is documented in one place rather than duplicated per module.
- **MLflow helpers**: thin wrappers so every module logs runs to the same tracking URI convention and the same minimal set of tags (module name, git commit, dataset subset identifier).
- Deliberately kept small. Module-specific logic (data loaders, model definitions, training loops) lives inside each module, not in `common/`, so modules stay independently runnable and independently reviewable.

## Module layout convention

Every module under `modules/<n>_<name>/` follows the same shape:

```
modules/<n>_<name>/
├── README.md       # data source, model architecture, evaluation plan, results
├── configs/        # YAML configs for training runs (no hardcoded hyperparameters in code)
├── src/            # module source, importable as its own package
├── tests/          # pytest: data-processing unit tests + one metric regression test
└── Dockerfile      # module-specific runtime, only the deps that module needs
```

## Evaluation philosophy

Every module reports two numbers, not one:

1. An **in-distribution** metric on a held-out split from the same source (batch/tissue/chromosome/protein-family family) as training.
2. A **transfer** metric on a deliberately out-of-distribution split — a different batch, tissue, cell type, chromosome set, or protein family than any training example.

The gap between the two is the headline result for that module, and is what `docs/impact.md` cites.

## MLOps layer

- **Serving**: FastAPI apps in `mlops/serving/`, one per served module (2 and 4), each loading its module's best checkpoint and exposing a `/predict` endpoint plus a `/metrics` endpoint for Prometheus.
- **Containerization**: each module has its own Dockerfile (training/eval), and each served module has a separate, smaller serving-only Dockerfile (inference deps only, no training deps).
- **Orchestration**: a Helm chart deploys both serving apps to a local `kind` cluster for demonstration purposes — this is not a production deployment target, it's evidence of Kubernetes/Helm competency applied to a real (if small) workload.
- **Observability**: Prometheus scrapes request count/latency and a model-specific metric (e.g. prediction confidence distribution) from each serving app.
- **CI/CD**: GitHub Actions builds and pushes module images on tagged releases; a separate scheduled workflow runs a cheap smoke-test pipeline weekly to catch silent breakage from upstream dependency changes.

## Why per-module extras instead of one monolithic environment

`pyproject.toml` defines one optional-dependency group per module (`cellpainting`, `gnn-scrna`, `genomic-fm`, `dti-gnn`) rather than a single dependency set, because the modules have materially different and occasionally conflicting heavy dependencies (e.g. `torch-geometric` vs `transformers` vs `rdkit` version constraints). Installing only the extra you need keeps CI fast and keeps a contributor's environment for one module from breaking on another module's dependency churn.
