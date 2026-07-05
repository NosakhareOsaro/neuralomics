# Contributing

NeuralOmics is solo-authored and solo-maintained. This document exists anyway, for two reasons: it's the single source of truth for the conventions used throughout the project's history, and it documents the review discipline applied even in the absence of a second reviewer.

## Branching

- `main` is always in a working state.
- Each module is developed on its own branch: `module/1-cellpainting`, `module/2-gnn-scrna`, `module/3-genomic-fm`, `module/4-dti-gnn`.
- Cross-cutting infrastructure (CI, shared `common/` utilities, root docs) is committed directly to `main` in small commits, since it isn't module-specific work awaiting review.
- Before merging a module branch to `main`, a PR-style description is written to `docs/pr-notes/<branch-name>.md` — data/model/eval summary, what was tested, what was deliberately left out of scope — before the merge, simulating the review record a second contributor's PR would leave.

## Commits

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `ci:`, scoped where useful, e.g. `feat(gnn-scrna): add GAT architecture and training loop`.
- One concern per commit. A commit message describing two unrelated changes means it should have been two commits.
- Non-trivial commits get a body explaining *why*, not just *what* — the diff already shows what changed.
- No AI/assistant attribution, signature, or co-author trailer in any commit, ever.
- History is never squashed or force-pushed without explicit sign-off from the repository owner first.

## Testing

- Every data-processing and model-utility function gets a pytest test. Target >80% coverage on non-notebook code.
- Every module includes at least one regression test asserting a key metric doesn't silently degrade below a fixed floor.
- `make test` runs lint, type-check, and the full test suite; it must pass before a module branch merges to `main`.

## Documentation

- Each module has its own `README.md`: data source (with any subsetting explicitly flagged, never silently downgraded), model architecture, evaluation plan, and results once available.
- Engineering decisions and their reasoning are narrated as they're made in `docs/build-log.md`, not reconstructed after the fact.
- Concrete, verifiable results (not aspirational claims) are added to `docs/impact.md` only once they come from a real experiment run logged in MLflow.

## Releases

- Semantic versioning. `v0.1.0` after Module 1 works end-to-end, incrementing through `v1.0.0` for the full MLOps-integrated release.
- Every tagged release has a corresponding GitHub Release with notes summarizing what shipped.
