# Data Sources

Every dataset used in this project is public. This file tracks provenance, licensing, and exactly what subset (if any) of each dataset is used, so substitutions made for compute reasons are documented rather than silently applied. Each module's own README links back here and adds module-specific detail (exact file lists, download commands, checksums).

## Module 1 — Cell Painting morphological profiling

**Dataset**: JUMP-CP (JUMP Cell Painting Consortium), hosted on the [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery) (AWS Open Data, `s3://cellpainting-gallery`, no-sign-request public bucket), Broad Institute.

- Full dataset: ~116TB across all batches/plates — not used in full.
- **Subset used** (documented explicitly here and in `modules/1_cellpainting/README.md` because it is a deliberate compute-driven substitution, not the full dataset): plate `BR00116991` (batch `2020_11_04_CPJUMP1`, platemap `JUMP-Target-1_compound_platemap`), restricted to 14 compounds (1,008 images, ~2.4GB) — to validate the pipeline end-to-end before scaling up. Exact compound list, well IDs, and gene targets are in `modules/1_cellpainting/configs/data.yaml`; full detail in `modules/1_cellpainting/README.md`. Verified directly against the live bucket, not assumed from documentation.
- **Feature extraction uses Broad's own precomputed CellProfiler profiles for this exact plate, not a locally-run CellProfiler pipeline.** Broad publishes these as gzipped CSV (not parquet, as an earlier version of this doc assumed) under `workspace/profiles/2020_11_04_CPJUMP1/BR00116991/` in the same bucket; this project uses the `_normalized_feature_select_batch.csv.gz` variant (838 pycytominer-selected features, all 384 wells, confirmed to include all 14 of our compounds). Full rationale for not running CellProfiler ourselves, and not writing a custom substitute, is in `modules/1_cellpainting/README.md`'s Feature extraction section. This only feeds the tabular baseline / UMAP-SHAP track — the CNN trains on raw images directly.
- **Tabular-baseline replicates**: `BR00116992`–`BR00116994`, three more plates in this same batch using the same platemap, confirmed to have identical well→compound assignment to `BR00116991` for our 14 wells (verified against the live bucket, not assumed — see `docs/build-log.md`). Used only to give the tabular baseline n=56 for leave-one-plate-out cross-validation; not used by the CNN.
- License: JUMP-CP data is released under CC0 (public domain dedication) by the Broad Institute / JUMP Cell Painting Consortium.

## Module 2 — GNN for scRNA-seq cell-type classification

**Primary dataset**: 10x Genomics PBMC (peripheral blood mononuclear cells), publicly distributed by 10x Genomics and bundled as `scanpy.datasets.pbmc3k()` / available directly from the [10x Genomics public datasets page](https://www.10xgenomics.com/resources/datasets).

**Transfer-target dataset**: a public human pancreas scRNA-seq dataset (candidate: Baron et al. 2016, GEO accession [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)) used for the cross-tissue zero-shot / fine-tuned transfer experiment. Final choice confirmed in `modules/2_gnn_scrna/README.md` when that module starts.

- License: 10x Genomics datasets are freely available for research use; GEO-deposited datasets are public domain / freely redistributable per NIH GEO policy.

## Module 3 — Genomic foundation model fine-tuning

**Fine-tuning target**: [ENCODE project](https://www.encodeproject.org/) data — candidate cis-regulatory elements (cCREs) for regulatory element classification, and GENCODE canonical splice-site annotations on the GRCh38 reference for splice-site prediction.

**Models**: [Nucleotide Transformer](https://huggingface.co/InstaDeepAI) (InstaDeepAI, HuggingFace) fine-tuned on the above; benchmarked against a [DNABERT-2](https://huggingface.co/zhihan1996/DNABERT-2-117M) baseline.

- Evaluation split: held out entire chromosomes (not a random row-level split), documented in `modules/3_genomic_fm/README.md` along with the leakage argument this avoids (sequence homology and regulatory redundancy within a chromosome make random splits optimistic).
- License: ENCODE and GENCODE data are public domain / freely available for research use.

## Module 4 — GNN drug-target interaction prediction

**Dataset**: [ChEMBL](https://www.ebi.ac.uk/chembl/) (EMBL-EBI), bioactivity records (pIC50) for drug-target pairs.

**Features**: Morgan fingerprints computed with [RDKit](https://www.rdkit.org/) (open source, BSD license) for compounds; [ESM-2](https://github.com/facebookresearch/esm) (Meta AI) protein language model embeddings for targets.

- Transfer split: an entire protein family held out from training, evaluated zero-shot, documented in `modules/4_dti_gnn/README.md`.
- License: ChEMBL data is available under a [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.

## General policy

- No dataset requiring a data use agreement, institutional access, or non-public credentials is used anywhere in this project.
- Any subset smaller than the full public dataset is stated explicitly, with the reason (compute/time budget) and the exact selection criteria, in both this file and the relevant module README — never presented as if it were the full dataset.
