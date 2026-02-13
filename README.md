# Ontology-Enhanced Breast Cancer Diagnosis (PyKEEN + Clustering + Meta-Learning)

This repository provides an end-to-end implementation of the pipeline described in the manuscript:

**Ontology-Enhanced Machine Learning Models for Breast Cancer Diagnosis**  
(OWL/RDF → KG embeddings with **PyKEEN/TransE** → **Leiden** + **HDBSCAN** clustering → semantic features + stacked meta-learning → SPARQL-based explanations)

## What the code does

1. Load an OWL/Turtle ontology and export **URI-only triples** (URI–URI–URI) to TSV for PyKEEN.
2. Train a KG embedding model (default: **TransE**) using PyKEEN.
3. Extract **patient embeddings**, project with **UMAP**, and build a **k-NN graph**.
4. Run **Leiden** community detection (reports modularity).
5. Run **HDBSCAN** clustering (reports silhouette; handles noise).
6. Export:
   - `Figure_Leiden.png` and `Figure_HDBSCAN.png` (separate images)
   - UMAP coordinates + cluster labels (`UMAP_Clusters.csv`)
   - Score + summary tables (`Cluster_Summaries.xlsx`)
   - SPARQL query templates (`SPARQL_Explanations.md`)
7. Optional: ML benchmarking + stacked meta-learner (see `scripts/04_train_classifiers_and_meta.py`)

## Quick start (Conda)

```bash
conda env create -f environment.yml
conda activate bc-ontology-ml
```

## Run clustering (single dataset)

```bash
python scripts/03_train_pykeen_and_cluster.py ^
  --input_ttl data/processed/Wisconsin_Categorized_KnowledgeGraph_enriched_final.ttl ^
  --out_dir outputs/wdbc ^
  --patient_prefix Patient_ ^
  --k 15 ^
  --umap_n_neighbors 20 ^
  --umap_min_dist 0.1 ^
  --pykeen_model TransE ^
  --epochs 50
```

## Folder structure

- `scripts/` runnable scripts (CLI)
- `src/` reusable modules
- `data/` (not committed by default)
- `outputs/` results (figures, CSV, XLSX)

## Data note

Do **not** upload large raw datasets to GitHub if licensing is unclear. Provide download instructions instead.


## SPARQL-based cluster explanations

After running clustering, generate templates + top-category patterns:

```bash
python scripts/05_sparql_cluster_explain.py --input_ttl <ttl> --clusters_csv <outputs>/UMAP_Clusters.csv --out_dir <outputs>
```
