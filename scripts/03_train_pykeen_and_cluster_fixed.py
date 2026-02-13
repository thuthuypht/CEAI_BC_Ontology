#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_train_pykeen_and_cluster.py (standalone)

End-to-end pipeline:
1) Load RDF/Turtle (.ttl) with rdflib (safe file loading on Windows)
2) Convert RDF triples -> PyKEEN TriplesFactory (keeps only (head, relation, tail) where tail is NOT a Literal by default)
3) Train a Knowledge Graph Embedding model with PyKEEN (default: TransE)
4) Extract entity embeddings (patients only by default)
5) Standardize -> UMAP 2D projection
6) Leiden community detection on k-NN graph (+ modularity)
7) HDBSCAN clustering (+ silhouette, ignoring noise when possible)
8) Export:
   - CSV/Excel: UMAP coords + cluster labels
   - PNG: Leiden UMAP scatter, HDBSCAN UMAP scatter
   - TXT/CSV: simple cluster explanations + SPARQL templates (optional execution)

Run (Windows cmd):
  python scripts\\03_train_pykeen_and_cluster.py ^
    --input_ttl data\\processed\\Wisconsin_Categorized_KnowledgeGraph_enriched_final.ttl ^
    --out_dir outputs\\wdbc ^
    --patient_prefix Patient_ ^
    --k 15 ^
    --umap_n_neighbors 20 ^
    --umap_min_dist 0.1 ^
    --pykeen_model TransE ^
    --epochs 50

Notes:
- If you are using PowerShell, replace ^ with ` (backtick) for line continuation.
- Recommended Python: 3.10 or 3.11 for best wheel availability on Windows.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------- Small utilities ---------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def _require(pkg: str, import_name: Optional[str] = None):
    """Import a package with a helpful error message."""
    name = import_name or pkg
    try:
        return __import__(name)
    except Exception as e:
        msg = (
            f"Missing/failed import: {name}\n"
            f"Error: {e}\n\n"
            "Install dependencies (example):\n"
            "  pip install rdflib pykeen scikit-learn umap-learn hdbscan igraph leidenalg networkx matplotlib openpyxl\n"
            "If you're on Windows, prefer Python 3.10/3.11 in a fresh conda env.\n"
        )
        raise RuntimeError(msg) from e

# Lazy imports (loaded only when needed)
rdflib = None

def load_graph_ttl(ttl_path: Path):
    global rdflib
    rdflib = rdflib or _require("rdflib")
    from rdflib import Graph

    g = Graph()

    # Windows-safe: open file handle so rdflib doesn't treat "D:/..." as a URL scheme "d:"
    with open(ttl_path, "rb") as f:
        g.parse(file=f, format="turtle")

    return g

def guess_base_iri(graph) -> str:
    """Try to guess a base IRI from Patient URIs or any URI in the graph."""
    from rdflib.term import URIRef
    for s in graph.subjects():
        if isinstance(s, URIRef):
            ss = str(s)
            if "#" in ss:
                return ss.split("#")[0] + "#"
            if ss.endswith("/") or ss.endswith("#"):
                return ss
            # fallback: trim last path segment
            if "/" in ss:
                return ss.rsplit("/", 1)[0] + "/"
    return "http://example.org/breastcancer#"

def graph_to_labeled_triples(
    graph,
    include_literals_as_nodes: bool = False,
    literal_node_prefix: str = "lit:",
) -> np.ndarray:
    """
    Convert rdflib Graph to labeled triples for PyKEEN.
    By default, drop triples with Literal objects (PyKEEN needs entity tails).
    """
    from rdflib.term import BNode, Literal, URIRef

    triples: List[Tuple[str, str, str]] = []
    for s, p, o in graph:
        if isinstance(s, Literal) or isinstance(p, Literal):
            continue  # invalid

        s_str = str(s)
        p_str = str(p)

        if isinstance(o, Literal):
            if not include_literals_as_nodes:
                continue
            # Turn literal into a node label (careful: can explode entity space)
            o_str = literal_node_prefix + str(o)
        else:
            o_str = str(o)

        # Optional: normalize blank nodes
        if isinstance(s, BNode):
            s_str = "_:" + str(s)
        if isinstance(o, BNode):
            o_str = "_:" + str(o)

        triples.append((s_str, p_str, o_str))

    if not triples:
        raise ValueError("No (subject, predicate, object) triples available for PyKEEN after filtering.")

    return np.asarray(triples, dtype=str)

def select_patients(entity_labels: List[str], patient_prefix: str) -> List[int]:
    """
    Select patient entity IDs by matching patient_prefix in the entity label/URI.
    """
    ids = [i for i, lab in enumerate(entity_labels) if patient_prefix in lab]
    return ids

# --------- PyKEEN training + embeddings ---------
@dataclass
class PykeenTrainConfig:
    model: str = "TransE"
    epochs: int = 50
    embedding_dim: int = 64
    batch_size: int = 256
    lr: float = 1e-3
    random_seed: int = 42

def train_pykeen_embeddings(triples: np.ndarray, cfg: PykeenTrainConfig):
    """
    Train a PyKEEN model and return (result, base_factory, entity_embeddings, entity_id_to_label).
    """
    _require("pykeen")
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    # Build a factory from labeled triples
    base_factory = TriplesFactory.from_labeled_triples(triples)

    # Split to satisfy pipeline's dataset checks (some versions require testing as well)
    training, testing, validation = base_factory.split(
        ratios=(0.8, 0.1, 0.1),
        random_state=cfg.random_seed,
    )

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=cfg.model,
        model_kwargs=dict(embedding_dim=cfg.embedding_dim),
        training_kwargs=dict(num_epochs=cfg.epochs, batch_size=cfg.batch_size),
        optimizer_kwargs=dict(lr=cfg.lr),
        random_seed=cfg.random_seed,
        device="cpu",  # keep CPU for portability
    )

    model = result.model

    # Extract entity embeddings robustly across PyKEEN versions
    rep = model.entity_representations[0]
    emb = None
    # Common cases
    if hasattr(rep, "weight"):
        try:
            emb = rep.weight.detach().cpu().numpy()
        except Exception:
            emb = None
    if emb is None and hasattr(rep, "_embeddings") and hasattr(rep._embeddings, "weight"):
        emb = rep._embeddings.weight.detach().cpu().numpy()
    if emb is None and hasattr(rep, "get_in_canonical_shape"):
        emb = rep.get_in_canonical_shape().detach().cpu().numpy()
    if emb is None:
        # Last resort: call representation (may work in some versions)
        emb = rep().detach().cpu().numpy()

    entity_id_to_label = base_factory.entity_id_to_label
    labels = [entity_id_to_label[i] for i in range(len(entity_id_to_label))]
    return result, base_factory, emb, labels

# --------- Clustering + projection ---------
def standardize(X: np.ndarray) -> np.ndarray:
    _require("sklearn", "sklearn")
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)

def project_umap(X: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    try:
        from umap import UMAP
    except Exception:
        try:
            from umap.umap_ import UMAP
        except Exception as e:
            raise RuntimeError("UMAP not available. Install: pip install umap-learn") from e
    um = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
    return um.fit_transform(X)

def build_knn_edges(X: np.ndarray, k: int) -> Tuple[List[Tuple[int,int]], List[float]]:
    _require("sklearn", "sklearn")
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples to build k-NN graph, got n={n}.")

    k_eff = min(k + 1, n)  # +1 to include self then drop it
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X)
    dists, inds = nn.kneighbors(X)

    edges: List[Tuple[int,int]] = []
    weights: List[float] = []
    for i in range(n):
        for j, d in zip(inds[i], dists[i]):
            if i == j:
                continue
            edges.append((i, int(j)))
            # Similarity weight from distance (avoid 0 div)
            weights.append(float(1.0 / (1.0 + d)))
    return edges, weights

def leiden_cluster(edges: List[Tuple[int,int]], weights: List[float], n_vertices: int, resolution: float, seed: int):
    _require("igraph", "igraph")
    _require("leidenalg")
    import igraph as ig
    import leidenalg

    g = ig.Graph(n=n_vertices, edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights

    # RBConfigurationVertexPartition is a common choice for weighted graphs
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"] if weights else None,
        resolution_parameter=resolution,
        seed=seed,
    )

    labels = np.asarray(part.membership, dtype=int)
    modularity = float(part.modularity)
    return labels, modularity

def hdbscan_cluster(X: np.ndarray, min_cluster_size: int, min_samples: Optional[int]):
    _require("hdbscan")
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    return labels

def silhouette_for_hdbscan(X: np.ndarray, labels: np.ndarray) -> float:
    _require("sklearn", "sklearn")
    from sklearn.metrics import silhouette_score

    # Ignore noise points (-1)
    mask = labels != -1
    if mask.sum() < 3:
        return -1.0
    uniq = np.unique(labels[mask])
    if len(uniq) < 2:
        return -1.0
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return -1.0

# --------- Plotting ---------
def plot_scatter(umap_xy: np.ndarray, labels: np.ndarray, title: str, out_png: Path):
    _require("matplotlib", "matplotlib")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.scatter(umap_xy[:, 0], umap_xy[:, 1], c=labels, s=10)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300)
    plt.close()

# --------- Cluster explanations (SPARQL templates + optional execution) ---------
def write_sparql_templates(base_iri: str, out_path: Path):
    q1 = f"""PREFIX bc: <{base_iri}>
SELECT ?feature ?category (COUNT(?patient) AS ?n)
WHERE {{
  ?patient a bc:Patient .
  ?patient ?feature ?category .
  FILTER(isIRI(?category))
  FILTER(REGEX(STR(?category), "Low|Med|Medium|High", "i"))
}}
GROUP BY ?feature ?category
ORDER BY DESC(?n)
LIMIT 30
"""
    q2 = f"""PREFIX bc: <{base_iri}>
SELECT ?patient ?feature ?category
WHERE {{
  ?patient a bc:Patient .
  ?patient ?feature ?category .
  FILTER(isIRI(?category))
  FILTER(REGEX(STR(?category), "High", "i"))
}}
LIMIT 50
"""
    ensure_dir(out_path.parent)
    out_path.write_text("### Query 1: top categorical features (Low/Med/High)\n" + q1 +
                        "\n### Query 2: example patients with 'High' categories\n" + q2,
                        encoding="utf-8")

def summarize_cluster_edges(graph, patient_uris: List[str], top_k: int = 15) -> pd.DataFrame:
    """
    Lightweight summary for a cluster: count (predicate, object) for each patient (object must be IRI, and match Low/Med/High).
    This avoids heavy SPARQL VALUES lists, but still produces interpretable patterns.
    """
    from rdflib.term import URIRef, Literal

    counts: Dict[Tuple[str, str], int] = {}
    patient_set = set(patient_uris)

    for s, p, o in graph:
        if str(s) not in patient_set:
            continue
        if isinstance(o, Literal):
            continue
        o_str = str(o)
        if not any(tok.lower() in o_str.lower() for tok in ["low", "med", "medium", "high"]):
            continue
        key = (str(p), o_str)
        counts[key] = counts.get(key, 0) + 1

    rows = [(pred, obj, n) for (pred, obj), n in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return pd.DataFrame(rows, columns=["predicate", "category_node", "count"])

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_ttl", required=True, help="Path to .ttl file (RDF/Turtle)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--patient_prefix", default="Patient_", help="Substring used to select patient entities (default: Patient_)")
    ap.add_argument("--include_literals_as_nodes", action="store_true", help="Convert literal objects into nodes (not recommended)")
    ap.add_argument("--k", type=int, default=15, help="k for k-NN graph in Leiden (default: 15)")
    ap.add_argument("--umap_n_neighbors", type=int, default=20)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--pykeen_model", default="TransE", help="PyKEEN model name, e.g., TransE, DistMult, ComplEx, RotatE")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--leiden_resolution", type=float, default=1.0)
    ap.add_argument("--hdbscan_min_cluster_size", type=int, default=15)
    ap.add_argument("--hdbscan_min_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    set_seed(args.seed)

    ttl_path = Path(args.input_ttl)
    out_dir = ensure_dir(Path(args.out_dir))
    ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "tables")
    ensure_dir(out_dir / "explanations")

    if not ttl_path.exists():
        raise FileNotFoundError(f"Input TTL not found: {ttl_path.resolve()}")

    print("📥 Loading RDF graph...")
    g = load_graph_ttl(ttl_path)
    print(f"🔹 Total triples: {len(g)}")

    base_iri = guess_base_iri(g)
    write_sparql_templates(base_iri, out_dir / "explanations" / "sparql_templates.txt")

    print("🔄 Converting RDF graph to labeled triples for PyKEEN (dropping Literal tails by default)...")
    triples = graph_to_labeled_triples(g, include_literals_as_nodes=args.include_literals_as_nodes)
    print(f"🔹 Triples for PyKEEN: {triples.shape[0]}")

    # Export TSV (optional, but handy for reproducibility)
    tsv_path = out_dir / "tables" / "triples_for_pykeen.tsv"
    pd.DataFrame(triples, columns=["head", "relation", "tail"]).to_csv(tsv_path, sep="\t", index=False)
    print(f"✅ Exported triples TSV: {tsv_path}")

    print("🧠 Training PyKEEN to learn KG embeddings...")
    cfg = PykeenTrainConfig(
        model=args.pykeen_model,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        random_seed=args.seed,
    )
    result, base_factory, entity_emb, entity_labels = train_pykeen_embeddings(triples, cfg)
    print(f"✅ Learned entity embeddings: {entity_emb.shape}")

    # Select patients
    patient_ids = select_patients(entity_labels, args.patient_prefix)
    if len(patient_ids) < 2:
        print(f"⚠️ Found only {len(patient_ids)} patient entities using prefix '{args.patient_prefix}'.")
        print("   Falling back to ALL entities for clustering (this is not ideal for your manuscript).")
        patient_ids = list(range(len(entity_labels)))

    patient_labels = [entity_labels[i] for i in patient_ids]
    X = entity_emb[np.array(patient_ids)]
    print(f"🔹 Samples used for clustering: {X.shape[0]} (patients)")

    Xs = standardize(X)
    umap_xy = project_umap(Xs, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, random_state=args.seed)

    # Leiden on k-NN graph
    print("🔗 Building k-NN graph + Leiden clustering...")
    edges, weights = build_knn_edges(Xs, k=args.k)
    leiden_labels, modularity = leiden_cluster(
        edges, weights, n_vertices=Xs.shape[0], resolution=args.leiden_resolution, seed=args.seed
    )
    print(f"✅ Leiden modularity: {modularity:.4f}")

    # HDBSCAN
    print("🧩 HDBSCAN clustering...")
    hdb_labels = hdbscan_cluster(Xs, min_cluster_size=args.hdbscan_min_cluster_size, min_samples=args.hdbscan_min_samples)
    sil = silhouette_for_hdbscan(umap_xy, hdb_labels)  # silhouette on 2D space for interpretability
    print(f"✅ HDBSCAN silhouette (UMAP, ignoring noise if possible): {sil:.4f}")

    # Save plots
    fig_leiden = out_dir / "figures" / "Fig4_Leiden_UMAP.png"
    fig_hdb = out_dir / "figures" / "Fig5_HDBSCAN_UMAP.png"
    plot_scatter(umap_xy, leiden_labels, "Leiden Clustering on PyKEEN Embeddings (UMAP)", fig_leiden)
    plot_scatter(umap_xy, hdb_labels, "HDBSCAN Clustering on PyKEEN Embeddings (UMAP)", fig_hdb)
    print(f"🖼️ Saved figures:\n - {fig_leiden}\n - {fig_hdb}")

    # Export table
    df = pd.DataFrame({
        "entity_label": patient_labels,
        "umap_1": umap_xy[:, 0],
        "umap_2": umap_xy[:, 1],
        "leiden_cluster": leiden_labels,
        "hdbscan_cluster": hdb_labels,
    })
    csv_path = out_dir / "tables" / "Clustering_UMAP_Results.csv"
    xlsx_path = out_dir / "tables" / "Clustering_UMAP_Results.xlsx"
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"📄 Saved tables:\n - {csv_path}\n - {xlsx_path}")
    except Exception as e:
        print(f"⚠️ Could not write Excel (missing openpyxl?). Saved CSV only. Details: {e}")
        print(f"📄 Saved table:\n - {csv_path}")

    # Save a short metrics file
    metrics_path = out_dir / "tables" / "Clustering_Metrics.txt"
    metrics_path.write_text(
        f"Leiden modularity (kNN graph): {modularity:.6f}\n"
        f"HDBSCAN silhouette (UMAP; noise excluded when possible): {sil:.6f}\n",
        encoding="utf-8"
    )
    print(f"📌 Saved metrics: {metrics_path}")

    # Explanations (lightweight counts on Low/Med/High object links)
    print("🧾 Generating lightweight cluster explanations (category links)...")
    expl_rows = []
    # Build patient URI list (use original labels)
    for cid in sorted(np.unique(leiden_labels)):
        members = df.loc[df["leiden_cluster"] == cid, "entity_label"].tolist()
        top_df = summarize_cluster_edges(g, members, top_k=15)
        top_df.insert(0, "leiden_cluster", cid)
        expl_rows.append(top_df)
    expl = pd.concat(expl_rows, ignore_index=True) if expl_rows else pd.DataFrame(columns=["leiden_cluster","predicate","category_node","count"])
    expl_csv = out_dir / "explanations" / "leiden_cluster_explanations.csv"
    expl.to_csv(expl_csv, index=False)
    print(f"✅ Saved explanations: {expl_csv}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
