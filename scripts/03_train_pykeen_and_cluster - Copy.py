#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_train_pykeen_and_cluster.py (standalone)

End-to-end pipeline:
1) Load RDF graph from .ttl / .owl / .rdf (rdflib)
2) Convert to labeled triples for PyKEEN
   - Supports literal handling with strategies: auto / drop / entity / categorize
3) Train one or more KGE models with PyKEEN (TransE / RotatE / ComplEx, etc.)
4) Extract patient embeddings
5) UMAP projection
6) Leiden clustering (on k-NN graph) + Modularity
7) HDBSCAN clustering + Silhouette
8) Export: TSV triples, embeddings, UMAP coords, cluster labels, scores, PNG figures (2 separate files)

Windows note:
- If your input path has spaces, wrap it in quotes.
- Prefer a single-line command to avoid CMD line-continuation issues.

Example (CMD, single line):
python scripts\\03_train_pykeen_and_cluster.py --input_rdf "data\\processed\\Breast Cancer Coimbra.owl" --out_dir outputs\\coimbra --patient_prefix Patient_ --models TransE RotatE ComplEx --epochs 50

"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# RDF
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF

# PyKEEN
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# ML / clustering / plotting
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import umap  # umap-learn
import igraph as ig
import leidenalg
import hdbscan
import matplotlib.pyplot as plt


# ----------------------------
# Utility helpers
# ----------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int) -> None:
    # Numpy
    np.random.seed(seed)
    # Torch (PyKEEN)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def safe_local_name(uri: str) -> str:
    # best-effort extraction of local name
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]

def is_patient_label(label: str, patient_prefix: str) -> bool:
    local = safe_local_name(label)
    return local.startswith(patient_prefix)

def literal_is_number(lit: Literal) -> bool:
    # Try numeric parse
    try:
        float(str(lit))
        return True
    except Exception:
        return False

def stable_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]

def parse_rdf_any(path: Path) -> Graph:
    """
    Robust RDF parser for .ttl/.owl/.rdf.
    Tries multiple formats to reduce "unknown url type" / parse failures.
    """
    g = Graph()

    # rdflib can infer format from extension sometimes, but be explicit & robust.
    suffix = path.suffix.lower()
    candidates: List[Optional[str]] = []

    if suffix in {".ttl", ".turtle"}:
        candidates = ["turtle", None]
    elif suffix in {".owl", ".rdf", ".xml"}:
        candidates = ["xml", "application/rdf+xml", None]
    elif suffix in {".nt"}:
        candidates = ["nt", None]
    else:
        candidates = [None, "turtle", "xml", "nt"]

    last_err = None
    for fmt in candidates:
        try:
            g.parse(str(path), format=fmt) if fmt else g.parse(str(path))
            return g
        except Exception as e:
            last_err = e
            g = Graph()

    raise RuntimeError(f"Failed to parse RDF file: {path}\nLast error: {last_err}")


@dataclass
class ConversionStats:
    total_triples: int
    kept_triples: int
    dropped_literal_triples: int
    literal_strategy: str
    numeric_literal_triples: int
    nonnumeric_literal_triples: int


def convert_graph_to_labeled_triples(
    g: Graph,
    patient_prefix: str,
    literal_strategy: str = "auto",
    numeric_bins: int = 3,
    schema_relations: bool = True,
) -> Tuple[List[Tuple[str, str, str]], ConversionStats]:
    """
    Convert rdflib Graph -> list of (head, relation, tail) strings for PyKEEN.

    literal_strategy:
      - "auto": if literal tails dominate, use "categorize", else "drop"
      - "drop": drop triples with Literal tails
      - "entity": turn literal tails into entity strings (may create many entities)
      - "categorize": discretize numeric literals into Low/Med/High (quantiles)
                     and create a small schema around measurements.

    Returns:
      triples: list[(h,r,t)]
      stats: ConversionStats
    """
    total = len(g)

    # Estimate literal tail ratio
    lit_cnt = 0
    num_lit_cnt = 0
    nonnum_lit_cnt = 0
    for _, _, o in g:
        if isinstance(o, Literal):
            lit_cnt += 1
            if literal_is_number(o):
                num_lit_cnt += 1
            else:
                nonnum_lit_cnt += 1

    chosen = literal_strategy
    if literal_strategy == "auto":
        # if >50% literal tails, categorization helps avoid losing most structure
        chosen = "categorize" if (lit_cnt / max(total, 1)) > 0.5 else "drop"

    # Precompute quantile thresholds per predicate for numeric literals (categorize)
    thresholds: Dict[str, Tuple[float, float]] = {}
    if chosen == "categorize":
        # collect numeric values per predicate from patient subjects only (when possible)
        pred_vals: Dict[str, List[float]] = {}
        for s, p, o in g:
            if not isinstance(o, Literal) or not literal_is_number(o):
                continue
            s_label = str(s)
            if patient_prefix and not is_patient_label(s_label, patient_prefix):
                continue
            p_local = safe_local_name(str(p))
            pred_vals.setdefault(p_local, []).append(float(str(o)))

        for p_local, vals in pred_vals.items():
            if len(vals) >= 10:
                q1 = float(np.quantile(vals, 1/3))
                q2 = float(np.quantile(vals, 2/3))
            elif len(vals) >= 3:
                q1 = float(np.quantile(vals, 0.33))
                q2 = float(np.quantile(vals, 0.66))
            else:
                # too few values -> fallback
                q1 = float(np.min(vals)) if vals else 0.0
                q2 = float(np.max(vals)) if vals else 0.0
            thresholds[p_local] = (q1, q2)

    triples: List[Tuple[str, str, str]] = []
    dropped_literal = 0

    # Optional: add lightweight schema nodes (helps embeddings)
    def add_schema():
        # keep schema minimal; you can expand later if needed
        triples.extend([
            ("Patient", "subClassOf", "Entity"),
            ("Measurement", "subClassOf", "Entity"),
            ("Attribute", "subClassOf", "Entity"),
            ("Category", "subClassOf", "Entity"),
            ("Low", "instanceOf", "Category"),
            ("Med", "instanceOf", "Category"),
            ("High", "instanceOf", "Category"),
        ])

    if schema_relations and chosen == "categorize":
        add_schema()

    for s, p, o in g:
        h = str(s)
        r = safe_local_name(str(p))

        if isinstance(o, URIRef):
            t = str(o)
            triples.append((h, r, t))
            continue

        # Literal tail
        if chosen == "drop":
            dropped_literal += 1
            continue

        if chosen == "entity":
            lit_text = str(o)
            # compress & sanitize
            lit_key = f"LIT::{r}::{stable_hash(lit_text)}"
            triples.append((h, r, lit_key))
            continue

        # chosen == "categorize"
        lit_text = str(o)
        if literal_is_number(o):
            val = float(lit_text)
            q1, q2 = thresholds.get(r, (val, val))
            if val <= q1:
                bucket = "Low"
            elif val <= q2:
                bucket = "Med"
            else:
                bucket = "High"

            # build measurement node to keep structure stable
            patient_local = safe_local_name(h)
            meas = f"MEAS::{patient_local}::{r}"
            attr = f"ATTR::{r}"
            cat = f"CAT::{r}::{bucket}"

            # Patient -> Measurement -> (Attribute, Category)
            triples.append((h, "hasMeasurement", meas))
            triples.append((meas, "measurementType", attr))
            triples.append((meas, "hasCategory", cat))
            triples.append((attr, "instanceOf", "Attribute"))
            triples.append((cat, "instanceOf", "Category"))
            triples.append((cat, "categoryLevel", bucket))
        else:
            # non-numeric literal -> entity token
            lit_key = f"LIT::{r}::{stable_hash(lit_text)}"
            triples.append((h, r, lit_key))

    stats = ConversionStats(
        total_triples=total,
        kept_triples=len(triples),
        dropped_literal_triples=dropped_literal,
        literal_strategy=chosen,
        numeric_literal_triples=num_lit_cnt,
        nonnumeric_literal_triples=nonnum_lit_cnt,
    )
    return triples, stats


def to_numpy_triples(triples: Sequence[Tuple[str, str, str]]) -> np.ndarray:
    arr = np.asarray(triples, dtype=str)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Triples array must have shape (n,3). Got: {arr.shape}")
    return arr


def save_triples_tsv(triples: Sequence[Tuple[str, str, str]], out_path: Path) -> None:
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")


def extract_patient_embeddings(result, tf: TriplesFactory, patient_prefix: str) -> Tuple[List[str], np.ndarray]:
    """
    Extract entity embeddings for patient entities (based on patient_prefix).
    Returns labels and 2D numpy array (n_patients, dim)
    """
    import torch

    id_to_entity = tf.entity_id_to_label
    patient_ids: List[int] = []
    patient_labels: List[str] = []
    for idx, label in id_to_entity.items():
        if is_patient_label(label, patient_prefix):
            patient_ids.append(int(idx))
            patient_labels.append(label)

    if len(patient_ids) == 0:
        raise ValueError(
            f"No patient entities found with prefix '{patient_prefix}'. "
            f"Example entity labels: {list(id_to_entity.values())[:5]}"
        )

    rep = result.model.entity_representations[0]
    device = next(result.model.parameters()).device
    idx_tensor = torch.as_tensor(patient_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        emb = rep(indices=idx_tensor)
    emb_np = emb.detach().cpu().numpy()

    # ComplEx may produce complex embeddings -> convert to real-valued features
    if np.iscomplexobj(emb_np):
        emb_np = np.concatenate([emb_np.real, emb_np.imag], axis=1).astype(np.float32, copy=False)

    return patient_labels, emb_np


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        n_components=2,
        metric="cosine",
        random_state=int(seed),
    )
    return reducer.fit_transform(X)


def leiden_on_knn(X: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, float]:
    """
    Leiden clustering on kNN graph built from X (scaled embeddings).
    Returns labels and modularity score.
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for Leiden clustering.")

    # ensure valid k for sklearn knn graph
    k_eff = int(min(max(2, k), n - 1))

    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X)
    knn_graph = nbrs.kneighbors_graph(X=None, mode="connectivity").tocoo()

    edges = list(zip(knn_graph.row.tolist(), knn_graph.col.tolist()))
    g_ig = ig.Graph(n=n, edges=edges, directed=False)
    g_ig.simplify(multiple=True, loops=True)

    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.ModularityVertexPartition,
        seed=int(seed),
    )
    labels = np.asarray(partition.membership, dtype=int)
    modularity = float(partition.modularity)
    return labels, modularity


def hdbscan_cluster(X: np.ndarray, min_cluster_size: int, min_samples: Optional[int], seed: int) -> Tuple[np.ndarray, float]:
    """
    HDBSCAN clustering on X (scaled embeddings).
    Returns labels and silhouette score computed on non-noise points when possible.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples) if min_samples is not None else None,
        metric="euclidean",
        prediction_data=False,
        gen_min_span_tree=False,
    )
    labels = clusterer.fit_predict(X)

    # Silhouette only meaningful if >=2 clusters (excluding noise) and enough points
    mask = labels != -1
    uniq = set(labels[mask].tolist())
    if len(uniq) >= 2 and mask.sum() >= 3:
        sil = float(silhouette_score(X[mask], labels[mask], metric="euclidean"))
    else:
        sil = -1.0
    return labels.astype(int), sil


def plot_clusters(coords: np.ndarray, labels: np.ndarray, title: str, out_png: Path) -> None:
    plt.figure(figsize=(10, 7))
    uniq = np.unique(labels)
    # noise first for readability
    uniq_sorted = sorted(uniq.tolist(), key=lambda x: (x != -1, x))
    for lab in uniq_sorted:
        m = labels == lab
        if lab == -1:
            plt.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.35, label="Noise (-1)")
        else:
            plt.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.75, label=f"C{lab}")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=1.2, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=420)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PyKEEN model(s) + clustering (Leiden/HDBSCAN) + UMAP.")
    p.add_argument("--input_rdf", required=True, help="Path to RDF file (.ttl/.owl/.rdf/.nt).")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--patient_prefix", default="Patient_", help="Prefix of patient entities (local name).")
    p.add_argument("--models", nargs="+", default=["TransE"], help="PyKEEN model names to run, e.g., TransE RotatE ComplEx")
    p.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension.")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs per model.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Literal handling
    p.add_argument("--literal_strategy", choices=["auto", "drop", "entity", "categorize"], default="auto",
                   help="How to handle literal tails when converting RDF to PyKEEN triples.")
    p.add_argument("--schema_relations", action="store_true", help="Add lightweight schema edges when categorizing literals.")
    p.add_argument("--export_triples_tsv", action="store_true", help="Export converted triples as TSV.")

    # UMAP
    p.add_argument("--umap_n_neighbors", type=int, default=20, help="UMAP n_neighbors.")
    p.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")

    # Leiden
    p.add_argument("--k", type=int, default=15, help="k for kNN graph used by Leiden clustering.")

    # HDBSCAN
    p.add_argument("--hdbscan_min_cluster_size", type=int, default=10, help="HDBSCAN min_cluster_size.")
    p.add_argument("--hdbscan_min_samples", type=int, default=None, help="HDBSCAN min_samples (optional).")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    set_seed(args.seed)

    in_path = Path(args.input_rdf)
    if not in_path.exists():
        raise FileNotFoundError(f"Input RDF file not found: {in_path}")

    out_dir = ensure_dir(Path(args.out_dir))

    print("📥 Loading RDF graph...")
    g = parse_rdf_any(in_path)
    print(f"🔹 Total triples: {len(g)}")

    print(f"🔁 Converting RDF graph to labeled triples for PyKEEN (literal_strategy={args.literal_strategy})...")
    triples, stats = convert_graph_to_labeled_triples(
        g=g,
        patient_prefix=args.patient_prefix,
        literal_strategy=args.literal_strategy,
        schema_relations=args.schema_relations,
    )
    print(f"🔹 Triples for PyKEEN: {len(triples)}")
    print(f"   - literal_strategy used: {stats.literal_strategy}")
    if stats.literal_strategy == "drop":
        print(f"   - dropped literal triples: {stats.dropped_literal_triples}")
    else:
        print(f"   - numeric literal triples: {stats.numeric_literal_triples}, non-numeric: {stats.nonnumeric_literal_triples}")

    # Optionally export triples
    tsv_path = out_dir / "pykeen_triples.tsv"
    if args.export_triples_tsv:
        print(f"💾 Saving labeled triples TSV: {tsv_path}")
        save_triples_tsv(triples, tsv_path)

    # Create TriplesFactory
    triples_arr = to_numpy_triples(triples)
    tf = TriplesFactory.from_labeled_triples(triples_arr, create_inverse_triples=False)

    # Split for PyKEEN pipeline (required: training+testing; validation recommended)
    train_tf, test_tf, val_tf = tf.split([0.8, 0.1, 0.1], random_state=args.seed)

    scores_rows = []

    for model_name in args.models:
        model_out = ensure_dir(out_dir / model_name)
        print(f"\n🧠 Training PyKEEN model: {model_name}")
        t0 = time.time()

        # Run pipeline
        result = pipeline(
            training=train_tf,
            testing=test_tf,
            validation=val_tf,
            model=model_name,
            model_kwargs=dict(embedding_dim=int(args.embedding_dim)),
            training_kwargs=dict(num_epochs=int(args.epochs)),
            random_seed=int(args.seed),
            # device="cpu",  # optionally expose as CLI flag if needed
        )
        train_time = time.time() - t0
        print(f"✅ Done training {model_name} in {train_time:.1f}s")

        # Extract embeddings
        patients, X = extract_patient_embeddings(result, tf=tf, patient_prefix=args.patient_prefix)
        print(f"👤 Patient embeddings: {X.shape[0]} patients, dim={X.shape[1]}")

        # Scale
        X_scaled = StandardScaler().fit_transform(X)

        # UMAP
        coords = run_umap(X_scaled, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, seed=args.seed)

        # Leiden
        leiden_labels, modularity = leiden_on_knn(X_scaled, k=args.k, seed=args.seed)
        n_leiden = len(set(leiden_labels.tolist()))
        print(f"🔷 Leiden: clusters={n_leiden}, modularity={modularity:.4f}")

        # HDBSCAN
        hdb_labels, sil = hdbscan_cluster(
            X_scaled,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            seed=args.seed,
        )
        n_hdb = len(set(hdb_labels.tolist())) - (1 if -1 in hdb_labels else 0)
        noise_ratio = float((hdb_labels == -1).mean())
        print(f"🔶 HDBSCAN: clusters={n_hdb}, noise_ratio={noise_ratio:.3f}, silhouette={sil:.4f}")

        # Export CSV (UMAP + labels)
        df = pd.DataFrame({
            "entity": patients,
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
            "leiden_cluster": leiden_labels,
            "hdbscan_cluster": hdb_labels,
        })
        csv_path = model_out / "umap_clusters.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"💾 Saved: {csv_path}")

        # Export embeddings
        emb_path = model_out / "patient_embeddings.npy"
        np.save(emb_path, X)
        print(f"💾 Saved: {emb_path}")

        # Plots (separate figures)
        fig_leiden = model_out / "Fig_Leiden_UMAP.png"
        fig_hdb = model_out / "Fig_HDBSCAN_UMAP.png"
        plot_clusters(coords, leiden_labels, f"Leiden Clustering on {model_name} Embeddings", fig_leiden)
        plot_clusters(coords, hdb_labels, f"HDBSCAN Clustering on {model_name} Embeddings", fig_hdb)
        print(f"🖼️ Saved: {fig_leiden}")
        print(f"🖼️ Saved: {fig_hdb}")

        scores_rows.append({
            "model": model_name,
            "embedding_dim": int(args.embedding_dim),
            "epochs": int(args.epochs),
            "patients": int(X.shape[0]),
            "leiden_clusters": int(n_leiden),
            "leiden_modularity": float(modularity),
            "hdbscan_clusters": int(n_hdb),
            "hdbscan_noise_ratio": float(noise_ratio),
            "hdbscan_silhouette": float(sil),
            "train_time_sec": float(train_time),
            "literal_strategy_used": stats.literal_strategy,
        })

    # Save comparison table
    scores_df = pd.DataFrame(scores_rows)
    scores_csv = out_dir / "clustering_model_comparison.csv"
    scores_df.to_csv(scores_csv, index=False, encoding="utf-8")
    print(f"\n📊 Saved model comparison: {scores_csv}")

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
