#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_export_tableX_missing.py

Exports missing downstream metrics for Table X:
- Meta-Learner F1 and ROC-AUC for each embedding model (TransE/RotatE/ComplEx)
using:
  (a) raw numeric literal features from KG
  (b) patient embeddings from each model folder/zip
  (c) Leiden + HDBSCAN cluster labels from each model folder/zip (one-hot)

Also parses (if available):
- PyKEEN MRR and Hits@10 (both, realistic) from pykeen_test_metrics.csv
"""

from __future__ import annotations
import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ModelArtifacts:
    model_name: str
    embeddings: pd.DataFrame     # patient_uri + emb dims
    clusters: pd.DataFrame       # patient_uri + leiden_cluster + hdbscan_cluster
    pykeen_mrr: Optional[float] = None
    pykeen_hits10: Optional[float] = None


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def guess_base_iri_from_graph(graph) -> str:
    from rdflib.term import URIRef
    for s in graph.subjects():
        if isinstance(s, URIRef):
            ss = str(s)
            if "#" in ss:
                return ss.split("#")[0] + "#"
            if "/" in ss:
                return ss.rsplit("/", 1)[0] + "/"
    return "http://example.org/breastcancer#"


def local_or_full_iri(pred: str, base_iri: str) -> str:
    return pred if pred.startswith("http://") or pred.startswith("https://") else base_iri + pred


def parse_pykeen_metrics_df(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Parse MRR + Hits@10 for Side=both and Rank_type=realistic."""
    try:
        mrr_row = df[(df["Side"] == "both") & (df["Rank_type"] == "realistic") &
                     (df["Metric"] == "adjusted_inverse_harmonic_mean_rank")]
        hits_row = df[(df["Side"] == "both") & (df["Rank_type"] == "realistic") &
                      (df["Metric"] == "hits_at_10")]
        mrr = float(mrr_row["Value"].iloc[0]) if len(mrr_row) else None
        hits10 = float(hits_row["Value"].iloc[0]) if len(hits_row) else None
        return mrr, hits10
    except Exception:
        return None, None


# -----------------------------
# RDF: load + extract X,y
# -----------------------------
def load_graph(kg_path: Path, kg_format: str):
    from rdflib import Graph
    g = Graph()
    with open(kg_path, "rb") as f:
        g.parse(file=f, format=kg_format)
    return g


def extract_patient_uris(graph, patient_prefix: str) -> List[str]:
    pats = set()
    for s in graph.subjects():
        ss = str(s)
        if patient_prefix in ss:
            pats.add(ss)
    return sorted(pats)


def detect_numeric_predicates(graph, patient_prefix: str) -> List[str]:
    from rdflib import Literal
    numeric_preds = set()
    for s, p, o in graph:
        if patient_prefix not in str(s):
            continue
        if not isinstance(o, Literal):
            continue
        if safe_float(str(o)) is not None:
            numeric_preds.add(str(p))
    return sorted(numeric_preds)


def build_Xy_from_graph(graph, patient_prefix: str, label_predicate_iri: str,
                        positive_label: str, numeric_preds: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from rdflib import URIRef
    label_p = URIRef(label_predicate_iri)

    patient_uris = extract_patient_uris(graph, patient_prefix)
    X_rows, y, kept = [], [], []

    for pu in patient_uris:
        s = URIRef(pu)

        lab = None
        for _, _, o in graph.triples((s, label_p, None)):
            lab = str(o)
            break
        if lab is None:
            continue

        row = []
        ok = True
        for p_iri in numeric_preds:
            p = URIRef(p_iri)
            val = None
            for _, _, o in graph.triples((s, p, None)):
                fv = safe_float(str(o))
                if fv is not None:
                    val = fv
                    break
            if val is None:
                ok = False
                break
            row.append(val)

        if not ok:
            continue

        X_rows.append(row)
        y.append(1 if lab == positive_label else 0)
        kept.append(pu)

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y, dtype=int)
    return X, y, kept


# -----------------------------
# Load artifacts from folder OR zip
# -----------------------------
def find_in_dir(root: Path, filename: str) -> Optional[Path]:
    for p in root.rglob(filename):
        return p
    return None


def load_artifacts_from_dir(model_dir: Path, model_name: str) -> ModelArtifacts:
    # embeddings
    emb_csv = find_in_dir(model_dir, "patient_embeddings.csv")
    emb_npy = find_in_dir(model_dir, "patient_embeddings.npy")

    # clusters
    clu_csv1 = find_in_dir(model_dir, "Clustering_UMAP_Results.csv")
    clu_csv2 = find_in_dir(model_dir, "umap_clusters.csv")

    # pykeen metrics optional
    pm_csv = find_in_dir(model_dir, "pykeen_test_metrics.csv")

    if emb_csv is None and emb_npy is None:
        raise FileNotFoundError(f"No patient_embeddings.(csv/npy) found under: {model_dir}")

    if clu_csv1 is None and clu_csv2 is None:
        raise FileNotFoundError(f"No clustering CSV found under: {model_dir}")

    # Read clusters first (needed for mapping when embeddings is npy)
    if clu_csv1 is not None:
        df_cl = pd.read_csv(clu_csv1)
        if "patient_uri" not in df_cl.columns:
            for cand in ["entity", "entity_label"]:
                if cand in df_cl.columns:
                    df_cl = df_cl.rename(columns={cand: "patient_uri"})
                    break
    else:
        df_cl = pd.read_csv(clu_csv2)
        if "entity" in df_cl.columns:
            df_cl = df_cl.rename(columns={"entity": "patient_uri"})

    # normalize cluster col names
    if "leiden_cluster" not in df_cl.columns:
        for cand in ["leiden", "leiden_label"]:
            if cand in df_cl.columns:
                df_cl = df_cl.rename(columns={cand: "leiden_cluster"})
                break
    if "hdbscan_cluster" not in df_cl.columns:
        for cand in ["hdbscan", "hdbscan_label"]:
            if cand in df_cl.columns:
                df_cl = df_cl.rename(columns={cand: "hdbscan_cluster"})
                break

    if "patient_uri" not in df_cl.columns or "leiden_cluster" not in df_cl.columns or "hdbscan_cluster" not in df_cl.columns:
        raise ValueError(f"Clusters CSV missing required columns. Columns={list(df_cl.columns)}")

    df_cl = df_cl[["patient_uri", "leiden_cluster", "hdbscan_cluster"]].copy()
    df_cl["patient_uri"] = df_cl["patient_uri"].astype(str)

    # embeddings
    if emb_csv is not None:
        df_emb = pd.read_csv(emb_csv)
        if "patient_uri" not in df_emb.columns:
            for cand in ["entity", "entity_label", "uri"]:
                if cand in df_emb.columns:
                    df_emb = df_emb.rename(columns={cand: "patient_uri"})
                    break
        if "patient_uri" not in df_emb.columns:
            raise ValueError(f"Embeddings CSV missing patient_uri. Columns={list(df_emb.columns)}")
        df_emb["patient_uri"] = df_emb["patient_uri"].astype(str)
    else:
        # npy requires mapping by order with umap_clusters.csv (entity column)
        if clu_csv2 is None:
            raise ValueError("Found patient_embeddings.npy but missing umap_clusters.csv for mapping URIs.")
        arr = np.load(emb_npy)
        df_map = pd.read_csv(clu_csv2)
        if "entity" not in df_map.columns:
            raise ValueError("umap_clusters.csv must contain column 'entity' to map embeddings.")
        if len(df_map) != arr.shape[0]:
            raise ValueError(f"Embedding rows ({arr.shape[0]}) != mapping rows ({len(df_map)}).")
        cols = ["patient_uri"] + [str(i) for i in range(arr.shape[1])]
        df_emb = pd.DataFrame(np.column_stack([df_map["entity"].values, arr]), columns=cols)
        df_emb["patient_uri"] = df_emb["patient_uri"].astype(str)
        for c in df_emb.columns[1:]:
            df_emb[c] = pd.to_numeric(df_emb[c], errors="coerce")

    mrr, hits10 = (None, None)
    if pm_csv is not None:
        dfm = pd.read_csv(pm_csv)
        mrr, hits10 = parse_pykeen_metrics_df(dfm)

    return ModelArtifacts(model_name=model_name, embeddings=df_emb, clusters=df_cl, pykeen_mrr=mrr, pykeen_hits10=hits10)


def load_artifacts_from_zip(zip_path: Path, model_name: str) -> ModelArtifacts:
    def members(z): return z.namelist()
    def find_first(mems, suffix):
        for m in mems:
            if m.endswith(suffix):
                return m
        return None

    with zipfile.ZipFile(zip_path, "r") as z:
        mems = members(z)

        emb_csv = find_first(mems, "patient_embeddings.csv")
        emb_npy = find_first(mems, "patient_embeddings.npy")
        clu_csv1 = find_first(mems, "Clustering_UMAP_Results.csv")
        clu_csv2 = find_first(mems, "umap_clusters.csv")
        pm_csv = find_first(mems, "pykeen_test_metrics.csv")

        if emb_csv is None and emb_npy is None:
            raise FileNotFoundError(f"No patient_embeddings in zip: {zip_path}")
        if clu_csv1 is None and clu_csv2 is None:
            raise FileNotFoundError(f"No clustering CSV in zip: {zip_path}")

        # clusters
        if clu_csv1 is not None:
            with z.open(clu_csv1) as f:
                df_cl = pd.read_csv(f)
            if "patient_uri" not in df_cl.columns:
                for cand in ["entity", "entity_label"]:
                    if cand in df_cl.columns:
                        df_cl = df_cl.rename(columns={cand: "patient_uri"})
                        break
        else:
            with z.open(clu_csv2) as f:
                df_cl = pd.read_csv(f)
            if "entity" in df_cl.columns:
                df_cl = df_cl.rename(columns={"entity": "patient_uri"})

        if "leiden_cluster" not in df_cl.columns:
            for cand in ["leiden", "leiden_label"]:
                if cand in df_cl.columns:
                    df_cl = df_cl.rename(columns={cand: "leiden_cluster"})
                    break
        if "hdbscan_cluster" not in df_cl.columns:
            for cand in ["hdbscan", "hdbscan_label"]:
                if cand in df_cl.columns:
                    df_cl = df_cl.rename(columns={cand: "hdbscan_cluster"})
                    break

        df_cl = df_cl[["patient_uri", "leiden_cluster", "hdbscan_cluster"]].copy()
        df_cl["patient_uri"] = df_cl["patient_uri"].astype(str)

        # embeddings
        if emb_csv is not None:
            with z.open(emb_csv) as f:
                df_emb = pd.read_csv(f)
            if "patient_uri" not in df_emb.columns:
                for cand in ["entity", "entity_label", "uri"]:
                    if cand in df_emb.columns:
                        df_emb = df_emb.rename(columns={cand: "patient_uri"})
                        break
            df_emb["patient_uri"] = df_emb["patient_uri"].astype(str)
        else:
            if clu_csv2 is None:
                raise ValueError("patient_embeddings.npy requires umap_clusters.csv mapping")
            with z.open(emb_npy) as f:
                arr = np.load(f)
            with z.open(clu_csv2) as f:
                df_map = pd.read_csv(f)
            if "entity" not in df_map.columns:
                raise ValueError("umap_clusters.csv must contain 'entity'")
            cols = ["patient_uri"] + [str(i) for i in range(arr.shape[1])]
            df_emb = pd.DataFrame(np.column_stack([df_map["entity"].values, arr]), columns=cols)
            df_emb["patient_uri"] = df_emb["patient_uri"].astype(str)
            for c in df_emb.columns[1:]:
                df_emb[c] = pd.to_numeric(df_emb[c], errors="coerce")

        mrr, hits10 = (None, None)
        if pm_csv is not None:
            with z.open(pm_csv) as f:
                dfm = pd.read_csv(f)
            mrr, hits10 = parse_pykeen_metrics_df(dfm)

    return ModelArtifacts(model_name=model_name, embeddings=df_emb, clusters=df_cl, pykeen_mrr=mrr, pykeen_hits10=hits10)


def load_model_artifacts(path: Path) -> ModelArtifacts:
    model_name = path.name.replace("model_", "").replace("Model_", "")
    if path.is_dir():
        return load_artifacts_from_dir(path, model_name=model_name)
    if path.is_file() and path.suffix.lower() == ".zip":
        return load_artifacts_from_zip(path, model_name=model_name)
    raise FileNotFoundError(f"Model path not found: {path}")


# -----------------------------
# Feature assembly (robust alignment)
# -----------------------------
def assemble_features(X_raw: np.ndarray, y: np.ndarray, patient_uris: List[str], art: ModelArtifacts,
                      use_embeddings: bool = True, use_clusters: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns X_full, y_used, and merged dataframe in the SAME row order.
    """
    df = pd.DataFrame({"patient_uri": patient_uris})
    df["row_idx"] = np.arange(len(df))

    # merge embeddings
    if use_embeddings:
        emb = art.embeddings.copy()
        emb["patient_uri"] = emb["patient_uri"].astype(str)
        df = df.merge(emb, on="patient_uri", how="left")

        emb_cols = [c for c in df.columns if c.isdigit()]
        if len(emb_cols) == 0:
            raise ValueError(f"No embedding columns detected for {art.model_name}")
        before = len(df)
        df = df.dropna(subset=emb_cols)
        dropped = before - len(df)
        if dropped > 0:
            print(f"⚠️ {art.model_name}: dropped {dropped} patients due to missing embeddings")

    # merge clusters
    if use_clusters:
        cl = art.clusters.copy()
        cl["patient_uri"] = cl["patient_uri"].astype(str)
        df = df.merge(cl, on="patient_uri", how="left")
        df["leiden_cluster"] = df["leiden_cluster"].fillna(-1).astype(int)
        df["hdbscan_cluster"] = df["hdbscan_cluster"].fillna(-1).astype(int)

    # build X and y aligned
    raw_idx = df["row_idx"].to_numpy()
    X_parts = [X_raw[raw_idx, :]]

    if use_embeddings:
        emb_cols = [c for c in df.columns if c.isdigit()]
        X_parts.append(df[emb_cols].to_numpy(dtype=float))

    if use_clusters:
        d1 = pd.get_dummies(df["leiden_cluster"], prefix="leiden", dtype=float)
        d2 = pd.get_dummies(df["hdbscan_cluster"], prefix="hdbscan", dtype=float)
        X_parts.append(pd.concat([d1, d2], axis=1).to_numpy(dtype=float))

    X_full = np.hstack(X_parts)
    y_used = y[raw_idx]
    return X_full, y_used, df


# -----------------------------
# Models + leakage-safe stacking
# -----------------------------
def build_base_models(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["LR"] = LogisticRegression(max_iter=3000, random_state=seed)
    models["kNN"] = KNeighborsClassifier(n_neighbors=7)
    models["SVM"] = SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=seed)
    models["RF"] = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    models["MLP"] = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=600, random_state=seed)

    # optional
    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            eval_metric="logloss", n_jobs=-1
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LGBM"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1
        )
    except Exception:
        pass

    return models


def proba_pos(model, X):
    p = model.predict_proba(X)
    return p[:, 1]


def stacking_meta_learner(X: np.ndarray, y: np.ndarray, base_models: Dict[str, object],
                         seed: int, test_size: float = 0.2, n_folds: int = 5) -> Dict[str, float]:
    """
    Hold-out test split; OOF predictions on training split for meta-features.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    base_names = list(base_models.keys())

    oof = np.zeros((X_tr.shape[0], len(base_names)), dtype=float)
    te_meta = np.zeros((X_te.shape[0], len(base_names)), dtype=float)

    for j, name in enumerate(base_names):
        # scale inside each model to avoid leakage
        base = Pipeline([("scaler", StandardScaler()), ("clf", base_models[name])])

        te_probs_folds = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = clone(base)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx, j] = proba_pos(m, X_tr[va_idx])
            te_probs_folds.append(proba_pos(m, X_te))

        te_meta[:, j] = np.mean(np.vstack(te_probs_folds), axis=0)

    meta = LogisticRegression(max_iter=3000, random_state=seed)
    meta.fit(oof, y_tr)
    prob = proba_pos(meta, te_meta)
    pred = (prob >= 0.5).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_te, pred)),
        "Precision": float(precision_score(y_te, pred, zero_division=0)),
        "Recall": float(recall_score(y_te, pred, zero_division=0)),
        "F1-score": float(f1_score(y_te, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_te, prob)),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_path", required=True)
    ap.add_argument("--kg_format", required=True, choices=["turtle", "xml"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patient_prefix", default="Patient_")
    ap.add_argument("--label_predicate", default="diagnosis")  # local or full IRI
    ap.add_argument("--positive_label", default="M")

    # STILL named model_zips for backward compatibility: can be folder OR zip
    ap.add_argument("--model_zips", nargs="+", required=True, help="Paths to model folders (or .zip) for TransE/RotatE/ComplEx")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--no_embeddings", action="store_true")
    ap.add_argument("--no_clusters", action="store_true")

    args = ap.parse_args()
    out_dir = ensure_dir(Path(args.out_dir))

    g = load_graph(Path(args.kg_path), args.kg_format)
    base_iri = guess_base_iri_from_graph(g)
    label_pred_iri = local_or_full_iri(args.label_predicate, base_iri)

    numeric_preds = detect_numeric_predicates(g, args.patient_prefix)
    if len(numeric_preds) == 0:
        raise ValueError("No numeric predicates detected. Check patient_prefix/kg_format.")

    X_raw, y, patient_uris = build_Xy_from_graph(
        g,
        patient_prefix=args.patient_prefix,
        label_predicate_iri=label_pred_iri,
        positive_label=args.positive_label,
        numeric_preds=numeric_preds
    )

    if len(patient_uris) < 10:
        raise ValueError(f"Too few usable patients after filtering: {len(patient_uris)}")

    base_models = build_base_models(args.seed)
    print(f"Base models used: {list(base_models.keys())}")

    rows = []
    details = {}

    for p in args.model_zips:
        path = Path(p)
        art = load_model_artifacts(path)

        X_full, y_used, df_merge = assemble_features(
            X_raw=X_raw,
            y=y,
            patient_uris=patient_uris,
            art=art,
            use_embeddings=(not args.no_embeddings),
            use_clusters=(not args.no_clusters),
        )

        metrics = stacking_meta_learner(
            X_full, y_used, base_models,
            seed=args.seed, test_size=args.test_size, n_folds=args.n_folds
        )

        row = {
            "DatasetKG": str(Path(args.kg_path).name),
            "Model": art.model_name,
            "PyKEEN_MRR(both,realistic)": art.pykeen_mrr,
            "PyKEEN_Hits@10(both,realistic)": art.pykeen_hits10,
            "Meta_F1": metrics["F1-score"],
            "Meta_ROC-AUC": metrics["ROC-AUC"],
            "Meta_Accuracy": metrics["Accuracy"],
            "Meta_Precision": metrics["Precision"],
            "Meta_Recall": metrics["Recall"],
            "n_patients_used": int(X_full.shape[0]),
            "n_features_full": int(X_full.shape[1]),
        }
        rows.append(row)
        details[art.model_name] = {"metrics": metrics, "n": int(X_full.shape[0]), "d": int(X_full.shape[1])}

        print(f"✅ {art.model_name}: F1={metrics['F1-score']:.4f} | AUC={metrics['ROC-AUC']:.4f} | n={X_full.shape[0]} | d={X_full.shape[1]}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "tableX_downstream_metrics.csv", index=False)
    try:
        with pd.ExcelWriter(out_dir / "tableX_downstream_metrics.xlsx") as w:
            df_out.to_excel(w, sheet_name="TableX", index=False)
    except Exception:
        pass

    (out_dir / "tableX_downstream_metrics_all.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("\n📄 Saved:")
    print(" -", out_dir / "tableX_downstream_metrics.csv")
    print(" -", out_dir / "tableX_downstream_metrics.xlsx")


if __name__ == "__main__":
    main()
