#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_ablation_table3.py

Ablation runner for Table 3 (semantic reasoning beyond correlations).

It extends the logic of the original export script (06_export_tableX_missing.py),
which already includes robust alignment and leakage-safe stacking fileciteturn41file0L1-L16 fileciteturn41file2L6-L43.

FEATURE SETTINGS:
  Baseline A: tabular/statistical features only (numeric literals from KG)
  Baseline B: + KG patient embeddings
  Baseline C: + clustering labels only (Leiden + HDBSCAN one-hot)
  Full (Ours): + SPARQL semantic rule features + stacking

Outputs:
  - table3_ablation.csv              (mean±std across seeds)
  - table3_ablation_per_seed.csv     (per-seed)
  - table3_ablation.xlsx
  - table3_ablation.json

SPARQL feature table format (CSV/XLSX):
  Must contain a patient identifier column named:
    patient_uri  (preferred)
  OR one of:
    entity, entity_label, uri
  plus one or more numeric columns (0/1 indicators or counts).
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

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
    embeddings: Optional[pd.DataFrame] = None     # patient_uri + emb dims
    clusters: Optional[pd.DataFrame] = None       # patient_uri + leiden_cluster + hdbscan_cluster
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
    """Best-effort guess of base IRI used in the KG."""
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


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".csv", ".tsv"]:
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


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
    """Detect predicates with numeric literal objects for patient subjects."""
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
    """Build X (numeric literal features), y (binary), and aligned patient URI list."""
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


def _normalize_patient_uri_column(df: pd.DataFrame) -> pd.DataFrame:
    if "patient_uri" not in df.columns:
        for cand in ["entity", "entity_label", "uri"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "patient_uri"})
                break
    if "patient_uri" not in df.columns:
        raise ValueError(f"Missing patient_uri column. Columns={list(df.columns)}")
    df["patient_uri"] = df["patient_uri"].astype(str)
    return df


def load_artifacts_from_dir(model_dir: Path, model_name: str) -> ModelArtifacts:
    emb_csv = find_in_dir(model_dir, "patient_embeddings.csv")
    emb_npy = find_in_dir(model_dir, "patient_embeddings.npy")

    clu_csv1 = find_in_dir(model_dir, "Clustering_UMAP_Results.csv")
    clu_csv2 = find_in_dir(model_dir, "umap_clusters.csv")

    pm_csv = find_in_dir(model_dir, "pykeen_test_metrics.csv")

    if emb_csv is None and emb_npy is None:
        raise FileNotFoundError(f"No patient_embeddings.(csv/npy) found under: {model_dir}")
    if clu_csv1 is None and clu_csv2 is None:
        raise FileNotFoundError(f"No clustering CSV found under: {model_dir}")

    # clusters
    df_cl = pd.read_csv(clu_csv1 if clu_csv1 is not None else clu_csv2)
    df_cl = _normalize_patient_uri_column(df_cl)

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
    df_cl["leiden_cluster"] = pd.to_numeric(df_cl["leiden_cluster"], errors="coerce").fillna(-1).astype(int)
    df_cl["hdbscan_cluster"] = pd.to_numeric(df_cl["hdbscan_cluster"], errors="coerce").fillna(-1).astype(int)

    # embeddings
    if emb_csv is not None:
        df_emb = pd.read_csv(emb_csv)
        df_emb = _normalize_patient_uri_column(df_emb)
    else:
        if clu_csv2 is None:
            raise ValueError("Found patient_embeddings.npy but missing umap_clusters.csv for mapping URIs.")
        arr = np.load(emb_npy)
        df_map = pd.read_csv(clu_csv2)
        df_map = _normalize_patient_uri_column(df_map)
        if len(df_map) != arr.shape[0]:
            raise ValueError(f"Embedding rows ({arr.shape[0]}) != mapping rows ({len(df_map)}).")
        cols = ["patient_uri"] + [str(i) for i in range(arr.shape[1])]
        df_emb = pd.DataFrame(np.column_stack([df_map["patient_uri"].values, arr]), columns=cols)
        df_emb["patient_uri"] = df_emb["patient_uri"].astype(str)
        for c in df_emb.columns[1:]:
            df_emb[c] = pd.to_numeric(df_emb[c], errors="coerce")

    for c in df_emb.columns:
        if c == "patient_uri":
            continue
        df_emb[c] = pd.to_numeric(df_emb[c], errors="coerce")

    return ModelArtifacts(model_name=model_name, embeddings=df_emb, clusters=df_cl)


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

        if emb_csv is None and emb_npy is None:
            raise FileNotFoundError(f"No patient_embeddings in zip: {zip_path}")
        if clu_csv1 is None and clu_csv2 is None:
            raise FileNotFoundError(f"No clustering CSV in zip: {zip_path}")

        # clusters
        with z.open(clu_csv1 if clu_csv1 is not None else clu_csv2) as f:
            df_cl = pd.read_csv(f)
        df_cl = _normalize_patient_uri_column(df_cl)

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
        df_cl["leiden_cluster"] = pd.to_numeric(df_cl["leiden_cluster"], errors="coerce").fillna(-1).astype(int)
        df_cl["hdbscan_cluster"] = pd.to_numeric(df_cl["hdbscan_cluster"], errors="coerce").fillna(-1).astype(int)

        # embeddings
        if emb_csv is not None:
            with z.open(emb_csv) as f:
                df_emb = pd.read_csv(f)
            df_emb = _normalize_patient_uri_column(df_emb)
        else:
            if clu_csv2 is None:
                raise ValueError("patient_embeddings.npy requires umap_clusters.csv mapping")
            with z.open(emb_npy) as f:
                arr = np.load(f)
            with z.open(clu_csv2) as f:
                df_map = pd.read_csv(f)
            df_map = _normalize_patient_uri_column(df_map)
            cols = ["patient_uri"] + [str(i) for i in range(arr.shape[1])]
            df_emb = pd.DataFrame(np.column_stack([df_map["patient_uri"].values, arr]), columns=cols)
            df_emb["patient_uri"] = df_emb["patient_uri"].astype(str)
            for c in df_emb.columns[1:]:
                df_emb[c] = pd.to_numeric(df_emb[c], errors="coerce")

        for c in df_emb.columns:
            if c == "patient_uri":
                continue
            df_emb[c] = pd.to_numeric(df_emb[c], errors="coerce")

    return ModelArtifacts(model_name=model_name, embeddings=df_emb, clusters=df_cl)


def load_model_artifacts(path: Path) -> ModelArtifacts:
    model_name = path.name.replace("model_", "").replace("Model_", "")
    if path.is_dir():
        return load_artifacts_from_dir(path, model_name=model_name)
    if path.is_file() and path.suffix.lower() == ".zip":
        return load_artifacts_from_zip(path, model_name=model_name)
    raise FileNotFoundError(f"Model path not found: {path}")


# -----------------------------
# SPARQL feature loading
# -----------------------------
def load_sparql_features(path: Path) -> pd.DataFrame:
    df = read_table(path)
    df = _normalize_patient_uri_column(df)
    feat_cols = [c for c in df.columns if c != "patient_uri"]

    keep = ["patient_uri"]
    for c in feat_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            df[c] = s.fillna(0.0)
            keep.append(c)

    if len(keep) == 1:
        raise ValueError(f"SPARQL feature table has no numeric feature columns: {path}")
    return df[keep].copy()


# -----------------------------
# Feature assembly
# -----------------------------
def assemble_features(
    X_raw: np.ndarray,
    y: np.ndarray,
    patient_uris: List[str],
    art: Optional[ModelArtifacts],
    sparql_df: Optional[pd.DataFrame],
    use_embeddings: bool,
    use_clusters: bool,
    use_sparql: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    df = pd.DataFrame({"patient_uri": patient_uris})
    df["row_idx"] = np.arange(len(df))

    # Start with tabular features
    raw_idx = df["row_idx"].to_numpy()
    X_parts: List[np.ndarray] = [X_raw[raw_idx, :]]

    if (use_embeddings or use_clusters) and art is None:
        raise ValueError("Embeddings/clusters requested but model artifacts not provided.")

    if use_embeddings:
        emb = art.embeddings.copy()
        emb["patient_uri"] = emb["patient_uri"].astype(str)
        df = df.merge(emb, on="patient_uri", how="left")

        emb_cols = [c for c in df.columns if c.isdigit()]
        if len(emb_cols) == 0:
            emb_cols = [c for c in df.columns if c not in ["patient_uri", "row_idx"] and pd.api.types.is_numeric_dtype(df[c])]
        if len(emb_cols) == 0:
            raise ValueError(f"No embedding columns detected for {art.model_name}")

        before = len(df)
        df = df.dropna(subset=emb_cols)
        dropped = before - len(df)
        if dropped > 0:
            print(f"⚠️ {art.model_name}: dropped {dropped} patients due to missing embeddings")

        raw_idx = df["row_idx"].to_numpy()
        X_parts = [X_raw[raw_idx, :], df[emb_cols].to_numpy(dtype=float)]

    if use_clusters:
        cl = art.clusters.copy()
        cl["patient_uri"] = cl["patient_uri"].astype(str)
        df = df.merge(cl, on="patient_uri", how="left")
        df["leiden_cluster"] = df["leiden_cluster"].fillna(-1).astype(int)
        df["hdbscan_cluster"] = df["hdbscan_cluster"].fillna(-1).astype(int)

        d1 = pd.get_dummies(df["leiden_cluster"], prefix="leiden", dtype=float)
        d2 = pd.get_dummies(df["hdbscan_cluster"], prefix="hdbscan", dtype=float)
        X_parts.append(pd.concat([d1, d2], axis=1).to_numpy(dtype=float))

    if use_sparql:
        if sparql_df is None:
            raise ValueError("use_sparql=True but no SPARQL feature table loaded.")
        sf = sparql_df.copy()
        sf["patient_uri"] = sf["patient_uri"].astype(str)
        df = df.merge(sf, on="patient_uri", how="left")

        sparql_cols = [c for c in sf.columns if c != "patient_uri"]
        df[sparql_cols] = df[sparql_cols].fillna(0.0)
        X_parts.append(df[sparql_cols].to_numpy(dtype=float))

    X_full = np.hstack(X_parts)
    y_used = y[df["row_idx"].to_numpy()]
    return X_full, y_used, df


# -----------------------------
# Models + leakage-safe stacking (from your 06 script logic) fileciteturn41file2L6-L43
# -----------------------------
def build_base_models(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["LR"] = LogisticRegression(max_iter=3000, random_state=seed)
    models["kNN"] = KNeighborsClassifier(n_neighbors=7)
    models["SVM"] = SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=seed)
    models["RF"] = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    models["MLP"] = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=600, random_state=seed)

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


def stacking_meta_learner(
    X: np.ndarray,
    y: np.ndarray,
    base_models: Dict[str, object],
    seed: int,
    test_size: float = 0.2,
    n_folds: int = 5,
) -> Dict[str, float]:

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    base_names = list(base_models.keys())

    oof = np.zeros((X_tr.shape[0], len(base_names)), dtype=float)
    te_meta = np.zeros((X_te.shape[0], len(base_names)), dtype=float)

    for j, name in enumerate(base_names):
        base = Pipeline([("scaler", StandardScaler()), ("clf", base_models[name])])

        te_probs_folds = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = clone(base)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx, j] = proba_pos(m, X_tr[va_idx])
            te_probs_folds.append(proba_pos(m, X_te))

        te_meta[:, j] = np.mean(np.vstack(te_probs_folds), axis=0)

    meta = LogisticRegression(max_iter=5000, random_state=seed)
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


def summarize_metrics(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed], dtype=float)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    return out


def run_setting(
    setting_name: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    patient_uris: List[str],
    art: Optional[ModelArtifacts],
    sparql_df: Optional[pd.DataFrame],
    seeds: List[int],
    test_size: float,
    n_folds: int,
    use_embeddings: bool,
    use_clusters: bool,
    use_sparql: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    X_full, y_used, _ = assemble_features(
        X_raw=X_raw,
        y=y,
        patient_uris=patient_uris,
        art=art,
        sparql_df=sparql_df,
        use_embeddings=use_embeddings,
        use_clusters=use_clusters,
        use_sparql=use_sparql,
    )

    per_seed = []
    for sd in seeds:
        base_models = build_base_models(sd)
        met = stacking_meta_learner(X_full, y_used, base_models, seed=sd, test_size=test_size, n_folds=n_folds)
        met["seed"] = sd
        per_seed.append(met)

    summ = summarize_metrics(per_seed)
    summ.update({
        "Setting": setting_name,
        "use_embeddings": bool(use_embeddings),
        "use_clusters": bool(use_clusters),
        "use_sparql": bool(use_sparql),
        "use_stacking": True,
        "n_patients_used": int(X_full.shape[0]),
        "n_features_full": int(X_full.shape[1]),
    })
    return summ, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True, help="e.g., WDBC or Coimbra")
    ap.add_argument("--kg_path", required=True)
    ap.add_argument("--kg_format", required=True, choices=["turtle", "xml"])
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--patient_prefix", default="Patient_")
    ap.add_argument("--label_predicate", default="diagnosis")
    ap.add_argument("--positive_label", default="M")

    ap.add_argument("--model_path", required=True,
                    help="Path to ONE model artifact folder (or .zip), e.g., outputs/wdbc/model_TransE")

    ap.add_argument("--sparql_features", default=None,
                    help="CSV/XLSX containing SPARQL rule features per patient (required for Full).")

    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="Recommended: 42 43 44 45 46")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_folds", type=int, default=5)

    ap.add_argument("--skip_full", action="store_true", help="Skip Full if SPARQL features not available.")

    args = ap.parse_args()
    out_dir = ensure_dir(Path(args.out_dir))

    # load KG
    g = load_graph(Path(args.kg_path), args.kg_format)
    base_iri = guess_base_iri_from_graph(g)
    label_pred_iri = local_or_full_iri(args.label_predicate, base_iri)

    numeric_preds = detect_numeric_predicates(g, args.patient_prefix)
    if len(numeric_preds) == 0:
        raise ValueError("No numeric predicates detected. Check --patient_prefix and KG format.")

    X_raw, y, patient_uris = build_Xy_from_graph(
        g,
        patient_prefix=args.patient_prefix,
        label_predicate_iri=label_pred_iri,
        positive_label=args.positive_label,
        numeric_preds=numeric_preds,
    )

    if len(patient_uris) < 10:
        raise ValueError(f"Too few usable patients after filtering: {len(patient_uris)}")

    # load artifacts (single model)
    art = load_model_artifacts(Path(args.model_path))

    # load SPARQL features (optional)
    sparql_df = None
    sparql_path = Path(args.sparql_features) if args.sparql_features else None
    if sparql_path is not None and sparql_path.exists():
        sparql_df = load_sparql_features(sparql_path)
        print(f"✅ Loaded SPARQL features: {sparql_path} | rows={len(sparql_df)} | cols={sparql_df.shape[1]}")
    else:
        if not args.skip_full:
            print("⚠️ No SPARQL feature file provided. Full setting will fail unless --skip_full.")
        else:
            print("ℹ️ No SPARQL feature file provided. Full setting will be skipped (--skip_full).")

    # settings
    settings = [
        ("Baseline A (Tabular only)", False, False, False),
        ("Baseline B (+Embeddings)",  True,  False, False),
        ("Baseline C (+Clusters)",    False, True,  False),
    ]
    if not args.skip_full:
        settings.append(("Full (Ours: +Emb +Clusters +SPARQL)", True, True, True))

    rows = []
    per_seed_rows = []
    details: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "kg_path": str(Path(args.kg_path)),
        "model_path": str(Path(args.model_path)),
        "sparql_features": str(sparql_path) if sparql_path is not None else None,
        "seeds": args.seeds,
        "test_size": args.test_size,
        "n_folds": args.n_folds,
        "positive_label": args.positive_label,
        "label_predicate_iri": label_pred_iri,
    }

    for setting_name, use_emb, use_clu, use_spq in settings:
        if use_spq and sparql_df is None:
            raise ValueError(
                "Full setting requires SPARQL features, but none were loaded.\n"
                "Pass --sparql_features <path_to_csv/xlsx> OR run with --skip_full."
            )

        summ, per_seed = run_setting(
            setting_name=setting_name,
            X_raw=X_raw,
            y=y,
            patient_uris=patient_uris,
            art=art,
            sparql_df=sparql_df,
            seeds=args.seeds,
            test_size=args.test_size,
            n_folds=args.n_folds,
            use_embeddings=use_emb,
            use_clusters=use_clu,
            use_sparql=use_spq,
        )

        summ_row = {"Dataset": args.dataset_name, "ArtifactsModel": art.model_name, **summ}
        rows.append(summ_row)

        for m in per_seed:
            per_seed_rows.append({
                "Dataset": args.dataset_name,
                "ArtifactsModel": art.model_name,
                "Setting": setting_name,
                "use_embeddings": use_emb,
                "use_clusters": use_clu,
                "use_sparql": use_spq,
                **m,
            })

        print(f"✅ {setting_name}: F1={summ['F1-score_mean']:.4f}±{summ['F1-score_std']:.4f} | "
              f"AUC={summ['ROC-AUC_mean']:.4f}±{summ['ROC-AUC_std']:.4f} | "
              f"n={summ['n_patients_used']} | d={summ['n_features_full']}")

    df_out = pd.DataFrame(rows)

    # deltas vs Baseline A
    base = df_out[df_out["Setting"].str.contains("Baseline A")].head(1)
    if len(base) == 1:
        for metric in ["F1-score_mean", "ROC-AUC_mean"]:
            b = float(base[metric].iloc[0])
            df_out[f"Δ{metric}_vs_A"] = df_out[metric].astype(float) - b

    out_csv = out_dir / "table3_ablation.csv"
    df_out.to_csv(out_csv, index=False)

    df_seed = pd.DataFrame(per_seed_rows)
    out_seed_csv = out_dir / "table3_ablation_per_seed.csv"
    df_seed.to_csv(out_seed_csv, index=False)

    try:
        with pd.ExcelWriter(out_dir / "table3_ablation.xlsx") as w:
            df_out.to_excel(w, sheet_name="Table3_mean_std", index=False)
            df_seed.to_excel(w, sheet_name="per_seed", index=False)
    except Exception:
        pass

    details["rows_mean_std"] = rows
    details["rows_per_seed"] = per_seed_rows
    (out_dir / "table3_ablation.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    print("\n📄 Saved:")
    print(" -", out_csv)
    print(" -", out_seed_csv)
    print(" -", out_dir / "table3_ablation.xlsx")
    print(" -", out_dir / "table3_ablation.json")


if __name__ == "__main__":
    main()
