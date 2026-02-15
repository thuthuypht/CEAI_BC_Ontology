#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_ablation_table3_improved.py

Improved ablation runner for Table 3.

This script extends the Table 3 ablation runner by:
  1) Applying PCA to KG embeddings (fit on training only).
  2) Adding HDBSCAN soft features (membership strength; outlier proxy = 1-strength),
     where HDBSCAN is fit on training only in the PCA embedding space, and test points
     are assigned via approximate_predict.
  3) Adding centroid-distance features (dist to Leiden centroid; dist to nearest HDBSCAN centroid;
     dist to assigned HDBSCAN centroid; dist to global centroid) computed using training centroids only.
  4) Tuning key hyperparameters via CV on the training split only (no test leakage):
       - PCA n_components (chosen by CV ROC-AUC with Logistic Regression)
       - Base model hyperparameters (small grids, CV ROC-AUC)
       - Meta-learner hyperparameter (LogReg C, CV ROC-AUC on OOF features)

Outputs (saved under --out_dir):
  - table3_ablation_improved.csv
  - table3_ablation_improved_per_seed.csv
  - table3_ablation_improved.json

Notes:
  - This script does NOT require PyKEEN. It uses already-exported artifacts:
      patient_embeddings.csv and Clustering_UMAP_Results.csv under --model_path.
  - If SPARQL features are not available, run with --skip_full.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


@dataclass
class ModelArtifacts:
    model_name: str
    embeddings: pd.DataFrame
    clusters: pd.DataFrame


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".csv", ".tsv"]:
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type: %s" % path)


def _normalize_patient_uri_column(df: pd.DataFrame) -> pd.DataFrame:
    if "patient_uri" not in df.columns:
        for cand in ["entity", "entity_label", "uri"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "patient_uri"})
                break
    if "patient_uri" not in df.columns:
        raise ValueError("Missing patient_uri column. Columns=%s" % list(df.columns))
    df["patient_uri"] = df["patient_uri"].astype(str)
    return df


def find_in_dir(root: Path, filename: str) -> Optional[Path]:
    for p in root.rglob(filename):
        return p
    return None


def load_graph(kg_path: Path, kg_format: str):
    from rdflib import Graph
    g = Graph()
    with open(kg_path, "rb") as f:
        g.parse(file=f, format=kg_format)
    return g


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


def build_Xy_from_graph(
    graph,
    patient_prefix: str,
    label_predicate_iri: str,
    positive_label: str,
    numeric_preds: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    from rdflib import URIRef
    label_p = URIRef(label_predicate_iri)

    patient_uris = extract_patient_uris(graph, patient_prefix)

    rows = []
    y = []

    for pu in patient_uris:
        s = URIRef(pu)

        lab = None
        for _, _, o in graph.triples((s, label_p, None)):
            lab = str(o)
            break
        if lab is None:
            continue

        feat = {}
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
            feat[p_iri] = val

        if not ok:
            continue

        rows.append({"patient_uri": pu, **feat})
        y.append(1 if lab == positive_label else 0)

    if len(rows) == 0:
        raise ValueError("No usable patients after filtering. Check patient_prefix/label_predicate.")

    df_tab = pd.DataFrame(rows)
    y = np.asarray(y, dtype=int)
    return df_tab, y


def load_artifacts_from_dir(model_dir: Path) -> ModelArtifacts:
    model_name = model_dir.name.replace("model_", "").replace("Model_", "")

    emb_csv = find_in_dir(model_dir, "patient_embeddings.csv")
    clu_csv = find_in_dir(model_dir, "Clustering_UMAP_Results.csv")

    if emb_csv is None:
        raise FileNotFoundError("No patient_embeddings.csv found under: %s" % model_dir)
    if clu_csv is None:
        raise FileNotFoundError("No Clustering_UMAP_Results.csv found under: %s" % model_dir)

    df_emb = pd.read_csv(emb_csv)
    df_emb = _normalize_patient_uri_column(df_emb)

    emb_cols = [c for c in df_emb.columns if c != "patient_uri"]
    new_cols = {}
    for c in emb_cols:
        new_cols[c] = "emb_%s" % str(c)
    df_emb = df_emb.rename(columns=new_cols)
    for c in df_emb.columns:
        if c != "patient_uri":
            df_emb[c] = pd.to_numeric(df_emb[c], errors="coerce")

    df_cl = pd.read_csv(clu_csv)
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

    return ModelArtifacts(model_name=model_name, embeddings=df_emb, clusters=df_cl)


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
        raise ValueError("SPARQL feature table has no numeric columns: %s" % path)
    return df[keep].copy()


def tune_pca_components(
    X_tab_tr: np.ndarray,
    X_emb_tr: np.ndarray,
    y_tr: np.ndarray,
    candidates: List[int],
    seed: int,
    cv_folds: int,
) -> int:
    best_k = None
    best_auc = -1.0
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for k in candidates:
        k_eff = int(min(k, X_emb_tr.shape[1]))
        if k_eff < 2:
            continue
        pca = PCA(n_components=k_eff, random_state=seed)
        Z = pca.fit_transform(X_emb_tr)
        X = np.hstack([X_tab_tr, Z])

        aucs = []
        for tr_idx, va_idx in skf.split(X, y_tr):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000, random_state=seed)),
            ])
            clf.fit(X[tr_idx], y_tr[tr_idx])
            prob = clf.predict_proba(X[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_tr[va_idx], prob))
        m = float(np.mean(aucs)) if len(aucs) else -1.0
        if m > best_auc:
            best_auc = m
            best_k = k_eff

    if best_k is None:
        best_k = int(min(16, X_emb_tr.shape[1]))
    return best_k


def fit_hdbscan_on_train(Z_train: np.ndarray, min_cluster_size: int, min_samples: Optional[int]):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(Z_train)
    strengths = getattr(clusterer, "probabilities_", None)
    if strengths is None:
        strengths = np.ones(len(labels), dtype=float)
    return clusterer, labels, strengths


def hdbscan_predict(clusterer, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import hdbscan
    labels, strengths = hdbscan.approximate_predict(clusterer, Z)
    strengths = np.asarray(strengths, dtype=float)
    return labels.astype(int), strengths


def centroid_distances(
    Z: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    label_prefix: str,
) -> pd.DataFrame:
    Z = np.asarray(Z, dtype=float)
    labels = np.asarray(labels, dtype=int)

    global_cent = Z[train_mask].mean(axis=0)
    centroids: Dict[int, np.ndarray] = {}
    for cid in sorted(np.unique(labels[train_mask])):
        if cid == -1:
            continue
        idx = np.where(train_mask & (labels == cid))[0]
        if len(idx) == 0:
            continue
        centroids[int(cid)] = Z[idx].mean(axis=0)

    dist_global = np.linalg.norm(Z - global_cent, axis=1)

    if len(centroids) == 0:
        return pd.DataFrame({
            "dist_to_assigned_%s_centroid" % label_prefix: dist_global,
            "dist_to_nearest_%s_centroid" % label_prefix: dist_global,
            "dist_to_global_centroid": dist_global,
        })

    cids = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in cids], axis=0)
    dmat = np.sqrt(((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    dist_nearest = dmat.min(axis=1)

    assigned = np.empty(Z.shape[0], dtype=float)
    cid_to_pos = {cid: j for j, cid in enumerate(cids)}
    for i in range(Z.shape[0]):
        cid = int(labels[i])
        if cid in cid_to_pos:
            assigned[i] = dmat[i, cid_to_pos[cid]]
        else:
            assigned[i] = dist_nearest[i]

    return pd.DataFrame({
        "dist_to_assigned_%s_centroid" % label_prefix: assigned,
        "dist_to_nearest_%s_centroid" % label_prefix: dist_nearest,
        "dist_to_global_centroid": dist_global,
    })


def build_base_model_grids(seed: int) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    grids: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {}

    grids["LR"] = (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=seed))]),
        {"clf__C": [0.1, 1.0, 10.0]},
    )
    grids["kNN"] = (
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [5, 7, 11]},
    )
    grids["SVM"] = (
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, kernel="rbf", random_state=seed))]),
        {"clf__C": [0.5, 2.0, 8.0], "clf__gamma": ["scale", "auto"]},
    )
    grids["RF"] = (
        Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1))]),
        {"clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 2, 4]},
    )
    grids["MLP"] = (
        Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=800, random_state=seed, early_stopping=True))]),
        {"clf__hidden_layer_sizes": [(64, 32), (128, 64)], "clf__alpha": [1e-4, 1e-3]},
    )

    try:
        from xgboost import XGBClassifier
        grids["XGB"] = (
            Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(
                n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                random_state=seed, eval_metric="logloss", n_jobs=-1
            ))]),
            {"clf__max_depth": [3, 5], "clf__reg_lambda": [1.0, 5.0]},
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        grids["LGBM"] = (
            Pipeline([("scaler", StandardScaler()), ("clf", LGBMClassifier(
                n_estimators=800, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
                random_state=seed, n_jobs=-1
            ))]),
            {"clf__num_leaves": [15, 31], "clf__reg_lambda": [0.0, 1.0, 5.0]},
        )
    except Exception:
        pass

    return grids


def tune_base_models(X_tr: np.ndarray, y_tr: np.ndarray, seed: int, cv_folds: int, n_jobs: int) -> Dict[str, Pipeline]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    tuned: Dict[str, Pipeline] = {}
    grids = build_base_model_grids(seed)

    for name, (pipe, grid) in grids.items():
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=skf, n_jobs=n_jobs, refit=True)
        gs.fit(X_tr, y_tr)
        tuned[name] = gs.best_estimator_

    return tuned


def stacking_with_tuned_models(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
    tuned_models: Dict[str, Pipeline],
    cv_folds: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    base_names = list(tuned_models.keys())

    oof = np.zeros((X_tr.shape[0], len(base_names)), dtype=float)
    te_meta = np.zeros((X_te.shape[0], len(base_names)), dtype=float)

    for j, name in enumerate(base_names):
        base_template = tuned_models[name]
        te_probs_folds = []

        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = Pipeline(steps=base_template.steps)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx, j] = m.predict_proba(X_tr[va_idx])[:, 1]
            te_probs_folds.append(m.predict_proba(X_te)[:, 1])

        te_meta[:, j] = np.mean(np.vstack(te_probs_folds), axis=0)

    meta_grid = [0.1, 1.0, 10.0]
    best_meta = None
    best_auc = -1.0
    best_C = 1.0

    for C in meta_grid:
        aucs = []
        for tr_idx, va_idx in skf.split(oof, y_tr):
            meta = LogisticRegression(max_iter=8000, random_state=seed, C=C)
            meta.fit(oof[tr_idx], y_tr[tr_idx])
            prob_va = meta.predict_proba(oof[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_tr[va_idx], prob_va))
        mauc = float(np.mean(aucs)) if len(aucs) else -1.0
        if mauc > best_auc:
            best_auc = mauc
            best_C = C
            best_meta = LogisticRegression(max_iter=8000, random_state=seed, C=C)

    best_meta.fit(oof, y_tr)
    prob = best_meta.predict_proba(te_meta)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_te, pred)),
        "Precision": float(precision_score(y_te, pred, zero_division=0)),
        "Recall": float(recall_score(y_te, pred, zero_division=0)),
        "F1-score": float(f1_score(y_te, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_te, prob)),
    }
    details = {
        "meta_C": float(best_C),
        "meta_cv_auc": float(best_auc),
        "base_models": base_names,
    }
    return metrics, details


def build_features_for_seed(
    df_tab: pd.DataFrame,
    y: np.ndarray,
    art: ModelArtifacts,
    seed: int,
    test_size: float,
    use_embeddings: bool,
    use_clusters: bool,
    use_sparql: bool,
    sparql_df: Optional[pd.DataFrame],
    pca_candidates: List[int],
    pca_cv_folds: int,
    hdb_min_cluster_size: int,
    hdb_min_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    df = df_tab.copy()
    df = df.merge(art.embeddings, on="patient_uri", how="inner")
    df = df.merge(art.clusters, on="patient_uri", how="left")

    if use_sparql:
        if sparql_df is None:
            raise ValueError("use_sparql=True but sparql_df is None")
        df = df.merge(sparql_df, on="patient_uri", how="left")
        sp_cols = [c for c in sparql_df.columns if c != "patient_uri"]
        df[sp_cols] = df[sp_cols].fillna(0.0)

    tab_index = {u: i for i, u in enumerate(df_tab["patient_uri"].astype(str).tolist())}
    y_aligned = np.array([y[tab_index[u]] for u in df["patient_uri"].astype(str).tolist()], dtype=int)

    idx_all = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx_all, test_size=test_size, stratify=y_aligned, random_state=seed)

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[tr_idx] = True

    tab_cols = [c for c in df_tab.columns if c != "patient_uri"]
    X_tab = df[tab_cols].to_numpy(dtype=float)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X_emb = df[emb_cols].to_numpy(dtype=float) if len(emb_cols) else None
    if use_embeddings and X_emb is None:
        raise ValueError("Embeddings requested but not found under model artifacts.")

    details: Dict[str, Any] = {}
    Z_pca = None
    if use_embeddings:
        chosen_pca = tune_pca_components(
            X_tab_tr=X_tab[tr_idx],
            X_emb_tr=X_emb[tr_idx],
            y_tr=y_aligned[tr_idx],
            candidates=pca_candidates,
            seed=seed,
            cv_folds=pca_cv_folds,
        )
        pca = PCA(n_components=chosen_pca, random_state=seed)
        pca.fit(X_emb[tr_idx])
        Z_pca = pca.transform(X_emb)
        details["pca_components"] = int(chosen_pca)
        details["pca_explained_var_sum"] = float(np.sum(pca.explained_variance_ratio_))

    clu_feat = None
    if use_clusters:
        if Z_pca is not None:
            Z_space = Z_pca
        else:
            Z_space = X_emb

        leiden_labels = df["leiden_cluster"].fillna(-1).astype(int).to_numpy()

        Z_train = Z_space[tr_idx]
        clusterer, hdb_labels_tr, hdb_strength_tr = fit_hdbscan_on_train(
            Z_train, min_cluster_size=hdb_min_cluster_size, min_samples=hdb_min_samples
        )

        hdb_labels_all = np.full(len(df), -1, dtype=int)
        hdb_strength_all = np.zeros(len(df), dtype=float)
        hdb_labels_all[tr_idx] = hdb_labels_tr
        hdb_strength_all[tr_idx] = hdb_strength_tr

        if len(te_idx) > 0:
            hdb_labels_te, hdb_strength_te = hdbscan_predict(clusterer, Z_space[te_idx])
            hdb_labels_all[te_idx] = hdb_labels_te
            hdb_strength_all[te_idx] = hdb_strength_te

        hdb_prob = hdb_strength_all
        hdb_outlier_proxy = 1.0 - hdb_strength_all

        leiden_train = leiden_labels[tr_idx]
        uniq_leiden = sorted([int(x) for x in np.unique(leiden_train) if x != -1])
        leiden_onehot = np.zeros((len(df), max(1, len(uniq_leiden))), dtype=float)
        if len(uniq_leiden) > 0:
            col_map = {cid: j for j, cid in enumerate(uniq_leiden)}
            for i in range(len(df)):
                cid = int(leiden_labels[i])
                if cid in col_map:
                    leiden_onehot[i, col_map[cid]] = 1.0

        dist_leiden = centroid_distances(Z_space, leiden_labels, train_mask=train_mask, label_prefix="leiden")
        dist_hdb = centroid_distances(Z_space, hdb_labels_all, train_mask=train_mask, label_prefix="hdbscan")

        clu_feat = np.hstack([
            leiden_onehot,
            hdb_prob.reshape(-1, 1),
            hdb_outlier_proxy.reshape(-1, 1),
            dist_leiden.to_numpy(dtype=float),
            dist_hdb.to_numpy(dtype=float),
        ])

        details["hdb_min_cluster_size"] = int(hdb_min_cluster_size)
        details["hdb_min_samples"] = None if hdb_min_samples is None else int(hdb_min_samples)
        details["hdb_n_clusters_train"] = int(len([c for c in np.unique(hdb_labels_tr) if c != -1]))
        details["hdb_noise_frac_train"] = float(np.mean(hdb_labels_tr == -1))

    sp_cols = []
    if use_sparql and sparql_df is not None:
        sp_cols = [c for c in sparql_df.columns if c != "patient_uri"]

    parts = [X_tab]
    if use_embeddings:
        parts.append(Z_pca)
    if use_clusters:
        parts.append(clu_feat)
    if use_sparql and len(sp_cols) > 0:
        parts.append(df[sp_cols].to_numpy(dtype=float))

    X = np.hstack(parts).astype(float)

    X_tr = X[tr_idx]
    X_te = X[te_idx]
    y_tr = y_aligned[tr_idx]
    y_te = y_aligned[te_idx]

    details["n_train"] = int(len(tr_idx))
    details["n_test"] = int(len(te_idx))
    details["n_features"] = int(X.shape[1])
    return X_tr, y_tr, X_te, y_te, details


def summarize_metrics(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed], dtype=float)
        out["%s_mean" % k] = float(np.mean(vals))
        out["%s_std" % k] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--kg_path", required=True)
    ap.add_argument("--kg_format", required=True, choices=["turtle", "xml"])
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--patient_prefix", default="Patient_")
    ap.add_argument("--label_predicate", default="diagnosis")
    ap.add_argument("--positive_label", default="M")

    ap.add_argument("--model_path", required=True)

    ap.add_argument("--sparql_features", default=None)
    ap.add_argument("--skip_full", action="store_true")

    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--pca_cv_folds", type=int, default=5)
    ap.add_argument("--pca_components", nargs="+", type=int, default=[8, 16, 32])

    ap.add_argument("--hdbscan_min_cluster_size", type=int, default=15)
    ap.add_argument("--hdbscan_min_samples", type=int, default=None)

    ap.add_argument("--n_jobs", type=int, default=-1)

    args = ap.parse_args()
    out_dir = ensure_dir(Path(args.out_dir))

    g = load_graph(Path(args.kg_path), args.kg_format)
    base_iri = guess_base_iri_from_graph(g)
    label_pred_iri = local_or_full_iri(args.label_predicate, base_iri)

    numeric_preds = detect_numeric_predicates(g, args.patient_prefix)
    if len(numeric_preds) == 0:
        raise ValueError("No numeric predicates detected. Check --patient_prefix and KG content.")

    df_tab, y = build_Xy_from_graph(
        g,
        patient_prefix=args.patient_prefix,
        label_predicate_iri=label_pred_iri,
        positive_label=args.positive_label,
        numeric_preds=numeric_preds,
    )

    art = load_artifacts_from_dir(Path(args.model_path))

    sparql_df = None
    if args.sparql_features is not None:
        sp_path = Path(args.sparql_features)
        if sp_path.exists():
            sparql_df = load_sparql_features(sp_path)

    settings = [
        ("Baseline A (Tabular only)", False, False, False),
        ("Baseline B (+Embeddings PCA)", True, False, False),
        ("Baseline C (+Clusters soft +dist)", True, True, False),
    ]
    if not args.skip_full:
        settings.append(("Full (Tabular +Emb PCA +Clusters +SPARQL +Stacking)", True, True, True))

    rows = []
    per_seed_rows = []
    detail_dump: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "kg_path": str(Path(args.kg_path)),
        "kg_format": args.kg_format,
        "model_path": str(Path(args.model_path)),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "cv_folds": args.cv_folds,
        "pca_candidates": args.pca_components,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "positive_label": args.positive_label,
        "label_predicate_iri": label_pred_iri,
        "settings": [],
    }

    for setting_name, use_emb, use_clu, use_spq in settings:
        if use_spq and sparql_df is None:
            raise ValueError("Full setting requires --sparql_features unless --skip_full is used.")

        per_seed_metrics = []
        per_seed_details = []

        for sd in args.seeds:
            X_tr, y_tr, X_te, y_te, feat_details = build_features_for_seed(
                df_tab=df_tab,
                y=y,
                art=art,
                seed=sd,
                test_size=args.test_size,
                use_embeddings=use_emb,
                use_clusters=use_clu,
                use_sparql=use_spq,
                sparql_df=sparql_df,
                pca_candidates=args.pca_components,
                pca_cv_folds=args.pca_cv_folds,
                hdb_min_cluster_size=args.hdbscan_min_cluster_size,
                hdb_min_samples=args.hdbscan_min_samples,
            )

            tuned_models = tune_base_models(X_tr, y_tr, seed=sd, cv_folds=args.cv_folds, n_jobs=args.n_jobs)
            met, meta_details = stacking_with_tuned_models(
                X_tr=X_tr,
                y_tr=y_tr,
                X_te=X_te,
                y_te=y_te,
                seed=sd,
                tuned_models=tuned_models,
                cv_folds=args.cv_folds,
            )

            met_row = {"seed": sd, **met}
            per_seed_metrics.append(met_row)

            per_seed_rows.append({
                "Dataset": args.dataset_name,
                "ArtifactsModel": art.model_name,
                "Setting": setting_name,
                "use_embeddings": bool(use_emb),
                "use_clusters": bool(use_clu),
                "use_sparql": bool(use_spq),
                "n_train": feat_details.get("n_train"),
                "n_test": feat_details.get("n_test"),
                "n_features": feat_details.get("n_features"),
                "pca_components": feat_details.get("pca_components"),
                "pca_explained_var_sum": feat_details.get("pca_explained_var_sum"),
                "hdb_n_clusters_train": feat_details.get("hdb_n_clusters_train"),
                "hdb_noise_frac_train": feat_details.get("hdb_noise_frac_train"),
                "meta_C": meta_details.get("meta_C"),
                "meta_cv_auc": meta_details.get("meta_cv_auc"),
                **met_row,
            })

            per_seed_details.append({"seed": sd, "features": feat_details, "meta": meta_details})

        summ = summarize_metrics(per_seed_metrics)
        n_features_mean = float(np.mean([r["n_features"] for r in per_seed_rows if r["Setting"] == setting_name and r["Dataset"] == args.dataset_name]))

        row = {
            "Dataset": args.dataset_name,
            "ArtifactsModel": art.model_name,
            "Setting": setting_name,
            "use_embeddings": bool(use_emb),
            "use_clusters": bool(use_clu),
            "use_sparql": bool(use_spq),
            "n_seeds": int(len(args.seeds)),
            "n_features_mean": float(n_features_mean),
            **summ,
        }
        rows.append(row)
        detail_dump["settings"].append({"Setting": setting_name, "per_seed_details": per_seed_details})

        print("%s: F1=%.4f +/- %.4f | AUC=%.4f +/- %.4f" % (
            setting_name,
            row["F1-score_mean"], row["F1-score_std"],
            row["ROC-AUC_mean"], row["ROC-AUC_std"],
        ))

    df_out = pd.DataFrame(rows)
    df_seed = pd.DataFrame(per_seed_rows)

    out_csv = out_dir / "table3_ablation_improved.csv"
    out_seed_csv = out_dir / "table3_ablation_improved_per_seed.csv"
    out_json = out_dir / "table3_ablation_improved.json"

    df_out.to_csv(out_csv, index=False)
    df_seed.to_csv(out_seed_csv, index=False)
    out_json.write_text(json.dumps(detail_dump, indent=2), encoding="utf-8")

    print("Saved: %s" % out_csv)
    print("Saved: %s" % out_seed_csv)
    print("Saved: %s" % out_json)


if __name__ == "__main__":
    main()
