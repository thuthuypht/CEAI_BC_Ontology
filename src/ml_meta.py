from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def _metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
    }


def build_models(seed: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
    ])

    models["k-NN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ])

    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
    ])

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=500, random_state=seed, n_jobs=-1
    )

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )

    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed)),
    ])

    return models


def simple_meta_learner(X: np.ndarray, y: np.ndarray, base_models: Dict[str, object], seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # OOF probs for training meta-learner
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    meta_train_cols = []
    meta_test_cols = []
    metrics = {}

    from sklearn.base import clone

    for name, model in base_models.items():
        # fit on full train -> evaluate on test
        model.fit(X_train, y_train)
        prob_test = model.predict_proba(X_test)[:, 1]
        pred_test = (prob_test >= 0.5).astype(int)
        metrics[name] = _metrics(y_test, prob_test, pred_test)

        # oof on train
        oof = np.zeros_like(y_train, dtype=float)
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m2 = clone(model)
            m2.fit(X_train[tr_idx], y_train[tr_idx])
            oof[va_idx] = m2.predict_proba(X_train[va_idx])[:, 1]

        meta_train_cols.append(oof.reshape(-1, 1))
        meta_test_cols.append(prob_test.reshape(-1, 1))

    X_meta_train = np.hstack(meta_train_cols)
    X_meta_test = np.hstack(meta_test_cols)

    meta = LogisticRegression(max_iter=2000, random_state=seed)
    meta.fit(X_meta_train, y_train)
    prob_meta = meta.predict_proba(X_meta_test)[:, 1]
    pred_meta = (prob_meta >= 0.5).astype(int)
    metrics["Our Meta-Learner"] = _metrics(y_test, prob_meta, pred_meta)

    return metrics
