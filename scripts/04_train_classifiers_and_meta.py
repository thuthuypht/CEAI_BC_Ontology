from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdflib import URIRef, Literal

from src.utils import ensure_dir, set_seed
from src.io_rdf import load_graph_ttl, extract_patients, detect_namespace
from src.ml_meta import build_models, simple_meta_learner
from src.attention_mlp import AttentionMLPClassifier, AttentionMLPConfig

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_ttl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_predicate", default="diagnosis", help="Predicate local name or full IRI")
    ap.add_argument("--positive_label", default="M")
    ap.add_argument("--patient_prefix", default="Patient_")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--do_shap", action="store_true", help="Compute SHAP for XGBoost/LightGBM (if available)")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    g = load_graph_ttl(args.input_ttl)
    base_ns = detect_namespace(g)
    pred_iri = args.label_predicate if args.label_predicate.startswith("http") else base_ns + args.label_predicate
    pred = URIRef(pred_iri)

    patients = extract_patients(g, patient_prefix=args.patient_prefix)

    # Collect numeric predicates automatically (raw statistical features)
    numeric_preds = set()
    for s, p, o in g:
        if args.patient_prefix not in str(s):
            continue
        if not isinstance(o, Literal):
            continue
        try:
            float(o)
            numeric_preds.add(str(p))
        except Exception:
            pass
    numeric_preds = sorted(numeric_preds)

    X_rows, y = [], []
    for pu in patients:
        s = URIRef(pu)

        # Label
        lab = None
        for _, _, o in g.triples((s, pred, None)):
            lab = str(o)
            break
        if lab is None:
            continue

        row = []
        ok = True
        for p_iri in numeric_preds:
            p = URIRef(p_iri)
            val = None
            for _, _, o in g.triples((s, p, None)):
                try:
                    val = float(o); break
                except Exception:
                    continue
            if val is None:
                ok = False; break
            row.append(val)

        if not ok:
            continue

        X_rows.append(row)
        y.append(1 if lab == args.positive_label else 0)

    X = np.array(X_rows, dtype=float)
    y = np.array(y, dtype=int)

    if X.shape[0] < 50:
        raise ValueError(f"Too few usable patients: {X.shape[0]}")

    # Baselines + meta-learner
    models = build_models(seed=args.seed)
    metrics = simple_meta_learner(X, y, models, seed=args.seed)

    # Attention-Enhanced MLP (feature-wise gating)
    att_cfg = AttentionMLPConfig(epochs=80, seed=args.seed, device="cpu")
    att = AttentionMLPClassifier(att_cfg)
    # train/test split consistent with ml_meta? We keep it simple here: fit on full data then CV is in paper; for repo, report holdout quickly.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    att.fit(X_train, y_train)
    prob = att.predict_proba(X_test)[:, 1]
    pred_att = (prob >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    metrics["Attention-Enhanced MLP"] = {
        "Accuracy": float(accuracy_score(y_test, pred_att)),
        "Precision": float(precision_score(y_test, pred_att, zero_division=0)),
        "Recall": float(recall_score(y_test, pred_att, zero_division=0)),
        "F1-score": float(f1_score(y_test, pred_att, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, prob)),
    }

    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    df.to_csv(str(Path(out_dir) / "ML_Results.csv"), index=False)
    with pd.ExcelWriter(str(Path(out_dir) / "ML_Results.xlsx"), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Metrics", index=False)

    # Confusion matrix for Attention MLP (example)
    cm = confusion_matrix(y_test, pred_att)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Attention-Enhanced MLP)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "ConfusionMatrix_AttentionMLP.png"), dpi=420)
    plt.close()

    # Optional SHAP for tree models
    if args.do_shap:
        try:
            import shap
            # XGBoost and LightGBM are in metrics if installed, but we need the fitted models.
            # For simplicity: fit a LightGBM if available, else XGBoost, then compute SHAP.
            tree_model = None
            name = None
            if "LightGBM" in models:
                tree_model = models["LightGBM"]
                name = "LightGBM"
            elif "XGBoost" in models:
                tree_model = models["XGBoost"]
                name = "XGBoost"
            if tree_model is not None:
                tree_model.fit(X_train, y_train)
                explainer = shap.TreeExplainer(tree_model)
                shap_values = explainer.shap_values(X_test)
                # Save a summary plot
                shap.summary_plot(shap_values, X_test, show=False)
                plt.tight_layout()
                plt.savefig(str(Path(out_dir) / f"SHAP_Summary_{name}.png"), dpi=420)
                plt.close()
        except Exception as e:
            print("⚠️ SHAP failed:", e)

    print("✅ Saved ML results to:", out_dir)
    print(df)


if __name__ == "__main__":
    main()
