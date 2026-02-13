from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.io_rdf import load_graph_ttl, detect_namespace
from src.sparql_explain import ExplainConfig, example_queries, top_categories_per_leiden_cluster
from src.utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_ttl", required=True)
    ap.add_argument("--clusters_csv", required=True, help="UMAP_Clusters.csv from script 03")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patient_prefix", default="Patient_")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)

    g = load_graph_ttl(args.input_ttl)
    base_ns = detect_namespace(g)

    df = pd.read_csv(args.clusters_csv)
    cfg = ExplainConfig(patient_prefix=args.patient_prefix)

    top = top_categories_per_leiden_cluster(g, df, cfg)
    out_csv = str(Path(out_dir) / "Leiden_TopCategories.csv")
    top.to_csv(out_csv, index=False)

    md = Path(out_dir) / "SPARQL_Explanation_Templates.md"
    qs = example_queries(base_ns)
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# SPARQL templates\n\nDetected namespace: `{base_ns}`\n\n")
        for title, q in qs.items():
            f.write(f"## {title}\n```sparql\n{q}\n```\n\n")

        f.write("## Top semantic categories per Leiden cluster\n")
        f.write(f"Saved to: `{Path(out_csv).name}`\n")

    print("✅ Saved:")
    print("-", out_csv)
    print("-", str(md))


if __name__ == "__main__":
    main()
