from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from rdflib import Graph, URIRef


@dataclass
class ExplainConfig:
    patient_prefix: str = "Patient_"
    category_suffixes: Tuple[str, ...] = ("_High", "_Medium", "_Low")
    top_k: int = 15


def example_queries(base_ns: str) -> Dict[str, str]:
    q1 = f"""PREFIX ex: <{base_ns}>
SELECT ?category (COUNT(?patient) AS ?n)
WHERE {{
  ?patient ?p ?category .
  FILTER(CONTAINS(STR(?patient), "Patient_"))
  FILTER(CONTAINS(STR(?category), "_High") || CONTAINS(STR(?category), "_Medium") || CONTAINS(STR(?category), "_Low"))
}}
GROUP BY ?category
ORDER BY DESC(?n)
LIMIT 20
"""

    q2 = f"""PREFIX ex: <{base_ns}>
SELECT ?diagnosis (COUNT(?patient) AS ?n)
WHERE {{
  ?patient ex:diagnosis ?diagnosis .
  FILTER(CONTAINS(STR(?patient), "Patient_"))
}}
GROUP BY ?diagnosis
ORDER BY DESC(?n)
"""
    return {"Top semantic categories": q1, "Diagnosis distribution": q2}


def top_categories_per_leiden_cluster(
    g: Graph,
    df_clusters: pd.DataFrame,
    cfg: ExplainConfig,
) -> pd.DataFrame:
    # df_clusters must contain: entity, leiden_cluster
    rows = []
    for _, r in df_clusters.iterrows():
        ent = str(r["entity"])
        if cfg.patient_prefix not in ent:
            continue
        cl = int(r["leiden_cluster"])
        s = URIRef(ent)
        for _, p, o in g.triples((s, None, None)):
            o_str = str(o)
            if o_str.startswith("http") and any(o_str.endswith(suf) for suf in cfg.category_suffixes):
                rows.append((cl, o_str))

    if not rows:
        return pd.DataFrame(columns=["leiden_cluster", "category", "count"])

    tmp = pd.DataFrame(rows, columns=["leiden_cluster", "category"])
    out = (tmp.groupby(["leiden_cluster", "category"]).size()
              .reset_index(name="count")
              .sort_values(["leiden_cluster", "count"], ascending=[True, False]))
    return out
