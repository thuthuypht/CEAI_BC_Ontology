from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF


@dataclass
class RdfExportConfig:
    keep_rdf_type: bool = True
    drop_literal_objects: bool = True
    drop_predicate_substrings: Tuple[str, ...] = ("comment", "label")
    keep_uri_objects_only: bool = True


def load_graph_ttl(path: str) -> Graph:
    g = Graph()
    # Safe for Windows paths like D:/... (avoid URI parsing issues)
    with open(path, "rb") as f:
        g.parse(file=f, format="turtle")
    return g


def iter_entity_triples(g: Graph, cfg: RdfExportConfig) -> Iterable[Tuple[str, str, str]]:
    for s, p, o in g:
        if cfg.keep_rdf_type is False and p == RDF.type:
            continue

        p_str = str(p).lower()
        if any(sub in p_str for sub in cfg.drop_predicate_substrings):
            continue

        if cfg.drop_literal_objects and isinstance(o, Literal):
            continue

        if cfg.keep_uri_objects_only and not isinstance(o, URIRef):
            continue

        if not isinstance(s, URIRef) or not isinstance(p, URIRef):
            continue

        yield str(s), str(p), str(o)


def export_triples_tsv(g: Graph, out_path: str, cfg: Optional[RdfExportConfig] = None) -> pd.DataFrame:
    cfg = cfg or RdfExportConfig()
    rows = list(iter_entity_triples(g, cfg))
    df = pd.DataFrame(rows, columns=["head", "relation", "tail"])
    df.to_csv(out_path, sep="\t", index=False, header=False)
    return df


def extract_patients(g: Graph, patient_prefix: str = "Patient_") -> List[str]:
    return sorted({str(s) for s in g.subjects() if patient_prefix in str(s)})


def detect_namespace(g: Graph) -> str:
    ns = dict(g.namespaces())
    if "ex" in ns:
        return str(ns["ex"])
    for _, uri in ns.items():
        if uri:
            return str(uri)
    return "http://example.org/"
