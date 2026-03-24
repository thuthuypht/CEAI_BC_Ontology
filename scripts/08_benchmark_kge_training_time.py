#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_benchmark_kge_training_time.py

Benchmark PyKEEN KGE training time (minimum wall-clock minutes) for:
  - TransE
  - RotatE
  - ComplEx

on one or more RDF/Turtle knowledge graphs (.ttl).

This script measures ONLY the KGE training stage runtime (PyKEEN pipeline call),
excluding TTL parsing and RDF-to-triples conversion.

Outputs:
  1) kge_training_time_runs.csv  (all runs)
  2) kge_training_time_min.csv   (min minutes per dataset/model, after optional warmup)

Example (Windows cmd):
  python scripts\08_benchmark_kge_training_time.py ^
    --datasets WDBC="D:\CEAI_BC_Ontology\data\processed\Wisconsin_Categorized_KnowledgeGraph_enriched_final.ttl" ^
             Coimbra="D:\CEAI_BC_Ontology\data\processed\Breast_Cancer_Coimbra.ttl" ^
    --out_dir "D:\CEAI_BC_Ontology\outputs_kge_time" ^
    --models TransE RotatE ComplEx ^
    --epochs 50 ^
    --embedding_dim 64 ^
    --batch_size 256 ^
    --lr 0.001 ^
    --repeats 3 ^
    --warmup 1 ^
    --seed 42

Notes:
- For fair comparison, all models are trained on the full triples factory (no split).
- Some PyKEEN versions require both training/testing factories. We automatically retry with
  testing=training and validation=training if needed.
- Use the same epochs/embedding_dim as your sensitivity experiments.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def safe_import(name: str):
    try:
        return __import__(name)
    except Exception as e:
        raise RuntimeError(
            "Missing/failed import: %s\nError: %s\n"
            "Install dependencies, e.g.:\n"
            "  pip install rdflib pykeen pandas numpy\n" % (name, e)
        ) from e


def load_graph_ttl(ttl_path: Path):
    safe_import("rdflib")
    from rdflib import Graph
    g = Graph()
    with open(ttl_path, "rb") as f:
        g.parse(file=f, format="turtle")
    return g


def graph_to_labeled_triples(graph) -> np.ndarray:
    from rdflib.term import BNode, Literal

    triples: List[Tuple[str, str, str]] = []
    for s, p, o in graph:
        if isinstance(o, Literal):
            continue
        s_str = "_:" + str(s) if isinstance(s, BNode) else str(s)
        o_str = "_:" + str(o) if isinstance(o, BNode) else str(o)
        triples.append((s_str, str(p), o_str))

    if not triples:
        raise ValueError("No triples for PyKEEN after filtering Literal tails.")
    return np.asarray(triples, dtype=str)


@dataclass
class KGEConfig:
    model: str
    epochs: int
    embedding_dim: int
    batch_size: int
    lr: float
    seed: int
    device: str = "cpu"


def run_pykeen_pipeline_full(factory, cfg: KGEConfig):
    """Run PyKEEN pipeline on the full factory (no split)."""
    safe_import("pykeen")
    from pykeen.pipeline import pipeline
    import inspect

    pipe_sig = inspect.signature(pipeline)

    common = dict(
        model=cfg.model,
        model_kwargs=dict(embedding_dim=cfg.embedding_dim),
        training_kwargs=dict(num_epochs=cfg.epochs, batch_size=cfg.batch_size),
        optimizer_kwargs=dict(lr=cfg.lr),
        random_seed=cfg.seed,
        device=cfg.device,
    )

    # Attempt 1: training only
    try:
        if "training" in pipe_sig.parameters:
            return pipeline(training=factory, **common)
        return pipeline(training_triples_factory=factory, **common)
    except Exception as e1:
        # Attempt 2: include testing/validation = training
        try:
            if "training" in pipe_sig.parameters:
                return pipeline(training=factory, testing=factory, validation=factory, **common)
            return pipeline(
                training_triples_factory=factory,
                testing_triples_factory=factory,
                validation_triples_factory=factory,
                **common
            )
        except Exception as e2:
            raise RuntimeError("PyKEEN pipeline failed.\nFirst error: %s\nSecond error: %s" % (e1, e2)) from e2


def benchmark_dataset(
    name: str,
    ttl_path: Path,
    models: List[str],
    repeats: int,
    warmup: int,
    base_seed: int,
    epochs: int,
    embedding_dim: int,
    batch_size: int,
    lr: float,
) -> pd.DataFrame:
    safe_import("pykeen")
    from pykeen.triples import TriplesFactory

    g = load_graph_ttl(ttl_path)
    triples = graph_to_labeled_triples(g)
    factory = TriplesFactory.from_labeled_triples(triples)

    rows = []
    for model in models:
        for r in range(repeats):
            run_seed = base_seed + r
            set_seed(run_seed)

            cfg = KGEConfig(
                model=model,
                epochs=epochs,
                embedding_dim=embedding_dim,
                batch_size=batch_size,
                lr=lr,
                seed=run_seed,
                device="cpu",
            )

            t0 = time.perf_counter()
            _ = run_pykeen_pipeline_full(factory, cfg)
            t1 = time.perf_counter()

            seconds = float(t1 - t0)
            minutes = float(seconds / 60.0)

            rows.append({
                "dataset": name,
                "ttl_path": str(ttl_path),
                "kge_model": model,
                "run_index": r,
                "seed": run_seed,
                "epochs": epochs,
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "lr": lr,
                "n_triples": int(factory.num_triples),
                "n_entities": int(factory.num_entities),
                "n_relations": int(factory.num_relations),
                "train_seconds": seconds,
                "train_minutes": minutes,
                "is_warmup": (r < warmup),
            })

    return pd.DataFrame(rows)


def parse_datasets(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Each --datasets entry must be Name=Path. Got: %s" % item)
        name, path = item.split("=", 1)
        p = Path(path.strip().strip('"'))
        out[name.strip()] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, help="List of Name=Path entries")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--models", nargs="+", default=["TransE", "RotatE", "ComplEx"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--repeats", type=int, default=3, help="Runs per model; minimum is computed over non-warmup runs")
    ap.add_argument("--warmup", type=int, default=1, help="Discard the first N runs per model when computing minimum")
    ap.add_argument("--seed", type=int, default=42, help="Base seed; run seeds are seed+run_index" )

    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    datasets = parse_datasets(args.datasets)

    all_runs = []
    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError("TTL not found: %s" % path)
        df_runs = benchmark_dataset(
            name=name,
            ttl_path=path,
            models=args.models,
            repeats=args.repeats,
            warmup=args.warmup,
            base_seed=args.seed,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        all_runs.append(df_runs)

    df_all = pd.concat(all_runs, ignore_index=True)

    df_non_warmup = df_all[df_all["is_warmup"] == False].copy()
    if df_non_warmup.empty:
        df_non_warmup = df_all.copy()

    df_min = (
        df_non_warmup
        .groupby(["dataset", "kge_model"], as_index=False)
        .agg(
            min_train_minutes=("train_minutes", "min"),
            min_train_seconds=("train_seconds", "min"),
            n_entities=("n_entities", "first"),
            n_relations=("n_relations", "first"),
            n_triples=("n_triples", "first"),
            epochs=("epochs", "first"),
            embedding_dim=("embedding_dim", "first"),
            batch_size=("batch_size", "first"),
            lr=("lr", "first"),
        )
        .sort_values(["dataset", "kge_model"])
    )

    runs_csv = out_dir / "kge_training_time_runs.csv"
    min_csv = out_dir / "kge_training_time_min.csv"

    df_all.to_csv(runs_csv, index=False)
    df_min.to_csv(min_csv, index=False)

    print("Saved:", runs_csv)
    print("Saved:", min_csv)
    print(df_min.to_string(index=False))


if __name__ == "__main__":
    main()
