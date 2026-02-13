from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


@dataclass
class PykeenConfig:
    model: str = "TransE"
    embedding_dim: int = 128
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    random_seed: int = 42
    device: Optional[str] = None  # "cpu" or "cuda:0"


def load_triples_factory_from_tsv(path_tsv: str, create_inverse_triples: bool = True) -> TriplesFactory:
    df = pd.read_csv(path_tsv, sep="\t", header=None, names=["head", "relation", "tail"])
    triples = df[["head", "relation", "tail"]].astype(str).values
    return TriplesFactory.from_labeled_triples(triples, create_inverse_triples=create_inverse_triples)


def split_triples_factory(tf: TriplesFactory, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
    training, testing, validation = tf.split(ratios=ratios, random_state=seed)
    return training, testing, validation


def train_pykeen(tf_train, tf_test, tf_valid, cfg: PykeenConfig):
    return pipeline(
        training=tf_train,
        testing=tf_test,
        validation=tf_valid,
        model=cfg.model,
        model_kwargs=dict(embedding_dim=cfg.embedding_dim),
        training_kwargs=dict(num_epochs=cfg.epochs, batch_size=cfg.batch_size),
        optimizer_kwargs=dict(lr=cfg.learning_rate),
        random_seed=cfg.random_seed,
        device=cfg.device,
    )


def get_entity_embeddings(result, tf_full: TriplesFactory):
    id_to_label = dict(tf_full.entity_id_to_label)
    model = result.model
    rep = model.entity_representations[0]

    emb = None
    try:
        emb = rep(indices=None).detach().cpu().numpy()
    except Exception:
        pass
    if emb is None:
        try:
            emb = rep().detach().cpu().numpy()
        except Exception:
            pass
    if emb is None:
        for attr in ("_embeddings", "embeddings"):
            if hasattr(rep, attr):
                w = getattr(rep, attr)
                if hasattr(w, "weight"):
                    emb = w.weight.detach().cpu().numpy()
                    break
    if emb is None:
        raise RuntimeError("Could not extract entity embeddings from this PyKEEN version.")
    return emb, id_to_label
