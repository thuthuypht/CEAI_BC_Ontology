from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import umap
import igraph as ig
import leidenalg
import hdbscan


@dataclass
class ClusterConfig:
    k: int = 15
    umap_n_neighbors: int = 20
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_random_state: int = 42
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: Optional[int] = None


def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def project_umap(X: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.umap_random_state,
    )
    return reducer.fit_transform(X)


def build_knn_graph(X_2d: np.ndarray, k: int) -> nx.Graph:
    n = X_2d.shape[0]
    k_eff = max(2, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X_2d)
    mat = nn.kneighbors_graph(X_2d, mode="connectivity")
    return nx.from_scipy_sparse_array(mat)


def leiden_communities(G: nx.Graph):
    edges = list(G.edges())
    ig_g = ig.Graph(edges=edges, directed=False)
    part = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition)
    labels = np.array(part.membership, dtype=int)
    modularity = float(part.quality())
    return labels, modularity


def hdbscan_clusters(X_2d: np.ndarray, cfg: ClusterConfig):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X_2d)

    mask = labels != -1
    if mask.sum() >= 3 and np.unique(labels[mask]).size >= 2:
        sil = float(silhouette_score(X_2d[mask], labels[mask], metric="euclidean"))
    else:
        sil = -1.0
    return labels, sil


def make_cluster_dataframe(entities, umap_xy, leiden_labels, hdb_labels) -> pd.DataFrame:
    return pd.DataFrame({
        "entity": list(entities),
        "umap_x": umap_xy[:, 0],
        "umap_y": umap_xy[:, 1],
        "leiden_cluster": leiden_labels,
        "hdbscan_cluster": hdb_labels,
    })
