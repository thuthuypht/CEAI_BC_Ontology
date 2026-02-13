from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter_clusters(df: pd.DataFrame, label_col: str, title: str, out_path: str, figsize: Tuple[int, int] = (8, 6)) -> None:
    plt.figure(figsize=figsize)
    labels = df[label_col].values
    for lab in np.unique(labels):
        sub = df[df[label_col] == lab]
        plt.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.85, label=str(lab))
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=420)
    plt.close()


def side_by_side(img1_path: str, img2_path: str, out_path: str, figsize: Tuple[int, int] = (14, 6), titles: Tuple[str, str] = ("Leiden", "HDBSCAN")) -> None:
    import matplotlib.image as mpimg
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)

    plt.figure(figsize=figsize)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img1); ax1.axis("off"); ax1.set_title(titles[0])
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(img2); ax2.axis("off"); ax2.set_title(titles[1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=420)
    plt.close()
