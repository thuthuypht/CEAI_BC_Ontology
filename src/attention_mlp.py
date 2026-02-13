from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureAttentionMLP(nn.Module):
    """A lightweight feature-wise attention MLP.

    - Learns a gate/attention weight per input feature (conditioned on the input).
    - Multiplies input features by attention weights, then feeds to an MLP.
    """

    def __init__(self, in_dim: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        w = self.att(x)              # (B, in_dim)
        xw = x * w                   # gated features
        logits = self.mlp(xw).squeeze(-1)
        return logits, w


@dataclass
class AttentionMLPConfig:
    hidden1: int = 128
    hidden2: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42


class AttentionMLPClassifier:
    """Sklearn-like wrapper for FeatureAttentionMLP."""

    def __init__(self, cfg: Optional[AttentionMLPConfig] = None):
        self.cfg = cfg or AttentionMLPConfig()
        self.model: Optional[FeatureAttentionMLP] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.cfg.seed)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        self.model = FeatureAttentionMLP(
            in_dim=X.shape[1],
            hidden1=self.cfg.hidden1,
            hidden2=self.cfg.hidden2,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.cfg.epochs):
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                yb = yb.to(self.cfg.device)

                opt.zero_grad()
                logits, _ = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.cfg.device)
            logits, _ = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        # sklearn-style: [:,0]=P(0), [:,1]=P(1)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.cfg.device)
            _, w = self.model(X_t)
            return w.cpu().numpy()
