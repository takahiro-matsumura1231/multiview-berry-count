"""Stage-2 aggregator: a transformer over view tokens, on cached features.

Each of the K views becomes one token: the frozen backbone's image feature concatenated
with that view's detection features. A CLS token reads out the count.

Why a transformer here rather than Attention-MIL pooling: attention pooling weights
each view independently, so it can say "trust this view" but not "these two views show the
same berries". Self-attention across view tokens can in principle represent redundancy
between overlapping viewpoints, which is exactly what counting a partially-occluded object
from multiple angles requires.

Detection features are concatenated, not fused by a learned projection, so an ablation that
removes a feature group removes exactly those input dimensions and nothing else.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ViewTokenCounter(nn.Module):
    """[B, K, image_dim + det_dim] -> [B] counts."""

    def __init__(self, input_dim: int, model_dim: int = 512, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, head_dropout: float = 0.2,
                 positional_encoding: bool = False, view_slots: int = 9) -> None:
        super().__init__()
        if positional_encoding and view_slots <= 0:
            raise ValueError("positional encoding requires a positive number of ordered view slots")
        self.positional_encoding = positional_encoding
        self.view_slots = view_slots
        self.project = nn.Linear(input_dim, model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        encoder = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=model_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(head_dropout), nn.Linear(256, 1),
        )
        if positional_encoding:
            # Fixed sinusoidal slot encoding. Non-persistent keeps checkpoint keys identical
            # to the permutation-invariant variant without slot encoding; CLS receives no slot encoding.
            positions = torch.arange(view_slots, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / model_dim)
            )
            pe = torch.zeros(view_slots, model_dim)
            pe[:, 0::2] = torch.sin(positions * frequencies)
            pe[:, 1::2] = torch.cos(positions * frequencies[:model_dim // 2])
            self.register_buffer("slot_encoding", pe, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        x = self.project(tokens)
        if self.positional_encoding:
            if x.shape[1] != self.view_slots:
                raise ValueError(f"expected {self.view_slots} ordered views, got {x.shape[1]}")
            x = x + self.slot_encoding.to(dtype=x.dtype)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)   # CLS first
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)


class FeatureScaler:
    """Per-feature standardisation fitted on the TRAINING FOLD only.

    This is small and easy to get wrong in the direction that flatters the result: fitting
    the mean and standard deviation on all data leaks distributional information from the
    evaluation fold into training. `fit` is only ever called with training bunches.

    Constant features (a group that never fires, or a padded dimension) would divide by zero;
    their scale is clamped to 1 so they pass through as zeros rather than NaNs.
    """

    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "FeatureScaler":
        self.mean = x.mean(dim=0)
        std = x.std(dim=0)
        self.std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler.transform before fit")
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict) -> "FeatureScaler":
        self.mean, self.std = state["mean"], state["std"]
        return self
