"""Classifier heads for cached SSL embeddings."""

from __future__ import annotations

import torch
from torch import nn


class SSLPooledMLP(nn.Module):
    """Small MLP classifier for pooled frozen SSL embeddings."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"SSLPooledMLP expects [batch, dim], got {tuple(x.shape)}")
        return self.classifier(x)
