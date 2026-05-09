"""Datasets and normalization helpers for cached SSL embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_ssl_cache(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TrainMeanStdNormalizer:
    """Feature normalizer fit on train embeddings only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        self.mean = mean.to(torch.float32)
        self.std = std.to(torch.float32).clamp_min(eps)
        self.eps = eps

    @classmethod
    def fit(cls, embeddings: torch.Tensor, eps: float = 1e-6) -> "TrainMeanStdNormalizer":
        if embeddings.ndim != 2:
            raise ValueError(f"expected [items, dim] embeddings, got {tuple(embeddings.shape)}")
        return cls(
            embeddings.to(torch.float32).mean(dim=0),
            embeddings.to(torch.float32).std(dim=0, unbiased=False),
            eps=eps,
        )

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        return (embeddings.to(torch.float32) - self.mean) / self.std

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "train_mean_std",
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
        }


class SSLEmbeddingDataset(Dataset):
    """Dataset backed by pooled SSL embedding cache items."""

    def __init__(
        self,
        cache: dict[str, Any],
        normalizer: TrainMeanStdNormalizer | None = None,
    ) -> None:
        self.cache = cache
        self.items = list(cache["items"])
        embeddings = [item["embedding"].to(torch.float32) for item in self.items]
        if not embeddings:
            raise ValueError("SSL embedding cache contains no items")
        self.embeddings = torch.stack(embeddings)
        if normalizer is not None:
            self.embeddings = normalizer.transform(self.embeddings)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = self.items[idx]
        return {
            "x": self.embeddings[idx],
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
            "file_id": item["file_id"],
            "path": item["path"],
            "system_id": item["system_id"],
        }


def class_weights_from_labels(labels: list[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2).to(torch.float32)
    if torch.any(counts == 0):
        return torch.ones(2, dtype=torch.float32)
    weights = counts.sum() / (2.0 * counts)
    return weights.to(torch.float32)
