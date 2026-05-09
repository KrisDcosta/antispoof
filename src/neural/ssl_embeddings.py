"""Utilities for frozen SSL embedding cache generation."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.reporting import ensure_dir


def encoder_slug(encoder_name: str) -> str:
    """Return a filesystem-friendly encoder identifier."""
    name = encoder_name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def pooled_mean_std(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Pool frame-level SSL states into concat(mean, std)."""
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state must have shape [batch, frames, hidden_dim], "
            f"got {tuple(last_hidden_state.shape)}"
        )
    mean = last_hidden_state.mean(dim=1)
    std = last_hidden_state.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=1)


def cache_path(config: dict, split: str) -> Path:
    ssl_config = config["ssl"]
    root = Path(config["cache"]["root"])
    slug = ssl_config.get("encoder_slug") or encoder_slug(ssl_config["encoder_name"])
    return root / slug / f"{split}.pt"


def ensure_cache_writable(path: str | os.PathLike, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"cache already exists: {path}; pass --overwrite to replace it")
    ensure_dir(path.parent)
    return path


def model_revision(encoder_name: str) -> str | None:
    try:
        from huggingface_hub import model_info

        return model_info(encoder_name).sha
    except Exception:
        return None


def transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return "unavailable"


def load_frozen_ssl_encoder(encoder_name: str, device: torch.device):
    """Load a Hugging Face SSL feature extractor/model pair with gradients disabled."""
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise ImportError(
            "Phase 3 SSL caching requires transformers. Install requirements-neural.txt."
        ) from exc

    processor = AutoFeatureExtractor.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


def build_cache_payload(
    *,
    config: dict,
    split: str,
    items: list[dict[str, Any]],
    processor_name: str,
    cache_device: torch.device,
    torch_dtype: torch.dtype,
) -> dict[str, Any]:
    ssl_config = config["ssl"]
    data_config = config["data"]
    return {
        "encoder_name": ssl_config["encoder_name"],
        "encoder_revision": ssl_config.get("encoder_revision") or model_revision(ssl_config["encoder_name"]),
        "processor_name": processor_name,
        "transformers_version": transformers_version(),
        "hidden_state_source": ssl_config.get("hidden_state_source", "last_hidden_state"),
        "cache_representation": ssl_config.get("cache_representation", "pooled_mean_std"),
        "sample_rate": int(data_config["sample_rate"]),
        "num_samples": int(data_config["num_samples"]),
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "cache_device": str(cache_device),
        "split": split,
        "items": items,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_cache(path: str | os.PathLike, payload: dict[str, Any]) -> None:
    torch.save(payload, Path(path))
