"""Check Phase 3 SSL configuration and local readiness without running WavLM."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neural.ssl_dataset import load_ssl_cache
from src.neural.ssl_embeddings import cache_path
from src.neural.train_utils import load_config


REQUIRED_PACKAGES = ["torch", "torchaudio", "transformers", "huggingface_hub"]
REQUIRED_CACHE_KEYS = [
    "encoder_name",
    "encoder_revision",
    "processor_name",
    "transformers_version",
    "hidden_state_source",
    "cache_representation",
    "sample_rate",
    "num_samples",
    "torch_dtype",
    "cache_device",
    "split",
    "items",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Phase 3 SSL experiment JSON config")
    parser.add_argument(
        "--check-caches",
        action="store_true",
        help="Validate existing pooled caches for train/dev/eval without creating them",
    )
    return parser.parse_args()


def validate_config(config: dict) -> list[str]:
    errors = []
    if config.get("track") != "external-pretrained/applied":
        errors.append("track must be external-pretrained/applied")
    if config["model"].get("type") != "ssl_pooled_mlp":
        errors.append("model.type must be ssl_pooled_mlp")
    if config["model"].get("input") != "wavlm_pooled_mean_std":
        errors.append("model.input must be wavlm_pooled_mean_std")
    if int(config["model"].get("input_dim", 0)) != 1536:
        errors.append("model.input_dim must be 1536 for WavLM-base-plus mean+std pooling")
    if config["ssl"].get("encoder_name") != "microsoft/wavlm-base-plus":
        errors.append("ssl.encoder_name must be microsoft/wavlm-base-plus for the first Phase 3 run")
    if config["ssl"].get("cache_representation") != "pooled_mean_std":
        errors.append("ssl.cache_representation must be pooled_mean_std")
    if config["ssl"].get("hidden_state_source") != "last_hidden_state":
        errors.append("ssl.hidden_state_source must be last_hidden_state")
    if config["ssl"].get("external_pretraining") is not True:
        errors.append("ssl.external_pretraining must be true")
    if int(config["data"].get("sample_rate", 0)) != 16000:
        errors.append("data.sample_rate must be 16000")
    if int(config["data"].get("num_samples", 0)) != 64600:
        errors.append("data.num_samples must be 64600")
    if config["training"].get("normalization") != "train_mean_std":
        errors.append("training.normalization must be train_mean_std")
    if config["training"].get("loss") != "weighted_cross_entropy":
        errors.append("training.loss must be weighted_cross_entropy")
    if "dev" not in config["data"].get("splits", []):
        errors.append("data.splits must include dev for checkpoint selection")
    if "eval" not in config["data"].get("splits", []):
        errors.append("data.splits must include eval for reportable metrics")
    return errors


def package_status() -> list[tuple[str, bool, str]]:
    statuses = []
    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            statuses.append((package, True, getattr(module, "__version__", "installed")))
        except Exception as exc:
            statuses.append((package, False, f"{type(exc).__name__}: {exc}"))
    return statuses


def validate_cache_payload(config: dict, split: str) -> list[str]:
    errors = []
    path = cache_path(config, split)
    if not path.exists():
        return [f"{split}: cache missing at {path}"]
    cache = load_ssl_cache(path)
    for key in REQUIRED_CACHE_KEYS:
        if key not in cache:
            errors.append(f"{split}: missing cache key {key}")
    if cache.get("split") != split:
        errors.append(f"{split}: cache split metadata is {cache.get('split')!r}")
    if cache.get("cache_representation") != "pooled_mean_std":
        errors.append(f"{split}: cache_representation must be pooled_mean_std")
    if cache.get("hidden_state_source") != "last_hidden_state":
        errors.append(f"{split}: hidden_state_source must be last_hidden_state")
    items = cache.get("items", [])
    if not items:
        errors.append(f"{split}: cache has no items")
        return errors
    embedding = items[0].get("embedding")
    if not isinstance(embedding, torch.Tensor):
        errors.append(f"{split}: first embedding is not a tensor")
    elif tuple(embedding.shape) != (int(config["model"]["input_dim"]),):
        errors.append(f"{split}: expected embedding shape {(int(config['model']['input_dim']),)}, got {tuple(embedding.shape)}")
    for item_key in ["file_id", "path", "label", "system_id", "embedding"]:
        if item_key not in items[0]:
            errors.append(f"{split}: first item missing {item_key}")
    return errors


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    errors = validate_config(config)
    print(f"Config: {args.config}")
    print("Config checks:", "OK" if not errors else "FAILED")
    for error in errors:
        print(f"  - {error}")

    print("Package checks:")
    package_errors = []
    for package, ok, detail in package_status():
        status = "OK" if ok else "MISSING"
        print(f"  - {package}: {status} ({detail})")
        if not ok:
            package_errors.append(f"missing package: {package}")

    cache_errors = []
    if args.check_caches:
        print("Cache checks:")
        for split in ["train", *config["data"]["splits"]]:
            split_errors = validate_cache_payload(config, split)
            if split_errors:
                for error in split_errors:
                    print(f"  - {error}")
                cache_errors.extend(split_errors)
            else:
                print(f"  - {split}: OK ({cache_path(config, split)})")

    all_errors = errors + package_errors + cache_errors
    if all_errors:
        raise SystemExit(1)
    print("Phase 3 SSL readiness: OK")


if __name__ == "__main__":
    main()
