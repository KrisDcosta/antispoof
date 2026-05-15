"""Cache pooled frozen SSL embeddings for ASVspoof 2019 LA."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import split_summary
from src.neural.dataset import ASVspoofWaveformDataset, load_limited_split
from src.neural.ssl_embeddings import (
    build_cache_payload,
    cache_path,
    ensure_cache_writable,
    load_frozen_ssl_encoder,
    pooled_mean_std,
    save_cache,
)
from src.neural.train_utils import load_config, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="SSL experiment JSON config")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split cache files")
    return parser.parse_args()


def make_loader(samples, config: dict) -> DataLoader:
    dataset = ASVspoofWaveformDataset(
        samples=samples,
        sample_rate=int(config["data"]["sample_rate"]),
        num_samples=int(config["data"]["num_samples"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def processor_name(processor) -> str:
    return getattr(processor, "name_or_path", None) or processor.__class__.__name__


@torch.inference_mode()
def cache_split(
    *,
    split: str,
    config: dict,
    processor,
    model,
    device: torch.device,
    overwrite: bool,
) -> None:
    split_start = time.time()
    limit_key = "train_limit" if split == "train" else "eval_limit"
    samples = load_limited_split(
        config["data"]["root"],
        split,
        config["data"].get(limit_key),
        seed=int(config["training"]["seed"]),
    )
    path = ensure_cache_writable(cache_path(config, split), overwrite=overwrite)
    print(f"{split.title()}: {split_summary(samples)}")
    print(f"Cache path: {path}")

    loader = make_loader(samples, config)
    items = []
    torch_dtype = next(model.parameters()).dtype
    for batch in tqdm(loader, desc=f"cache ssl {split}"):
        arrays = [x.detach().cpu().numpy() for x in batch["x"]]
        processed = processor(
            arrays,
            sampling_rate=int(config["data"]["sample_rate"]),
            return_tensors="pt",
            padding=True,
        )
        processed = {key: value.to(device) for key, value in processed.items()}
        outputs = model(**processed)
        hidden = outputs.last_hidden_state
        embeddings = pooled_mean_std(hidden).detach().cpu().to(torch.float32)
        for idx, embedding in enumerate(embeddings):
            items.append({
                "file_id": batch["file_id"][idx],
                "path": batch["path"][idx],
                "label": int(batch["label"][idx].item()),
                "system_id": batch["system_id"][idx],
                "embedding": embedding,
            })

    payload = build_cache_payload(
        config=config,
        split=split,
        items=items,
        processor_name=processor_name(processor),
        cache_device=device,
        torch_dtype=torch_dtype,
    )
    payload["cache_seconds"] = time.time() - split_start
    save_cache(path, payload)
    print(f"Saved {len(items)} pooled embeddings: {path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"]["seed"]))

    if config["ssl"].get("cache_representation") != "pooled_mean_std":
        raise ValueError("SSL cache generation currently supports pooled_mean_std caches")
    if config["ssl"].get("hidden_state_source", "last_hidden_state") != "last_hidden_state":
        raise ValueError("SSL cache generation currently supports last_hidden_state")

    device = select_device(config["training"]["device"])
    processor, model = load_frozen_ssl_encoder(config["ssl"]["encoder_name"], device)
    print(f"Encoder: {config['ssl']['encoder_name']}")
    print(f"Device: {device}")

    for split in ["train", *config["data"]["splits"]]:
        cache_split(
            split=split,
            config=config,
            processor=processor,
            model=model,
            device=device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
