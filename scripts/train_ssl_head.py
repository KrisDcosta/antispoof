"""Train and evaluate a classifier head on cached SSL embeddings."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neural.evaluation import metrics_from_score_rows, score_loader
from src.neural.ssl_dataset import (
    SSLEmbeddingDataset,
    TrainMeanStdNormalizer,
    class_weights_from_labels,
    load_ssl_cache,
)
from src.neural.ssl_embeddings import cache_path
from src.neural.ssl_models import SSLPooledMLP
from src.neural.train_utils import (
    build_run_id,
    count_parameters,
    create_run_dirs,
    current_commit,
    environment_payload,
    load_config,
    select_device,
    set_seed,
    write_model_card,
)
from src.reporting import (
    save_per_attack_plot,
    save_roc_plot,
    save_score_distribution,
    write_json,
    write_scores_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Phase 3 SSL experiment JSON config")
    return parser.parse_args()


def make_loader(dataset: SSLEmbeddingDataset, config: dict, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(config: dict) -> SSLPooledMLP:
    model_config = config["model"]
    if model_config["type"] != "ssl_pooled_mlp":
        raise ValueError(f"unsupported SSL head type: {model_config['type']}")
    return SSLPooledMLP(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.3)),
    )


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    training = config["training"]
    if str(training.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("Phase 3 SSL head currently supports adamw")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in tqdm(loader, desc="train ssl head"):
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y.long())
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(y)
        total_items += len(y)
    return total_loss / max(total_items, 1)


def evaluate_split(
    model,
    loader,
    split: str,
    dirs: dict[str, Path],
    run_id: str,
    *,
    save_scores: bool,
) -> dict:
    rows = score_loader(model, loader, next(model.parameters()).device)
    metrics, attack_eers, summary = metrics_from_score_rows(rows)
    if rows and save_scores:
        write_scores_csv(dirs["scores"] / f"{split}_scores.csv", rows)
    if attack_eers:
        write_per_attack_csv(dirs["run"] / f"per_attack_{split}.csv", attack_eers)
    save_roc_plot(dirs["plots"] / f"roc_{split}.png", metrics, f"{run_id} ROC ({split})")
    save_score_distribution(
        dirs["plots"] / f"score_distribution_{split}.png",
        rows,
        f"{run_id} score distribution ({split})",
    )
    save_per_attack_plot(
        dirs["plots"] / f"per_attack_eer_{split}.png",
        attack_eers,
        f"{run_id} per-attack EER ({split})",
    )
    return {
        "metrics": metrics,
        "per_attack_eer": attack_eers,
        "split_summary": summary,
    }


def write_per_attack_csv(path: Path, attack_eers: dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system_id", "eer"])
        writer.writeheader()
        for system_id, eer in sorted(attack_eers.items()):
            writer.writerow({"system_id": system_id, "eer": eer})


def load_datasets(config: dict):
    train_cache_path = cache_path(config, "train")
    train_cache = load_ssl_cache(train_cache_path)
    raw_train_dataset = SSLEmbeddingDataset(train_cache)
    normalizer = TrainMeanStdNormalizer.fit(raw_train_dataset.embeddings)
    train_dataset = SSLEmbeddingDataset(train_cache, normalizer=normalizer)
    eval_datasets = {}
    eval_caches = {}
    for split in config["data"]["splits"]:
        split_cache_path = cache_path(config, split)
        cache = load_ssl_cache(split_cache_path)
        eval_caches[split] = cache
        eval_datasets[split] = SSLEmbeddingDataset(cache, normalizer=normalizer)
    return train_dataset, eval_datasets, normalizer, {"train": train_cache, **eval_caches}


def cache_metadata(cache: dict) -> dict:
    keys = [
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
    ]
    return {key: cache.get(key) for key in keys}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"]["seed"]))

    run_id = build_run_id(config)
    dirs = create_run_dirs(config["output"]["root"], run_id)
    shutil.copy2(args.config, dirs["run"] / "config.json")

    device = select_device(config["training"]["device"])
    train_dataset, eval_datasets, normalizer, caches = load_datasets(config)
    model = build_model(config).to(device)
    params = count_parameters(model)
    write_json(dirs["run"] / "environment.json", environment_payload(args.config, device, params))
    torch.save(normalizer.state_dict(), dirs["run"] / "normalization.pt")

    labels = [int(item["label"]) for item in caches["train"]["items"]]
    class_weights = class_weights_from_labels(labels).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(config, model)

    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Parameters: {params:,}")
    print(f"Class weights [spoof, bonafide]: {[float(x) for x in class_weights.detach().cpu()]}")

    train_loader = make_loader(train_dataset, config, shuffle=True)
    eval_loaders = {
        split: make_loader(dataset, config, shuffle=False)
        for split, dataset in eval_datasets.items()
    }

    start = time.time()
    history = []
    best_dev_eer = float("inf")
    best_path = dirs["checkpoints"] / "best.pt"
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_result = {"epoch": epoch, "train_loss": loss}
        if "dev" in eval_loaders:
            dev_rows = score_loader(model, eval_loaders["dev"], device)
            dev_metrics, _, _ = metrics_from_score_rows(dev_rows)
            epoch_result["dev_eer"] = dev_metrics["eer"]
            print(f"Epoch {epoch}: loss={loss:.4f} dev_eer={dev_metrics['eer'] * 100:.2f}%")
            if dev_metrics["eer"] < best_dev_eer:
                best_dev_eer = dev_metrics["eer"]
                if config["output"].get("save_checkpoint", True):
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "run_id": run_id,
                        "epoch": epoch,
                        "dev_eer": best_dev_eer,
                        "normalization": normalizer.state_dict(),
                    }, best_path)
        else:
            print(f"Epoch {epoch}: loss={loss:.4f}")
        history.append(epoch_result)

    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    final_results = {}
    for split, loader in eval_loaders.items():
        final_results[split] = evaluate_split(
            model,
            loader,
            split,
            dirs,
            run_id,
            save_scores=bool(config["output"].get("save_scores", True)),
        )
        metrics = final_results[split]["metrics"]
        print(
            f"{split}: EER={metrics['eer'] * 100:.2f}% "
            f"accuracy={metrics['accuracy'] * 100:.2f}% threshold={metrics['threshold']:.4f}"
        )

    train_cache_meta = cache_metadata(caches["train"])
    metrics_payload = {
        "run_id": run_id,
        "track": config["track"],
        "model_name": config["run_name"],
        "model_family": "ssl_pooled_mlp",
        "feature_or_input": config["model"]["input"],
        "external_pretraining": True,
        "config_path": args.config,
        "commit": current_commit(),
        "seed": config["training"]["seed"],
        "model_parameters": params,
        "training_seconds": time.time() - start,
        "history": history,
        "splits": final_results,
        "encoder_name": train_cache_meta["encoder_name"],
        "encoder_revision": train_cache_meta["encoder_revision"],
        "processor_name": train_cache_meta["processor_name"],
        "transformers_version": train_cache_meta["transformers_version"],
        "hidden_state_source": train_cache_meta["hidden_state_source"],
        "cache_representation": train_cache_meta["cache_representation"],
        "sample_rate": train_cache_meta["sample_rate"],
        "num_samples": train_cache_meta["num_samples"],
        "normalization": config["training"].get("normalization", "train_mean_std"),
        "loss": config["training"].get("loss", "weighted_cross_entropy"),
        "class_weights": [float(x) for x in class_weights.detach().cpu()],
        "cache_paths": {
            split: str(cache_path(config, split))
            for split in ["train", *config["data"]["splits"]]
        },
    }
    write_json(dirs["run"] / "metrics.json", metrics_payload)

    model_card_payload = {
        "model_name": config["run_name"],
        "run_id": run_id,
        "track": config["track"],
        "model_family": "ssl_pooled_mlp",
        "input": config["model"]["input"],
        "external_pretraining": "WavLM/wav2vec2 frozen external SSL encoder; not protocol-comparable",
        "parameters": params,
        "splits": {split: result["metrics"] for split, result in final_results.items()},
        "command": f"python scripts/train_ssl_head.py --config {args.config}",
    }
    write_model_card(dirs["run"] / "model_card.md", model_card_payload)
    print(f"Artifacts saved: {dirs['run']}")


if __name__ == "__main__":
    main()
