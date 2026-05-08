"""Train and evaluate PyTorch neural countermeasures for ASVspoof 2019 LA."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import split_summary
from src.neural.dataset import ASVspoofSpectrogramDataset, ASVspoofWaveformDataset, load_limited_split
from src.neural.evaluation import metrics_from_score_rows, score_loader
from src.neural.models import build_model
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
from src.neural.transforms import LogMelTransform
from src.reporting import (
    save_per_attack_plot,
    save_roc_plot,
    save_score_distribution,
    write_json,
    write_scores_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Neural experiment JSON config")
    return parser.parse_args()


def make_loader(samples, config: dict, transform, *, shuffle: bool) -> DataLoader:
    model_input = config["model"]["input"]
    if model_input == "logmel":
        dataset = ASVspoofSpectrogramDataset(
            samples=samples,
            transform=transform,
            sample_rate=int(config["data"]["sample_rate"]),
            clip_seconds=float(config["data"]["clip_seconds"]),
        )
    elif model_input == "waveform":
        dataset = ASVspoofWaveformDataset(
            samples=samples,
            sample_rate=int(config["data"]["sample_rate"]),
            num_samples=int(config["data"]["num_samples"]),
        )
    else:
        raise ValueError(f"unsupported neural input type: {model_input}")
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in tqdm(loader, desc="train neural"):
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.ndim == 2 and logits.shape[1] == 2:
            loss = criterion(logits, y.long())
        else:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(y)
        total_items += len(y)
    return total_loss / max(total_items, 1)


def build_transform(config: dict):
    if config["model"]["input"] == "waveform":
        return None
    if config["model"]["input"] == "logmel":
        return LogMelTransform(
            sample_rate=int(config["data"]["sample_rate"]),
            n_mels=int(config["model"]["n_mels"]),
        )
    raise ValueError(f"unsupported neural input type: {config['model']['input']}")


def build_configured_model(config: dict) -> torch.nn.Module:
    model_config = dict(config["model"])
    model_type = str(model_config.pop("type"))
    model_config.pop("input", None)
    dropout = float(model_config.pop("dropout", 0.3))
    if model_type == "lcnn":
        model_config.pop("n_mels", None)
    return build_model(model_type, dropout=dropout, **model_config)


def build_criterion(config: dict) -> torch.nn.Module:
    loss_name = str(config["training"].get("loss", "bce")).lower()
    if loss_name == "cross_entropy":
        class_weights = config["training"].get("class_weights")
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return torch.nn.CrossEntropyLoss(weight=weight)
    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    raise ValueError(f"unsupported loss type: {loss_name}")


def move_criterion_to_device(criterion: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return criterion.to(device)


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    training = config["training"]
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    lr = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(training.get("betas", [0.9, 0.999])),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(training.get("betas", [0.9, 0.999])),
        )
    raise ValueError(f"unsupported optimizer type: {optimizer_name}")


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    scheduler_name = str(config["training"].get("scheduler", "none")).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(config["training"].get("min_learning_rate", 0.0)),
        )
    raise ValueError(f"unsupported scheduler type: {scheduler_name}")


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
        scores_path = dirs["scores"] / f"{split}_scores.csv"
        write_scores_csv(scores_path, rows)
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"]["seed"]))

    run_id = build_run_id(config)
    dirs = create_run_dirs(config["output"]["root"], run_id)
    shutil.copy2(args.config, dirs["run"] / "config.json")

    device = select_device(config["training"]["device"])
    transform = build_transform(config)
    model = build_configured_model(config).to(device)
    params = count_parameters(model)
    write_json(dirs["run"] / "environment.json", environment_payload(args.config, device, params))

    train_samples = load_limited_split(
        config["data"]["root"],
        "train",
        config["data"].get("train_limit"),
        seed=int(config["training"]["seed"]),
    )
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Parameters: {params:,}")
    print(f"Train: {split_summary(train_samples)}")

    train_loader = make_loader(train_samples, config, transform, shuffle=True)
    criterion = move_criterion_to_device(build_criterion(config), device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    start = time.time()
    history = []
    best_dev_eer = float("inf")
    best_path = dirs["checkpoints"] / "best.pt"
    eval_loaders = {}
    for split in config["data"]["splits"]:
        samples = load_limited_split(
            config["data"]["root"],
            split,
            config["data"].get("eval_limit"),
            seed=int(config["training"]["seed"]) + 10,
        )
        print(f"{split.title()}: {split_summary(samples)}")
        eval_loaders[split] = make_loader(samples, config, transform, shuffle=False)

    final_results = {}
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if scheduler is not None:
            scheduler.step()
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
                    }, best_path)
        else:
            print(f"Epoch {epoch}: loss={loss:.4f}")
        history.append(epoch_result)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

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

    metrics_payload = {
        "run_id": run_id,
        "track": config["track"],
        "model_name": config["run_name"],
        "model_family": config["model"]["type"],
        "feature_or_input": config["model"]["input"],
        "external_pretraining": False,
        "config_path": args.config,
        "commit": current_commit(),
        "seed": config["training"]["seed"],
        "model_parameters": params,
        "training_seconds": time.time() - start,
        "history": history,
        "splits": final_results,
    }
    write_json(dirs["run"] / "metrics.json", metrics_payload)

    model_card_payload = {
        "model_name": config["run_name"],
        "run_id": run_id,
        "track": config["track"],
        "model_family": config["model"]["type"],
        "input": config["model"]["input"],
        "external_pretraining": "none",
        "parameters": params,
        "splits": {split: result["metrics"] for split, result in final_results.items()},
        "command": f"python scripts/train_neural.py --config {args.config}",
    }
    write_model_card(dirs["run"] / "model_card.md", model_card_payload)
    print(f"Artifacts saved: {dirs['run']}")


if __name__ == "__main__":
    main()
