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
from src.neural.dataset import ASVspoofSpectrogramDataset, load_limited_split
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
    dataset = ASVspoofSpectrogramDataset(
        samples=samples,
        transform=transform,
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
    )
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
        loss = criterion(logits, y)
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
    transform = LogMelTransform(
        sample_rate=int(config["data"]["sample_rate"]),
        n_mels=int(config["model"]["n_mels"]),
    )
    model = build_model(
        config["model"]["type"],
        dropout=float(config["model"].get("dropout", 0.3)),
    ).to(device)
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
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

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
