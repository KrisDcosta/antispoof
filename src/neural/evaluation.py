"""Evaluation helpers for neural anti-spoofing models."""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate import compute_metrics
from src.reporting import per_attack_eer, split_summary_from_rows


@torch.no_grad()
def score_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc="score neural"):
        x = batch["x"].to(device)
        logits = model(x)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        for idx, score in enumerate(scores):
            label = int(labels[idx])
            rows.append({
                "file_id": batch["file_id"][idx],
                "path": batch["path"][idx],
                "label": label,
                "label_name": "bonafide" if label == 1 else "spoof",
                "system_id": batch["system_id"][idx],
                "score": float(score),
            })
    return rows


def metrics_from_score_rows(rows: list[dict[str, object]]) -> tuple[dict, dict[str, float], dict[str, object]]:
    y_true = np.array([int(row["label"]) for row in rows])
    y_scores = np.array([float(row["score"]) for row in rows])
    metrics = compute_metrics(y_true, y_scores)
    attack_eers = per_attack_eer(rows)
    summary = split_summary_from_rows(rows)
    return metrics, attack_eers, summary
