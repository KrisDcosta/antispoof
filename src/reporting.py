"""Utilities for writing baseline metrics, score files, and plots."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate import compute_eer


def ensure_dir(path: str | os.PathLike) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | os.PathLike, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def write_scores_csv(path: str | os.PathLike, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty score file")
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def per_attack_eer(rows: list[dict[str, object]]) -> dict[str, float]:
    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    systems = np.array([str(r["system_id"]) for r in rows])
    bonafide_mask = labels == 1
    results: dict[str, float] = {}

    for system_id in sorted(s for s in set(systems) if s != "-"):
        attack_mask = systems == system_id
        mask = bonafide_mask | attack_mask
        if len(np.unique(labels[mask])) < 2:
            continue
        eer, _ = compute_eer(labels[mask], scores[mask])
        results[system_id] = float(eer)
    return results


def split_summary_from_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = [int(r["label"]) for r in rows]
    systems = [str(r["system_id"]) for r in rows if str(r["system_id"]) != "-"]
    return {
        "total": len(rows),
        "bonafide": int(sum(labels)),
        "spoof": int(len(labels) - sum(labels)),
        "systems": dict(sorted(Counter(systems).items())),
    }


def save_roc_plot(path: str | os.PathLike, metrics: dict, title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(metrics["fpr"], metrics["tpr"], label=f"EER={metrics['eer'] * 100:.2f}%")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_score_distribution(path: str | os.PathLike, rows: list[dict[str, object]], title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    bonafide = [float(r["score"]) for r in rows if int(r["label"]) == 1]
    spoof = [float(r["score"]) for r in rows if int(r["label"]) == 0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(spoof, bins=80, alpha=0.65, label="spoof", density=True)
    ax.hist(bonafide, bins=80, alpha=0.65, label="bonafide", density=True)
    ax.set_xlabel("GMM log-likelihood ratio score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_per_attack_plot(path: str | os.PathLike, attack_eers: dict[str, float], title: str) -> None:
    if not attack_eers:
        return
    path = Path(path)
    ensure_dir(path.parent)
    systems = sorted(attack_eers)
    eers = [attack_eers[s] * 100 for s in systems]

    fig, ax = plt.subplots(figsize=(max(8, len(systems) * 0.65), 4.8))
    bars = ax.bar(systems, eers, color="#4C78A8")
    ax.set_xlabel("Attack system")
    ax.set_ylabel("EER (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25, f"{val:.1f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_eer_comparison_plot(
    path: str | os.PathLike,
    rows: list[dict[str, object]],
    title: str,
) -> None:
    """Grouped EER comparison by method and split."""
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)

    splits = [split for split in ["dev", "eval"] if any(r["split"] == split for r in rows)]
    methods = [r["method"] for r in rows]
    methods = list(dict.fromkeys(methods))
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(splits))

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.3), 5))
    for idx, split in enumerate(splits):
        values = []
        for method in methods:
            match = next((r for r in rows if r["method"] == method and r["split"] == split), None)
            values.append(np.nan if match is None else float(match["eer_percent"]))
        offset = (idx - (len(splits) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=split)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.25,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if value > 20 else 0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("EER (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_eval_vs_baseline_plot(
    path: str | os.PathLike,
    project_rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
    title: str,
) -> None:
    """Plot eval EER for project methods against standard baselines."""
    rows = []
    for row in baseline_rows:
        if row.get("eval_eer_percent") is not None:
            rows.append({
                "method": row["name"],
                "eer_percent": float(row["eval_eer_percent"]),
                "group": "standard",
            })
    for row in project_rows:
        if row["split"] == "eval":
            rows.append({
                "method": row["method"],
                "eer_percent": float(row["eer_percent"]),
                "group": "project",
            })
    if not rows:
        return

    path = Path(path)
    ensure_dir(path.parent)
    colors = ["#72B7B2" if row["group"] == "standard" else "#4C78A8" for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.25), 5))
    bars = ax.bar(x, [row["eer_percent"] for row in rows], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([row["method"] for row in rows], rotation=25, ha="right")
    ax.set_ylabel("Eval EER (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["eer_percent"] + 0.25,
            f"{row['eer_percent']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
