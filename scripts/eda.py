"""Preliminary EDA for ASVspoof 2019 LA protocol and audio durations."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import Sample, load_split, split_summary
from src.reporting import ensure_dir, write_json


def sample_durations(samples: list[Sample], n: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    selected = samples if n <= 0 or n >= len(samples) else rng.sample(samples, n)
    rows = []
    failed = 0
    for sample in selected:
        try:
            info = sf.info(sample.path)
            duration = float(info.frames / info.samplerate)
        except Exception:
            failed += 1
            continue
        rows.append({
            "file_id": sample.file_id,
            "label": sample.label,
            "label_name": "bonafide" if sample.label == 1 else "spoof",
            "system_id": sample.system_id,
            "duration_sec": duration,
            "samplerate": int(info.samplerate),
            "frames": int(info.frames),
        })
    if failed:
        print(f"  [!] skipped {failed} files while reading audio metadata")
    return rows


def summarize_durations(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
    durations = sorted(float(row["duration_sec"]) for row in rows)
    n = len(durations)
    return {
        "count": n,
        "min_sec": durations[0],
        "median_sec": percentile(durations, 0.50),
        "p90_sec": percentile(durations, 0.90),
        "max_sec": durations[-1],
        "mean_sec": sum(durations) / n,
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def plot_class_counts(path: Path, summaries: dict[str, dict[str, object]]) -> None:
    splits = list(summaries)
    bonafide = [int(summaries[split]["bonafide"]) for split in splits]
    spoof = [int(summaries[split]["spoof"]) for split in splits]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(splits))
    ax.bar(x, spoof, label="spoof", color="#E45756")
    ax.bar(x, bonafide, bottom=spoof, label="bonafide", color="#4C78A8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(splits)
    ax.set_ylabel("Files")
    ax.set_title("ASVspoof 2019 LA class balance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attack_counts(path: Path, split_samples: dict[str, list[Sample]]) -> None:
    system_ids = sorted({
        sample.system_id
        for samples in split_samples.values()
        for sample in samples
        if sample.system_id != "-"
    })
    splits = list(split_samples)
    width = 0.8 / max(1, len(splits))
    x = list(range(len(system_ids)))

    fig, ax = plt.subplots(figsize=(max(9, len(system_ids) * 0.55), 4.8))
    for idx, split in enumerate(splits):
        counts = Counter(sample.system_id for sample in split_samples[split] if sample.system_id != "-")
        offset = (idx - (len(splits) - 1) / 2) * width
        ax.bar([v + offset for v in x], [counts[s] for s in system_ids], width=width, label=split)

    ax.set_xticks(x)
    ax.set_xticklabels(system_ids, rotation=45, ha="right")
    ax.set_ylabel("Spoof files")
    ax.set_title("Attack-system distribution by split")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_duration_hist(path: Path, duration_rows: dict[str, list[dict[str, object]]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, rows in duration_rows.items():
        durations = [float(row["duration_sec"]) for row in rows]
        if durations:
            ax.hist(durations, bins=40, alpha=0.45, density=True, label=split)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Sampled utterance duration distribution")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/LA", help="Path to data/LA")
    parser.add_argument("--output", default="results/eda")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "eval"], default=["train", "dev", "eval"])
    parser.add_argument(
        "--duration-samples-per-split",
        type=int,
        default=500,
        help="Number of files per split for duration metadata; use 0 for all files",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output)
    plot_dir = ensure_dir(out / "plots")

    split_samples = {split: load_split(args.data, split) for split in args.splits}
    protocol_summaries = {split: split_summary(samples) for split, samples in split_samples.items()}
    duration_rows = {
        split: sample_durations(samples, args.duration_samples_per_split, args.seed)
        for split, samples in split_samples.items()
    }
    duration_summaries = {
        split: summarize_durations(rows)
        for split, rows in duration_rows.items()
    }

    payload = {
        "data_root": args.data,
        "splits": args.splits,
        "duration_samples_per_split": args.duration_samples_per_split,
        "seed": args.seed,
        "protocol_summary": protocol_summaries,
        "duration_summary": duration_summaries,
    }
    write_json(out / "eda_summary.json", payload)
    plot_class_counts(plot_dir / "class_balance.png", protocol_summaries)
    plot_attack_counts(plot_dir / "attack_distribution.png", split_samples)
    plot_duration_hist(plot_dir / "duration_distribution.png", duration_rows)

    print("EDA summary")
    for split in args.splits:
        print(f"  {split}: {protocol_summaries[split]}")
        print(f"       durations: {duration_summaries[split]}")
    print(f"  Summary saved: {out / 'eda_summary.json'}")
    print(f"  Plots saved  : {plot_dir}")


if __name__ == "__main__":
    main()
