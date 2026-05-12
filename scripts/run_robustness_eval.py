"""Run Phase 5 robustness evaluation for frozen ASVspoof 2019 LA systems."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_neural import build_configured_model, build_transform
from scripts.train_ssl_head import build_model as build_ssl_head
from src.dataset import Sample, load_split, split_summary
from src.evaluate import compute_metrics
from src.neural.dataset import class_balanced_limit
from src.neural.evaluation import bonafide_scores_from_logits
from src.neural.ssl_dataset import TrainMeanStdNormalizer
from src.neural.ssl_embeddings import load_frozen_ssl_encoder, pooled_mean_std
from src.neural.transforms import crop_or_pad
from src.neural.train_utils import current_commit, load_config, select_device, set_seed
from src.reporting import ensure_dir, per_attack_eer, split_summary_from_rows, write_json, write_scores_csv


@dataclass(frozen=True)
class Corruption:
    name: str
    type: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FusionStats:
    lcnn_mean: float
    lcnn_std: float
    wavlm_mean: float
    wavlm_std: float
    alpha: float = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Phase 5 robustness JSON config")
    return parser.parse_args()


def read_json(path: str | os.PathLike) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_config(config: dict[str, Any]) -> None:
    required = [
        Path(config["data_root"]) / "ASVspoof2019_LA_eval" / "flac",
        Path(config["data_root"]) / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
        Path(config["systems"]["lcnn"]["config_path"]),
        Path(config["systems"]["lcnn"]["checkpoint_path"]),
        Path(config["systems"]["wavlm"]["config_path"]),
        Path(config["systems"]["wavlm"]["checkpoint_path"]),
        Path(config["systems"]["fusion"]["summary_path"]),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Phase 5 robustness missing required artifacts:\n{detail}")
    if config.get("split", "eval") != "eval":
        raise ValueError("Phase 5 robustness is eval-only for this implementation")
    if not config.get("corruptions"):
        raise ValueError("config must define at least one corruption")


def load_samples(config: dict[str, Any]) -> list[Sample]:
    samples = load_split(config["data_root"], "eval", limit=None)
    limit = config.get("limit")
    if limit is None or int(limit) <= 0 or int(limit) >= len(samples):
        return samples
    return class_balanced_limit(samples, int(limit), int(config.get("seed", 42)))


def parse_corruptions(config: dict[str, Any]) -> list[Corruption]:
    corruptions = []
    for item in config["corruptions"]:
        name = str(item["name"])
        ctype = str(item["type"])
        params = {key: value for key, value in item.items() if key not in {"name", "type"}}
        corruptions.append(Corruption(name=name, type=ctype, params=params))
    return corruptions


def apply_corruption(
    waveform: torch.Tensor,
    corruption: Corruption | dict[str, Any],
    *,
    sample_rate: int = 16_000,
    seed: int = 42,
    file_id: str = "",
) -> torch.Tensor:
    if isinstance(corruption, dict):
        corruption = Corruption(
            name=str(corruption["name"]),
            type=str(corruption["type"]),
            params={key: value for key, value in corruption.items() if key not in {"name", "type"}},
        )
    waveform = waveform.to(torch.float32).flatten()
    ctype = corruption.type
    if ctype == "identity":
        return waveform.clone()
    if ctype == "gain":
        factor = 10.0 ** (float(corruption.params["db"]) / 20.0)
        return waveform * factor
    if ctype == "clipping":
        threshold = float(corruption.params["threshold"])
        if threshold <= 0:
            raise ValueError("clipping threshold must be positive")
        return waveform.clamp(-threshold, threshold)
    if ctype == "resample":
        target_sr = int(corruption.params["target_sample_rate"])
        if target_sr <= 0:
            raise ValueError("target_sample_rate must be positive")
        degraded = AF.resample(waveform, sample_rate, target_sr)
        return AF.resample(degraded, target_sr, sample_rate)
    if ctype == "noise":
        snr_db = float(corruption.params["snr_db"])
        signal_power = waveform.pow(2).mean().clamp_min(1e-12)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        stable = sum(ord(ch) for ch in f"{file_id}:{corruption.name}")
        generator = torch.Generator(device=waveform.device).manual_seed(int(seed) + stable)
        noise = torch.randn(waveform.shape, generator=generator, device=waveform.device, dtype=waveform.dtype)
        noise = noise * torch.sqrt(noise_power / noise.pow(2).mean().clamp_min(1e-12))
        return waveform + noise
    if ctype == "codec":
        raise NotImplementedError("codec corruption is optional and not enabled in this runner")
    raise ValueError(f"unsupported corruption type: {ctype}")


def load_waveform(sample: Sample, sample_rate: int, target_samples: int) -> torch.Tensor:
    audio, sr = sf.read(sample.path, dtype="float32", always_2d=False)
    waveform = torch.from_numpy(np.asarray(audio))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    if sr != sample_rate:
        waveform = AF.resample(waveform, sr, sample_rate)
    return crop_or_pad(waveform, target_samples)


def batched(items: list[Any], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield items[idx:idx + batch_size]


def score_lcnn(
    *,
    samples: list[Sample],
    corruption: Corruption,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, object]]:
    lcnn_config = load_config(config["systems"]["lcnn"]["config_path"])
    checkpoint = load_checkpoint(Path(config["systems"]["lcnn"]["checkpoint_path"]), device)
    model = build_configured_model(lcnn_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = build_transform(lcnn_config).to(device)
    sample_rate = int(config["sample_rate"])
    target_samples = int(round(sample_rate * float(lcnn_config["data"]["clip_seconds"])))
    batch_size = int(config.get("batch_size", lcnn_config["training"].get("batch_size", 32)))
    rows = []
    with torch.inference_mode():
        for batch in tqdm(list(batched(samples, batch_size)), desc=f"score lcnn {corruption.name}"):
            features = []
            for sample in batch:
                waveform = load_waveform(sample, sample_rate, target_samples)
                waveform = apply_corruption(
                    waveform,
                    corruption,
                    sample_rate=sample_rate,
                    seed=int(config.get("seed", 42)),
                    file_id=sample.file_id,
                )
                waveform = crop_or_pad(waveform, target_samples).to(device)
                features.append(transform(waveform))
            logits = model(torch.stack(features))
            scores = bonafide_scores_from_logits(logits).detach().cpu().numpy()
            for sample, score in zip(batch, scores):
                rows.append(score_row(sample, score, "lcnn", corruption))
    return rows


def normalizer_from_checkpoint(checkpoint: dict[str, Any]) -> TrainMeanStdNormalizer:
    state = checkpoint.get("normalization")
    if not state:
        raise ValueError("WavLM checkpoint does not contain train normalization stats")
    # The checkpoint may be loaded directly onto an accelerator, while pooled
    # embeddings are intentionally moved to CPU before normalization. Keep the
    # saved train statistics on CPU to avoid cross-device arithmetic.
    return TrainMeanStdNormalizer(
        state["mean"].detach().cpu(),
        state["std"].detach().cpu(),
        eps=float(state.get("eps", 1e-6)),
    )


def score_wavlm(
    *,
    samples: list[Sample],
    corruption: Corruption,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, object]]:
    wavlm_config = load_config(config["systems"]["wavlm"]["config_path"])
    checkpoint = load_checkpoint(Path(config["systems"]["wavlm"]["checkpoint_path"]), device)
    head = build_ssl_head(wavlm_config).to(device)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()
    normalizer = normalizer_from_checkpoint(checkpoint)
    processor, encoder = load_frozen_ssl_encoder(wavlm_config["ssl"]["encoder_name"], device)
    sample_rate = int(config["sample_rate"])
    target_samples = int(wavlm_config["data"]["num_samples"])
    batch_size = int(config.get("ssl_batch_size", min(8, wavlm_config["training"].get("batch_size", 8))))
    rows = []
    with torch.inference_mode():
        for batch in tqdm(list(batched(samples, batch_size)), desc=f"score wavlm {corruption.name}"):
            arrays = []
            for sample in batch:
                waveform = load_waveform(sample, sample_rate, target_samples)
                waveform = apply_corruption(
                    waveform,
                    corruption,
                    sample_rate=sample_rate,
                    seed=int(config.get("seed", 42)),
                    file_id=sample.file_id,
                )
                waveform = crop_or_pad(waveform, target_samples)
                arrays.append(waveform.detach().cpu().numpy())
            processed = processor(arrays, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            processed = {key: value.to(device) for key, value in processed.items()}
            hidden = encoder(**processed).last_hidden_state
            embeddings = pooled_mean_std(hidden).detach().cpu().to(torch.float32)
            x = normalizer.transform(embeddings).to(device)
            scores = bonafide_scores_from_logits(head(x)).detach().cpu().numpy()
            for sample, score in zip(batch, scores):
                rows.append(score_row(sample, score, "wavlm", corruption))
    return rows


def score_row(sample: Sample, score: float, model: str, corruption: Corruption) -> dict[str, object]:
    return {
        "file_id": sample.file_id,
        "path": sample.path,
        "label": int(sample.label),
        "label_name": "bonafide" if int(sample.label) == 1 else "spoof",
        "system_id": sample.system_id,
        "score": float(score),
        "model": model,
        "condition": corruption.name,
        "corruption_type": corruption.type,
        "severity": severity_label(corruption),
    }


def severity_label(corruption: Corruption) -> str:
    if not corruption.params:
        return "clean"
    return ",".join(f"{key}={value}" for key, value in sorted(corruption.params.items()))


def load_fusion_stats(summary_path: str | os.PathLike) -> FusionStats:
    path = Path(summary_path)
    payload = read_json(path)
    stats = payload.get("score_normalization")
    if stats is None:
        full_metrics = path.parent.parent / path.stem.replace("_summary", "") / "metrics.json"
        if full_metrics.exists():
            stats = read_json(full_metrics).get("score_normalization")
    if not stats:
        raise ValueError(
            f"Phase 4 fusion normalization stats not found in {path}; "
            "restore results/fusion/<run_id>/metrics.json or provide a summary containing score_normalization"
        )
    rule = payload.get("methods", {}).get("weighted_mean", {}).get("rule", {})
    return FusionStats(
        lcnn_mean=float(stats["lcnn"]["mean"]),
        lcnn_std=float(stats["lcnn"]["std"]),
        wavlm_mean=float(stats["wavlm"]["mean"]),
        wavlm_std=float(stats["wavlm"]["std"]),
        alpha=float(rule.get("alpha", 0.7)),
    )


def apply_frozen_fusion_score(lcnn_score: float, wavlm_score: float, stats: FusionStats) -> float:
    z_lcnn = (float(lcnn_score) - stats.lcnn_mean) / stats.lcnn_std
    z_wavlm = (float(wavlm_score) - stats.wavlm_mean) / stats.wavlm_std
    return stats.alpha * z_lcnn + (1.0 - stats.alpha) * z_wavlm


def fusion_rows(
    lcnn_rows: list[dict[str, object]],
    wavlm_rows: list[dict[str, object]],
    stats: FusionStats,
) -> list[dict[str, object]]:
    by_id = {str(row["file_id"]): row for row in wavlm_rows}
    rows = []
    for lcnn in lcnn_rows:
        file_id = str(lcnn["file_id"])
        wavlm = by_id[file_id]
        if int(lcnn["label"]) != int(wavlm["label"]) or str(lcnn["system_id"]) != str(wavlm["system_id"]):
            raise ValueError(f"LCNN/WavLM metadata mismatch for {file_id}")
        row = dict(lcnn)
        row["model"] = "fusion"
        row["score"] = apply_frozen_fusion_score(float(lcnn["score"]), float(wavlm["score"]), stats)
        row["lcnn_score"] = float(lcnn["score"])
        row["wavlm_score"] = float(wavlm["score"])
        row["z_lcnn"] = (float(lcnn["score"]) - stats.lcnn_mean) / stats.lcnn_std
        row["z_wavlm"] = (float(wavlm["score"]) - stats.wavlm_mean) / stats.wavlm_std
        row["fusion_rule"] = f"{stats.alpha:.1f}*z_lcnn+{1.0 - stats.alpha:.1f}*z_wavlm"
        rows.append(row)
    return rows


def summarize_condition(rows: list[dict[str, object]], clean_eer: float | None) -> tuple[dict[str, object], list[dict[str, object]]]:
    y_true = np.array([int(row["label"]) for row in rows], dtype=int)
    y_score = np.array([float(row["score"]) for row in rows], dtype=float)
    metrics = compute_metrics(y_true, y_score)
    eer = float(metrics["eer"])
    delta = None if clean_eer is None else eer - clean_eer
    ratio = None if clean_eer is None or clean_eer <= 0 else eer / clean_eer
    first = rows[0]
    summary = {
        "condition": first["condition"],
        "corruption_type": first["corruption_type"],
        "severity": first["severity"],
        "model": first["model"],
        "eer": eer,
        "threshold": float(metrics["threshold"]),
        "accuracy": float(metrics["accuracy"]),
        "delta_vs_clean": delta,
        "ratio_vs_clean": ratio,
        "split_summary": split_summary_from_rows(rows),
    }
    attack_rows = [
        {
            "condition": first["condition"],
            "model": first["model"],
            "attack_id": attack,
            "eer": eer_value,
        }
        for attack, eer_value in per_attack_eer(rows).items()
    ]
    return summary, attack_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, samples: list[Sample], corruptions: list[Corruption]) -> None:
    rows = [
        {
            "file_id": sample.file_id,
            "path": sample.path,
            "label": sample.label,
            "system_id": sample.system_id,
            "condition": corruption.name,
            "corruption_type": corruption.type,
            "severity": severity_label(corruption),
            "audio_written": False,
        }
        for corruption in corruptions
        for sample in samples
    ]
    write_csv(path, rows)


def save_eer_by_condition(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    x = np.arange(len(conditions))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(9, len(conditions) * 1.1), 5))
    for idx, model in enumerate(models):
        values = [
            next((float(row["eer"]) * 100 for row in rows if row["condition"] == condition and row["model"] == model), np.nan)
            for condition in conditions
        ]
        ax.bar(x + (idx - (len(models) - 1) / 2) * width, values, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.set_ylabel("EER (%)")
    ax.set_title("Phase 5 robustness EER by condition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_relative_degradation(path: Path, rows: list[dict[str, object]]) -> None:
    rows = [row for row in rows if row.get("delta_vs_clean") is not None]
    if not rows:
        return
    ensure_dir(path.parent)
    labels = [f"{row['model']} / {row['condition']}" for row in rows]
    values = [float(row["delta_vs_clean"]) * 100 for row in rows]
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.45), 5))
    ax.bar(np.arange(len(rows)), values, color="#4C78A8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("EER change vs clean (percentage points)")
    ax.set_title("Relative robustness degradation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_per_attack_heatmap(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    attacks = sorted({str(row["attack_id"]) for row in rows})
    keys = list(dict.fromkeys(f"{row['model']} / {row['condition']}" for row in rows))
    matrix = np.full((len(keys), len(attacks)), np.nan)
    for row in rows:
        matrix[keys.index(f"{row['model']} / {row['condition']}"), attacks.index(str(row["attack_id"]))] = float(row["eer"]) * 100
    fig, ax = plt.subplots(figsize=(max(9, len(attacks) * 0.65), max(5, len(keys) * 0.32)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(attacks)))
    ax.set_xticklabels(attacks)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_title("Per-attack EER heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("EER (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compact_summary(
    config: dict[str, Any],
    run_dir: Path,
    condition_rows: list[dict[str, object]],
    skipped: list[dict[str, str]],
    elapsed: float,
) -> dict[str, object]:
    primary = {
        f"{row['model']}::{row['condition']}": {
            "eer": row["eer"],
            "delta_vs_clean": row["delta_vs_clean"],
            "ratio_vs_clean": row["ratio_vs_clean"],
        }
        for row in condition_rows
    }
    return {
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": current_commit(),
        "split": config.get("split", "eval"),
        "limit": config.get("limit"),
        "phase": "phase5_robustness_eval",
        "accepted_phase4_fusion_rule": "fused_score = 0.7 * z_lcnn + 0.3 * z_wavlm",
        "elapsed_seconds": elapsed,
        "artifact_dir": str(run_dir),
        "primary": primary,
        "skipped_corruptions": skipped,
    }


def model_card(config: dict[str, Any], condition_rows: list[dict[str, object]], skipped: list[dict[str, str]]) -> str:
    lines = [
        f"# Model Card: {config['run_id']}",
        "",
        "## Scope",
        "",
        "- Task: ASVspoof 2019 LA eval robustness under deterministic synthetic corruptions.",
        "- Systems: LCNN, frozen WavLM pooled MLP, and frozen LCNN+WavLM score fusion.",
        "- No retraining, fusion retuning, or normalization fitting on corrupted eval is performed.",
        "",
        "## Primary Metrics",
        "",
        "| Model | Condition | EER | Delta vs clean |",
        "|---|---|---:|---:|",
    ]
    for row in condition_rows:
        delta = row["delta_vs_clean"]
        delta_text = "n/a" if delta is None else f"{float(delta) * 100:.2f} pp"
        lines.append(f"| {row['model']} | {row['condition']} | {float(row['eer']) * 100:.2f}% | {delta_text} |")
    if skipped:
        lines.extend(["", "## Skipped Corruptions", ""])
        for item in skipped:
            lines.append(f"- {item['condition']}: {item['reason']}")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- These are controlled synthetic corruptions, not a full real-world robustness claim.",
        "- Eval is used only for measurement under predeclared conditions.",
        "",
    ])
    return "\n".join(lines)


def run(config_path: str | os.PathLike) -> dict[str, object]:
    start = time.time()
    config = load_config(str(config_path))
    set_seed(int(config.get("seed", 42)))
    validate_config(config)

    run_dir = ensure_dir(Path(config["output_root"]) / config["run_id"])
    plots_dir = ensure_dir(run_dir / "plots")
    scores_dir = ensure_dir(run_dir / "scores")
    metrics_dir = ensure_dir(Path(config["output_root"]) / "metrics")
    shutil.copy2(config_path, run_dir / "config.json")

    samples = load_samples(config)
    corruptions = parse_corruptions(config)
    fusion_stats = load_fusion_stats(config["systems"]["fusion"]["summary_path"])
    if abs(fusion_stats.alpha - 0.7) > 1e-9:
        raise ValueError(f"expected frozen fusion alpha 0.7, got {fusion_stats.alpha}")

    device = select_device(str(config.get("device", "auto")))
    print(f"Run ID: {config['run_id']}")
    print(f"Device: {device}")
    print(f"Eval samples: {split_summary(samples)}")

    skipped: list[dict[str, str]] = []
    all_lcnn_rows: list[dict[str, object]] = []
    all_wavlm_rows: list[dict[str, object]] = []
    all_fusion_rows: list[dict[str, object]] = []
    condition_metrics: list[dict[str, object]] = []
    attack_metrics: list[dict[str, object]] = []
    clean_eers: dict[str, float] = {}

    write_manifest(run_dir / "corruption_manifest.csv", samples, corruptions)

    for corruption in corruptions:
        if corruption.type == "codec":
            skipped.append({"condition": corruption.name, "reason": "codec corruption is optional and not enabled"})
            continue
        lcnn_rows = score_lcnn(samples=samples, corruption=corruption, config=config, device=device)
        wavlm_rows = score_wavlm(samples=samples, corruption=corruption, config=config, device=device)
        fused_rows = fusion_rows(lcnn_rows, wavlm_rows, fusion_stats)
        for model_rows in [lcnn_rows, wavlm_rows, fused_rows]:
            model_name = str(model_rows[0]["model"])
            clean_eer = None if corruption.type == "identity" else clean_eers.get(model_name)
            summary, per_attack = summarize_condition(model_rows, clean_eer)
            if corruption.type == "identity":
                clean_eers[model_name] = float(summary["eer"])
            condition_metrics.append(summary)
            attack_metrics.extend(per_attack)
        all_lcnn_rows.extend(lcnn_rows)
        all_wavlm_rows.extend(wavlm_rows)
        all_fusion_rows.extend(fused_rows)
        print(
            f"{corruption.name}: "
            f"LCNN={condition_metrics[-3]['eer'] * 100:.2f}% "
            f"WavLM={condition_metrics[-2]['eer'] * 100:.2f}% "
            f"Fusion={condition_metrics[-1]['eer'] * 100:.2f}%"
        )

    write_scores_csv(scores_dir / "lcnn_scores.csv", all_lcnn_rows)
    write_scores_csv(scores_dir / "wavlm_scores.csv", all_wavlm_rows)
    write_scores_csv(scores_dir / "fusion_scores.csv", all_fusion_rows)
    write_csv(run_dir / "per_condition_metrics.csv", condition_metrics)
    write_csv(run_dir / "per_attack_condition_metrics.csv", attack_metrics)
    save_eer_by_condition(plots_dir / "eer_by_condition.png", condition_metrics)
    save_relative_degradation(plots_dir / "relative_degradation.png", condition_metrics)
    save_per_attack_heatmap(plots_dir / "per_attack_heatmap.png", attack_metrics)

    elapsed = time.time() - start
    metrics_payload = {
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": current_commit(),
        "config_path": str(config_path),
        "split_summary": split_summary(samples),
        "fusion_rule": {
            "type": "weighted_mean",
            "alpha_lcnn": fusion_stats.alpha,
            "alpha_wavlm": 1.0 - fusion_stats.alpha,
            "normalization_source": config["systems"]["fusion"]["summary_path"],
        },
        "per_condition": condition_metrics,
        "per_attack_condition": attack_metrics,
        "skipped_corruptions": skipped,
        "elapsed_seconds": elapsed,
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    summary = compact_summary(config, run_dir, condition_metrics, skipped, elapsed)
    write_json(metrics_dir / f"{config['run_id']}_summary.json", summary)
    (run_dir / "model_card.md").write_text(model_card(config, condition_metrics, skipped))
    print(f"Artifacts saved: {run_dir}")
    return summary


def main() -> None:
    try:
        run(parse_args().config)
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        raise SystemExit(f"Phase 5 robustness aborted: {exc}") from exc


if __name__ == "__main__":
    main()
