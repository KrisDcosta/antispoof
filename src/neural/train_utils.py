"""Shared utilities for neural training scripts."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.reporting import ensure_dir


def load_config(path: str | os.PathLike) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_run_id(config: dict) -> str:
    seed = config["training"]["seed"]
    epochs = config["training"]["epochs"]
    return f"{config['run_name']}_seed{seed}_{epochs}ep"


def create_run_dirs(output_root: str | os.PathLike, run_id: str) -> dict[str, Path]:
    run_dir = ensure_dir(Path(output_root) / run_id)
    dirs = {
        "run": run_dir,
        "plots": ensure_dir(run_dir / "plots"),
        "scores": ensure_dir(run_dir / "scores"),
        "checkpoints": ensure_dir(run_dir / "checkpoints"),
    }
    return dirs


def environment_payload(config_path: str, device: torch.device, model_parameters: int) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "commit": current_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "device": str(device),
        "model_parameters": model_parameters,
    }


def write_model_card(path: str | os.PathLike, payload: dict) -> None:
    lines = [
        f"# Model Card: {payload['model_name']}",
        "",
        "## Summary",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Track: {payload['track']}",
        f"- Model family: {payload['model_family']}",
        f"- Input representation: {payload['input']}",
        f"- Dataset: ASVspoof 2019 Logical Access",
        f"- External pretrained components: {payload['external_pretraining']}",
        f"- Parameters: {payload['parameters']:,}",
        "",
        "## Results",
        "",
        "| Split | EER | Accuracy | Threshold |",
        "|---|---:|---:|---:|",
    ]
    for split, metrics in payload["splits"].items():
        lines.append(
            f"| {split} | {metrics['eer'] * 100:.2f}% | "
            f"{metrics['accuracy'] * 100:.2f}% | {metrics['threshold']:.4f} |"
        )
    lines.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "## Limitations",
        "",
        "- Eval EER is the primary generalization metric.",
        "- This protocol-comparable neural run does not use external pretrained speech models.",
        "- Accuracy is secondary because ASVspoof splits are class-imbalanced.",
        "- The detector should not be treated as definitive forensic proof.",
        "",
    ])
    Path(path).write_text("\n".join(lines))

