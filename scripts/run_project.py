"""Run the complete ASVspoof 2019 LA experiment from a JSON config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/asvspoof2019_gmm.json",
        help="Path to experiment config JSON",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Do not regenerate EDA artifacts",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not run model training and scoring",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not regenerate result tables and comparison plots",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional class-balanced utterance cap for smoke runs",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())

    py = sys.executable
    data = config["data"]
    output = config["output"]

    if not args.skip_eda:
        run([py, "scripts/eda.py", "--data", data, "--output", "results/eda"])

    if not args.skip_train:
        train_cmd = [
            py,
            "scripts/train_eval.py",
            "--data",
            data,
            "--output",
            output,
            "--cache-dir",
            config["cache_dir"],
            "--features",
            *config["features"],
            "--splits",
            *config["splits"],
            "--gmm-components",
            str(config["gmm_components"]),
            "--covariance-type",
            config["covariance_type"],
            "--max-iter",
            str(config["max_iter"]),
            "--reg-covar",
            str(config["reg_covar"]),
            "--max-frames-per-class",
            str(config["max_frames_per_class"] or 0),
            "--seed",
            str(config["seed"]),
        ]
        if args.limit is not None:
            train_cmd.extend(["--limit", str(args.limit)])
        if not config.get("standardize", True):
            train_cmd.append("--no-standardize")
        run(train_cmd)

    if not args.skip_summary:
        run([
            py,
            "scripts/summarize_results.py",
            "--results",
            output,
            "--output",
            config["summary_output"],
        ])


if __name__ == "__main__":
    main()
