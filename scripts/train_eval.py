"""
Protocol-correct GMM-LLR baseline for ASVspoof 2019 LA.

This entrypoint trains one bonafide GMM and one spoof GMM on frame-level
features, then scores each utterance with the average log-likelihood ratio.
Scores are saved so plots, metrics, and per-attack analysis are reproducible.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_split, split_summary
from src.evaluate import compute_metrics, print_results
from src.gmm_baseline import FRAME_FEATURES, GMMConfig, score_samples, train_gmm
from src.reporting import (
    ensure_dir,
    per_attack_eer,
    save_per_attack_plot,
    save_roc_plot,
    save_score_distribution,
    split_summary_from_rows,
    write_json,
    write_scores_csv,
)


def load_limited_split(data_root: str, split: str, limit: int | None, seed: int):
    samples = load_split(data_root, split, limit=None)
    if limit is None:
        return samples
    return class_balanced_limit(samples, limit, seed)


def class_balanced_limit(samples, limit: int, seed: int):
    if limit <= 0 or limit >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    by_label = {
        1: [sample for sample in samples if sample.label == 1],
        0: [sample for sample in samples if sample.label == 0],
    }
    if not by_label[1] or not by_label[0]:
        return list(rng.choice(samples, size=limit, replace=False))

    per_class = limit // 2
    remainder = limit - (per_class * 2)
    selected = []
    for label in [1, 0]:
        take = min(per_class, len(by_label[label]))
        selected.extend(rng.choice(by_label[label], size=take, replace=False).tolist())
    if remainder:
        remaining = [sample for sample in samples if sample not in selected]
        if remaining:
            selected.extend(rng.choice(remaining, size=min(remainder, len(remaining)), replace=False).tolist())
    rng.shuffle(selected)
    return selected


def metrics_from_rows(rows: list[dict[str, object]]) -> dict:
    y_true = np.array([int(row["label"]) for row in rows])
    y_scores = np.array([float(row["score"]) for row in rows])
    return compute_metrics(y_true, y_scores)


def enrich_rows(
    rows: list[dict[str, object]],
    *,
    split: str,
    feature: str,
    classifier: str,
) -> list[dict[str, object]]:
    for row in rows:
        row["split"] = split
        row["feature"] = feature
        row["classifier"] = classifier
    return rows


def run_one_feature(args: argparse.Namespace, feature: str) -> dict[str, dict]:
    print(f"\n{'=' * 72}")
    print(f"Feature: {feature} | classifier: gmm-llr")

    train_samples = load_limited_split(args.data, "train", args.limit, args.seed)
    print(f"  Train: {split_summary(train_samples)}")

    config = GMMConfig(
        feature=feature,
        n_components=args.gmm_components,
        covariance_type=args.covariance_type,
        max_iter=args.max_iter,
        reg_covar=args.reg_covar,
        seed=args.seed,
        max_frames_per_class=args.max_frames_per_class,
        standardize=not args.no_standardize,
    )
    bundle = train_gmm(train_samples, config, cache_root=args.cache_dir)

    run_id = gmm_run_id(config)
    model_dir = ensure_dir(Path(args.output) / "models")
    model_path = model_dir / f"{run_id}.joblib"
    joblib.dump(bundle, model_path)
    print(f"  Model saved: {model_path}")

    feature_results: dict[str, dict] = {}
    for split in args.splits:
        samples = load_limited_split(args.data, split, args.limit, args.seed + 10)
        print(f"  {split.title()}: {split_summary(samples)}")
        rows = enrich_rows(
            score_samples(samples, bundle, cache_root=args.cache_dir),
            split=split,
            feature=feature,
            classifier="gmm-llr",
        )
        if not rows:
            raise RuntimeError(f"no scores produced for split '{split}'")

        metrics = metrics_from_rows(rows)
        attack_eers = per_attack_eer(rows)
        result = {
            "run_id": run_id,
            "feature": feature,
            "classifier": "gmm-llr",
            "split": split,
            "data_root": args.data,
            "config": asdict(config),
            "model_path": str(model_path),
            "split_summary": split_summary_from_rows(rows),
            "metrics": metrics,
            "per_attack_eer": attack_eers,
        }
        feature_results[split] = result

        scores_path = Path(args.output) / "scores" / f"{run_id}_{split}_scores.csv"
        metrics_path = Path(args.output) / "metrics" / f"{run_id}_{split}_metrics.json"
        roc_path = Path(args.output) / "plots" / f"{run_id}_{split}_roc.png"
        dist_path = Path(args.output) / "plots" / f"{run_id}_{split}_score_distribution.png"
        attack_path = Path(args.output) / "plots" / f"{run_id}_{split}_per_attack_eer.png"

        write_scores_csv(scores_path, rows)
        write_json(metrics_path, result)
        save_roc_plot(roc_path, metrics, f"{feature.upper()} GMM-LLR ROC ({split})")
        save_score_distribution(
            dist_path,
            rows,
            f"{feature.upper()} GMM-LLR score distribution ({split})",
        )
        save_per_attack_plot(
            attack_path,
            attack_eers,
            f"{feature.upper()} GMM-LLR per-attack EER ({split})",
        )

        print_results(f"{feature} gmm-llr {split}", metrics)
        print(f"  Scores saved : {scores_path}")
        print(f"  Metrics saved: {metrics_path}")

    return feature_results


def gmm_run_id(config: GMMConfig) -> str:
    std = "std" if config.standardize else "raw"
    frame_limit = "allframes" if config.max_frames_per_class is None else f"{config.max_frames_per_class}frames"
    return (
        f"gmm_{config.feature}_{config.n_components}c_"
        f"{config.covariance_type}_{std}_{frame_limit}_seed{config.seed}"
    )


def write_ablation_summary(output_dir: str, all_results: dict[str, dict[str, dict]]) -> None:
    summary = {}
    for feature, split_results in all_results.items():
        summary[feature] = {}
        for split, result in split_results.items():
            summary[feature][split] = {
                "eer": result["metrics"]["eer"],
                "threshold": result["metrics"]["threshold"],
                "accuracy_at_eer_threshold": result["metrics"]["accuracy"],
                "metrics_path": str(
                    Path(output_dir)
                    / "metrics"
                    / f"{result['run_id']}_{split}_metrics.json"
                ),
            }
    write_json(Path(output_dir) / "metrics" / "ablation_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/LA", help="Path to data/LA")
    parser.add_argument("--output", default="results/baseline", help="Output directory")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory for cached frame-level features",
    )
    parser.add_argument("--feature", choices=sorted(FRAME_FEATURES), default="cqcc")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=sorted(FRAME_FEATURES),
        default=None,
        help="Feature list for ablations; overrides --feature and --ablation",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the standard baseline ablation: LFCC, MFCC, and CQCC",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "eval"],
        default=["dev"],
        help="Evaluation splits to score after training",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deterministic class-balanced sample cap per split for smoke tests",
    )
    parser.add_argument("--gmm-components", type=int, default=64)
    parser.add_argument("--covariance-type", choices=["diag", "full", "tied", "spherical"], default="diag")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument(
        "--max-frames-per-class",
        type=int,
        default=300_000,
        help="Subsample training frames per class; use 0 for all frames",
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_frames_per_class == 0:
        args.max_frames_per_class = None
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)
    if args.features:
        features = args.features
    elif args.ablation:
        features = ["lfcc", "mfcc", "cqcc"]
    else:
        features = [args.feature]

    all_results = {}
    for feature in features:
        all_results[feature] = run_one_feature(args, feature)
        if args.ablation or args.features:
            write_ablation_summary(args.output, all_results)
    if args.ablation or args.features:
        write_ablation_summary(args.output, all_results)


if __name__ == "__main__":
    main()
