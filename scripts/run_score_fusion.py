"""Run score-level fusion for ASVspoof 2019 LA systems.

The script fits every score transformation on dev only, freezes the chosen
fusion rule by dev EER, then applies the frozen rules to eval scores.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import compute_metrics
from src.neural.train_utils import current_commit
from src.reporting import ensure_dir, per_attack_eer, split_summary_from_rows, write_json, write_scores_csv


DEFAULT_LCNN_RUN = "results/neural/lcnn_logmel_full_seed42_30ep"
DEFAULT_WAVLM_RUN = "results/neural/ssl_wavlm_pooled_full_seed42_50ep"
DEFAULT_LFCC_DEV = "results/baseline/scores/gmm_lfcc_64c_diag_std_300000frames_seed42_dev_scores.csv"
DEFAULT_LFCC_EVAL = "results/baseline/scores/gmm_lfcc_64c_diag_std_300000frames_seed42_eval_scores.csv"


@dataclass(frozen=True)
class ScoreSource:
    name: str
    dev_path: Path
    eval_path: Path
    invert: bool = False


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Fusion run ID")
    parser.add_argument("--output-root", default="results/fusion", help="Fusion output root")
    parser.add_argument("--lcnn-run", default=DEFAULT_LCNN_RUN, help="LCNN run directory")
    parser.add_argument("--lcnn-dev", default=None, help="Override LCNN dev score CSV")
    parser.add_argument("--lcnn-eval", default=None, help="Override LCNN eval score CSV")
    parser.add_argument("--wavlm-run", default=DEFAULT_WAVLM_RUN, help="WavLM run directory")
    parser.add_argument("--wavlm-dev", default=None, help="Override WavLM dev score CSV")
    parser.add_argument("--wavlm-eval", default=None, help="Override WavLM eval score CSV")
    parser.add_argument("--lfcc-dev", default=DEFAULT_LFCC_DEV, help="Optional LFCC dev score CSV")
    parser.add_argument("--lfcc-eval", default=DEFAULT_LFCC_EVAL, help="Optional LFCC eval score CSV")
    parser.add_argument("--include-lfcc", action="store_true", help="Also run a three-way LFCC fusion check")
    parser.add_argument("--invert-source", action="append", default=[], help="Source name to invert before fusion")
    parser.add_argument("--alpha-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    return parser.parse_args()


def source_from_run(name: str, run_dir: str, inverted: set[str]) -> ScoreSource:
    base = Path(run_dir) / "scores"
    return ScoreSource(
        name=name,
        dev_path=base / "dev_scores.csv",
        eval_path=base / "eval_scores.csv",
        invert=name in inverted,
    )


def source_from_paths(name: str, dev_path: str, eval_path: str, inverted: set[str]) -> ScoreSource:
    return ScoreSource(name, Path(dev_path), Path(eval_path), name in inverted)


def lfcc_source(dev_path: str, eval_path: str, inverted: set[str]) -> ScoreSource:
    return ScoreSource("lfcc_gmm", Path(dev_path), Path(eval_path), "lfcc_gmm" in inverted)


def read_score_csv(path: Path, *, invert: bool = False) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"score file not found: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"file_id", "label", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = {}
        for row in reader:
            file_id = str(row["file_id"])
            if file_id in rows:
                raise ValueError(f"{path} contains duplicate file_id: {file_id}")
            score = float(row["score"])
            row["score"] = -score if invert else score
            row["label"] = int(row["label"])
            row.setdefault("system_id", "-")
            row.setdefault("label_name", "bonafide" if int(row["label"]) == 1 else "spoof")
            rows[file_id] = row
    if not rows:
        raise ValueError(f"empty score file: {path}")
    return rows


def align_scores(sources: list[ScoreSource], split: str) -> list[dict[str, object]]:
    loaded = {
        source.name: read_score_csv(
            source.dev_path if split == "dev" else source.eval_path,
            invert=source.invert,
        )
        for source in sources
    }
    common_ids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    if not common_ids:
        raise ValueError(f"no common file_id values found for {split}")

    dropped = {name: len(rows) - len(common_ids) for name, rows in loaded.items()}
    if any(count for count in dropped.values()):
        print(f"{split}: aligning on {len(common_ids)} common files; dropped {dropped}")

    primary_name = sources[0].name
    aligned = []
    for file_id in common_ids:
        base = loaded[primary_name][file_id]
        row = {
            "file_id": file_id,
            "path": base.get("path", ""),
            "label": int(base["label"]),
            "label_name": base.get("label_name", "bonafide" if int(base["label"]) == 1 else "spoof"),
            "system_id": base.get("system_id", "-"),
        }
        for source in sources:
            other = loaded[source.name][file_id]
            if int(other["label"]) != int(row["label"]):
                raise ValueError(f"label mismatch for {file_id}: {primary_name} vs {source.name}")
            if str(other.get("system_id", "-")) != str(row["system_id"]):
                raise ValueError(f"system_id mismatch for {file_id}: {primary_name} vs {source.name}")
            row[f"{source.name}_score"] = float(other["score"])
        aligned.append(row)
    return aligned


def fit_dev_normalization(rows: list[dict[str, object]], source_names: list[str]) -> dict[str, NormalizationStats]:
    stats = {}
    for name in source_names:
        scores = np.array([float(row[f"{name}_score"]) for row in rows], dtype=float)
        std = float(scores.std(ddof=0))
        if std <= 0.0:
            raise ValueError(f"cannot normalize constant score source: {name}")
        stats[name] = NormalizationStats(mean=float(scores.mean()), std=std)
    return stats


def add_normalized_scores(
    rows: list[dict[str, object]],
    source_names: list[str],
    stats: dict[str, NormalizationStats],
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        fused = dict(row)
        for name in source_names:
            score = float(row[f"{name}_score"])
            source_stats = stats[name]
            fused[f"z_{name}"] = (score - source_stats.mean) / source_stats.std
        output.append(fused)
    return output


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([int(row["label"]) for row in rows], dtype=int)


def feature_matrix(rows: list[dict[str, object]], source_names: list[str]) -> np.ndarray:
    return np.array([[float(row[f"z_{name}"]) for name in source_names] for row in rows], dtype=float)


def rows_with_score(rows: list[dict[str, object]], scores: np.ndarray, method: str) -> list[dict[str, object]]:
    output = []
    for row, score in zip(rows, scores):
        fused = dict(row)
        fused["score"] = float(score)
        fused["fusion_method"] = method
        output.append(fused)
    return output


def summarize_rows(rows: list[dict[str, object]], dev_threshold: float | None = None) -> dict[str, object]:
    y_true = labels(rows)
    y_score = np.array([float(row["score"]) for row in rows], dtype=float)
    metrics = compute_metrics(y_true, y_score)
    accuracy_at_dev_threshold = None
    if dev_threshold is not None:
        accuracy_at_dev_threshold = float(((y_score >= dev_threshold).astype(int) == y_true).mean())
    return {
        "eer": float(metrics["eer"]),
        "threshold": float(metrics["threshold"]),
        "accuracy": float(metrics["accuracy"]),
        "accuracy_at_dev_threshold": accuracy_at_dev_threshold,
        "confusion_matrix": metrics["confusion_matrix"],
        "split_summary": split_summary_from_rows(rows),
        "per_attack_eer": per_attack_eer(rows),
    }


def mean_scores(rows: list[dict[str, object]], source_names: list[str]) -> np.ndarray:
    return feature_matrix(rows, source_names).mean(axis=1)


def weighted_scores(rows: list[dict[str, object]], left: str, right: str, alpha: float) -> np.ndarray:
    return np.array([
        alpha * float(row[f"z_{left}"]) + (1.0 - alpha) * float(row[f"z_{right}"])
        for row in rows
    ])


def select_weighted_alpha(
    rows: list[dict[str, object]],
    left: str,
    right: str,
    alpha_grid: list[float],
) -> tuple[float, list[dict[str, float]]]:
    y_true = labels(rows)
    candidates = []
    for alpha in alpha_grid:
        scores = weighted_scores(rows, left, right, alpha)
        metrics = compute_metrics(y_true, scores)
        candidates.append({"alpha": float(alpha), "dev_eer": float(metrics["eer"]), "dev_threshold": float(metrics["threshold"])})
    best = min(candidates, key=lambda item: (item["dev_eer"], abs(item["alpha"] - 0.5)))
    return float(best["alpha"]), candidates


def fit_logistic_fusion(rows: list[dict[str, object]], source_names: list[str]) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(feature_matrix(rows, source_names), labels(rows))
    return model


def logistic_scores(model: LogisticRegression, rows: list[dict[str, object]], source_names: list[str]) -> np.ndarray:
    return model.predict_proba(feature_matrix(rows, source_names))[:, 1]


def evaluate_fusion_methods(
    dev_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    source_names: list[str],
    alpha_grid: list[float],
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]], dict[str, object]]:
    methods: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    methods["mean"] = (
        mean_scores(dev_rows, source_names),
        mean_scores(eval_rows, source_names),
        {"type": "mean", "sources": source_names},
    )

    if len(source_names) == 2:
        alpha, alpha_candidates = select_weighted_alpha(dev_rows, source_names[0], source_names[1], alpha_grid)
        methods["weighted_mean"] = (
            weighted_scores(dev_rows, source_names[0], source_names[1], alpha),
            weighted_scores(eval_rows, source_names[0], source_names[1], alpha),
            {"type": "weighted_mean", "alpha": alpha, "left": source_names[0], "right": source_names[1], "alpha_candidates": alpha_candidates},
        )

    logistic = fit_logistic_fusion(dev_rows, source_names)
    methods["logistic_regression"] = (
        logistic_scores(logistic, dev_rows, source_names),
        logistic_scores(logistic, eval_rows, source_names),
        {
            "type": "logistic_regression",
            "sources": source_names,
            "class_weight": "balanced",
            "coef": logistic.coef_[0].tolist(),
            "intercept": logistic.intercept_.tolist(),
        },
    )

    summaries = {}
    score_rows = {}
    for name, (dev_scores, eval_scores, rule) in methods.items():
        dev_scored = rows_with_score(dev_rows, dev_scores, name)
        dev_summary = summarize_rows(dev_scored)
        eval_scored = rows_with_score(eval_rows, eval_scores, name)
        eval_summary = summarize_rows(eval_scored, dev_threshold=float(dev_summary["threshold"]))
        summaries[name] = {"rule": rule, "dev": dev_summary, "eval": eval_summary}
        score_rows[name] = {"dev": dev_scored, "eval": eval_scored}

    selected_name = min(summaries, key=lambda method: summaries[method]["dev"]["eer"])
    selected = {"method": selected_name, "rule": summaries[selected_name]["rule"]}
    return summaries, score_rows, selected


def write_per_attack_csv(path: Path, method_summaries: dict[str, dict[str, object]], split: str) -> None:
    systems = sorted({
        system_id
        for summary in method_summaries.values()
        for system_id in summary[split]["per_attack_eer"]
    })
    fieldnames = ["system_id", *method_summaries.keys()]
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for system_id in systems:
            row = {"system_id": system_id}
            for method, summary in method_summaries.items():
                value = summary[split]["per_attack_eer"].get(system_id)
                row[method] = "" if value is None else value
            writer.writerow(row)


def save_score_distributions(path: Path, rows_by_method: dict[str, list[dict[str, object]]]) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(len(rows_by_method), 1, figsize=(8, max(4, 3.2 * len(rows_by_method))), sharex=False)
    if len(rows_by_method) == 1:
        axes = [axes]
    for ax, (method, rows) in zip(axes, rows_by_method.items()):
        bonafide = [float(row["score"]) for row in rows if int(row["label"]) == 1]
        spoof = [float(row["score"]) for row in rows if int(row["label"]) == 0]
        ax.hist(spoof, bins=80, alpha=0.65, label="spoof", density=True)
        ax.hist(bonafide, bins=80, alpha=0.65, label="bonafide", density=True)
        ax.set_title(method)
        ax.set_ylabel("Density")
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("Bonafide score")
    axes[0].legend()
    fig.suptitle("Fusion score distributions on eval")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_per_attack_plot(path: Path, method_summaries: dict[str, dict[str, object]], split: str) -> None:
    systems = sorted({
        system_id
        for summary in method_summaries.values()
        for system_id in summary[split]["per_attack_eer"]
    })
    if not systems:
        return
    methods = list(method_summaries)
    x = np.arange(len(systems))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(10, len(systems) * 0.8), 5))
    for idx, method in enumerate(methods):
        values = [method_summaries[method][split]["per_attack_eer"].get(system_id, np.nan) * 100 for system_id in systems]
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("EER (%)")
    ax.set_title(f"Per-attack EER ({split})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gain_loss_plot(
    path: Path,
    method_summaries: dict[str, dict[str, object]],
    selected_method: str,
    baselines: dict[str, dict[str, float]],
) -> None:
    selected = method_summaries[selected_method]["eval"]["per_attack_eer"]
    systems = sorted(selected)
    if not systems:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(systems) * 0.75), 5))
    x = np.arange(len(systems))
    width = 0.8 / max(1, len(baselines))
    for idx, (name, attack_eers) in enumerate(baselines.items()):
        values = [(attack_eers.get(system_id, np.nan) - selected.get(system_id, np.nan)) * 100 for system_id in systems]
        offset = (idx - (len(baselines) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=f"{name} minus fusion")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("EER gain vs baseline (percentage points)")
    ax.set_title(f"Fusion gain/loss by attack: {selected_method}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compact_summary(
    run_id: str,
    selected_method: str,
    method_summaries: dict[str, dict[str, object]],
    source_names: list[str],
) -> dict[str, object]:
    methods = {
        method: {
            "dev_eer": summary["dev"]["eer"],
            "eval_eer": summary["eval"]["eer"],
            "eval_accuracy_at_dev_threshold": summary["eval"]["accuracy_at_dev_threshold"],
            "rule": summary["rule"],
        }
        for method, summary in method_summaries.items()
    }
    selected_eval = method_summaries[selected_method]["eval"]
    selected_attacks = selected_eval["per_attack_eer"]
    wavlm_target = 0.0508
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": current_commit(),
        "sources": source_names,
        "selected_method": selected_method,
        "selected_dev_eer": method_summaries[selected_method]["dev"]["eer"],
        "selected_eval_eer": selected_eval["eer"],
        "beats_wavlm_5_08_eval_eer": bool(selected_eval["eer"] < wavlm_target),
        "result_strength": result_strength(float(selected_eval["eer"])),
        "a17_a18_a19_eval_eer": {attack: selected_attacks.get(attack) for attack in ["A17", "A18", "A19"]},
        "selected_eval_per_attack_eer": selected_attacks,
        "methods": methods,
    }


def result_strength(eval_eer: float) -> str:
    if eval_eer <= 0.0475:
        return "strong"
    if eval_eer < 0.0508:
        return "moderate"
    return "weak"


def model_card(run_id: str, selected_method: str, method_summaries: dict[str, dict[str, object]], source_names: list[str]) -> str:
    selected = method_summaries[selected_method]
    lines = [
        f"# Model Card: {run_id}",
        "",
        "## Summary",
        "",
        "- Task: ASVspoof 2019 LA score-level countermeasure fusion",
        f"- Sources: {', '.join(source_names)}",
        f"- Selected rule: `{selected_method}` by dev EER",
        "- External pretrained components: inherited from WavLM source when included",
        "- Normalization: z-score using dev statistics only",
        "",
        "## Results",
        "",
        "| Method | Dev EER | Eval EER | Eval accuracy at dev threshold |",
        "|---|---:|---:|---:|",
    ]
    for method, summary in method_summaries.items():
        eval_acc = summary["eval"]["accuracy_at_dev_threshold"]
        eval_acc_text = "n/a" if eval_acc is None else f"{eval_acc * 100:.2f}%"
        lines.append(
            f"| {method} | {summary['dev']['eer'] * 100:.2f}% | "
            f"{summary['eval']['eer'] * 100:.2f}% | {eval_acc_text} |"
        )
    lines.extend([
        "",
        "## Selected Eval Per-Attack EER",
        "",
        "| Attack | EER |",
        "|---|---:|",
    ])
    for system_id, eer in sorted(selected["eval"]["per_attack_eer"].items()):
        lines.append(f"| {system_id} | {eer * 100:.2f}% |")
    lines.extend([
        "",
        "## Guardrails",
        "",
        "- Fusion rules and score normalization are fit on dev only.",
        "- Eval is used only after the predeclared fusion rules are frozen.",
        "- Raw score CSVs are kept under the run artifact directory and should stay out of git if large.",
        "- This result should not be described as SOTA.",
        "",
    ])
    return "\n".join(lines)


def parse_alpha_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid:
        raise ValueError("alpha grid cannot be empty")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in grid):
        raise ValueError("alpha values must be in [0, 1]")
    return grid


def run_fusion(
    run_id: str,
    output_root: Path,
    sources: list[ScoreSource],
    alpha_grid: list[float],
) -> dict[str, object]:
    source_names = [source.name for source in sources]
    dev_aligned = align_scores(sources, "dev")
    eval_aligned = align_scores(sources, "eval")
    stats = fit_dev_normalization(dev_aligned, source_names)
    dev_rows = add_normalized_scores(dev_aligned, source_names, stats)
    eval_rows = add_normalized_scores(eval_aligned, source_names, stats)
    method_summaries, score_rows, selected = evaluate_fusion_methods(dev_rows, eval_rows, source_names, alpha_grid)

    run_dir = ensure_dir(output_root / run_id)
    plots_dir = ensure_dir(run_dir / "plots")
    scores_dir = ensure_dir(run_dir / "scores")
    metrics_dir = ensure_dir(output_root / "metrics")

    selected_method = str(selected["method"])
    write_scores_csv(scores_dir / "dev_scores.csv", selected_score_rows(score_rows[selected_method]["dev"], source_names))
    write_scores_csv(scores_dir / "eval_scores.csv", selected_score_rows(score_rows[selected_method]["eval"], source_names))
    write_per_attack_csv(run_dir / "per_attack_dev.csv", method_summaries, "dev")
    write_per_attack_csv(run_dir / "per_attack_eval.csv", method_summaries, "eval")
    save_score_distributions(
        plots_dir / "score_distributions.png",
        {method: rows["eval"] for method, rows in score_rows.items()},
    )
    save_per_attack_plot(plots_dir / "per_attack_eer.png", method_summaries, "eval")
    baseline_attacks = {
        source.name: source_per_attack(eval_rows, source.name)
        for source in sources
    }
    save_gain_loss_plot(plots_dir / "fusion_gain_loss.png", method_summaries, selected_method, baseline_attacks)

    config_payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": current_commit(),
        "sources": [
            {"name": source.name, "dev_path": str(source.dev_path), "eval_path": str(source.eval_path), "invert": source.invert}
            for source in sources
        ],
        "score_normalization": {
            name: {"mean": stats[name].mean, "std": stats[name].std}
            for name in source_names
        },
        "alpha_grid": alpha_grid,
        "selected": selected,
    }
    metrics_payload = {
        **compact_summary(run_id, selected_method, method_summaries, source_names),
        "score_normalization": config_payload["score_normalization"],
        "method_summaries": method_summaries,
    }
    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(metrics_dir / f"{run_id}_summary.json", compact_summary(run_id, selected_method, method_summaries, source_names))
    (run_dir / "model_card.md").write_text(model_card(run_id, selected_method, method_summaries, source_names))
    return metrics_payload


def selected_score_rows(rows: list[dict[str, object]], source_names: list[str]) -> list[dict[str, object]]:
    field_order = ["file_id", "path", "label", "label_name", "system_id", "score", "fusion_method"]
    source_fields = []
    for name in source_names:
        source_fields.extend([f"{name}_score", f"z_{name}"])
    output = []
    for row in rows:
        projected = {field: row.get(field, "") for field in [*field_order, *source_fields]}
        output.append(projected)
    return output


def source_per_attack(rows: list[dict[str, object]], source_name: str) -> dict[str, float]:
    source_rows = []
    for row in rows:
        item = {
            "file_id": row["file_id"],
            "label": row["label"],
            "system_id": row["system_id"],
            "score": row[f"{source_name}_score"],
        }
        source_rows.append(item)
    return per_attack_eer(source_rows)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or "lcnn_wavlm_score_fusion_seed42"
    inverted = set(args.invert_source)
    if bool(args.lcnn_dev) != bool(args.lcnn_eval):
        raise SystemExit("Score fusion aborted: provide both --lcnn-dev and --lcnn-eval, or neither")
    if bool(args.wavlm_dev) != bool(args.wavlm_eval):
        raise SystemExit("Score fusion aborted: provide both --wavlm-dev and --wavlm-eval, or neither")
    lcnn = (
        source_from_paths("lcnn", args.lcnn_dev, args.lcnn_eval, inverted)
        if args.lcnn_dev and args.lcnn_eval
        else source_from_run("lcnn", args.lcnn_run, inverted)
    )
    wavlm = (
        source_from_paths("wavlm", args.wavlm_dev, args.wavlm_eval, inverted)
        if args.wavlm_dev and args.wavlm_eval
        else source_from_run("wavlm", args.wavlm_run, inverted)
    )
    sources = [
        lcnn,
        wavlm,
    ]
    if args.include_lfcc:
        sources.append(lfcc_source(args.lfcc_dev, args.lfcc_eval, inverted))

    try:
        metrics = run_fusion(run_id, Path(args.output_root), sources, parse_alpha_grid(args.alpha_grid))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Score fusion aborted: {exc}") from exc
    selected = metrics["selected_method"]
    print(f"Selected fusion: {selected}")
    print(f"Dev EER : {metrics['selected_dev_eer'] * 100:.2f}%")
    print(f"Eval EER: {metrics['selected_eval_eer'] * 100:.2f}%")
    print(f"Artifacts: {Path(args.output_root) / run_id}")


if __name__ == "__main__":
    main()
