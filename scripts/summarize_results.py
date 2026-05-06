"""Create final result tables and comparison plots from saved metrics JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reporting import (
    ensure_dir,
    save_eer_comparison_plot,
    save_eval_vs_baseline_plot,
    write_json,
)


def load_project_results(metrics_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        data = json.loads(path.read_text())
        metrics = data["metrics"]
        feature = data["feature"]
        classifier = data["classifier"]
        split = data["split"]
        rows.append({
            "method": f"{feature.upper()} {classifier.upper()}",
            "feature": feature,
            "classifier": classifier,
            "split": split,
            "eer_percent": float(metrics["eer"]) * 100.0,
            "threshold": float(metrics["threshold"]),
            "accuracy_percent": float(metrics["accuracy"]) * 100.0,
            "metrics_path": str(path),
            "run_id": data["run_id"],
        })
    return rows


def load_standard_baselines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return list(data.get("asvspoof2019_la", []))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_project_table(rows: list[dict[str, object]]) -> str:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    splits = [split for split in ["dev", "eval"] if any(row["split"] == split for row in rows)]
    lines = ["| Method | " + " | ".join(f"{split.title()} EER" for split in splits) + " |",
             "|---|" + "|".join("---:" for _ in splits) + "|"]
    for method in methods:
        values = []
        for split in splits:
            match = next((row for row in rows if row["method"] == method and row["split"] == split), None)
            values.append("pending" if match is None else f"{match['eer_percent']:.2f}%")
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def best_eval_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    eval_rows = [row for row in rows if row["split"] == "eval"]
    if not eval_rows:
        return None
    return min(eval_rows, key=lambda row: row["eer_percent"])


def matching_standard_eval_gap(
    project_row: dict[str, object] | None,
    baselines: list[dict[str, object]],
) -> str | None:
    if not project_row:
        return None
    feature = str(project_row["feature"]).upper()
    for row in baselines:
        name = str(row.get("name", "")).upper()
        if feature in name and row.get("eval_eer_percent") is not None:
            gap = float(project_row["eer_percent"]) - float(row["eval_eer_percent"])
            direction = "above" if gap >= 0 else "below"
            return (
                f"- {project_row['method']} is {abs(gap):.2f} percentage points "
                f"{direction} the matching published eval reference."
            )
    return None


def ablation_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        feature = str(row["feature"])
        split = str(row["split"])
        summary.setdefault(feature, {})[split] = {
            "accuracy_at_eer_threshold": float(row["accuracy_percent"]) / 100.0,
            "eer": float(row["eer_percent"]) / 100.0,
            "metrics_path": str(row["metrics_path"]),
            "threshold": float(row["threshold"]),
        }
    return summary


def markdown_baseline_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    lines = [
        "| Reference System | Dev EER | Eval EER | Source |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        source = row["source"]
        if row.get("source_url"):
            source = f"[{source}]({row['source_url']})"
        lines.append(
            f"| {row['name']} | {row['dev_eer_percent']:.2f}% | "
            f"{row['eval_eer_percent']:.2f}% | {source} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/baseline", help="Baseline result directory")
    parser.add_argument("--output", default=None, help="Summary output directory")
    parser.add_argument("--baselines", default="references/standard_baselines.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results)
    output_dir = ensure_dir(args.output or results_dir / "summary")
    metrics_dir = results_dir / "metrics"
    plot_dir = ensure_dir(output_dir / "plots")

    project_rows = load_project_results(metrics_dir)
    baseline_rows = load_standard_baselines(Path(args.baselines))
    if not project_rows:
        raise FileNotFoundError(f"no metrics JSON files found in {metrics_dir}")

    write_csv(output_dir / "project_results.csv", project_rows)
    write_json(output_dir / "project_results.json", {"results": project_rows})
    write_json(output_dir / "standard_baselines.json", {"baselines": baseline_rows})
    write_json(metrics_dir / "ablation_summary.json", ablation_summary(project_rows))
    save_eer_comparison_plot(
        plot_dir / "project_eer_by_split.png",
        project_rows,
        "Project GMM-LLR EER by split",
    )
    save_eval_vs_baseline_plot(
        plot_dir / "eval_eer_vs_standard_baselines.png",
        project_rows,
        baseline_rows,
        "Eval EER vs published ASVspoof 2019 references",
    )

    best_eval = best_eval_row(project_rows)
    quality_note = (
        f"- Best project eval result: {best_eval['method']} at {best_eval['eer_percent']:.2f}% EER."
        if best_eval
        else "- Eval results are not available yet."
    )
    gap_note = matching_standard_eval_gap(best_eval, baseline_rows)

    report = [
        "# ASVspoof 2019 LA Results",
        "",
        "## Project Results",
        "",
        markdown_project_table(project_rows),
        "",
        "## Published Reference Systems",
        "",
        markdown_baseline_table(baseline_rows) or "No published reference file found.",
        "",
        "## Notes",
        "",
        quality_note,
        *( [gap_note] if gap_note else [] ),
        "- LFCC is the current strongest project feature; eval remains harder than dev because eval attacks are unseen.",
        "- Project CQCC uses the repository's simplified librosa CQT + DCT extraction.",
        "- Published references use the official ASVspoof 2019 recipes.",
        "- Eval is the primary generalization split because attacks A07-A19 are unseen during training.",
        "",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(report))

    print(f"Summary saved: {output_dir / 'RESULTS.md'}")
    print(f"Project CSV  : {output_dir / 'project_results.csv'}")
    print(f"Plots        : {plot_dir}")


if __name__ == "__main__":
    main()
