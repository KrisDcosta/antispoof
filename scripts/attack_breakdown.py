"""Compute per-attack EER from a saved score CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reporting import per_attack_eer, save_per_attack_plot, write_json


def read_scores(path: str | os.PathLike) -> list[dict[str, object]]:
    with Path(path).open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"score file is empty: {path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="CSV produced by scripts/train_eval.py")
    parser.add_argument("--output", default="results/baseline/plots")
    parser.add_argument("--name", default=None, help="Optional name for output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_scores(args.scores)
    attack_eers = per_attack_eer(rows)
    if not attack_eers:
        raise ValueError("no attack systems found in score file")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = args.name or Path(args.scores).stem
    write_json(output / f"{stem}_per_attack_eer.json", attack_eers)
    save_per_attack_plot(output / f"{stem}_per_attack_eer.png", attack_eers, f"Per-attack EER: {stem}")

    print("Per-attack EER")
    for system_id, eer in sorted(attack_eers.items(), key=lambda item: item[1]):
        print(f"  {system_id}: {eer * 100:.2f}%")
    print(f"Saved: {output / f'{stem}_per_attack_eer.json'}")
    print(f"Saved: {output / f'{stem}_per_attack_eer.png'}")


if __name__ == "__main__":
    main()
