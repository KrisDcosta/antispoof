"""
Fix D: Per-attack EER breakdown.

Loads dev set via protocol files, runs a trained classifier,
groups predictions by attack system ID (A01-A19), computes EER per system.
Shows which TTS/VC systems fool the detector and which are caught.

Usage:
    python scripts/attack_breakdown.py --data data/LA --feature wcqcc
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_split, split_summary
from src.features import mfcc_vec, cqcc_vec, wcqcc_vec, wcqcc_azcr_vec, SR
from src.evaluate import compute_eer

FEATURE_FNS = {
    "mfcc":       mfcc_vec,
    "cqcc":       cqcc_vec,
    "wcqcc":      wcqcc_vec,
    "wcqcc+azcr": wcqcc_azcr_vec,
}


def extract(samples, feature_fn, desc=""):
    X, y, sys_ids, failed = [], [], [], 0
    for s in tqdm(samples, desc=desc):
        try:
            audio, sr = librosa.load(s.path, sr=SR, mono=True)
            X.append(feature_fn(audio, sr))
            y.append(s.label)
            sys_ids.append(s.system_id)
        except Exception:
            failed += 1
    if failed:
        print(f"  [!] {failed} files skipped")
    return np.array(X), np.array(y), sys_ids


def per_attack_eer(y_true, y_scores, system_ids):
    """
    Compute EER separately for each attack system vs all bonafide samples.
    Returns dict: system_id -> EER float
    """
    bonafide_mask = np.array(y_true) == 1
    results = {}

    unique_systems = sorted(set(s for s in system_ids if s != "-"))
    for sys_id in unique_systems:
        attack_mask = np.array(system_ids) == sys_id
        mask = bonafide_mask | attack_mask

        yt = np.array(y_true)[mask]
        ys = np.array(y_scores)[mask]

        if len(np.unique(yt)) < 2:
            continue
        try:
            eer, _ = compute_eer(yt, ys)
            results[sys_id] = eer
        except Exception:
            pass

    return results


def plot_per_attack(results: dict, feature_name: str, output_dir: str):
    systems = sorted(results.keys())
    eers = [results[s] * 100 for s in systems]

    colors = ["#d62728" if e > 20 else "#2ca02c" if e < 5 else "#ff7f0e"
              for e in eers]

    fig, ax = plt.subplots(figsize=(max(8, len(systems) * 0.7), 5))
    bars = ax.bar(systems, eers, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=5,  color="green",  linestyle="--", linewidth=0.8, label="EER=5% (good)")
    ax.axhline(y=20, color="red",    linestyle="--", linewidth=0.8, label="EER=20% (poor)")
    ax.set_xlabel("Attack System ID")
    ax.set_ylabel("EER (%)")
    ax.set_title(f"Per-Attack EER — {feature_name}\n"
                 "Green < 5% (caught), Orange 5–20%, Red > 20% (evades detector)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"attack_breakdown_{feature_name.replace('+','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/LA")
    parser.add_argument("--feature", choices=list(FEATURE_FNS), default="wcqcc")
    parser.add_argument("--output",  default="results")
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    feature_fn = FEATURE_FNS[args.feature]

    train_samples = load_split(args.data, "train", args.limit)
    dev_samples   = load_split(args.data, "dev",   args.limit)

    print(f"Train: {split_summary(train_samples)}")
    print(f"Dev  : {split_summary(dev_samples)}")

    X_train, y_train, _          = extract(train_samples, feature_fn, "train")
    X_dev,   y_dev,   sys_ids    = extract(dev_samples,   feature_fn, "dev")

    print("Training classifier...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    y_scores = clf.predict_proba(X_dev)[:, 1]

    print("\nPer-attack EER:")
    attack_eers = per_attack_eer(y_dev, y_scores, sys_ids)
    for sys_id, eer in sorted(attack_eers.items(), key=lambda x: x[1]):
        bar = "█" * int(eer * 2)
        print(f"  {sys_id}  {eer*100:5.1f}%  {bar}")

    plot_per_attack(attack_eers, args.feature, args.output)


if __name__ == "__main__":
    main()
