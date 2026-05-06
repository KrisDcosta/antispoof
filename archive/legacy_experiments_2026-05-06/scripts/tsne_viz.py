"""
Fix F: t-SNE visualization of feature space.

Two plots:
  1. Bonafide vs Spoof (binary) — shows overall separability
  2. Per attack system — shows which TTS/VC systems cluster away from bonafide

Usage:
    python scripts/tsne_viz.py --data data/LA --feature wcqcc --split dev
"""

import argparse
import os
import sys

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_split
from src.features import mfcc_vec, cqcc_vec, wcqcc_vec, SR

FEATURE_FNS = {
    "mfcc":  mfcc_vec,
    "cqcc":  cqcc_vec,
    "wcqcc": wcqcc_vec,
}

# Colorblind-friendly palette for up to 20 attack systems
ATTACK_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9a6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9",
]


def extract(samples, feature_fn, limit=None):
    X, labels, sys_ids = [], [], []
    samples = samples[:limit] if limit else samples
    for s in tqdm(samples, desc="extracting"):
        try:
            audio, sr = librosa.load(s.path, sr=SR, mono=True)
            X.append(feature_fn(audio, sr))
            labels.append(s.label)
            sys_ids.append(s.system_id)
        except Exception:
            pass
    return np.array(X), np.array(labels), sys_ids


def run_tsne(X):
    print("Running t-SNE (this takes a minute)...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=42, n_jobs=-1)
    return tsne.fit_transform(X)


def plot_binary(coords, labels, feature_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(labels == 1, "#2196F3", "#F44336")
    ax.scatter(coords[:, 0], coords[:, 1], c=colors,
               s=6, alpha=0.6, linewidths=0)
    legend = [
        mpatches.Patch(color="#2196F3", label="Bonafide"),
        mpatches.Patch(color="#F44336", label="Spoof"),
    ]
    ax.legend(handles=legend, fontsize=10)
    ax.set_title(f"t-SNE Feature Space — {feature_name}\nBonafide vs Spoof")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"tsne_binary_{feature_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_attack(coords, labels, sys_ids, feature_name, output_dir):
    unique_systems = sorted(set(s for s in sys_ids if s != "-"))
    sys_color_map = {s: ATTACK_COLORS[i % len(ATTACK_COLORS)]
                     for i, s in enumerate(unique_systems)}

    fig, ax = plt.subplots(figsize=(10, 7))

    # Bonafide first (background layer)
    bonafide_mask = np.array(labels) == 1
    ax.scatter(coords[bonafide_mask, 0], coords[bonafide_mask, 1],
               c="#2196F3", s=5, alpha=0.4, linewidths=0, label="Bonafide")

    # Each attack system
    for sys_id in unique_systems:
        mask = np.array(sys_ids) == sys_id
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=sys_color_map[sys_id], s=6, alpha=0.7,
                   linewidths=0, label=sys_id)

    ax.legend(fontsize=7, markerscale=2, ncol=2,
              loc="upper right", framealpha=0.8)
    ax.set_title(f"t-SNE Feature Space — {feature_name}\nPer Attack System")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(output_dir, f"tsne_attacks_{feature_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/LA")
    parser.add_argument("--feature", choices=list(FEATURE_FNS), default="wcqcc")
    parser.add_argument("--split",   choices=["train", "dev"], default="dev")
    parser.add_argument("--output",  default="results")
    parser.add_argument("--limit",   type=int, default=2000,
                        help="Max samples for t-SNE (default 2000 for speed)")
    args = parser.parse_args()

    feature_fn = FEATURE_FNS[args.feature]
    samples = load_split(args.data, args.split)

    X, labels, sys_ids = extract(samples, feature_fn, limit=args.limit)
    coords = run_tsne(X)

    plot_binary(coords, labels, args.feature, args.output)
    plot_per_attack(coords, labels, sys_ids, args.feature, args.output)


if __name__ == "__main__":
    main()
