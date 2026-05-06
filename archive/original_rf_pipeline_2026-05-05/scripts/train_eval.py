"""
Train and evaluate anti-spoofing classifiers on ASVspoof 2019 LA.

Fix A: loads train/dev from official protocol files — no random splits, LA only.
Fix B: EER computed from predict_proba() continuous scores, not binary predict().

Usage:
    python scripts/train_eval.py --data data/LA --feature mfcc
    python scripts/train_eval.py --data data/LA --feature wcqcc
    python scripts/train_eval.py --data data/LA --ablation        # runs all features
"""

import argparse
import os
import sys

import numpy as np
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_split, split_summary
from src.features import mfcc_vec, cqcc_vec, wcqcc_vec, wcqcc_azcr_vec, SR
from src.evaluate import compute_metrics, print_results


FEATURE_FNS = {
    "mfcc":        mfcc_vec,
    "cqcc":        cqcc_vec,
    "wcqcc":       wcqcc_vec,
    "wcqcc+azcr":  wcqcc_azcr_vec,
}


def extract_features(samples, feature_fn, desc="extracting"):
    X, y, failed = [], [], 0
    for s in tqdm(samples, desc=desc):
        try:
            audio, sr = librosa.load(s.path, sr=SR, mono=True)
            vec = feature_fn(audio, sr)
            X.append(vec)
            y.append(s.label)
        except Exception as e:
            failed += 1
    if failed:
        print(f"  [!] {failed} files skipped due to errors")
    return np.array(X), np.array(y)


def run_feature(data_root, feature_name, output_dir, limit=None):
    feature_fn = FEATURE_FNS[feature_name]

    print(f"\nLoading splits from protocol files...")
    train_samples = load_split(data_root, "train", limit=limit)
    dev_samples   = load_split(data_root, "dev",   limit=limit)

    print(f"  Train: {split_summary(train_samples)}")
    print(f"  Dev  : {split_summary(dev_samples)}")

    X_train, y_train = extract_features(train_samples, feature_fn, f"[{feature_name}] train")
    X_dev,   y_dev   = extract_features(dev_samples,   feature_fn, f"[{feature_name}] dev")

    print(f"  Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # Fix B: use predict_proba for EER, not predict
    y_scores = clf.predict_proba(X_dev)[:, 1]   # P(bonafide)
    y_pred   = clf.predict(X_dev)

    metrics = compute_metrics(y_dev, y_scores, y_pred)
    print_results(feature_name, metrics)

    # Save model for demo.py
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"rf_{feature_name.replace('+','_')}.pkl")
    joblib.dump(clf, model_path)
    print(f"  Model saved: {model_path}")

    _save_roc(feature_name, metrics, output_dir)

    return metrics


def _save_roc(feature_name, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(metrics["fpr"], metrics["tpr"],
            label=f"EER={metrics['eer']*100:.2f}%")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {feature_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"roc_{feature_name.replace('+', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC saved: {path}")


def run_ablation(data_root, output_dir, limit=None):
    results = {}
    for name in FEATURE_FNS:
        results[name] = run_feature(data_root, name, output_dir, limit)
    _save_ablation_plot(results, output_dir)
    return results


def _save_ablation_plot(results, output_dir):
    names = list(results.keys())
    eers  = [results[n]["eer"] * 100 for n in names]
    accs  = [results[n]["accuracy"] * 100 for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x - 0.2, eers, 0.35, label="EER (%)", color="salmon")
    ax.bar(x + 0.2, [100 - a for a in accs], 0.35,
           label="Error rate (%)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("% (lower is better)")
    ax.set_title("Ablation: EER and Error Rate by Feature")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "ablation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nAblation plot saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/LA",
                        help="Path to data/LA/ directory")
    parser.add_argument("--feature", choices=list(FEATURE_FNS),
                        default="mfcc",
                        help="Feature to use (ignored if --ablation)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run all features and produce comparison plot")
    parser.add_argument("--output", default="results",
                        help="Directory for plots and logs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per split (for quick testing)")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(args.data, args.output, args.limit)
    else:
        run_feature(args.data, args.feature, args.output, args.limit)


if __name__ == "__main__":
    main()
