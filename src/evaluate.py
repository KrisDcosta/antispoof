"""
Evaluation metrics for speech anti-spoofing.

EER (Equal Error Rate) is the primary metric used in ASVspoof evaluations.
It is computed from continuous probability scores — NOT from binary predictions.
Using classifier.predict() instead of predict_proba() gives a meaningless EER.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate from ground-truth labels and continuous scores.

    Args:
        y_true:   binary labels, 1 = bonafide, 0 = spoof
        y_scores: bonafide probability from classifier.predict_proba()[:, 1]
                  Higher score = more likely bonafide.

    Returns:
        (eer, threshold): EER as a fraction [0, 1] and the decision threshold.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) != 2:
        raise ValueError("EER requires both bonafide and spoof samples")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Prefer interpolation when the ROC is well behaved; fall back to the
    # closest operating point for degenerate curves from small smoke tests.
    try:
        eer = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0.0, 1.0)
        threshold = float(interp1d(fpr, thresholds)(eer))
    except ValueError:
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        threshold = float(thresholds[idx])
    return float(eer), threshold


def compute_metrics(y_true: np.ndarray,
                    y_scores: np.ndarray,
                    y_pred: np.ndarray = None) -> dict:
    """
    Compute all metrics given true labels, probability scores, and binary preds.

    Args:
        y_true:   ground-truth labels (1 = bonafide, 0 = spoof)
        y_scores: bonafide probabilities from predict_proba()[:, 1]
        y_pred:   binary predictions from predict()

    Returns:
        dict with keys: eer, threshold, accuracy, confusion_matrix,
                        fpr (array), tpr (array), thresholds (array)
    """
    eer, threshold = compute_eer(y_true, y_scores)
    if y_pred is None:
        y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=1)

    return {
        "eer": eer,
        "threshold": threshold,
        "accuracy": acc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
    }


def print_results(feature_name: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(f"\n{'=' * 40}")
    print(f"Feature: {feature_name}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  EER      : {metrics['eer']:.4f}  ({metrics['eer']*100:.2f}%)")
    print(f"  Threshold: {metrics['threshold']:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"             Spoof   Bonafide")
    print(f"  Spoof    {cm[0,0]:7d}  {cm[0,1]:7d}")
    print(f"  Bonafide {cm[1,0]:7d}  {cm[1,1]:7d}")
