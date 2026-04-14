"""
Evaluation Metrics for Next-Action Prediction
===============================================
Multi-label metrics: Top-k Accuracy, F1, Mean Average Precision.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    """Compute all evaluation metrics for multi-label prediction.

    Parameters
    ----------
    preds : np.ndarray, shape (N, C)
        Predicted probabilities (after sigmoid) for each class.
    targets : np.ndarray, shape (N, C)
        Ground truth multi-hot binary vectors.
    threshold : float
        Threshold for converting probabilities to binary predictions.

    Returns
    -------
    dict
        Dictionary of metric name to value.
    """
    N, C = preds.shape

    # Binary predictions
    pred_binary = (preds >= threshold).astype(np.float32)

    # Top-1 Accuracy: is the highest-scored predicted class actually active?
    top1_correct = 0
    for i in range(N):
        top1_idx = np.argmax(preds[i])
        if targets[i, top1_idx] > 0:
            top1_correct += 1
    top1_acc = top1_correct / N

    # Top-5 Accuracy: is at least one of top-5 predicted classes active?
    top5_correct = 0
    for i in range(N):
        top5_idx = np.argsort(preds[i])[-5:]
        if np.any(targets[i, top5_idx] > 0):
            top5_correct += 1
    top5_acc = top5_correct / N

    # Exact Match: predicted set exactly equals target set
    exact_match = np.mean(np.all(pred_binary == targets, axis=1))

    # Sample-level F1 (average over samples)
    f1_samples = f1_score(targets, pred_binary, average="samples", zero_division=0)

    # Macro F1 (average over classes)
    f1_macro = f1_score(targets, pred_binary, average="macro", zero_division=0)

    # Micro F1
    f1_micro = f1_score(targets, pred_binary, average="micro", zero_division=0)

    # Precision and Recall (micro)
    precision = precision_score(targets, pred_binary, average="micro", zero_division=0)
    recall = recall_score(targets, pred_binary, average="micro", zero_division=0)

    # Mean Average Precision (mAP)
    # Compute AP per class, then average
    aps = []
    for c in range(C):
        if targets[:, c].sum() > 0:
            ap = average_precision_score(targets[:, c], preds[:, c])
            aps.append(ap)
    mAP = np.mean(aps) if aps else 0.0

    return {
        "top1_acc": round(top1_acc, 4),
        "top5_acc": round(top5_acc, 4),
        "exact_match": round(exact_match, 4),
        "f1_samples": round(f1_samples, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP": round(mAP, 4),
    }


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Top-1 Accuracy:  {metrics['top1_acc']:.4f}")
    print(f"  Top-5 Accuracy:  {metrics['top5_acc']:.4f}")
    print(f"  Exact Match:     {metrics['exact_match']:.4f}")
    print(f"  F1 (samples):    {metrics['f1_samples']:.4f}")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):      {metrics['f1_micro']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  mAP:             {metrics['mAP']:.4f}")
    print(f"{'=' * 50}")
