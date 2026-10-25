"""Calibration and evaluation metrics: ECE, AUROC, Brier Score, and Reliability Diagrams."""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def compute_ece(
    confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) with equal-width bins in [0, 1].

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        return 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            weight = bin_count / n
            ece += weight * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = (low + high) / 2.0

        bin_details.append({
            "bin": i + 1,
            "low": float(low),
            "high": float(high),
            "count": bin_count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
        })

    return float(ece), bin_details


def compute_all_metrics(confidences: List[float], labels: List[bool], n_bins: int = 10) -> Dict[str, Any]:
    """Compute Accuracy, ECE, AUROC, and Brier score."""
    conf_arr = np.asarray(confidences, dtype=float)
    label_arr = np.asarray(labels, dtype=int)
    n = len(label_arr)

    if n == 0:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "auroc": 0.5, "brier": 0.0}

    accuracy = float(np.mean(label_arr))
    ece, bin_details = compute_ece(conf_arr, label_arr, n_bins=n_bins)
    brier = float(brier_score_loss(label_arr, conf_arr))

    # AUROC requires both classes (0 and 1) to be present
    unique_labels = np.unique(label_arr)
    if len(unique_labels) < 2:
        auroc = 0.5
    else:
        try:
            auroc = float(roc_auc_score(label_arr, conf_arr))
        except Exception:
            auroc = 0.5

    return {
        "n": n,
        "accuracy": accuracy,
        "ece": ece,
        "auroc": auroc,
        "brier": brier,
        "bin_details": bin_details,
    }
