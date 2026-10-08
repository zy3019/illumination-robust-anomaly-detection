"""Exact binary AUROC with tied scores, plus fixed-threshold confusion counts."""
import numpy as np


def auroc(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid binary labels or scores")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    average_ranks = (starts + 1 + ends) / 2
    positive_counts = np.add.reduceat(labels[order].astype(np.int64), starts)
    rank_sum = float(average_ranks @ positive_counts)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    auc = auroc(labels, scores)
    predicted = scores > threshold
    normal = labels == 0
    anomalous = labels == 1
    fp = int((predicted & normal).sum())
    tp = int((predicted & anomalous).sum())
    n = int(normal.sum())
    p = int(anomalous.sum())
    return {"image_auroc": auc, "false_positives": fp, "normal_images": n,
            "true_positives": tp, "anomalous_images": p,
            "fpr": fp / n if n else None, "recall": tp / p if p else None}
