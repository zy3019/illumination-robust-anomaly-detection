"""Exact binary AUROC via negative-score order statistics, preserving tie handling."""
import numpy as np


def exact_pixel_auroc(labels, scores):
    labels=np.asarray(labels).reshape(-1)
    scores=np.asarray(scores).reshape(-1)
    if len(labels)!=len(scores) or not np.isin(labels,[0,1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid binary labels or scores')
    positive=scores[labels==1]
    negative=np.sort(scores[labels==0])
    if not len(positive) or not len(negative):
        return None
    left=np.searchsorted(negative,positive,side='left')
    right=np.searchsorted(negative,positive,side='right')
    return float((left.sum(dtype=np.float64)+right.sum(dtype=np.float64))/(2*len(positive)*len(negative)))
