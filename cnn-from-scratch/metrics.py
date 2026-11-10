import numpy as np
from scipy.stats import rankdata
from data import xray_batches

def score_dataset(model, X, boxes, bs=64):
    # P(tumor) and label for every image, on clean (non-augmented) data, in eval mode
    scores, labels = [], []
    for xb, yb in xray_batches(X, boxes, bs, train=False):
        scores.append(model.predict_proba(xb)[:, 1])
        labels.append(yb)
    return np.concatenate(scores), np.concatenate(labels)

def auc(scores, labels):
    # Probability a random tumor image scores higher than a random clean one (threshold-free)
    n_pos = int(np.sum(labels == 1))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)                                    # average ranks handle ties
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def binary_metrics(scores, labels, threshold=0.5):
    pred, pos = scores >= threshold, labels == 1
    tp, fn = int(np.sum(pred & pos)), int(np.sum(~pred & pos))
    fp, tn = int(np.sum(pred & ~pos)), int(np.sum(~pred & ~pos))
    sens, spec = tp / max(1, tp + fn), tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, sensitivity=sens, specificity=spec, precision=prec,
                f1=2 * prec * sens / max(1e-12, prec + sens), balanced_acc=(sens + spec) / 2)

def pick_threshold(scores, labels, min_sensitivity=0.90):
    # Highest threshold that still catches at least min_sensitivity of the tumors (maximizes specificity)
    pos = np.sort(scores[labels == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    return float(pos[int(np.ceil(min_sensitivity * len(pos))) - 1])

def bootstrap_ci(scores, labels, stat, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(labels), len(labels))
        if labels[i].min() != labels[i].max():              # resample must contain both classes
            vals.append(stat(scores[i], labels[i]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (float("nan"), float("nan"))