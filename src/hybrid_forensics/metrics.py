import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(prob1, y, thr: float = 0.5):
    """prob1 = P(manipulated). Threshold is fixed at 0.5 unless a validation-chosen value is passed."""
    prob1, y = np.asarray(prob1, float), np.asarray(y)
    pred = (prob1 >= thr).astype(int)
    authentic = y == 0
    return {
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob1)) if len(set(y)) > 1 else float("nan"),
        "fpr_authentic": float(pred[authentic].mean()) if authentic.any() else float("nan"),
        "ece": ece(prob1, y), "brier": float(((prob1 - y) ** 2).mean()), "nll": nll(prob1, y),
    }


def ece(prob1, y, bins: int = 15):
    conf = np.maximum(prob1, 1 - prob1)
    correct = ((prob1 >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def nll(prob1, y, eps: float = 1e-7):
    p = np.clip(np.where(y == 1, prob1, 1 - prob1), eps, 1)
    return float(-np.log(p).mean())


def dice_iou(pred_mask, true_mask, eps: float = 1e-6):
    """Per-image Dice and IoU; callers pass only images that really contain a manipulated region."""
    p, t = pred_mask.astype(bool), true_mask.astype(bool)
    inter = (p & t).sum((1, 2))
    dice = (2 * inter + eps) / (p.sum((1, 2)) + t.sum((1, 2)) + eps)
    iou = (inter + eps) / ((p | t).sum((1, 2)) + eps)
    return dice, iou
