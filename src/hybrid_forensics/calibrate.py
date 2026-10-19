"""Fit a single temperature on VALIDATION probabilities, then report ECE/NLL/Brier on test before and after.
usage: python -m hybrid_forensics.calibrate runs/<run>"""
import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import classification_metrics


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_T(p, T):
    return 1 / (1 + np.exp(-_logit(p) / T))


def fit_temperature(p_val, y_val):
    def nll(T):
        q = np.clip(apply_T(p_val, T), 1e-7, 1 - 1e-7)
        return -np.log(np.where(y_val == 1, q, 1 - q)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def selective(p, y, coverages=(1.0, 0.8, 0.6)):
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    out = {}
    for c in coverages:
        keep = order[: int(round(c * len(p)))]
        out[str(c)] = float(((p[keep] >= 0.5).astype(int) == y[keep]).mean())
    return out


def main(run):
    z = np.load(f"{run}/probs.npz")
    T = fit_temperature(z["val_p_fused"], z["val_y"])
    res = {"temperature": T}
    for split in ("val", "test", "ood_removal"):
        p, y = z[f"{split}_p_fused"], z[f"{split}_y"]
        res[split] = {"before": classification_metrics(p, y), "after": classification_metrics(apply_T(p, T), y),
                      "selective_accuracy_at_coverage_after": selective(apply_T(p, T), y)}
    json.dump(res, open(f"{run}/calibration.json", "w"), indent=1)
    print(json.dumps({k: (v if k == "temperature" else {kk: {m: round(x, 4) for m, x in vv.items() if m in ("ece", "nll", "brier")}
                                                           for kk, vv in v.items() if kk in ("before", "after")}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
