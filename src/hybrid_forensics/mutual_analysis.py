"""Descriptive statistics for the mutual-learning and transition reports (reports 07, 08).

Everything is computed from saved test probabilities (seen-tool subset) and from results.json.
Question answered here: does B improve because it learned something, or because it copies A, and does
alignment cost A anything? Descriptive only; the pre-declared decision rule lives in compare.py.
usage: python -m hybrid_forensics.mutual_analysis > reports/evidence/mutual_analysis.md
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from .analysis import SEEN, subset

VARIANTS = ["single_a", "single_b", "indep", "ml", "ml_hem", "cfr_adaptive", "cfr_sudden", "cfr_fixed", "cfr_scratch"]


def _stats(run):
    pa, y, _ = subset(run, "test", SEEN, "A")
    pb, _, _ = subset(run, "test", SEEN, "B")
    wa, wb = (pa >= .5) != y.astype(bool), (pb >= .5) != y.astype(bool)
    la, lb = np.log(np.clip(pa, 1e-6, 1 - 1e-6) / (1 - np.clip(pa, 1e-6, 1 - 1e-6))), np.log(np.clip(pb, 1e-6, 1 - 1e-6) / (1 - np.clip(pb, 1e-6, 1 - 1e-6)))
    both = (wa & wb).sum()
    return {"aucA": roc_auc_score(y, pa), "aucB": roc_auc_score(y, pb),
            "errA": wa.mean(), "errB": wb.mean(), "both_wrong": both / len(y),
            # of B's errors, the share that are also A's errors: high means B fails where A fails
            "B_err_shared_with_A": both / max(wb.sum(), 1),
            "agree": ((pa >= .5) == (pb >= .5)).mean(), "logit_corr": float(np.corrcoef(la, lb)[0, 1]),
            "mean_abs_prob_gap": float(np.abs(pa - pb).mean())}


def main():
    print("## Branch behaviour on test (authentic + splice + copy_move), mean over seeds\n")
    cols = ["aucA", "aucB", "errA", "errB", "both_wrong", "B_err_shared_with_A", "agree", "logit_corr", "mean_abs_prob_gap"]
    print("| variant | n seeds | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
    for v in VARIANTS:
        runs = [f"runs/{v}_s{s}" for s in range(3) if os.path.exists(f"runs/{v}_s{s}/probs.npz")]
        ss = [_stats(r) for r in runs]
        print(f"| {v} | {len(ss)} | " + " | ".join(f"{np.mean([s[c] for s in ss]):.3f}" for c in cols) + " |")
    print("\n## Transition: first epoch at which the CFR weight applied, and what blocked eligibility before it\n")
    for v in ("cfr_adaptive", "cfr_sudden", "cfr_fixed"):
        for s in range(3):
            p = f"runs/{v}_s{s}/results.json"
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            first = {}
            for t in r["transition"]:
                for k, ok in t["checks"].items():
                    if not ok and k not in first.setdefault(t["epoch"], []):
                        first[t["epoch"]].append(k)
            blocked = {e: ks for e, ks in first.items() if ks}
            print(f"- {v} s{s}: trigger epoch {r['trigger_epoch']}, failed checks per epoch before/at trigger: "
                  + "; ".join(f"e{e}: {','.join(ks)}" for e, ks in sorted(blocked.items()) if r['trigger_epoch'] is None or e <= r['trigger_epoch']))
    print("\n## Training-time pair statistics at the last epoch (validation)\n")
    for v in ("indep", "ml", "ml_hem", "cfr_adaptive", "cfr_sudden", "cfr_fixed", "cfr_scratch"):
        rows = []
        for s in range(3):
            p = f"runs/{v}_s{s}/results.json"
            if os.path.exists(p):
                h = json.load(open(p))["history"][-1]["val"]["pair"]
                rows.append((h["disagreement"], h["prob_corr"]))
        if rows:
            print(f"- {v}: disagreement {np.mean([r[0] for r in rows]):.3f}, prob corr {np.mean([r[1] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
