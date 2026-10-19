"""Post-hoc subset metrics from saved probabilities, ordered as the manifest rows of each split.

Primary metric (declared in reports/02): test AUROC over authentic + splice + copy_move only.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEEN = ("authentic", "splice", "copy_move")


def split_frame(split, manifest="data/forge/manifest.csv"):
    return pd.read_csv(manifest).query("split == @split").reset_index(drop=True)


def subset(run, split, tools, key="fused"):
    z = np.load(os.path.join(run, "probs.npz"))
    df = split_frame(split)
    k = f"{split}_p_{key}"
    if k not in z:
        k = [n for n in z.files if n.startswith(f"{split}_p_")][0]
    keep = df.manipulation.isin(tools).values
    assert len(df) == len(z[k]) and (z[f"{split}_y"] == df.label.values).all(), "probs.npz order does not match manifest"
    return z[k][keep], df.label.values[keep], df[keep]


def primary(run, key="fused"):
    p, y, _ = subset(run, "test", SEEN, key)
    return float(roc_auc_score(y, p))


def by_tool(run, key="fused"):
    out = {}
    for tool in ("splice", "copy_move", "removal"):
        p, y, _ = subset(run, "test", ("authentic", tool), key)
        out[tool] = float(roc_auc_score(y, p))
    return out


if __name__ == "__main__":
    for run in sorted(glob.glob("runs/*/results.json")):
        d = os.path.dirname(run)
        keys = [k for k in np.load(os.path.join(d, "probs.npz")).files if k.startswith("test_p_")]
        for k in keys:
            kk = k.replace("test_p_", "")
            print(f"{os.path.basename(d):<16} {kk:<6} primary(test seen tools) {primary(d, kk):.3f}  by tool {({a: round(b, 3) for a, b in by_tool(d, kk).items()})}")
