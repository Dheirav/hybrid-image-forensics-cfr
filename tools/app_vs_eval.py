#!/usr/bin/env python3
"""Does the application path reproduce the evaluation probabilities of the same checkpoint (brief phase 4 and 8)?
Runs infer.Predictor (no calibration) on every seen-tool test image and compares with the saved probs.npz.
usage: tools/app_vs_eval.py runs/<run>  -> reports/evidence/app_vs_eval_probs.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from hybrid_forensics.infer import Predictor

run = sys.argv[1]
d = np.load(f"{run}/probs.npz")
df = pd.read_csv("data/forge/manifest.csv").query("split=='test'").reset_index(drop=True)
assert (df.label.values == d["test_y"]).all(), "row order differs from the saved probabilities"
P = Predictor.from_checkpoint(f"{run}/best.pt")
got = {"fused": [], "A": [], "B": []}
for p in df.path:
    r = P.predict(p)
    got["fused"].append(r["p_manipulated"]); got["A"].append(r["branch_probs"]["visual"]); got["B"].append(r["branch_probs"]["forensic"])
out = {"run": run, "n": len(df)}
for k in got:
    a, b = np.array(got[k]), d[f"test_p_{k}"]
    out[k] = {"max_abs_diff": float(np.abs(a - b).max()), "mean_abs_diff": float(np.abs(a - b).mean()), "label_flips_at_0.5": int(((a >= .5) != (b >= .5)).sum())}
json.dump(out, open("reports/evidence/app_vs_eval_probs.json", "w"), indent=1)
print(json.dumps(out, indent=1))
