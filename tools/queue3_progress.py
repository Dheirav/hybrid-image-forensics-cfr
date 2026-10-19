#!/usr/bin/env python3
"""Stage 3 queue progress (15 runs in runs_extra/): finished out of total, elapsed, ETA from the measured mean time per
finished run. If none has finished there is no rate yet, so no ETA is shown. usage: tools/queue3_progress.py [--watch]"""
import json, os, sys, time

NAMES = [n for s in (0, 1, 2) for n in (f"joint_concat_s{s}", f"joint_gate_s{s}", f"cfr_scratch_noalign_s{s}", f"ml_lam0.25_s{s}", f"ml_lam4_s{s}")]

def show():
    d = "runs_extra"
    starts = [json.load(open(f"{d}/{n}/progress.json"))["t0"] for n in NAMES if os.path.exists(f"{d}/{n}/progress.json")]
    if not starts:
        print(f"0/{len(NAMES)} runs started"); return
    done = [n for n in NAMES if os.path.exists(f"{d}/{n}/results.json")]
    cur = ""
    for n in NAMES:
        p = f"{d}/{n}/progress.json"
        if n not in done and os.path.exists(p):
            s = json.load(open(p)); cur = f" | now {n}: epoch {s['epoch']+1}/{s['epochs']}"; break
    if done:
        per = sum(json.load(open(f"{d}/{n}/results.json"))["train_seconds"] for n in done) / len(done)
        eta = f"ETA ~{int((len(NAMES)-len(done))*per/60)} min (mean {per/60:.1f} min per finished run)"
    else:
        eta = "no ETA yet: no run has finished"
    print(f"{len(done)}/{len(NAMES)} runs finished, elapsed {int((time.time()-min(starts))/60)} min, {eta}{cur}")

if __name__ == "__main__":
    while True:
        show()
        if "--watch" not in sys.argv: break
        time.sleep(30)
