#!/usr/bin/env python3
"""Whole-queue progress for tools/queue_stage2.sh: finished runs out of 24, elapsed, and an ETA from the
measured mean time per finished run (plus the live step progress of the current run). usage: tools/queue_progress.py [--watch]"""
import glob, json, os, sys, time

NAMES = [f"{v}_s{s}" for s in (0, 1, 2) for v in ("ml", "cfr_adaptive", "ml_hem")] + [f"indep_s{s}" for s in (1, 2)] + \
        [f"{v}_s{s}" for s in (0, 1, 2) for v in ("cfr_sudden", "cfr_fixed", "cfr_scratch")] + \
        [f"{v}_s{s}" for s in (1, 2) for v in ("single_a", "single_b")]

def show():
    # queue start = earliest run start among the queue's runs (log ctime moves on every write)
    start = min(json.load(open(f"runs/{n}/progress.json"))["t0"] for n in NAMES if os.path.exists(f"runs/{n}/progress.json"))
    done = [n for n in NAMES if os.path.exists(f"runs/{n}/results.json")]
    el = time.time() - start
    cur = ""
    for n in NAMES:
        p = f"runs/{n}/progress.json"
        if n not in done and os.path.exists(p):
            s = json.load(open(p)); cur = f" | now {n}: epoch {s['epoch']+1}/{s['epochs']}"; break
    if done:
        per = sum(json.load(open(f"runs/{n}/results.json"))["train_seconds"] for n in done) / len(done)
        eta = f"ETA ~{int((len(NAMES)-len(done))*per/60)} min (mean {per/60:.1f} min per finished run)"
    else:
        eta = "no ETA yet: no run has finished"
    print(f"{len(done)}/{len(NAMES)} runs finished, elapsed {int(el/60)} min, {eta}{cur}")

if __name__ == "__main__":
    while True:
        show()
        if "--watch" not in sys.argv: break
        time.sleep(10)
