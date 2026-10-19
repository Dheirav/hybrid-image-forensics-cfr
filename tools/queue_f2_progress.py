#!/usr/bin/env python3
"""forge2 queue progress (36 runs in runs_f2/): finished out of total, elapsed, ETA from the measured mean time per finished run.
No ETA until a run has finished. usage: tools/queue_f2_progress.py [--watch]"""
import glob, json, os, sys, time
TOTAL = 36
def show():
    prog = glob.glob("runs_f2/*/progress.json")
    if not prog:
        print(f"0/{TOTAL} runs started"); return
    t0 = min(json.load(open(p))["t0"] for p in prog)
    done = glob.glob("runs_f2/*/results.json")
    cur = ""
    for p in sorted(prog):
        s = json.load(open(p))
        if not s.get("finished"): cur = f" | now {p.split('/')[1]}: epoch {s['epoch']+1}/{s['epochs']}"
    if done:
        per = sum(json.load(open(d))["train_seconds"] for d in done) / len(done)
        eta = f"ETA ~{int((TOTAL-len(done))*per/60)} min (mean {per/60:.1f} min per finished run)"
    else:
        eta = "no ETA yet: no run has finished"
    print(f"{len(done)}/{TOTAL} runs finished, elapsed {int((time.time()-t0)/60)} min, {eta}{cur}")
if __name__ == "__main__":
    while True:
        show()
        if "--watch" not in sys.argv: break
        time.sleep(30)
