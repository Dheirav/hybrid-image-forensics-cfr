#!/usr/bin/env python3
"""Show step/epoch progress for training runs from their progress.json, with a measured rate.
usage: tools/run_progress.py runs/*/progress.json   (add --watch to refresh every 5 s)
The ETA is remaining steps / (steps done / elapsed) for that run; it is absent until a run has taken steps."""
import glob, json, sys, time

def show(paths):
    for p in paths:
        try:
            s = json.load(open(p))
        except Exception:
            continue
        el = (s["time"] if s.get("finished") else time.time()) - s["t0"]
        rate = s["step"] / el if el > 0 and s["step"] else 0
        eta = "done" if s.get("finished") else (f"ETA ~{int((s['total']-s['step'])/rate)}s" if rate else "no ETA yet")
        print(f"{p.split('/')[-2]:<22} epoch {s['epoch']+ (0 if s.get('finished') else 1)}/{s['epochs']}  step {s['step']}/{s['total']}  "
              f"{rate:.2f} step/s  elapsed {int(el)}s  {eta}")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--watch"] or glob.glob("runs/*/progress.json")
    while True:
        show(sorted(args))
        if "--watch" not in sys.argv: break
        print("-"); time.sleep(5)
