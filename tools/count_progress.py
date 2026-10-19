#!/usr/bin/env python3
"""Progress for any job that writes one output file per item: done/total, elapsed, and an ETA from the measured rate.
usage: tools/count_progress.py STAMPFILE 'GLOB' TOTAL [--watch]
STAMPFILE is touched by the launcher when the job starts, so elapsed is measured from it. With nothing finished there is no rate, so no ETA."""
import glob, os, sys, time

def show(stamp, pattern, total):
    done = len(glob.glob(pattern))
    el = time.time() - os.path.getmtime(stamp)
    eta = f"ETA ~{int((total - done) * el / done / 60)} min (measured {el / done:.0f} s per item)" if done else "no ETA yet: nothing finished"
    print(f"{done}/{total} done, elapsed {int(el / 60)} min {int(el % 60)} s, {eta}")

if __name__ == "__main__":
    stamp, pattern, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    while True:
        show(stamp, pattern, total)
        if "--watch" not in sys.argv: break
        time.sleep(10)
