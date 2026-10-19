#!/usr/bin/env python3
"""One-off: copy an identical set of already-installed packages into this project's .venv.

Why this exists: the NVIDIA/torch wheels download at ~170 KB/s here (about 6 hours for 4 GB).
This copies files listed in each package's RECORD, so .venv becomes self-contained. It does
not link to the source environment and nothing imports from it afterwards.
usage: seed_venv_from_local.py SOURCE_SITE_PACKAGES DEST_SITE_PACKAGES
"""
import csv, shutil, subprocess, sys
from pathlib import Path

WANT = ("torch torchvision triton cuda_bindings cuda_pathfinder filelock fsspec jinja2 markupsafe mpmath "
        "networkx sympy typing_extensions numpy scipy scikit_learn joblib threadpoolctl pillow "
        "opencv_python_headless pandas python_dateutil pytz tzdata six pyyaml packaging tqdm pygments").split()
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dists = [d for d in src.glob("*.dist-info") if d.name[:-len(".dist-info")].rsplit("-", 1)[0].lower().replace("-", "_") in WANT
         or d.name.lower().startswith("nvidia_")]
n = 0
for d in sorted(dists):
    for row in csv.reader(open(d / "RECORD")):
        rel = row[0]
        if rel.startswith("..") or rel.endswith(".pyc") or "__pycache__" in rel:
            continue
        s, t = src / rel, dst / rel
        if s.is_file():
            t.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "--reflink=auto", "--update=none", str(s), str(t)], check=True)
            n += 1
print(f"copied {n} files from {len(dists)} distributions")
