#!/usr/bin/env python3
"""External TEST-ONLY set of AI-edited and AI-generated images (reports/04, 11, 12 addenda).

Why it exists: the brief asks for AI-edited and fully generated cases, and the network here (~100 KB/s) rules out
downloading any. The only copy on this machine is a slice of OpenSDI inside the old Deepfake_Detection project's data folder.
It is used READ-ONLY and ZERO-SHOT: no training, no model selection, no labels or numbers from that project. Files are
copied here so the evaluation does not depend on that folder afterwards. Not for redistribution (OpenSDI terms apply).

Excluded: any image whose COCO id is among the 2000 images that COCO-Forge was built from (source-level leakage).
usage: python tools/build_external.py"""
import json, os, re, shutil, random
import pandas as pd
from PIL import Image

SRC = "/home/dheirav/Code/Deepfake_Detection/deepfake-project/data_sources/heldout"
OUT = "data/external"
PER = 100          # per generator and kind; real uses all of sd2/real
coco_ids = {int(re.findall(r"\d+", f)[0]) for f in os.listdir("data/coco_raw/images")}
rng = random.Random(0)
rows = []
def add(gen, kind, name, mask_dir):
    src = f"{SRC}/{gen}/{kind}/{name}"
    cid = int(re.findall(r"\d+", name)[0])
    if cid in coco_ids: return
    dst = f"{OUT}/{kind}/{gen}/{name}"
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
    mpath = ""
    if kind == "ai_edited":
        m = f"{SRC}/{mask_dir}/{os.path.splitext(name)[0]}.png"
        mpath = f"{OUT}/masks/{gen}/{os.path.basename(m)}"
        os.makedirs(os.path.dirname(mpath), exist_ok=True); shutil.copy(m, mpath)
    im = Image.open(dst)
    q = im.quantization if hasattr(im, "quantization") else None
    qsum = sum(q[0]) if q else -1   # proxy for JPEG quality: lower sum = higher quality
    rows.append(dict(path=dst, source_id=cid, label=0 if kind == "real" else 1, kind=kind, mask_path=mpath,
                     generator_or_tool=gen if kind != "real" else "real_" + gen, format=im.format,
                     width=im.size[0], height=im.size[1], file_bytes=os.path.getsize(dst), qtable_sum=qsum, split="external_test"))
for gen in ("flux", "sd15", "sd3", "sdxl"):
    fs = sorted(os.listdir(f"{SRC}/{gen}/ai_edited")); rng.shuffle(fs)
    for n in fs[:PER + 5]: add(gen, "ai_edited", n, f"{gen}_masks")
for gen in ("flux", "sd2", "sd3", "sdxl"):
    fs = sorted(os.listdir(f"{SRC}/{gen}/ai_generated")); rng.shuffle(fs)
    for n in fs[:PER + 5]: add(gen, "ai_generated", n, "")
for n in sorted(os.listdir(f"{SRC}/sd2/real")): add("sd2", "real", n, "")
df = pd.DataFrame(rows)
# cap per generator and kind after exclusions so groups stay balanced
df = df.groupby(["kind", "generator_or_tool"], group_keys=False).apply(lambda g: g.head(PER if g.name[0] != "real" else len(g)))
df.to_csv(f"{OUT}/manifest.csv", index=False)
print(df.groupby(["kind", "generator_or_tool"]).size())
print(df.groupby("kind")[["width", "height", "file_bytes", "qtable_sum"]].describe().T.round(1).to_string())
