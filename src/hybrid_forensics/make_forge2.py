"""Build 'forge2': COCO-Forge with a harder, tool-disjoint protocol, from the same source images and the same split of sources.

Why: Model A saturates COCO-Forge (0.994) and every method sits within 0.01 of it, so the benchmark cannot rank them. The forge_hard test
(`make_hard`) showed all 14 trained variants are at chance on Poisson-blended edits and at 0.87 to 0.90 on a content-preserving re-paste. Training on
some of those tools while holding others out gives a benchmark where there is headroom on seen tools and a real tool-disjoint test.

Training tools (rotated over sources, one edit per source, as in COCO-Forge so the training cost is unchanged):
  splice_feather, copy_move_feather, splice_poisson, copy_move_poisson
Never trained on, test only: self_resample (resampling traces without new scene content), removal_telea (the COCO-Forge removal tool),
  removal_ns (Navier-Stokes inpainting, a different inpainting algorithm).
Splits: train / val use the four seen tools, rotated. test holds EVERY tool for every source (authentic + 7 edits), so tools are paired by source.
ood_removal reuses the COCO-Forge ood sources (authentic + Telea removal) so train.py's four-split loop runs unchanged.
Sources, donors and split membership are the COCO-Forge ones; crops are drawn afresh, so no image is shared with COCO-Forge or forge_hard.
"""
import json
import os
import random

import cv2
import numpy as np
import pandas as pd

from .make_dataset import CROP, JPEG_Q, _load
from .make_hard import feather_paste, poisson_paste, self_resample

SEEN = ("splice_feather", "copy_move_feather", "splice_poisson", "copy_move_poisson")
UNSEEN = ("self_resample", "removal_telea", "removal_ns")


def make_edit(tool, img, obj, dimg, dobj, rng):
    if tool == "splice_feather": return feather_paste(img, dimg, dobj, rng)
    if tool == "splice_poisson": return poisson_paste(img, dimg, dobj, rng)
    if tool == "copy_move_feather": return feather_paste(img, img, obj, rng, avoid=obj)
    if tool == "copy_move_poisson": return poisson_paste(img, img, obj, rng, avoid=obj)
    if tool == "self_resample": return self_resample(img, obj)
    grown = cv2.dilate(obj, np.ones((7, 7), np.uint8))
    flag = cv2.INPAINT_TELEA if tool == "removal_telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, grown, 3, flag), grown


def build(raw_dir="data/coco_raw", out_dir="data/forge2", base_manifest="data/forge/manifest.csv", seed=11):
    recs = {r["image_id"]: r for r in json.load(open(os.path.join(raw_dir, "selection.json")))}
    base = pd.read_csv(base_manifest)
    rng = random.Random(seed)
    for d in ("images", "masks"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    rows = []

    def write(r, split, kind, arr, m, donor):
        q = rng.choice(JPEG_Q)
        name = f"{r['image_id']:012d}_{kind}"
        path = os.path.join(out_dir, "images", name + ".jpg")
        cv2.imwrite(path, arr, [cv2.IMWRITE_JPEG_QUALITY, q])
        mpath = ""
        if m is not None:
            mpath = os.path.join(out_dir, "masks", name + ".png")
            cv2.imwrite(mpath, m * 255)
        rows.append({"path": path, "source_id": r["image_id"], "label": int(kind != "authentic"), "mask_path": mpath, "generator_or_tool": kind,
                     "manipulation": kind, "format": "jpeg", "jpeg_quality": q, "dimensions": f"{CROP}x{CROP}", "split": split,
                     "donor_source_id": donor, "coco_license": r["license"]})

    for split in ("train", "val", "test", "ood_removal"):
        ids = sorted(base[(base.split == split) & (base.label == 0)].source_id.unique())
        pool = [recs[i] for i in ids]
        for i, r in enumerate(pool):
            img, obj = _load(r, raw_dir, rng)
            donor = pool[(i + 1 + rng.randrange(len(pool) - 1)) % len(pool)]
            dimg, dobj = _load(donor, raw_dir, rng)
            tools = {"train": [SEEN[i % 4]], "val": [SEEN[i % 4]], "test": list(SEEN + UNSEEN), "ood_removal": ["removal_telea"]}[split]
            write(r, split, "authentic", img, None, None)
            for tool in tools:
                man, m = make_edit(tool, img, obj, dimg, dobj, rng)
                write(r, split, tool, man, m, donor["image_id"] if tool.startswith("splice") else None)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(d.groupby(["split", "manipulation"]).size())
