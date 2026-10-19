"""Zero-shot evaluation of trained runs on the external AI-edited / AI-generated set (tools/build_external.py).

Nothing here is trained or tuned on that set: threshold 0.5, the checkpoint chosen on COCO-Forge validation, the same
tiling and padding as infer.py (no resizing). Image score = max over 384 px windows, as in the application.
Dice is computed for AI-edited images only (those have masks), from the fused branch masks stitched back to image size.
usage: python -m hybrid_forensics.external_eval runs/<run> [runs/<run> ...]   -> runs/<run>/external.json
       python -m hybrid_forensics.external_eval --report   -> markdown table over everything found in runs/ and runs_extra/"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .infer import TILE, tile_starts
from .metrics import dice_iou
from .models import ForensicBranch, FusionHead, VisualBranch

MAN = "data/external/manifest.csv"


def load(run, device="cuda"):
    sd = torch.load(os.path.join(run, "best.pt"), map_location="cpu")
    models = {}
    for k, cls in (("A", VisualBranch), ("B", ForensicBranch)):
        if k in sd:
            m = cls(2); m.load_state_dict(sd[k]); models[k] = m.to(device).eval()
    head = None
    if "head" in sd:
        kind = json.load(open(os.path.join(run, "results.json")))["args"]["variant"].split("_")[1]
        head = FusionHead(kind).to(device).eval(); head.load_state_dict(sd["head"])
    return models, head


@torch.no_grad()
def score_image(models, head, rgb, device="cuda"):
    h, w = rgb.shape[:2]
    ph, pw = max(TILE - h, 0), max(TILE - w, 0)
    pad = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric") if (ph or pw) else rgb
    x = torch.from_numpy(np.ascontiguousarray(pad)).permute(2, 0, 1).float()[None] / 255.0
    H, W = x.shape[-2:]
    tiles, pos = [], []
    for y0 in tile_starts(H):
        for x0 in tile_starts(W):
            tiles.append(x[..., y0:y0 + TILE, x0:x0 + TILE]); pos.append((y0, x0))
    t = torch.cat(tiles).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outs = {k: m(t) for k, m in models.items()}
    p = {k: F.softmax(o["logits"].float(), 1)[:, 1] for k, o in outs.items()}
    keys = list(models)
    if head is not None:
        p["fused"] = F.softmax(head(outs["A"], outs["B"]), 1)[:, 1]
    elif len(keys) == 2:
        p["fused"] = 0.5 * p["A"] + 0.5 * p["B"]
    else:
        p["fused"] = p[keys[0]]
    heat, count = torch.zeros(1, 1, H, W), torch.zeros(1, 1, H, W)
    for i, (y0, x0) in enumerate(pos):
        ms = [torch.sigmoid(outs[k]["mask_logits"][i:i + 1].float()).cpu() for k in keys]
        heat[..., y0:y0 + TILE, x0:x0 + TILE] += sum(ms) / len(ms)
        count[..., y0:y0 + TILE, x0:x0 + TILE] += 1
    heat = heat / count
    return {k: float(v.max()) for k, v in p.items()}, heat[0, 0, :h, :w].numpy()


def evaluate_run(run, crop=False):
    """crop=True scores the central 384 px window only: one window, no padding, no resizing. It is the control for
    the seams that symmetric padding of 512 px images to 768 px could introduce (written to external_crop.json)."""
    man = pd.read_csv(MAN)
    models, head = load(run)
    rows = []
    for r in man.itertuples():
        rgb = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB)
        off = (rgb.shape[0] - TILE) // 2 if crop else 0
        if crop:
            rgb = rgb[off:off + TILE, off:off + TILE]
        s, heat = score_image(models, head, rgb)
        row = dict(path=r.path, **{f"p_{k}": v for k, v in s.items()})
        if isinstance(r.mask_path, str) and r.mask_path:
            gt = cv2.imread(r.mask_path, 0) > 127
            if crop:
                gt = gt[off:off + TILE, off:off + TILE]
            d, i = dice_iou((heat >= 0.5)[None], gt[None])
            row.update(dice=float(d[0]), iou=float(i[0]))
        rows.append(row)
    res = man.merge(pd.DataFrame(rows), on="path")
    key = "p_fused"
    out = {"n": len(res)}
    real = res[res.kind == "real"]
    out["fpr_real"] = float((real[key] >= .5).mean())
    for kind in ("ai_edited", "ai_generated"):
        fk = res[res.kind == kind]
        both = pd.concat([real, fk])
        out[f"{kind}_auroc"] = float(roc_auc_score(both.label, both[key]))
        out[f"{kind}_recall"] = float((fk[key] >= .5).mean())
        for g, gg in fk.groupby("generator_or_tool"):
            b = pd.concat([real, gg])
            out[f"{kind}:{g}_auroc"] = float(roc_auc_score(b.label, b[key]))
            out[f"{kind}:{g}_recall"] = float((gg[key] >= .5).mean())
    ed = res[res.kind == "ai_edited"]
    out["ai_edited_dice"] = float(ed.dice.mean())
    # an image-level detector that scores every image high looks fine on recall, so the real-image rate is reported next to it
    out["mean_score_real"], out["mean_score_fake"] = float(real[key].mean()), float(res[res.kind != "real"][key].mean())
    json.dump(out, open(os.path.join(run, "external_crop.json" if crop else "external.json"), "w"), indent=1)
    np.savez(os.path.join(run, "external_crop_probs.npz" if crop else "external_probs.npz"), **{c: res[c].values for c in res.columns if c.startswith("p_")})
    return out


def report():
    rows = {}
    for f in sorted(glob.glob("runs*/*/external_crop.json" if "--crop" in sys.argv else ("runs*/*/external_padded.json" if "--padded" in sys.argv else "runs*/*/external.json"))):
        run = os.path.dirname(f)
        name = os.path.basename(run).rsplit("_s", 1)
        rows.setdefault(name[0], []).append(json.load(open(f)))
    cols = ["fpr_real", "ai_edited_auroc", "ai_edited_recall", "ai_edited_dice", "ai_generated_auroc", "ai_generated_recall"]
    print("| variant | seeds | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
    for v, rs in rows.items():
        print(f"| {v} | {len(rs)} | " + " | ".join(f"{np.mean([r[c] for r in rs]):.3f}" for c in cols) + " |")
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report()
    else:
        crop = "--crop" in sys.argv
        for r in [a for a in sys.argv[1:] if a != "--crop"]:
            print(r, json.dumps({k: round(v, 3) for k, v in evaluate_run(r, crop).items() if not k.startswith(("ai_edited:", "ai_generated:"))}))
