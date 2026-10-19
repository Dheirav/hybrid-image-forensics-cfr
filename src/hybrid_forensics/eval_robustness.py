"""Degradation robustness on the test split. Degradations are applied to BOTH classes, so a model that
merely flags 'processed' images would be penalised through the authentic false-positive rate.
Kept separate from the unseen-tool (ood_removal) test, as the brief requires.
usage: python -m hybrid_forensics.eval_robustness runs/<run>/best.pt out.json"""
import json
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import classification_metrics, dice_iou
from .models import ForensicBranch, VisualBranch


def jpeg(q):
    return lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)


def resize(s):
    return lambda im: cv2.resize(im, (int(round(im.shape[1] * s / 8) * 8), int(round(im.shape[0] * s / 8) * 8)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def blur(sig):
    return lambda im: cv2.GaussianBlur(im, (0, 0), sig)


def noise(sd):
    rng = np.random.default_rng(0)
    return lambda im: np.clip(im + rng.normal(0, sd, im.shape), 0, 255).astype(np.uint8)


def chain(*fs):
    def f(im):
        for g in fs:
            im = g(im)
        return im
    return f


DEGRADATIONS = {"clean": lambda im: im, "jpeg_q70": jpeg(70), "jpeg_q50": jpeg(50), "recompress_q90_q60": chain(jpeg(90), jpeg(60)),
                "resize_0.75": resize(0.75), "resize_1.25": resize(1.25), "blur_s1": blur(1.0), "noise_s5": noise(5),
                "chain_resize0.75_jpeg70_blur0.8": chain(resize(0.75), jpeg(70), blur(0.8))}


@torch.no_grad()
def main(ckpt, out, manifest="data/forge/manifest.csv", split="test"):
    sd = torch.load(ckpt, map_location="cpu")
    models = {"A": VisualBranch(2), "B": ForensicBranch(2)}
    for k, m in models.items():
        m.load_state_dict(sd[k]); m.cuda().eval()
    df = pd.read_csv(manifest)
    df = df[df.split == split].reset_index(drop=True)
    res = {}
    for name, fn in DEGRADATIONS.items():
        pf, y, dice, iou = [], [], [], []
        for r in df.itertuples():
            im = fn(cv2.imread(r.path))
            x = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = {k: m(x) for k, m in models.items()}
            p = 0.5 * (F.softmax(o["A"]["logits"].float(), 1)[0, 1] + F.softmax(o["B"]["logits"].float(), 1)[0, 1])
            pf.append(p.item()); y.append(r.label)
            if r.label == 1:
                heat = 0.5 * (torch.sigmoid(o["A"]["mask_logits"].float()) + torch.sigmoid(o["B"]["mask_logits"].float()))[0, 0].cpu().numpy()
                gt = cv2.resize(cv2.imread(r.mask_path, 0), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                d, i = dice_iou((heat >= .5)[None], gt[None]); dice.append(d[0]); iou.append(i[0])
        m = classification_metrics(np.array(pf), np.array(y))
        m.update(dice=float(np.mean(dice)), iou=float(np.mean(iou)))
        res[name] = m
        print(name, {k: round(v, 3) for k, v in m.items() if k in ("auroc", "macro_f1", "fpr_authentic", "dice")}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
