"""End-to-end inference: image file in, validated prediction + localisation out.

Uses the same models, class order (0 = authentic, 1 = manipulated) and fusion (probability average) as
evaluation. Images are never resized: they are processed in 384 px windows (the last one anchored to the edge), which is the training
size, and a 384x384 image goes through exactly the evaluation path. For larger images the image-level
score is the maximum over windows and the heat map is stitched from the windows.
"""
import argparse
import json
import os
import time

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .models import ForensicBranch, VisualBranch

TILE = 384
ALLOWED = {".jpg", ".jpeg", ".png"}
MIN_SIDE, MAX_SIDE, MAX_ASPECT = 128, 4096, 4.0
LIMITS = ("This is a model prediction, not proof. It was trained only on synthetic conventional edits "
          "(splice, copy-move, object removal) built from COCO photographs. It has not been tested on "
          "AI-edited or fully AI-generated images and says nothing about them. A low score does not show "
          "that an image is authentic.")


def tile_starts(n):
    """Window start offsets along one axis. The last window is anchored to the image edge (windows overlap) so no padding
    is needed when n >= TILE. Symmetric padding to a multiple of 384 put mirror seams inside the image and raised the
    false-positive rate on authentic 512 px images from 0.06 to 0.19 (reports/13, entry 13); only images shorter than a
    window are padded now."""
    if n <= TILE:
        return [0]
    starts = list(range(0, n - TILE + 1, TILE))
    if starts[-1] != n - TILE:
        starts.append(n - TILE)
    return starts


class InputError(ValueError):
    pass


def validate(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED:
        raise InputError(f"unsupported file type '{ext}', expected one of {sorted(ALLOWED)}")
    if not os.path.isfile(path):
        raise InputError("file not found")
    # OpenCV silently returns a grey-filled partial image for a truncated JPEG, which would be scored as
    # if it were real content. PIL's strict decode raises instead, so use it as the integrity gate.
    try:
        with Image.open(path) as im:
            im.load()
    except Exception:
        raise InputError("file could not be decoded as an image (corrupted, truncated or not an image)")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("file could not be decoded as an image (corrupted or not an image)")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise InputError(f"image is {w}x{h}; supported side length is {MIN_SIDE} to {MAX_SIDE} px")
    if max(h, w) / min(h, w) > MAX_ASPECT:
        raise InputError(f"aspect ratio {max(h, w) / min(h, w):.1f}:1 exceeds the supported {MAX_ASPECT:.0f}:1")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Predictor:
    def __init__(self, models, fuse_w=0.5, temperature=1.0, device="cuda"):
        self.models, self.fuse_w, self.T, self.device = models, fuse_w, temperature, device
        for m in models.values():
            m.to(device).eval()

    @classmethod
    def from_checkpoint(cls, ckpt, calibration=None, device="cuda"):
        sd = torch.load(ckpt, map_location="cpu")
        models = {"A": VisualBranch(2), "B": ForensicBranch(2)}
        for k, m in models.items():
            m.load_state_dict(sd[k])
        T = json.load(open(calibration))["temperature"] if calibration and os.path.exists(calibration) else 1.0
        return cls(models, temperature=T, device=device)

    @torch.no_grad()
    def _tiles(self, rgb):
        h, w = rgb.shape[:2]
        ph, pw = max(TILE - h, 0), max(TILE - w, 0)
        padded = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric") if (ph or pw) else rgb
        x = torch.from_numpy(np.ascontiguousarray(padded)).permute(2, 0, 1).float()[None] / 255.0
        H, W = x.shape[-2:]
        heat, count, probs = torch.zeros(1, 1, H, W), torch.zeros(1, 1, H, W), []
        for y0 in tile_starts(H):
            for x0 in tile_starts(W):
                t = x[..., y0:y0 + TILE, x0:x0 + TILE].to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    outs = {k: m(t) for k, m in self.models.items()}
                pa, pb = (F.softmax(outs[k]["logits"].float(), 1)[:, 1] for k in ("A", "B"))
                fused = (self.fuse_w * pa + (1 - self.fuse_w) * pb).clamp(1e-6, 1 - 1e-6)
                if self.T != 1.0:  # temperature fitted on validation data only (see calibrate.py)
                    fused = torch.sigmoid(torch.logit(fused) / self.T)
                probs.append({"A": pa.item(), "B": pb.item(), "fused": fused.item()})
                m = self.fuse_w * torch.sigmoid(outs["A"]["mask_logits"].float()) + (1 - self.fuse_w) * torch.sigmoid(outs["B"]["mask_logits"].float())
                heat[..., y0:y0 + TILE, x0:x0 + TILE] += m.cpu()
                count[..., y0:y0 + TILE, x0:x0 + TILE] += 1
        return probs, (heat / count)[0, 0, :h, :w].numpy()

    def predict(self, path):
        t0 = time.time()
        rgb = validate(path)
        probs, heat = self._tiles(rgb)
        top = max(probs, key=lambda d: d["fused"])
        return {"p_manipulated": top["fused"], "p_authentic": 1 - top["fused"],
                "label": "manipulated" if top["fused"] >= 0.5 else "authentic",
                "branch_probs": {"visual": top["A"], "forensic": top["B"]},
                "branches_agree": (top["A"] >= .5) == (top["B"] >= .5),
                "windows": len(probs), "size": [rgb.shape[1], rgb.shape[0]],
                "heatmap": heat, "mask": (heat >= 0.5).astype(np.uint8),
                "mask_area_fraction": float((heat >= 0.5).mean()),
                "seconds": time.time() - t0, "limitations": LIMITS}


def export(result, path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    heat = cv2.applyColorMap((result["heatmap"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(out_dir, stem + "_overlay.png"), cv2.addWeighted(img, 0.6, heat, 0.4, 0))
    cv2.imwrite(os.path.join(out_dir, stem + "_mask.png"), result["mask"] * 255)
    summary = {k: v for k, v in result.items() if k not in ("heatmap", "mask")}
    json.dump(summary, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calibration", default=None); ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    try:
        r = Predictor.from_checkpoint(a.ckpt, a.calibration).predict(a.image)
        print(json.dumps(export(r, a.image, a.out), indent=1))
    except InputError as e:
        print(json.dumps({"error": str(e)})); raise SystemExit(2)
