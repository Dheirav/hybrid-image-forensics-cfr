#!/usr/bin/env python3
"""End-to-end test matrix for the real inference path (infer.Predictor on a trained checkpoint).
Records completion, class, confidence, whether a mask was produced, Dice/IoU where ground truth exists,
error text and seconds per case. usage: tools/e2e_matrix.py runs/<run>/best.pt [calibration.json]
Cases come from the TEST split (never trained on); files are written to outputs/e2e_inputs/."""
import json, os, sys, time
import cv2, numpy as np, pandas as pd

sys.path.insert(0, "src")
from hybrid_forensics.infer import InputError, Predictor, export
from hybrid_forensics.metrics import dice_iou

ckpt = sys.argv[1]; cal = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs("outputs/e2e_inputs", exist_ok=True)
df = pd.read_csv("data/forge/manifest.csv").query("split == 'test'").reset_index(drop=True)
pick = lambda tool, i=0: df[df.manipulation == tool].iloc[i]
gt_of = lambda r: (cv2.imread(r.mask_path, 0) > 127) if isinstance(r.mask_path, str) and r.mask_path else None
cases = []  # (name, path, gt_mask or None, note)

def add(name, img, gt=None, ext=".png", note="", raw=None):
    p = f"outputs/e2e_inputs/{name}{ext}"
    if raw is not None: open(p, "wb").write(raw)
    else: cv2.imwrite(p, img)
    cases.append((name, p, gt, note))

au, sp, cm, rm = pick("authentic"), pick("splice"), pick("copy_move"), pick("removal")
areas = df[df.label == 1].assign(a=lambda d: [(cv2.imread(m, 0) > 127).mean() for m in d.mask_path]).sort_values("a")
small, large = areas.iloc[0], areas.iloc[-1]
add("01_authentic", cv2.imread(au.path), None, ".jpg", "authentic crop"); cases[-1] = (cases[-1][0], cases[-1][1], np.zeros((384, 384), bool), "authentic crop")
add("02_conventional_splice", cv2.imread(sp.path), gt_of(sp), ".jpg")
add("03_conventional_copy_move", cv2.imread(cm.path), gt_of(cm), ".jpg")
add("04_conventional_removal_unseen_tool", cv2.imread(rm.path), gt_of(rm), ".jpg", "tool never seen in training")
add("05_small_region", cv2.imread(small.path), gt_of(small), ".jpg", f"mask {small.a:.1%} of frame")
add("06_large_region", cv2.imread(large.path), gt_of(large), ".jpg", f"mask {large.a:.1%} of frame")
im, g = cv2.imread(sp.path), gt_of(sp)
q = cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], 1)
add("07_jpeg_q40_of_splice", q, g, ".jpg", "recompressed at q40")
r = cv2.resize(im, (288, 288), interpolation=cv2.INTER_AREA); add("08_resized_0.75_of_splice", r, cv2.resize(g.astype(np.uint8), (288, 288), interpolation=cv2.INTER_NEAREST) > 0, ".png", "resized to 288 px")
add("09_blurred_of_splice", cv2.GaussianBlur(im, (0, 0), 1.5), g, ".png", "gaussian blur sigma 1.5")
add("10a_unsupported_gif", None, None, ".gif", "unsupported type", raw=b"GIF89a\x01\x00\x01\x00")
add("10b_corrupted_jpeg", None, None, ".jpg", "truncated file", raw=open(sp.path, "rb").read()[:2000])
mos = np.vstack([np.hstack([cv2.imread(df[df.manipulation == "authentic"].iloc[i].path) for i in (1, 2, 3)]),
                 np.hstack([cv2.imread(sp.path)] + [cv2.imread(df[df.manipulation == "authentic"].iloc[i].path) for i in (4, 5)])])
gm = np.zeros(mos.shape[:2], bool); gm[384:, :384] = g
add("11_large_mosaic_1152x768", mos, gm, ".png", "5 authentic tiles + 1 spliced tile; tile seams are unseen in training")
add("12a_aspect_3to1", cv2.hconcat([cv2.imread(au.path)] * 3), None, ".png", "1152x384 (3:1), tiled not resized")
add("12b_aspect_6to1_rejected", np.zeros((120, 720, 3), np.uint8) + 90, None, ".png", "outside supported aspect")
add("13a_too_small_64px", cv2.resize(im, (64, 64)), None, ".png", "outside supported size")
add("13b_too_large_5000px", np.zeros((300, 5000, 3), np.uint8) + 90, None, ".png", "outside supported size")
# AI cases come from the external test-only set (tools/build_external.py): 512 px OpenSDI images the model was never trained on
ext = pd.read_csv("data/external/manifest.csv")
for name, kind, gen in (("14a_ai_edited_sdxl", "ai_edited", "sdxl"), ("14b_ai_edited_flux", "ai_edited", "flux"),
                        ("14c_ai_generated_sd3", "ai_generated", "sd3"), ("14d_real_photo_external", "real", "real_sd2")):
    r = ext[(ext.kind == kind) & (ext.generator_or_tool == gen)].iloc[0]
    gt = (cv2.imread(r.mask_path, 0) > 127) if isinstance(r.mask_path, str) and r.mask_path else (np.zeros((512, 512), bool) if kind == "real" else None)
    cases.append((name, r.path, gt, f"external {kind} ({gen}), 512 px, zero-shot"))

pred = Predictor.from_checkpoint(ckpt, cal)
rows = []
for name, path, gt, note in cases:
    t = time.time(); row = {"case": name, "note": note}
    try:
        res = pred.predict(path); export(res, path, "outputs/e2e_results")
        row.update(completed=True, label=res["label"], p_manipulated=round(res["p_manipulated"], 4), windows=res["windows"],
                   branches_agree=bool(res["branches_agree"]), localisation_generated=True, mask_area=round(res["mask_area_fraction"], 4))
        if gt is not None and gt.any():
            d, i = dice_iou((res["heatmap"] >= .5)[None], gt[None]); row.update(dice=round(float(d[0]), 3), iou=round(float(i[0]), 3))
    except InputError as e:
        row.update(completed=False, error=str(e), localisation_generated=False)
    except Exception as e:  # anything else is a bug in the pipeline, recorded not hidden
        row.update(completed=False, error=f"UNEXPECTED {type(e).__name__}: {e}", localisation_generated=False)
    row["seconds"] = round(time.time() - t, 3); rows.append(row)
json.dump(rows, open("reports/evidence/e2e_results.json", "w"), indent=1)
for r in rows: print(r)
