"""Train and evaluate: single branches, and A+B pairs under several training regimes.

variants (pairs):
  indep         A and B trained side by side, no alignment (the independent-ensemble control)
  ml            mutual learning throughout, constant alignment weight
  ml_hem        mutual learning + self-hardness example weighting, no CFR
  cfr_adaptive  mutual-learning warm-up, validation-triggered, gradual switch to CFR
  cfr_sudden    same trigger, but CFR weight jumps to 1
  cfr_fixed     switch at a pre-set epoch, ignoring validation
  cfr_scratch   CFR from the first step, no warm-up
  joint_concat  A and B plus a classifier on the concatenated embeddings (no alignment term)
  joint_gate    A and B plus an input-dependent gate over the two branch logits (no alignment term)
Use --tag to run a variant under a different setting (e.g. --lam 0 --tag _noalign) without overwriting the headline runs.
Single runs: --variant single_a | single_b.

All runs of a seed share data order and initialisation of each branch. Model selection uses the
validation split only; the test and ood_removal splits are evaluated once, at the end, with the
best-validation checkpoint.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .controller import TransitionController
from .data import ForgeDataset
from .losses import cfr_terms, hem_terms, masked_seg_loss, weighted_model_loss
from .metrics import classification_metrics, dice_iou, ece
from .models import ForensicBranch, FusionHead, VisualBranch

DEV = "cuda"


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_branches(variant, seed, pretrained):
    seed_all(seed); a = VisualBranch(2, pretrained=pretrained)
    seed_all(seed + 1000); b = ForensicBranch(2)
    return a.to(DEV), b.to(DEV)


@torch.no_grad()
def predict(models, loader, head=None):
    """Returns per-branch P(manipulated) and, for manipulated images, per-branch mask probabilities."""
    for m in models.values():
        m.eval()
    probs = {k: [] for k in models}; masks = {k: [] for k in models}; ys, gts = [], []
    for batch in loader:
        x = batch["x"].to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = {k: m(x) for k, m in models.items()}
        if head is not None:
            with torch.autocast("cuda", enabled=False):
                probs.setdefault("fused", []).append(F.softmax(head(outs["A"], outs["B"]), 1)[:, 1].cpu())
        for k, o in outs.items():
            probs[k].append(F.softmax(o["logits"].float(), 1)[:, 1].cpu())
            # float16 / uint8: the forge2 test split is 2000 images and float32 masks for every branch held 6+ GB on a 7.8 GB machine.
            # Masks are only ever compared with a 0.5 threshold, so the rounding changes a Dice only if a probability sits within 5e-4 of 0.5.
            masks[k].append(torch.sigmoid(o["mask_logits"].float()).half().cpu())
        ys.append(batch["y"]); gts.append((batch["mask"] > 0.5).to(torch.uint8))
    cat = lambda v: torch.cat(v).numpy()
    return {k: cat(v) for k, v in probs.items()}, {k: cat(v) for k, v in masks.items()}, cat(ys), cat(gts)


def evaluate(models, loader, fuse_w=0.5, head=None):
    if head is not None:
        head.eval()
    probs, masks, y, gt = predict(models, loader, head)
    res, keys = {}, list(models)
    if len(keys) == 2:
        if head is None:
            probs["fused"] = fuse_w * probs[keys[0]] + (1 - fuse_w) * probs[keys[1]]
        masks["fused"] = fuse_w * masks[keys[0]] + (1 - fuse_w) * masks[keys[1]]
    manip = y == 1
    for k in probs:
        r = classification_metrics(probs[k], y)
        if manip.any():
            d, i = dice_iou(masks[k][manip, 0] >= 0.5, gt[manip, 0] >= 0.5)
            r.update(dice=float(d.mean()), iou=float(i.mean()))
        res[k] = r
    if len(keys) == 2:
        ca = (probs[keys[0]] >= .5) == y; cb = (probs[keys[1]] >= .5) == y
        res["pair"] = {"disagreement": float(((probs[keys[0]] >= .5) != (probs[keys[1]] >= .5)).mean()),
                       "complementary": float((ca != cb).mean()),
                       "prob_corr": float(np.corrcoef(probs[keys[0]], probs[keys[1]])[0, 1])}
    return res, probs, y


def run(args):
    out = os.path.join(args.out, f"{args.variant}{args.tag}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    man = pd.read_csv(args.manifest)
    mk = lambda split, aug, shuf, bs=None: DataLoader(
        ForgeDataset(man, split, augment=aug), batch_size=bs or args.bs, shuffle=shuf, num_workers=args.workers if shuf else 2,
        drop_last=shuf, pin_memory=True,
        # only the train loader keeps workers alive: 4 loaders x 4 persistent workers exhausted this 7.8 GB machine
        persistent_workers=shuf and args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed) if shuf else None,
        worker_init_fn=lambda w: np.random.seed((torch.initial_seed() + w) % 2 ** 32))
    tr, va, te, ood = mk("train", True, True), mk("val", False, False, 32), mk("test", False, False, 32), mk("ood_removal", False, False, 32)

    pair = not args.variant.startswith("single")
    a, b = build_branches(args.variant, args.seed, not args.no_pretrain)
    models = {"A": a, "B": b} if pair else {("A" if args.variant == "single_a" else "B"): a if args.variant == "single_a" else b}
    head = FusionHead(args.variant.split("_")[1]).to(DEV) if args.variant.startswith("joint_") else None
    params = [p for m in list(models.values()) + ([head] if head is not None else []) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.15)

    ctrl = None
    if args.variant in ("cfr_adaptive", "cfr_sudden", "cfr_fixed"):
        ctrl = TransitionController({"cfr_adaptive": "adaptive", "cfr_sudden": "sudden", "cfr_fixed": "fixed"}[args.variant],
                                    fixed_epoch=args.fixed_epoch)
    gamma = 1.0 if args.variant == "cfr_scratch" else 0.0
    lam = 0.0 if args.variant in ("indep", "joint_concat", "joint_gate") else args.lam

    prog, hist, best, t0, step = os.path.join(out, "progress.json"), [], -1.0, time.time(), 0
    for epoch in range(args.epochs):
        for m in models.values():
            m.train()
        if head is not None:
            head.train()
        tl, n = 0.0, 0
        for batch in tr:
            x, y = batch["x"].to(DEV, non_blocking=True), batch["y"].to(DEV)
            mask, hm = batch["mask"].to(DEV), batch["has_mask"].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = {k: m(x) for k, m in models.items()}
            seg = sum(masked_seg_loss(o["mask_logits"].float(), mask, hm) for o in outs.values())
            if not pair:
                (o,) = outs.values(); loss = F.cross_entropy(o["logits"].float(), y) + seg
            else:
                la, lb = outs["A"]["logits"].float(), outs["B"]["logits"].float()
                if args.variant in ("indep", "ml", "joint_concat", "joint_gate"):
                    ones = torch.ones_like(y, dtype=torch.float)
                    wa = wb = aa = ab = ones
                elif args.variant == "ml_hem":
                    (wa, aa), (wb, ab) = hem_terms(la, lb, y, args.beta), hem_terms(lb, la, y, args.beta)
                else:  # cfr family: blend between mutual-learning terms and CFR terms by gamma
                    ones = torch.ones_like(y, dtype=torch.float)
                    ca, cb = cfr_terms(la, lb, y, args.beta, args.floor), cfr_terms(lb, la, y, args.beta, args.floor)
                    wa, aa = (1 - gamma) * ones + gamma * ca[0], (1 - gamma) * ones + gamma * ca[1]
                    wb, ab = (1 - gamma) * ones + gamma * cb[0], (1 - gamma) * ones + gamma * cb[1]
                loss_a, _ = weighted_model_loss(la, lb, y, wa, aa, lam, args.T)
                loss_b, _ = weighted_model_loss(lb, la, y, wb, ab, lam, args.T)
                loss = loss_a + loss_b + seg
                if head is not None:  # joint fusion output is trained on the label too; branch heads keep their own CE above
                    loss = loss + F.cross_entropy(head(outs["A"], outs["B"]), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); sched.step(); step += 1
            tl += loss.item() * len(y); n += len(y)
            if step % 5 == 0:
                json.dump({"step": step, "total": total_steps, "epoch": epoch, "epochs": args.epochs, "t0": t0,
                           "time": time.time()}, open(prog, "w"))
        res, _, _ = evaluate(models, va, head=head)
        rec = {"epoch": epoch, "train_loss": tl / n, "gamma": gamma, "val": res}
        sel = res["fused" if pair else list(models)[0]]["auroc"]
        if pair:
            stats = {"auroc_a": res["A"]["auroc"], "auroc_b": res["B"]["auroc"], "train_loss": tl / n,
                     "complementary": res["pair"]["complementary"], "disagreement": res["pair"]["disagreement"],
                     "ece_fused": res["fused"]["ece"]}
            if ctrl:
                gamma = ctrl.update(epoch, stats)
                rec["controller"] = ctrl.log[-1]
        hist.append(rec)
        if sel > best:
            best = sel
            torch.save({**{k: m.state_dict() for k, m in models.items()}, **({"head": head.state_dict()} if head is not None else {})},
                       os.path.join(out, "best.pt"))
            rec["is_best"] = True
        print(f"[{args.variant} s{args.seed}] epoch {epoch+1}/{args.epochs} loss {tl/n:.3f} val_auroc "
              f"{ {k: round(v['auroc'], 3) for k, v in res.items() if 'auroc' in v} } gamma {gamma:.2f}", flush=True)

    # final: best-validation checkpoint on all held-out splits, exactly once
    sd = torch.load(os.path.join(out, "best.pt"))
    for k, m in models.items():
        m.load_state_dict(sd[k])
    if head is not None:
        head.load_state_dict(sd["head"])
    final, saved = {}, {}
    for name, loader in (("val", va), ("test", te), ("ood_removal", ood)):
        res, probs, y = evaluate(models, loader, head=head)
        final[name] = res
        saved.update({f"{name}_y": y, **{f"{name}_p_{k}": v for k, v in probs.items()}})
    np.savez(os.path.join(out, "probs.npz"), **saved)  # kept so paired bootstrap and calibration need no retraining
    json.dump({"args": vars(args), "history": hist, "final": final,
               "transition": ctrl.log if ctrl else None, "trigger_epoch": ctrl.trigger_epoch if ctrl else None,
               "train_seconds": time.time() - t0}, open(os.path.join(out, "results.json"), "w"), indent=1, default=float)
    json.dump({"step": total_steps, "total": total_steps, "epoch": args.epochs, "epochs": args.epochs, "t0": t0,
               "time": time.time(), "finished": True}, open(prog, "w"))
    print("finished", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--floor", type=float, default=0.25)
    ap.add_argument("--fixed_epoch", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the run directory, for runs outside the headline set")
    ap.add_argument("--manifest", default="data/forge/manifest.csv")
    ap.add_argument("--out", default="runs")
    run(ap.parse_args())
