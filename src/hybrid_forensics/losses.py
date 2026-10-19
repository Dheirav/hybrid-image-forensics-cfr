"""Losses. Reductions are stated explicitly because a wrong reduction silently rescales lambda."""
import torch
import torch.nn.functional as F


def masked_seg_loss(mask_logits, mask, has_mask, eps: float = 1.0):
    """BCE + Dice over samples with a trusted mask only. Returns 0 (with grad graph) if none."""
    keep = has_mask.bool()
    if not keep.any():
        return mask_logits.sum() * 0.0
    lg, tg = mask_logits[keep], mask[keep]
    bce = F.binary_cross_entropy_with_logits(lg, tg)                      # mean over pixels and samples
    p = torch.sigmoid(lg).flatten(1)
    t = tg.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)   # per sample
    return bce + dice.mean()


def kl_to_partner(student_logits, partner_logits, T: float = 2.0):
    """KL(p_partner^T || p_student^T), batch-mean, times T^2.

    The partner is detached: it is a target here, not something this loss may move. The T^2
    factor keeps the gradient magnitude comparable across temperatures.
    """
    with torch.no_grad():
        target = F.softmax(partner_logits.detach() / T, dim=1)
    log_p = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(log_p, target, reduction="batchmean") * (T * T)


def mutual_losses(out_a, out_b, y, lam_align: float, T: float = 2.0):
    """L_A = CE(p_A, y) + lam * KL(p_B || p_A); L_B symmetric. Ground-truth CE is always on."""
    ce_a, ce_b = F.cross_entropy(out_a["logits"], y), F.cross_entropy(out_b["logits"], y)
    kl_a = kl_to_partner(out_a["logits"], out_b["logits"], T)
    kl_b = kl_to_partner(out_b["logits"], out_a["logits"], T)
    return ce_a + lam_align * kl_a, ce_b + lam_align * kl_b, {"kl_a": kl_a.detach(), "kl_b": kl_b.detach()}


# --- per-sample variants used by hard-example mining and CFR ---------------------------------

def per_sample_kl(student_logits, partner_logits, T: float = 2.0):
    """KL(p_partner^T || p_student^T) per sample, times T^2. Partner is a detached target."""
    with torch.no_grad():
        target = F.softmax(partner_logits.detach() / T, dim=1)
    log_p = F.log_softmax(student_logits / T, dim=1)
    return (target * (target.clamp_min(1e-8).log() - log_p)).sum(1) * (T * T)


def _true_prob(logits, y):
    return F.softmax(logits.detach(), dim=1).gather(1, y[:, None]).squeeze(1)


def _normalise(w):
    return w / w.mean().clamp_min(1e-8)  # mean weight 1, so weighting cannot rescale the loss


def hem_terms(logits_self, logits_partner, y, beta: float = 2.0):
    """Hard-example mining baseline: weight by the model's OWN error, align on every sample."""
    w = _normalise(1 + beta * (1 - _true_prob(logits_self, y)))
    return w, torch.ones_like(w)


def cfr_terms(logits_self, logits_partner, y, beta: float = 2.0, floor: float = 0.25, w_max: float = 4.0):
    """Competitive Forensic Reasoning weights for one model (self) against its partner.

    challenge c = max(0, p_t(partner) - p_t(self)): the partner is more right about the true label,
    which exposes this model's weak prediction. CE on those samples is up-weighted (correction under
    ground truth). Alignment toward the partner is gated by the same ground-truth comparison: it is
    strong only where the partner is more correct, and never below `floor`, so Model B stays aligned
    with Model A without being pulled onto A's mistakes. Weights are clipped and renormalised.
    This is competitive hard-example correction, not a formal game-theoretic objective.
    """
    ps, pp = _true_prob(logits_self, y), _true_prob(logits_partner, y)
    challenge = (pp - ps).clamp_min(0)
    w = _normalise((1 + beta * challenge).clamp_max(w_max))
    gate = torch.where(pp > ps, pp, torch.zeros_like(pp))
    return w, floor + (1 - floor) * gate


def weighted_model_loss(logits_self, logits_partner, y, w_ce, w_align, lam: float, T: float = 2.0):
    ce = (F.cross_entropy(logits_self, y, reduction="none") * w_ce).mean()
    kl = (per_sample_kl(logits_self, logits_partner, T) * w_align).mean()
    return ce + lam * kl, kl.detach()
