import torch

from hybrid_forensics.losses import cfr_terms, hem_terms, weighted_model_loss


def _logits(pt, y):
    """2-class logits whose true-class probability is pt."""
    p = torch.tensor(pt)
    l1 = torch.log(torch.where(y == 1, p, 1 - p))
    l0 = torch.log(torch.where(y == 1, 1 - p, p))
    return torch.stack([l0, l1], 1)


def test_cfr_weights_target_samples_where_partner_is_more_right():
    y = torch.tensor([1, 1, 1, 1])
    me, partner = _logits([0.2, 0.9, 0.2, 0.9], y), _logits([0.9, 0.2, 0.2, 0.9], y)
    w, a = cfr_terms(me, partner, y, beta=2.0, floor=0.25)
    assert w[0] > w[1] and w[0] > w[3]            # exposed sample gets the most correction
    assert abs(w.mean().item() - 1) < 1e-5         # weighting does not rescale the loss
    assert a[0] > a[1]                              # align strongly only toward the more-correct partner
    assert (a >= 0.25 - 1e-6).all()                 # alignment never switched off


def test_cfr_does_not_align_toward_a_partner_that_is_wrong():
    y = torch.tensor([1])
    me, partner = _logits([0.9], y), _logits([0.1], y)
    _, a = cfr_terms(me, partner, y, floor=0.25)
    assert a.item() == 0.25                         # only the floor, no ground-truth-gated pull


def test_hem_ignores_partner_and_aligns_everywhere():
    y = torch.tensor([1, 1])
    me = _logits([0.2, 0.9], y)
    w1, a1 = hem_terms(me, _logits([0.9, 0.2], y), y)
    w2, _ = hem_terms(me, _logits([0.1, 0.1], y), y)
    assert torch.allclose(w1, w2) and (a1 == 1).all() and w1[0] > w1[1]


def test_weighted_loss_keeps_ground_truth_and_detaches_partner():
    y = torch.tensor([0, 1, 1])
    a = torch.randn(3, 2, requires_grad=True)
    b = torch.randn(3, 2, requires_grad=True)
    w, al = cfr_terms(a, b, y)
    loss, _ = weighted_model_loss(a, b, y, w, al, lam=1.0)
    loss.backward()
    assert a.grad is not None and b.grad is None
