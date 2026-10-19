"""Software-correctness tests. Passing these says the code runs as specified; it says nothing
about whether the models detect manipulation on real data."""
import torch
import torch.nn.functional as F

from hybrid_forensics.forensic_features import BayarConv, BlockDCT, SRMResidual
from hybrid_forensics.losses import kl_to_partner, masked_seg_loss, mutual_losses
from hybrid_forensics.models import ForensicBranch, VisualBranch

torch.manual_seed(0)
H = W = 64


def _batch(n=4):
    return torch.rand(n, 3, H, W)


def test_srm_shape_and_range():
    out = SRMResidual()(_batch())
    assert out.shape == (4, 9, H, W)
    assert out.abs().max() <= 1.0 + 1e-6


def test_bayar_constraint_holds_even_after_weights_are_perturbed():
    conv = BayarConv()
    with torch.no_grad():
        conv.weight.add_(torch.randn_like(conv.weight))
    w = conv.constrained_weight()
    assert torch.allclose(w[:, :, 2, 2], torch.full((3, 3), -1.0), atol=1e-5)
    assert torch.allclose(w.sum(dim=(2, 3)) - w[:, :, 2, 2], torch.ones(3, 3), atol=1e-4)


def test_dct_matches_scipy_and_keeps_block_correspondence():
    from scipy.fft import dctn
    x = _batch(1)
    out = BlockDCT()(x)
    assert out.shape == (1, 64, H // 8, W // 8)
    luma = (0.299 * x[0, 0] + 0.587 * x[0, 1] + 0.114 * x[0, 2]) * 255 - 128
    ref = dctn(luma[8:16, 16:24].numpy(), norm="ortho")          # block row 1, col 2
    got = out[0, :, 1, 2]
    comp = torch.sign(torch.tensor(ref)) * torch.log1p(torch.tensor(ref).abs()) / 5
    assert torch.allclose(got, comp.flatten().float(), atol=1e-3)


def test_dct_spatial_locality():
    # Changing one 8x8 block must change only that block's cell.
    x = _batch(1)
    y = x.clone()
    y[:, :, 8:16, 16:24] += 0.3
    d = (BlockDCT()(x) - BlockDCT()(y)).abs().sum(1)[0]
    assert d[1, 2] > 0 and d.sum() == d[1, 2]


def test_branches_output_contract_and_gradients():
    for net in (VisualBranch(num_classes=3), ForensicBranch(num_classes=3)):
        out = net(_batch())
        assert out["logits"].shape == (4, 3)
        assert out["mask_logits"].shape == (4, 1, H, W)
        (out["logits"].sum() + out["mask_logits"].sum()).backward()
        dead = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"no gradient reached: {dead[:5]}"


def test_forensic_branch_has_no_gradient_into_fixed_filters():
    net = ForensicBranch()
    assert not any(p.requires_grad for p in net.srm.parameters())
    assert not any(p.requires_grad for p in net.dct.parameters())
    assert net.bayar.weight.requires_grad


def test_masked_seg_loss_ignores_samples_without_mask():
    lg = torch.randn(2, 1, 16, 16, requires_grad=True)
    m = torch.zeros(2, 1, 16, 16)
    m[0, :, 4:8, 4:8] = 1
    both = masked_seg_loss(lg, m, torch.tensor([1, 0]))
    only = masked_seg_loss(lg[:1], m[:1], torch.tensor([1]))
    assert torch.allclose(both, only)
    assert masked_seg_loss(lg, m, torch.tensor([0, 0])).item() == 0.0


def test_kl_is_detached_from_partner_and_has_right_direction():
    a = torch.randn(8, 3, requires_grad=True)
    b = torch.randn(8, 3, requires_grad=True)
    kl_to_partner(a, b).backward()
    assert a.grad is not None and b.grad is None          # partner is a target only
    assert kl_to_partner(a, a.detach()).item() < 1e-6       # zero when they agree
    p, q = F.softmax(b / 2, 1), F.softmax(a / 2, 1)
    manual = (p * (p.log() - q.log())).sum(1).mean() * 4     # KL(partner || student) * T^2
    assert torch.allclose(kl_to_partner(a, b), manual, atol=1e-5)


def test_mutual_losses_keep_ground_truth_term_when_alignment_is_off():
    a, b = torch.randn(8, 3, requires_grad=True), torch.randn(8, 3, requires_grad=True)
    y = torch.randint(0, 3, (8,))
    la, lb, _ = mutual_losses({"logits": a}, {"logits": b}, y, lam_align=0.0)
    assert torch.allclose(la, F.cross_entropy(a, y)) and torch.allclose(lb, F.cross_entropy(b, y))


def test_tiny_batch_overfit_is_a_software_check_only():
    # Can the forensic branch drive a 4-image batch's loss down? Says nothing about generalisation.
    net = ForensicBranch(num_classes=2)
    x = _batch(4)
    y = torch.tensor([0, 1, 0, 1])
    m = torch.zeros(4, 1, H, W)
    m[1, :, 16:40, 16:40] = 1
    m[3, :, 8:24, 30:50] = 1
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    first = None
    for _ in range(60):
        out = net(x)
        loss = F.cross_entropy(out["logits"], y) + masked_seg_loss(out["mask_logits"], m, y)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first


def test_fusion_head_gradients_reach_both_branches_and_gate_starts_even():
    import torch
    from hybrid_forensics.models import ForensicBranch, FusionHead, VisualBranch
    torch.manual_seed(0)
    a, b = VisualBranch(2), ForensicBranch(2)
    x = torch.rand(2, 3, 64, 64)
    for kind in ("concat", "gate"):
        head = FusionHead(kind)
        oa, ob = a(x), b(x)
        z = head(oa, ob)
        assert z.shape == (2, 2)
        z.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.parameters()), kind
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.parameters()), kind
        a.zero_grad(); b.zero_grad()
    a.eval(); b.eval()  # dropout and batch-norm would make the two forward passes differ
    g = FusionHead("gate")
    out = g(a(x), b(x))
    assert torch.allclose(g.last_gate, torch.full_like(g.last_gate, 0.5))
    # at initialisation the gate output equals the fixed 0.5 / 0.5 mix of the branch logits
    assert torch.allclose(out, 0.5 * a(x)["logits"] + 0.5 * b(x)["logits"], atol=1e-5)
