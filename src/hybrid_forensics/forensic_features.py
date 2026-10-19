"""Fixed and constrained forensic transforms that keep pixel-level spatial correspondence.

SRM and DCT are fixed. Bayar is learnable but re-projected onto its constraint on every
forward pass, otherwise the optimiser drifts it back into an ordinary conv that learns image
content instead of residuals.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _srm_kernels() -> torch.Tensor:
    # The three SRM high-pass kernels used by RGB-N style detectors, padded to 5x5.
    k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
    k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    k3 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
                       [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0
    return torch.stack([k1, k2, k3]).unsqueeze(1)  # (3, 1, 5, 5)


class SRMResidual(nn.Module):
    """Fixed SRM residuals, applied per colour channel, truncated to [-t, t]. Output 9 channels."""

    def __init__(self, truncate: float = 2.0):
        super().__init__()
        self.register_buffer("kernels", _srm_kernels().repeat(3, 1, 1, 1))  # (9, 1, 5, 5)
        self.truncate = truncate

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        b, c, h, w = x.shape
        # Work on 0..255 scale so the truncation threshold keeps its usual meaning.
        x = (x * 255.0).reshape(b * c, 1, h, w)
        out = F.conv2d(x, self.kernels[:3], padding=2).reshape(b, c * 3, h, w)
        return out.clamp(-self.truncate, self.truncate) / self.truncate


class BayarConv(nn.Module):
    """Constrained conv: centre weight -1, remaining weights of each filter sum to +1."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, k: int = 5):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * 0.05)

    def constrained_weight(self) -> torch.Tensor:
        w = self.weight
        c = self.k // 2
        mask = torch.ones_like(w)
        mask[:, :, c, c] = 0.0
        w = w * mask
        w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-8)  # off-centre taps sum to 1
        centre = torch.zeros_like(w)
        centre[:, :, c, c] = -1.0
        return w * mask + centre

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return F.conv2d(x * 255.0, self.constrained_weight(), padding=self.k // 2) / 8.0


class BlockDCT(nn.Module):
    """8x8 block DCT of luma. Output (B, 64, H/8, W/8): one coefficient per channel.

    Resolution drops by 8 on purpose: each output cell is exactly one JPEG block, so the
    forensic branch can inject it at stride 8 without any misalignment.
    """

    def __init__(self):
        super().__init__()
        basis = torch.zeros(64, 1, 8, 8)
        for u in range(8):
            for v in range(8):
                cu = math.sqrt(1 / 8) if u == 0 else math.sqrt(2 / 8)
                cv = math.sqrt(1 / 8) if v == 0 else math.sqrt(2 / 8)
                for i in range(8):
                    for j in range(8):
                        basis[u * 8 + v, 0, i, j] = (
                            cu * cv * math.cos((2 * i + 1) * u * math.pi / 16)
                            * math.cos((2 * j + 1) * v * math.pi / 16))
        self.register_buffer("basis", basis)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        assert x.shape[-1] % 8 == 0 and x.shape[-2] % 8 == 0, "H and W must be divisible by 8"
        luma = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 255.0 - 128.0
        coef = F.conv2d(luma, self.basis, stride=8)
        return torch.sign(coef) * torch.log1p(coef.abs()) / 5.0  # compress the heavy tail
