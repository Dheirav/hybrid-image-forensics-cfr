"""Model A (visual) and Model B (forensic), each with a classifier head and a mask decoder.

Both branches return the same dict so that mutual learning, fusion and the application can
treat them interchangeably.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .forensic_features import BayarConv, BlockDCT, SRMResidual

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class MaskDecoder(nn.Module):
    """Light top-down decoder over four feature maps (strides 4, 8, 16, 32) -> 1-ch logits."""

    def __init__(self, in_chs, width: int = 64):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in in_chs])
        self.smooth = nn.ModuleList([_cbr(width, width) for _ in in_chs[:-1]])
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, feats, out_size):
        x = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[i](x + self.lat[i](feats[i]))
        return F.interpolate(self.out(x), size=out_size, mode="bilinear", align_corners=False)


class VisualBranch(nn.Module):
    """Model A: EfficientNet-B0 on normalised RGB."""

    taps = (2, 3, 5, 8)  # torchvision efficientnet_b0 feature indices at strides 4, 8, 16, 32
    chs = (24, 40, 112, 1280)

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = torchvision.models.efficientnet_b0(weights=w).features
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.chs[-1], num_classes))
        self.decoder = MaskDecoder(self.chs)

    def forward(self, x):
        h = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.taps:
                feats.append(h)
        pooled = feats[-1].mean(dim=(2, 3))
        return {"logits": self.head(pooled), "mask_logits": self.decoder(feats, x.shape[-2:]),
                "embedding": pooled}


class ForensicBranch(nn.Module):
    """Model B: SRM + Bayar residuals at full resolution, block DCT injected at stride 8."""

    chs = (32, 64, 128, 256)

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.srm, self.bayar, self.dct = SRMResidual(), BayarConv(3, 3, 5), BlockDCT()
        self.stem = nn.Sequential(_cbr(9 + 3, 24, 1), _cbr(24, 32, 2), _cbr(32, 32, 2))      # -> stride 4
        self.s8 = nn.Sequential(_cbr(32, 64, 2), _cbr(64, 64))
        self.dct_proj = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fuse8 = _cbr(128, 64)
        self.s16 = nn.Sequential(_cbr(64, 128, 2), _cbr(128, 128))
        self.s32 = nn.Sequential(_cbr(128, 256, 2), _cbr(256, 256))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, num_classes))
        self.decoder = MaskDecoder(self.chs)

    def forward(self, x):
        res = torch.cat([self.srm(x), self.bayar(x)], dim=1)
        f4 = self.stem(res)
        f8 = self.fuse8(torch.cat([self.s8(f4), self.dct_proj(self.dct(x))], dim=1))
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        pooled = f32.mean(dim=(2, 3))
        return {"logits": self.head(pooled), "mask_logits": self.decoder([f4, f8, f16, f32], x.shape[-2:]),
                "embedding": pooled}


class FusionHead(nn.Module):
    """Learned fusion of the two branches (Phase 4 items 3 and 5 of the brief).

    concat: classifier on [LN(e_A), LN(e_B)]. The branch embeddings differ in size (1280 vs 256) and scale, so each is
            layer-normalised first, otherwise the larger one dominates the first layer by construction.
    gate:   input-dependent weights over the two branch logits, g = softmax(MLP([e_A, e_B])), z = g_A z_A + g_B z_B.
            Initialised at 0.5/0.5 so training starts as the fixed-weight fusion and has to learn any departure from it.
    Gradients flow into both branches (embeddings and logits are not detached); the per-branch heads are kept
    under their own cross-entropy by train.py so neither branch can go dead behind the fusion.
    """

    def __init__(self, kind: str, d_a: int = 1280, d_b: int = 256, num_classes: int = 2):
        super().__init__()
        assert kind in ("concat", "gate")
        self.kind = kind
        self.ln_a, self.ln_b = nn.LayerNorm(d_a), nn.LayerNorm(d_b)
        if kind == "concat":
            self.net = nn.Sequential(nn.Linear(d_a + d_b, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, num_classes))
        else:
            self.net = nn.Sequential(nn.Linear(d_a + d_b, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.last_gate = None

    def forward(self, out_a, out_b):
        z = torch.cat([self.ln_a(out_a["embedding"].float()), self.ln_b(out_b["embedding"].float())], dim=1)
        if self.kind == "concat":
            return self.net(z)
        g = torch.softmax(self.net(z), dim=1)
        self.last_gate = g.detach()
        return g[:, :1] * out_a["logits"].float() + g[:, 1:] * out_b["logits"].float()
