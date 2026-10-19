"""Software tests of the inference interface with untrained weights (no claim about accuracy)."""
import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hybrid_forensics.infer import InputError, Predictor, validate
from hybrid_forensics.models import ForensicBranch, VisualBranch


@pytest.fixture(scope="module")
def predictor():
    torch.manual_seed(0)
    return Predictor({"A": VisualBranch(2), "B": ForensicBranch(2)})


def _img(tmp_path, name, w, h, ext=".png"):
    p = tmp_path / (name + ext)
    cv2.imwrite(str(p), np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))
    return str(p)


def test_rejects_unsupported_corrupt_small_large_and_extreme_aspect(tmp_path):
    (tmp_path / "a.gif").write_bytes(b"GIF89a")
    (tmp_path / "bad.jpg").write_bytes(b"not an image at all")
    for path, frag in [(str(tmp_path / "a.gif"), "unsupported"), (str(tmp_path / "bad.jpg"), "decoded"),
                       (str(tmp_path / "missing.png"), "not found"), (_img(tmp_path, "s", 64, 64), "supported side"),
                       (_img(tmp_path, "l", 5000, 200), "supported side"), (_img(tmp_path, "w", 1500, 300), "aspect")]:
        with pytest.raises(InputError, match=frag):
            validate(path)


def test_rejects_truncated_jpeg(tmp_path):
    full = _img(tmp_path, "full", 384, 384, ".jpg")
    (tmp_path / "cut.jpg").write_bytes(open(full, "rb").read()[:3000])
    with pytest.raises(InputError, match="decoded"):
        validate(str(tmp_path / "cut.jpg"))


def test_384_input_matches_the_evaluation_path(predictor, tmp_path):
    p = _img(tmp_path, "t", 384, 384)
    r = predictor.predict(p)
    x = torch.from_numpy(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda() / 255
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pa = F.softmax(predictor.models["A"](x)["logits"].float(), 1)[0, 1].item()
        pb = F.softmax(predictor.models["B"](x)["logits"].float(), 1)[0, 1].item()
    assert r["windows"] == 1 and abs(r["p_manipulated"] - (pa + pb) / 2) < 1e-6


def test_odd_sizes_give_full_size_heatmap_and_valid_probabilities(predictor, tmp_path):
    for w, h in [(500, 333), (900, 300), (384, 1000), (130, 130)]:
        r = predictor.predict(_img(tmp_path, f"o{w}x{h}", w, h, ".jpg"))
        assert r["heatmap"].shape == (h, w) and r["mask"].shape == (h, w)
        assert 0 <= r["p_manipulated"] <= 1 and abs(r["p_manipulated"] + r["p_authentic"] - 1) < 1e-9
        assert "not proof" in r["limitations"]


def test_tiling_covers_the_image_without_padding_when_it_is_larger_than_a_window():
    from hybrid_forensics.infer import TILE, tile_starts
    for n in (384, 385, 512, 768, 1000, 1152, 4096):
        st = tile_starts(n)
        assert st[0] == 0 and st[-1] + TILE == max(n, TILE), n    # last window ends exactly at the edge
        assert all(b - a <= TILE for a, b in zip(st, st[1:])), n    # no gaps between windows
    assert tile_starts(200) == [0] and tile_starts(384) == [0]


def test_512_px_image_uses_four_windows_and_returns_a_mask_of_image_size(predictor, tmp_path):
    r = predictor.predict(_img(tmp_path, "m", 512, 512))
    assert r["windows"] == 4 and r["heatmap"].shape == (512, 512)
