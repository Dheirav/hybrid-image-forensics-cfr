"""Hybrid RGB + forensic image manipulation detection with mutual learning and CFR.

Conventions used everywhere in this package:
  * images are float tensors in [0, 1], RGB, shape (B, 3, H, W), H and W divisible by 8
  * masks are (B, 1, H, W), 1 = manipulated pixel, 0 = untouched
  * a sample without a trustworthy mask has has_mask = 0 and is ignored by the mask loss
"""
