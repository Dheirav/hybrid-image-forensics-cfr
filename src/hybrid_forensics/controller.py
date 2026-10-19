"""Validation-driven transition from mutual learning to CFR.

The criteria are declared here, before any run, and the controller only ever reads validation
statistics. It never forces a transition: if the conditions are not met the run stays in mutual
learning and the log says why.
"""
from dataclasses import dataclass, field

DEFAULTS = dict(min_warmup=3, patience=2, ramp_epochs=4, min_auroc=0.60, max_auroc_delta=0.03,
                max_loss_increase=0.10, min_complementary=0.03, max_disagreement=0.35, max_ece=0.20)


@dataclass
class TransitionController:
    kind: str                       # "adaptive" | "sudden" | "fixed" | "off"
    fixed_epoch: int = 6            # used by kind == "fixed"
    cfg: dict = field(default_factory=lambda: dict(DEFAULTS))
    trigger_epoch: int | None = None
    _streak: int = 0
    _prev: dict | None = None
    log: list = field(default_factory=list)

    def eligibility(self, epoch, s):
        c, p = self.cfg, self._prev
        checks = {
            "past_warmup": epoch >= c["min_warmup"],
            "both_learn_signal": min(s["auroc_a"], s["auroc_b"]) >= c["min_auroc"],
            "val_stable": p is not None and max(abs(s["auroc_a"] - p["auroc_a"]), abs(s["auroc_b"] - p["auroc_b"])) <= c["max_auroc_delta"],
            "training_stable": s["train_loss"] == s["train_loss"] and (p is None or s["train_loss"] <= p["train_loss"] * (1 + c["max_loss_increase"])),
            "complementary_errors": s["complementary"] >= c["min_complementary"],
            "aligned_not_collapsed": s["disagreement"] <= c["max_disagreement"],
            "calibrated_enough": s["ece_fused"] <= c["max_ece"],
        }
        return checks

    def update(self, epoch, stats):
        """Call once per epoch after validation; returns the CFR weight gamma for the NEXT epoch."""
        checks = self.eligibility(epoch, stats)
        eligible = all(checks.values())
        if self.kind == "off":
            gamma, trig = 0.0, False
        elif self.kind == "fixed":
            if self.trigger_epoch is None and epoch + 1 >= self.fixed_epoch:
                self.trigger_epoch = epoch + 1
            trig = self.trigger_epoch is not None
        else:
            self._streak = self._streak + 1 if eligible else 0
            if self.trigger_epoch is None and self._streak >= self.cfg["patience"]:
                self.trigger_epoch = epoch + 1
            trig = self.trigger_epoch is not None
        if self.kind != "off":
            if not trig:
                gamma = 0.0
            elif self.kind == "sudden":
                gamma = 1.0
            else:
                gamma = min(1.0, (epoch + 1 - self.trigger_epoch + 1) / self.cfg["ramp_epochs"])
        self.log.append({"epoch": epoch, "eligible": eligible, "checks": checks, "triggered": trig,
                         "trigger_epoch": self.trigger_epoch, "gamma_next": gamma})
        self._prev = stats
        return gamma
