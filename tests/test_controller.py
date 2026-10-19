from hybrid_forensics.controller import TransitionController


def stats(**kw):
    s = dict(auroc_a=0.8, auroc_b=0.75, train_loss=0.5, complementary=0.1, disagreement=0.1, ece_fused=0.05)
    s.update(kw)
    return s


def run(c, seq):
    return [c.update(e, s) for e, s in enumerate(seq)]


def test_adaptive_waits_for_warmup_and_patience_then_ramps():
    c = TransitionController("adaptive")
    g = run(c, [stats()] * 9)
    assert g[:4] == [0.0] * 4                    # epochs 0..3: warm-up / patience not met
    assert c.trigger_epoch is not None and c.trigger_epoch >= 4
    assert 0 < g[c.trigger_epoch - 1] < 1 <= g[-1]   # gradual ramp reaches 1
    assert all(b >= a for a, b in zip(g, g[1:]))


def test_adaptive_is_never_forced_when_conditions_fail():
    c = TransitionController("adaptive")
    g = run(c, [stats(complementary=0.0)] * 12)   # models never make complementary errors
    assert c.trigger_epoch is None and set(g) == {0.0}


def test_unstable_validation_blocks_transition():
    c = TransitionController("adaptive")
    seq = [stats(auroc_a=0.6 + 0.1 * (e % 2)) for e in range(10)]   # val AUROC oscillates by 0.1
    run(c, seq)
    assert c.trigger_epoch is None


def test_sudden_jumps_to_one_and_fixed_ignores_stats():
    s = TransitionController("sudden"); run(s, [stats()] * 8)
    assert s.log[-1]["gamma_next"] == 1.0
    f = TransitionController("fixed", fixed_epoch=3); run(f, [stats(complementary=0.0)] * 6)
    assert f.trigger_epoch == 3                   # fixed schedule does not look at validation


def test_off_never_triggers():
    c = TransitionController("off"); run(c, [stats()] * 8)
    assert c.trigger_epoch is None
