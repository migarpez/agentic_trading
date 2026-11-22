import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from agentic_trading.model.metrics import (
    baseline_mix,
    bucket_test,
    daily_ic,
    fit_har,
    nw_tstat,
    predict_har,
    qlike,
    rank_residual,
    sign_by_confidence,
    success_checks,
    vs_mix,
)


def panel(days=200, n=50, signal=0.3, seed=0):
    rng = np.random.default_rng(seed)
    d = np.repeat(pd.bdate_range("2010-01-01", periods=days), n)
    p = rng.normal(size=days * n)
    y = signal * p + rng.normal(size=days * n)
    return d, p, y


def test_daily_ic_equals_spearman():
    d, p, y = panel()
    ic = daily_ic(d, p, y)
    day = d[0]
    m = d == day
    assert ic[day] == pytest.approx(spearmanr(p[m], y[m]).statistic, rel=1e-9)
    assert len(ic) == 200 and ic.mean() > 0.2


def test_nw_tstat_iid_close_to_classic():
    x = np.random.default_rng(1).normal(0.1, 1, 5000)
    m, t = nw_tstat(x, 5)
    assert m == pytest.approx(x.mean())
    assert t == pytest.approx(x.mean() / (x.std() / np.sqrt(len(x))), rel=0.1)


def test_nw_penalizes_overlap():
    e = np.random.default_rng(2).normal(size=5020)
    x = np.convolve(e, np.ones(20) / 20, "valid") + 0.05   # medias solapadas de 20
    _, t_nw = nw_tstat(x, 20)
    _, t_naive = nw_tstat(x, 0)
    assert t_nw < t_naive / 2


def test_bucket_and_sign():
    d, p, y = panel(signal=0.5)
    b = bucket_test(d, p, y, lags=1)
    assert all(np.diff(b["buckets"]) > 0) and b["t"] > 3
    s = np.abs(np.random.default_rng(3).normal(size=len(p)))
    acc = sign_by_confidence(p, s, y)
    assert len(acc) == 5 and min(acc) > 0.5


def test_har_recovers_and_qlike_minimum():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20000, 3))
    lv = 0.1 + x @ np.array([0.2, 0.3, 0.4]) + rng.normal(0, 0.3, 20000)
    v = np.exp(lv / 2)[:, None].repeat(3, 1)
    har = fit_har(x, v, np.ones_like(v, bool))
    np.testing.assert_allclose(har["coefs"][0][0], [0.1, 0.2, 0.3, 0.4], atol=0.02)
    assert qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(0.0)
    assert predict_har(har, x).shape == (20000, 3)


def test_baseline_mix_by_hand():
    # Día 1: reversión 3,1,2 → rangos 1, 1/3, 2/3; momentum 1,2,3 → 1/3, 2/3, 1.
    # Día 2 (2 activos): reversión 5,−5 → 1, 1/2; momentum 0,9 → 1/2, 1.
    d = np.array(["a", "a", "a", "b", "b"])
    rev = np.array([3.0, 1.0, 2.0, 5.0, -5.0])
    mom = np.array([1.0, 2.0, 3.0, 0.0, 9.0])
    np.testing.assert_allclose(baseline_mix(d, rev, mom),
                               [2 / 3, 1 / 2, 5 / 6, 3 / 4, 3 / 4])


def test_nw_tstat_by_hand_lags():
    x = np.array([1.0, 2.0, 0.0, 3.0])            # media 1,5; e = −0,5, 0,5, −1,5, 1,5
    # γ0 = 5/4; γ1 = (−0,25 − 0,75 − 2,25)/4 = −13/16; γ2 = (0,75 + 0,75)/4 = 3/8
    g0, g1, g2 = 5 / 4, -13 / 16, 3 / 8
    for lags, var in ((0, g0), (1, g0 + 2 * (1 / 2) * g1),
                      (2, g0 + 2 * (2 / 3) * g1 + 2 * (1 / 3) * g2)):
        m, t = nw_tstat(x, lags)
        assert m == pytest.approx(1.5)
        assert t == pytest.approx(1.5 / np.sqrt(var / 4))


def test_vs_mix_uses_given_lags():
    e = np.random.default_rng(5).normal(size=1020)
    days = pd.bdate_range("2010-01-01", periods=1000)
    mix = pd.Series(0.02, index=days)
    model = pd.Series(np.convolve(e, np.ones(21) / 21, "valid")[:1000] * 0.05 + 0.021,
                      index=days)
    for lags in (5, 10):
        r = vs_mix(model, mix, lags)
        m, t = nw_tstat((model - mix).to_numpy(), lags)
        assert r["dif. IC con la mezcla"] == pytest.approx(m)
        assert r["t dif. IC mezcla"] == pytest.approx(t)
    assert vs_mix(model, mix, 10)["t dif. IC mezcla"] < vs_mix(model, mix, 0)["t dif. IC mezcla"]


def fake_report(dif=None, t=None, har=True, conf=(0.55, 0.5)):
    """Informe mínimo para success_checks: dif/t por horizonte de μ (5 y 10)."""
    dif, t = dif or {5: 0.0, 10: 0.0}, t or {5: 0.0, 10: 0.0}
    rep = {}
    for h in (5, 10, 20):
        rep[h] = {"IC modelo": 0.04, "IC mezcla": 0.03, "QLIKE modelo": 0.2,
                  "signo por confianza": list(conf)}
        if h in dif:
            rep[h].update({"dif. IC con la mezcla": dif[h], "t dif. IC mezcla": t[h]})
        if har:
            rep[h]["QLIKE HAR"] = 0.25
    return rep


@pytest.mark.parametrize("dif, t, ok", [
    ({5: 0.02, 10: 0.0}, {5: 2.5, 10: 0.0}, True),       # pasa a h = 5
    ({5: -0.01, 10: 0.02}, {5: -1.0, 10: 2.1}, True),    # pasa a h = 10
    ({5: 0.02, 10: 0.02}, {5: 1.9, 10: 1.99}, False),    # positiva pero t ≤ 2
    ({5: 0.02, 10: 0.02}, {5: 2.0, 10: 2.0}, False),     # t = 2 no basta (> 2)
    ({5: -0.02, 10: -0.02}, {5: 3.0, 10: 3.0}, False),   # t alto con diferencia negativa
    ({5: np.nan, 10: np.nan}, {5: np.nan, 10: np.nan}, False),
])
def test_success_checks_mu(dif, t, ok):
    c = success_checks(fake_report(dif, t))
    assert c["μ"]["ok"] is ok
    assert set(c["μ"]["por horizonte"]) == {5, 10}
    r5 = c["μ"]["por horizonte"][5]
    assert {"IC μ", "IC mezcla", "dif.", "t NW"} <= set(r5)


def test_success_checks_mu_ignores_h20():
    rep = fake_report()
    rep[20].update({"dif. IC con la mezcla": 0.05, "t dif. IC mezcla": 9.0})
    assert not success_checks(rep)["μ"]["ok"]


def test_success_checks_without_confidence():
    good = ({5: 0.02, 10: 0.0}, {5: 2.5, 10: 0.0})
    a = success_checks(fake_report(*good, conf=(0.55, 0.5)))
    b = success_checks(fake_report(*good, conf=(0.4, 0.6)))   # acierto al revés
    assert set(a) == {"μ", "σ"}
    assert a == b
    rep = fake_report(*good)
    for r in rep.values():
        del r["signo por confianza"]
    assert success_checks(rep) == a


def test_success_checks_sigma():
    assert success_checks(fake_report())["σ"]["ok"]
    assert not success_checks(fake_report(har=False))["σ"]["ok"]   # sin HAR, σ falla
    rep = fake_report()
    rep[20]["QLIKE HAR"] = 0.19                                     # peor que HAR a h = 20
    assert not success_checks(rep)["σ"]["ok"]


def test_rank_residual_removes_baseline():
    d, p, _y = panel(days=300, n=60, signal=0.0, seed=1)
    rng = np.random.default_rng(2)
    z = rng.normal(size=len(p))
    x = 2 * z + p                                        # x = parte de z + parte propia (p)
    e = rank_residual(d, x, z)
    rz = pd.Series(z).groupby(d).rank(pct=True)
    cz = rz - rz.groupby(d).transform("mean")
    np.testing.assert_allclose((e * cz).groupby(d).sum(), 0.0, atol=1e-9)  # MCO: ⟂ rangos de z
    assert daily_ic(d, x, z).mean() > 0.8 and abs(daily_ic(d, e, z).mean()) < 0.1
    assert daily_ic(d, e, p).mean() > 0.3                # conserva la parte propia
