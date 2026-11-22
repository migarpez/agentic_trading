"""Opciones de mu_tree (22-11-2026): market, target_rank, xsec y tail_weight.

La referencia bit a bit (fixtures/mu_options_off_ref.npz) se generó con el código de antes de
añadir las opciones (22-11-2026, PyTorch 2.14.1 en CPU, esta máquina), con la configuración de
final20 en pequeño (MU_REF_CONFIG). En otra CPU u otra versión de PyTorch puede cambiar en los
últimos bits sin que el código haya cambiado.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.test_coarse import synthetic_data

REF = Path(__file__).parent / "fixtures" / "mu_options_off_ref.npz"
MU_REF_CONFIG = {"epochs": 2, "epoch_frac": 0.5, "batch": 128, "threads": 1, "target": "rel",
                 "window": 63, "coarse": "month", "lam": 1.0}


@pytest.fixture
def rel1(monkeypatch):
    """L_μ con cualquier número de muestras por fecha (3 activos sintéticos)."""
    import agentic_trading.model.train as tr
    monkeypatch.setattr(tr, "REL_MIN_SAMPLES", 1)
    return tr


def with_market(tmp_path):
    """Dataset sintético con regiones (A, B: EE. UU.; C: Europa) y vela del mercado."""
    from agentic_trading.data.market import build_market
    data, frames = synthetic_data(tmp_path / "ds")
    clean = tmp_path / "clean"
    clean.mkdir()
    for s, df in frames.items():
        df.rename_axis("date").to_parquet(clean / f"{s}.parquet")
    uni = pd.DataFrame({"symbol": ["A", "B", "C"], "asset_class": "equity",
                        "country": ["US", "US", "DE"]})
    build_market(data, clean, uni, min_stocks=1)
    data._regions = np.array([0, 0, 1])
    return data, frames


def test_options_off_bit_identical(tmp_path, rel1):
    ref = np.load(REF)
    data, _ = synthetic_data(tmp_path)
    idx = np.arange(len(data.samples))
    model, hist = rel1.train_model(data, idx, rel1.TrainConfig(**MU_REF_CONFIG), seed=7,
                                   log=lambda *_: None)
    out = {f"hist_{k}": np.array([r[k] for r in hist]) for k in ("loss", "l_mu", "l_sigma")}
    out["drift"] = np.asarray(model.drift)
    p = rel1.predict(model, data, idx)
    for c in p.columns.drop("sample"):
        out[c] = p[c].to_numpy()
    assert sorted(out) == sorted(ref.files)
    for k in ref.files:
        assert out[k].dtype == ref[k].dtype and np.array_equal(out[k], ref[k]), k


def test_date_ranks_and_tail_weights(tmp_path):
    from scipy.stats import norm
    data, _ = synthetic_data(tmp_path)
    idx = np.flatnonzero(data.samples["asset"].to_numpy() != 2)[::2]      # una parte
    z, n = data.date_ranks(idx)
    rows = data.samples["row"].to_numpy()
    out = np.setdiff1d(np.arange(len(data.samples)), idx)
    assert np.isnan(z[out]).all() and (n[out] == 0).all()
    for d, g in data.samples.iloc[idx].groupby("date"):
        i = g.index.to_numpy()
        for j, h in enumerate(data.horizons):
            y = np.asarray(data.arr[f"y_{h}"])[rows[i]]
            r = pd.Series(y).rank().to_numpy()
            np.testing.assert_allclose(z[i, j], norm.ppf((r - 0.5) / len(i)), rtol=1e-5)
        assert (n[i] == len(i)).all()
    # pesos de la cola: media 1 por fecha; el decil alto pesa 3 veces más que el resto
    big = np.arange(len(data.samples))
    w = data.tail_weights(big, 3.0)
    dates = data.samples["date"].to_numpy()
    means = pd.DataFrame(w).groupby(dates).mean().to_numpy()
    np.testing.assert_allclose(means, 1.0, rtol=1e-5)
    for _, g in data.samples.groupby("date"):
        i = g.index.to_numpy()
        for j in range(3):
            vals = np.unique(np.round(w[i, j], 5))
            assert len(vals) <= 2 and (len(vals) == 1 or np.isclose(vals[1] / vals[0], 3.0))


def test_market_rows_mean_flat_and_end():
    from agentic_trading.data.market import market_rows
    d = pd.bdate_range("2019-12-23", periods=6)          # 23-12-2019 … 30-12-2019
    rng = np.random.default_rng(0)

    def frame(dates, valid=None):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        df = pd.DataFrame({"open": c * 1.001, "high": c * 1.01, "low": c * 0.99, "close": c},
                          index=dates)
        df["valid"] = True if valid is None else valid
        return df

    f = [frame(d), frame(d), frame(d[[0, 1, 2, 4, 5]]), frame(d.append(pd.DatetimeIndex(
        ["2020-01-02"])))]
    f[1].loc[d[3], "valid"] = False
    regions, stock = np.array([0, 0, 1, 0]), np.array([True, True, True, False])
    m = market_rows(f, regions, stock, end="2019-12-31", min_stocks=1)
    n = [len(x) for x in f]
    off = np.r_[0, np.cumsum(n)]

    def lr(df, k, t):
        return np.log(df[k].iloc[t] / df["close"].iloc[t - 1])

    mc = m["m_log_close"][off[0]:off[1]]
    # día 2 (US): media de A y B; día 3: solo A (B no válida); día 4: solo A (B, sin día anterior)
    np.testing.assert_allclose(mc[2] - mc[1], np.mean([lr(f[0], "close", 2),
                                                       lr(f[1], "close", 2)]))
    np.testing.assert_allclose(mc[3] - mc[2], lr(f[0], "close", 3))
    np.testing.assert_allclose(mc[4] - mc[3], lr(f[0], "close", 4))
    mh = m["m_log_high"][off[0]:off[1]]
    np.testing.assert_allclose(mh[2] - mc[1], np.mean([lr(f[0], "high", 2),
                                                       lr(f[1], "high", 2)]))
    # B recibe la misma vela que A en todas sus filas (también en la suya no válida)
    np.testing.assert_array_equal(m["m_log_close"][off[1]:off[2]], mc)
    # Europa (solo C): el 26-12 C no cotiza y su día siguiente usa el cierre del 25 como anterior
    ce = m["m_log_close"][off[2]:off[3]]
    np.testing.assert_allclose(ce[3] - ce[2], np.log(f[2]["close"].iloc[3] / f[2]["close"].iloc[2]))
    # el no acción (cripto) recibe la de EE. UU. y nada después del fin del desarrollo
    cr = {k: v[off[3]:off[4]] for k, v in m.items()}
    np.testing.assert_array_equal(cr["m_log_close"][:6], mc)
    assert all(np.isnan(v[6]) for v in cr.values())
    # día sin vela de la región: plana en el último cierre (cripto, fin de semana)
    g = [frame(d[:2]), frame(pd.DatetimeIndex([d[0], d[1], pd.Timestamp("2019-12-28")]))]
    m2 = market_rows(g, np.array([0, 0]), np.array([True, False]), min_stocks=1)
    for k in ("m_log_open", "m_log_high", "m_log_low", "m_log_close"):
        assert m2[k][4] == m2["m_log_close"][1]


def test_batch_market_channels(tmp_path):
    data, _ = with_market(tmp_path)
    idx = np.arange(len(data.samples))[::50]
    plain = data.batch(idx, window=63, coarse="month")
    b = data.batch(idx, window=63, coarse="month", market=True)
    assert b["x"].shape == (len(idx), 63, 9) and b["xc"].shape == (len(idx), 12, 9)
    np.testing.assert_array_equal(b["x"][:, :, :5], plain["x"])
    np.testing.assert_array_equal(b["xc"][:, :, :5], plain["xc"])
    assert (b["x"][:, -1, 8] == 0).all() and (b["xc"][:, -1, 8] == 0).all()
    rows = data.samples["row"].to_numpy()[idx]
    m, sig = data.market, np.asarray(data.arr["sigma"])[rows]
    i = 3
    ref = (np.asarray(m["m_log_high"])[rows[i] - 62:rows[i] + 1]
           - np.asarray(m["m_log_close"])[rows[i]]) / sig[i]
    np.testing.assert_allclose(b["x"][i, :, 6], ref, rtol=1e-5)
    hi = np.asarray(m["m_log_high"])[rows[i] - 251:rows[i] + 1].reshape(12, 21).max(1)
    np.testing.assert_allclose(b["xc"][i, :, 6], (hi - np.asarray(m["m_log_close"])[rows[i]])
                               / sig[i], rtol=1e-5)


def test_group_mean_and_xsec_head():
    from agentic_trading.model.tcn import TCN, group_mean
    h = torch.arange(12.0).reshape(6, 2)
    g = torch.tensor([3, 1, 3, 1, 1, 7])
    out = group_mean(h, g)
    np.testing.assert_allclose(out[0], h[[0, 2]].mean(0))
    np.testing.assert_allclose(out[1], h[[1, 3, 4]].mean(0))
    np.testing.assert_allclose(out[5], h[5])
    torch.manual_seed(0)
    net = TCN(dilations=(1, 2, 4, 8, 16), coarse="month", market=True, xsec=True)
    x, s, xc = torch.randn(6, 63, 9), torch.randn(6, 1), torch.randn(6, 12, 9)
    mu, _ = net(x, s, xc, group=g)
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    mu2, _ = net(x[perm], s[perm], xc[perm], group=g[perm])
    torch.testing.assert_close(mu2, mu[perm])                  # no depende del orden
    with pytest.raises(ValueError):
        net(x, s, xc)


def test_joint_loss_weights():
    from agentic_trading.model.tcn import joint_loss
    torch.manual_seed(0)
    mu, logv, y = torch.randn(8, 3), torch.randn(8, 3), torch.randn(8, 3)
    v, m = torch.rand(8, 3) + 0.5, torch.ones(8, 3, dtype=torch.bool)
    base = joint_loss(mu, logv, y, v, m)
    same = joint_loss(mu, logv, y, v, m, weight_mu=torch.ones(8, 3),
                      horizon_w=[1 / 5, 1 / 10, 1 / 20])
    torch.testing.assert_close(base[1], same[1])
    w = torch.rand(8, 3)
    _, l_mu, _ = joint_loss(mu, logv, y, v, m, weight_mu=w, horizon_w=[1, 1, 1])
    torch.testing.assert_close(l_mu, (w * (mu - y) ** 2).mean(0).sum())


@pytest.mark.parametrize("opts", [{"market": True}, {"target_rank": True}, {"xsec": True},
                                  {"tail_weight": 3.0},
                                  {"market": True, "target_rank": True, "xsec": True,
                                   "tail_weight": 3.0}])
def test_options_train_and_predict(tmp_path, rel1, opts):
    data, _ = with_market(tmp_path)
    idx = np.arange(len(data.samples))
    cfg = rel1.TrainConfig(**{**MU_REF_CONFIG, "epochs": 1, **opts})
    model, hist = rel1.train_model(data, idx, cfg, seed=3, log=lambda *_: None)
    assert np.isfinite([r["loss"] for r in hist]).all()
    p = rel1.predict(model, data, idx[::3])
    assert np.isfinite(p.drop(columns="sample").to_numpy()).all()
    assert (p["sample"].to_numpy() == idx[::3]).all()
    if opts.get("target_rank"):
        assert getattr(model, "drift", None) is None
    # checkpoint: mismas predicciones al cargar
    path = tmp_path / "m.pt"
    rel1.save_model(model, path, cfg, 3, hist)
    m2, _ = rel1.load_model(path)
    pd.testing.assert_frame_equal(rel1.predict(m2, data, idx[::3]), p)
    if opts.get("xsec"):
        # con lotes de días enteros, el tamaño del lote no cambia nada
        q = rel1.predict(model, data, idx[::3], batch=7)
        np.testing.assert_allclose(q["mu_5"], p["mu_5"], rtol=1e-5, atol=1e-6)
        # al barajar las muestras, cada una conserva su predicción
        sh = np.random.default_rng(0).permutation(idx[::3])
        r = rel1.predict(model, data, sh).set_index("sample").loc[idx[::3]]
        np.testing.assert_allclose(r["mu_5"], p["mu_5"], rtol=1e-5, atol=1e-6)


def test_xsec_batches_are_whole_dates(tmp_path, rel1, monkeypatch):
    data, _ = with_market(tmp_path)
    idx = np.arange(len(data.samples))
    seen = []
    orig = rel1._batch

    def spy(d, b_idx, g):
        seen.append(np.asarray(b_idx).copy())
        return orig(d, b_idx, g)

    monkeypatch.setattr(rel1, "_batch", spy)
    cfg = rel1.TrainConfig(**{**MU_REF_CONFIG, "epochs": 1, "xsec": True, "bootstrap": False})
    rel1.train_model(data, idx, cfg, seed=0, log=lambda *_: None)
    dates = data.samples["date"].to_numpy()
    n_dates = len(np.unique(dates))
    assert len(seen) == round(n_dates * cfg.epoch_frac)
    for b in seen:
        assert len(np.unique(dates[b])) == 1
        assert len(b) == (dates == dates[b[0]]).sum() and len(np.unique(b)) == len(b)


# ----------------------------------------------------------------------------- métricas

def tiny_market():
    from agentic_trading.agent.simulator import Market
    dates = pd.bdate_range("2016-01-04", periods=8)
    op = np.array([[10, 20, 30], [11, 20, np.nan], [12, 22, np.nan], [13, 22, np.nan],
                   [14, 24, 33], [15, 24, 36], [16, 26, 39], [17, 26, 42]], float)
    return Market(dates, ["A", "B", "C"], op, op, op, op)


def test_open_prices_returns():
    from agentic_trading.model.mu_metrics import OpenPrices
    op = OpenPrices(tiny_market())
    d = op.market.dates
    cols = np.array([0, 1, 2])
    r = op.returns(d[0], d[3], cols)           # apertura de d[1] a la de d[4]
    np.testing.assert_allclose(r, np.log([14 / 11, 24 / 20, 33 / 33]))
    r = op.returns(d[5], d[7], cols)           # sin vela de salida: NaN
    assert np.isnan(r).all()
    r = op.returns(d[1], d[2], np.array([2]))  # sin velas entre medias: misma vela, 0
    assert r[0] == 0.0


def test_tail_series_and_paired():
    from agentic_trading.model.mu_metrics import compare_rule, paired, tail_series
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2001-01-05", periods=300, freq="W-FRI")
    n = 50
    wf = pd.DataFrame({"date": np.repeat(dates, n), "ret_oo": rng.normal(0, 0.02, 300 * n)})
    wf["ret_cc"] = wf["ret_oo"]
    good = wf["ret_oo"].to_numpy() + rng.normal(0, 0.01, len(wf))     # informativa
    s = tail_series(wf, good, n=20)
    g = wf.assign(mu=good)
    top = g[g.groupby("date")["mu"].rank(ascending=False, method="first") <= 20]
    ref = 1e4 * (top.groupby("date")["ret_oo"].mean() - wf.groupby("date")["ret_oo"].mean())
    pd.testing.assert_series_equal(s["top20_oo"], ref, check_names=False)
    assert (s["top20_oo"] > 0).mean() > 0.9
    bad = tail_series(wf, rng.normal(size=len(wf)), n=20)
    p = paired(s["top20_oo"], bad["top20_oo"])
    assert p["mean"] > 0 and p["t"] > 5 and p["n"] == 300
    assert sum(b["n"] for b in p["blocks"].values()) == 300            # 2001–2006: cv0, cv1
    std = {"total": {"IC": 0.04, "QLIKE": 0.20}}
    v = compare_rule(p, {"total": {"IC": 0.039, "QLIKE": 0.2019}}, std)
    # solo hay semanas en dos bloques: falla la condición de los bloques (y 2016–19 sin datos)
    assert v["conditions"]["1_top20_total"] and not v["conditions"]["2_top20_blocks"]
    assert v["conditions"]["3_ic"] and v["conditions"]["4_qlike"] and not v["wins"]
    v = compare_rule(p, {"total": {"IC": 0.0379, "QLIKE": 0.2021}}, std)
    assert not v["conditions"]["3_ic"] and not v["conditions"]["4_qlike"]


def test_compare_rule_blocks():
    from agentic_trading.model.mu_metrics import BLOCK_NAMES, compare_rule
    std = {"total": {"IC": 0.04, "QLIKE": 0.2}}

    def top(means, mean=1.0, t=1.0):
        return {"mean": mean, "t": t, "blocks": {b: {"mean": m} for b, m in zip(BLOCK_NAMES, means)}}
    assert compare_rule(top([1, 1, 1, -1, 0]), std, std)["wins"]           # 3 de 5, 2016–19 = 0
    assert not compare_rule(top([1, 1, 1, 1, -0.1]), std, std)["wins"]     # 2016–19 negativo
    assert compare_rule(top([1, 1, -1, -1, 1]), std, std)["wins"]          # 3 de 5
    assert not compare_rule(top([1, -1, -1, -1, 1]), std, std)["wins"]     # 2 de 5
    assert compare_rule(top([1, -1, -1, 1, 1]), std, std)["wins"]
    assert not compare_rule(top([1, 1, 1, 1, 1], t=0.99), std, std)["wins"]
    assert not compare_rule(top([1, 1, 1, 1, 1], mean=0.0, t=5), std, std)["wins"]


def test_recency_weights_and_training(tmp_path, rel1):
    from agentic_trading.model.tcn import joint_loss
    data, _ = synthetic_data(tmp_path)
    idx = np.arange(len(data.samples))[::2]
    w = rel1.recency_weights(data.samples, idx, 2.0)
    assert np.isnan(np.delete(w, idx)).all()
    np.testing.assert_allclose(w[idx].mean(), 1.0, rtol=1e-5)
    d = data.samples["date"].to_numpy()[idx]
    last, two = d == d.max(), d <= d.max() - np.timedelta64(731, "D")
    assert w[idx][two].max() <= 0.5 * w[idx][last].min() * 1.001        # 2 años: la mitad o menos
    torch.manual_seed(0)
    mu, logv, y = torch.randn(8, 3), torch.randn(8, 3), torch.randn(8, 3)
    v, m = torch.rand(8, 3) + 0.5, torch.ones(8, 3, dtype=torch.bool)
    base = joint_loss(mu, logv, y, v, m)
    one = joint_loss(mu, logv, y, v, m, weight=torch.ones(8))
    torch.testing.assert_close(base[0], one[0])
    cfg = rel1.TrainConfig(**{**MU_REF_CONFIG, "epochs": 1, "recency_halflife": 4.0})
    _model, hist = rel1.train_model(data, idx, cfg, seed=1, log=lambda *_: None)
    assert np.isfinite(hist[0]["loss"])
