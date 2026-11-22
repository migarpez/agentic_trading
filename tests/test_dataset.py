import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.dataset import (
    WindowData,
    asset_arrays,
    asset_info,
    build_dataset,
    dataset_id,
    split_mask,
    yang_zhang,
)
from tests.test_audit import synthetic


def clean_frame(n=800, seed=0):
    df = synthetic(n, seed)[["open", "high", "low", "close", "volume"]]
    df = df.set_axis(pd.bdate_range("2013-01-01", periods=n))
    df["valid"] = True
    df["ohlc_repaired"] = False
    return df


def yz_reference(df, t, n):
    """Yang-Zhang directo, tal como está escrito en el handoff."""
    d = df.iloc[t + 1:t + n + 1]
    prev_c = df["close"].iloc[t:t + n].to_numpy()
    o = np.log(d["open"].to_numpy() / prev_c)
    c = np.log(d["close"] / d["open"]).to_numpy()
    u = np.log(d["high"] / d["open"]).to_numpy()
    lo = np.log(d["low"] / d["open"]).to_numpy()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = o.var(ddof=1) + k * c.var(ddof=1) + (1 - k) * np.mean(u * (u - c) + lo * (lo - c))
    return np.sqrt(var)


def test_targets_match_reference():
    df = clean_frame()
    a = asset_arrays(df)
    lc = np.log(df["close"].to_numpy())
    for t in (251, 400, 779):
        sigma = np.std(np.diff(lc[t - 251:t + 1]), ddof=1)
        assert a["sigma"][t] == pytest.approx(sigma)
        for h in (5, 10, 20):
            assert a[f"y_{h}"][t] == pytest.approx((lc[t + h] - lc[t]) / sigma)
            assert a[f"v_{h}"][t] == pytest.approx(yz_reference(df, t, h) / sigma)
    assert np.isnan(a["sigma"][250])


def test_yang_zhang_scale():
    # Días con salto nocturno σ_o y trayectoria intradía browniana σ_c (1000 pasos; con menos,
    # H y L discretos se quedan cortos y σ_YZ sale sesgada a la baja):
    # σ_YZ debe estimar √(σ_o² + σ_c²), la volatilidad diaria total.
    rng = np.random.default_rng(3)
    days, steps, s_o, s_c = 10000, 1000, 0.006, 0.014
    o_jump = rng.normal(0, s_o, days)
    path = np.cumsum(rng.normal(0, s_c / np.sqrt(steps), (days, steps)), axis=1)
    path = np.c_[np.zeros(days), path]
    lc = np.cumsum(o_jump + path[:, -1])
    lo = lc - path[:, -1]
    lh, ll = lo + path.max(1), lo + path.min(1)
    yz = yang_zhang(lo, lh, ll, lc, 20)
    assert np.nanmean(yz ** 2) ** 0.5 == pytest.approx(np.hypot(s_o, s_c), rel=0.03)


def test_sample_validity_and_sigma_mask():
    df = clean_frame()
    df.iloc[500, df.columns.get_loc("valid")] = False
    df.iloc[300, df.columns.get_loc("ohlc_repaired")] = True
    a = asset_arrays(df)
    ok = a["sample_ok"]
    assert not ok[:251].any() and not ok[-20:].any()
    assert not ok[500 - 20:500 + 252].any()          # la vela 500 en ventana o en target
    assert ok[500 - 21] and ok[500 + 252]
    assert not a["mask_sigma_5"][295] and a["mask_sigma_5"][294] and a["mask_sigma_5"][300]


def test_build_and_batch(tmp_path):
    frames = {"A": clean_frame(800, 1), "B": clean_frame(700, 2)}
    frames["B"].iloc[400:403, frames["B"].columns.get_loc("volume")] = 0.0
    uni = pd.DataFrame({"symbol": list(frames)})
    meta = build_dataset(uni, frames.__getitem__, tmp_path)
    assert meta["n_samples"] == (800 - 271) + (700 - 271)

    data = WindowData(tmp_path)
    b = data.batch(np.arange(len(data.samples)))
    assert b["x"].shape == (meta["n_samples"], 252, 5) and np.isfinite(b["x"]).all()
    np.testing.assert_allclose(b["x"][:, -1, 3], 0.0, atol=1e-6)   # C_t / C_t
    assert b["y"].shape == b["v"].shape == (meta["n_samples"], 3)

    # muestra concreta contra el cálculo directo
    i = 10
    s = data.samples.iloc[i]
    df = frames[meta["symbols"][s.asset]]
    t = df.index.get_loc(s.date)
    w = df.iloc[t - 251:t + 1]
    sigma = np.std(np.diff(np.log(w["close"])), ddof=1)
    np.testing.assert_allclose(b["x"][i, :, 0], np.log(w["open"] / w["close"].iloc[-1]) / sigma,
                               rtol=1e-5)
    med = np.median(w["volume"][w["volume"] > 0])
    np.testing.assert_allclose(b["x"][i, :, 4], np.log(w["volume"] / med), rtol=1e-4, atol=1e-5)
    assert b["scalar"][i, 0] == pytest.approx(np.log(sigma), rel=1e-5)

    # volumen 0 -> v = 0
    sb = data.samples[(data.samples.asset == 1)]
    j = sb.index[(sb.date >= frames["B"].index[402]) & (sb.date <= frames["B"].index[402])][0]
    assert (data.batch(np.array([j]))["x"][0, -1, 4]) == 0.0


def test_split_mask_respects_gap():
    dates = pd.bdate_range("2019-12-01", "2020-01-31")
    s = pd.DataFrame({"date": dates, "target_end": dates + pd.offsets.BDay(20)})
    dev = split_mask(s, "dev")
    assert s.target_end[dev].max() <= pd.Timestamp("2019-12-31")
    assert not split_mask(s, "test")[s.date < "2020-01-01"].any()


def test_test_split_is_locked(tmp_path):
    build_dataset(pd.DataFrame({"symbol": ["A"]}), lambda s: clean_frame(), tmp_path)
    with pytest.raises(PermissionError):
        WindowData(tmp_path).split("test")
    WindowData(tmp_path, allow_test=True).split("test")


def test_zero_yang_zhang_is_masked():
    df = clean_frame()
    df.iloc[401:406, :4] = df["close"].iloc[400]  # 5 velas planas tras t=400
    a = asset_arrays(df)
    assert a["v_5"][400] == 0 and not a["mask_sigma_5"][400]


def test_relative_target_has_zero_daily_mean(tmp_path):
    frames = {"A": clean_frame(800, 1), "B": clean_frame(700, 2), "C": clean_frame(750, 3)}
    build_dataset(pd.DataFrame({"symbol": list(frames)}), frames.__getitem__, tmp_path)
    data = WindowData(tmp_path)
    y_mean, n = data.date_means()
    y = data.batch(np.arange(len(data.samples)))["y"]
    dates = data.samples["date"].to_numpy()
    rel = pd.DataFrame(y - y_mean).groupby(dates).mean()
    np.testing.assert_allclose(rel.to_numpy(), 0.0, atol=1e-5)
    counts = data.samples.groupby("date")["asset"].transform("size").to_numpy()
    assert (n == counts).all() and set(np.unique(n)) == {1, 2, 3}


def test_window_keeps_samples_and_last_bars(tmp_path):
    frames = {"A": clean_frame(800, 1), "B": clean_frame(700, 2)}
    build_dataset(pd.DataFrame({"symbol": list(frames)}), frames.__getitem__, tmp_path)
    data = WindowData(tmp_path)
    idx = np.arange(len(data.samples))
    full, short = data.batch(idx), data.batch(idx, window=63)
    assert short["x"].shape == (len(idx), 63, 5)
    np.testing.assert_array_equal(short["x"], full["x"][:, -63:])
    for k in ("scalar", "y", "v", "mask_sigma", "asset", "date"):
        np.testing.assert_array_equal(short[k], full[k])
    with pytest.raises(ValueError):
        data.batch(idx, window=300)


def test_date_means_use_only_given_samples(tmp_path):
    frames = {"A": clean_frame(800, 1), "B": clean_frame(700, 2), "C": clean_frame(750, 3)}
    build_dataset(pd.DataFrame({"symbol": list(frames)}), frames.__getitem__, tmp_path)
    data = WindowData(tmp_path)
    smp = data.samples
    per_day = smp.groupby("date")["asset"].transform("size").to_numpy()
    on_d = np.flatnonzero(smp["date"].to_numpy() == smp["date"].to_numpy()[per_day == 3][0])
    out, keep = on_d[0], on_d[1:]                       # una muestra del día fuera de idx
    train = np.setdiff1d(np.arange(len(smp)), [out])
    m1, n1 = data.date_means(train)
    rows = smp["row"].to_numpy()
    y5 = np.asarray(data.arr["y_5"])[rows[keep]].astype(float)
    np.testing.assert_allclose(m1[keep, 0], y5.mean(), rtol=1e-6)
    assert (n1[keep] == 2).all() and n1[out] == 0 and np.isnan(m1[out]).all()
    for h in data.horizons:                             # cambiar su target no cambia las medias
        y = np.array(data.arr[f"y_{h}"])
        y[rows[out]] += 100.0
        data.arr[f"y_{h}"] = y
    m2, n2 = data.date_means(train)
    np.testing.assert_array_equal(m2[train], m1[train])
    np.testing.assert_array_equal(n2, n1)
    m_all, _ = data.date_means()                        # con todas las muestras, sí la cambia
    assert (m_all[keep, 0] > m1[keep, 0] + 10).all()


def test_asset_info_by_symbol_not_position():
    uni = pd.DataFrame({"symbol": ["B", "C", "A", "D"],
                        "asset_class": ["equity", "crypto", "equity", "equity"],
                        "country": ["FR", None, "US", "DE"]})
    info = asset_info(["A", "B", "C", "X"], uni)      # orden del dataset, no del universo
    assert info["symbol"].tolist() == ["A", "B", "C", "X"]
    assert info["group"].tolist() == ["US", "FR", "crypto", "?"]
    assert info["region"].tolist() == ["US", "Europa", "crypto", "?"]


def test_dataset_id_tracks_data_and_params(tmp_path):
    build_dataset(pd.DataFrame({"symbol": ["A"]}), lambda s: clean_frame(800, 1), tmp_path / "a")
    build_dataset(pd.DataFrame({"symbol": ["A"]}), lambda s: clean_frame(700, 1), tmp_path / "b")
    ids = [dataset_id(WindowData(tmp_path / k).meta) for k in "ab"]
    assert ids[0] != ids[1] and ids[0] == dataset_id(WindowData(tmp_path / "a").meta)
