import numpy as np
import pandas as pd

from agentic_trading.data.audit import audit_asset
from agentic_trading.data.clean import CleanParams, as_audit_input, clean_asset, usable_samples
from tests.test_audit import scale_prices, synthetic


def run(df, symbol="X", **kw):
    flags = audit_asset(df, symbol)
    out, act = clean_asset(df, flags, symbol, params=CleanParams(**kw))
    post = audit_asset(as_audit_input(out), symbol)
    return out, pd.DataFrame(act, columns=["symbol", "date", "action", "reason", "detail"]), post


def no_errors(post):
    return post[post.severity == "error"].empty


def test_clean_series_untouched():
    df = synthetic()
    out, act, _post = run(df)
    assert act.empty and out["valid"].all() and len(out) == len(df)
    np.testing.assert_allclose(out["close"], df["close"])


def test_unrecorded_split_is_repaired():
    df = synthetic()
    ref = df["close"].copy()
    scale_prices(df, 800, 0.5, volume=2)
    out, act, post = run(df)
    assert "rescale_history" in set(act.action) and no_errors(post)
    np.testing.assert_allclose(out["close"], ref * 0.5)


def test_unit_shift_is_repaired():
    df = scale_prices(synthetic(), 600, 0.01)
    _out, act, post = run(df, "X.L")
    assert "rescale_history" in set(act.action) and no_errors(post)


def test_spike_bar_is_dropped():
    df = scale_prices(synthetic(), 500, 3.0, end=500)
    out, _act, post = run(df)
    assert df.index[500] not in out.index and len(out) == len(df) - 1 and no_errors(post)


def test_stale_run_keeps_first_bar():
    df = synthetic()
    df.iloc[100:108, :5] = df["close"].iloc[99]
    out, _, _ = run(df)
    assert df.index[99] in out.index
    assert not out.index.isin(df.index[100:108]).any()


def test_ohlc_open_out_of_range_is_fixed_and_marked():
    df = synthetic()
    df.iloc[300, df.columns.get_loc("open")] = df["high"].iloc[300] * 1.3
    out, _act, post = run(df)
    d = df.index[300]
    assert out.loc[d, "open"] <= out.loc[d, "high"] and no_errors(post)
    assert out["valid"].all() and out.loc[d, "ohlc_repaired"] and out["ohlc_repaired"].sum() == 1
    out2, _, _ = run(df, ohlc_large="mask")
    assert not out2.loc[d, "valid"] and out2.loc[d, "ohlc_repaired"]


def test_volume_jump_truncates_history():
    df = synthetic()
    df.iloc[:400, df.columns.get_loc("volume")] /= 1000
    out, act, _ = run(df)
    assert "truncate" in set(act.action)
    assert abs((out.index[0] - df.index[400]).days) <= 5


def test_dividend_adjustment():
    df = synthetic()
    y = 0.02
    df.iloc[700, df.columns.get_loc("dividends")] = df["close"].iloc[699] * y
    out, _, _ = run(df)
    np.testing.assert_allclose(out["close"].iloc[:700], df["close"].iloc[:700] * (1 - y))
    np.testing.assert_allclose(out["close"].iloc[700:], df["close"].iloc[700:])


def test_usable_samples():
    v = pd.Series(True, index=range(300))
    assert usable_samples(v, 252, 20) == 300 - 272 + 1
    v.iloc[150] = False
    assert usable_samples(v, 252, 20) == 0


def test_yahoo_adj_close_is_ignored():
    # El factor sale de la columna de dividendos: un adj_close de Yahoo corrupto (negativo) o con
    # los dividendos ÷100 (Londres) no afecta.
    df = synthetic()
    y = 0.03
    df.iloc[700, df.columns.get_loc("dividends")] = df["close"].iloc[699] * y
    df["adj_close"] = df["close"] * np.where(np.arange(len(df)) < 700, 1 - y / 100, 1.0)
    df.iloc[:50, df.columns.get_loc("adj_close")] *= -1
    out, act, _ = run(df, "X.L")
    np.testing.assert_allclose(out["close"].iloc[:700], df["close"].iloc[:700] * (1 - y))
    assert act.empty


def test_bad_dividends_are_ignored():
    df = synthetic()
    df.iloc[900, df.columns.get_loc("dividends")] = df["close"].iloc[899] * 0.30  # sin caída
    df.iloc[1200, df.columns.get_loc("dividends")] = df["close"].iloc[1199] * 3.0  # imposible
    out, act, _ = run(df)
    np.testing.assert_allclose(out["close"], df["close"])
    assert (act.action == "ignore_dividend").sum() == 2


def test_special_dividend_that_matches_the_drop_is_applied():
    df = synthetic()
    y = 0.30
    df.iloc[900, df.columns.get_loc("dividends")] = df["close"].iloc[899] * y
    scale_prices(df, 900, 1 - y)                       # el precio cae lo que paga
    out, act, post = run(df)
    assert "ignore_dividend" not in set(act.action)
    assert abs(np.log(out["close"]).diff().iloc[900]) < 0.1 and no_errors(post)


def test_dividend_on_dropped_bar_still_counts():
    df = synthetic()
    d = df.index[700]
    df.loc[d, ["open", "high", "low", "close", "adj_close"]] = df["close"].iloc[699]
    df.loc[d, "volume"] = 0.0                          # sesión sin negociación: se quita
    y = 0.02
    df.loc[d, "dividends"] = df["close"].iloc[699] * y
    out, _, _ = run(df)
    assert d not in out.index
    np.testing.assert_allclose(out["close"].iloc[:700], df["close"].iloc[:700] * (1 - y))


def test_bar_after_long_gap_is_invalid():
    df = synthetic()
    df = df.drop(df.index[500:515])                    # 15 sesiones = 21 días naturales
    out, act, _ = run(df)
    after = df.index[500]
    assert not out.loc[after, "valid"] and out["valid"].sum() == len(out) - 1
    assert ((act.action == "mask") & (act.reason == "gap")).sum() == 1


def test_coarse_price_year_is_masked():
    df = synthetic()
    year = df.index[600].year
    m = df.index.year == year
    grid = 0.05 * df.loc[m, "close"].median()          # precios en escalones del 5 %
    cols = ["open", "high", "low", "close", "adj_close"]
    df.loc[m, cols] = (df.loc[m, cols] / grid).round() * grid
    out, _, _ = run(df)
    assert not out.loc[out.index.year == year, "valid"].any()
    assert out.loc[out.index.year != year, "valid"].all()


def test_jump_at_badly_recorded_corporate_action_is_masked():
    from agentic_trading.data.audit import audit_universe
    from agentic_trading.data.clean import clean_universe
    df = scale_prices(synthetic(), 800, 0.6)  # salto que no casa con el split registrado (< extremo)
    df.iloc[800, df.columns.get_loc("stock_splits")] = 0.5
    uni = pd.DataFrame({"symbol": ["X"], "asset_class": ["equity"], "country": ["US"]})
    flags, _ = audit_universe(uni, lambda s: df)
    frames, actions, _, _ = clean_universe(uni, flags, lambda s: df)
    assert not frames["X"].loc[df.index[800], "valid"] and frames["X"]["valid"].sum() == len(df) - 1
    assert (actions.reason == "corporate_action_jump").sum() == 1


def test_unit_shift_keeps_volume():
    df = scale_prices(synthetic(), 600, 0.01)
    vol = df["volume"].copy()
    out, _, _ = run(df, "X.L")
    np.testing.assert_allclose(out["volume"], vol)  # peniques -> libras no cambia el nº de acciones


def test_tiny_ohlc_violation_is_fixed_exactly():
    df = synthetic()
    df.iloc[300, df.columns.get_loc("open")] = df["high"].iloc[300] * (1 + 5e-5)  # < tolerancia
    out, _, _ = run(df)
    assert (out["high"] >= np.maximum(out["open"], out["close"])).all()
    assert (out["low"] <= np.minimum(out["open"], out["close"])).all()


def test_second_pass_drops_bar_that_becomes_no_trade():
    from agentic_trading.data.clean import clean_universe
    df = synthetic()
    d = df.index[400]
    c = df["close"].iloc[400]
    df.loc[d, ["high", "low", "adj_close"]] = c          # H = L = C, volumen 0 ...
    df.loc[d, "volume"] = 0.0
    df.loc[d, "open"] = c * 0.97                          # ... y una apertura fuera de rango
    uni = pd.DataFrame({"symbol": ["X"], "asset_class": ["equity"], "country": ["US"]})
    from agentic_trading.data.audit import audit_universe
    flags, _ = audit_universe(uni, lambda s: df)
    frames, actions, _, post = clean_universe(uni, flags, lambda s: df)
    assert d not in frames["X"].index                     # tras corregir la apertura es no_trade
    assert actions["pass"].max() == 2 and post.query("check == 'no_trade'").empty


def test_extreme_move_bar_is_masked():
    df = scale_prices(synthetic(), 700, 7.6)
    out, _act, _post = run(df)
    assert not out.at[df.index[700], "valid"] and out["valid"].sum() == len(out) - 1


def test_absurd_wick_is_repaired_and_marked():
    df = synthetic()
    d = df.index[400]
    df.loc[d, "low"] = df.loc[d, "close"] / 700  # mínimo ÷700, como Shell en 2019
    out, act, post = run(df)
    body = min(df.at[d, "open"], df.at[d, "close"]) * out.at[d, "adj_factor"]
    assert np.isclose(out.at[d, "low"], body) and out.at[d, "ohlc_repaired"]
    assert out.at[d, "valid"] and "fix_wick" in set(act.action)
    assert post[post.check == "wick_outlier"].empty


def test_isolated_pence_open_is_repaired():
    df = synthetic()
    d = df.index[400]
    df.loc[d, ["open", "low"]] = df.loc[d, "close"] / 100  # apertura ÷100 (y su mínimo)
    out, act, post = run(df, "X.L")
    prev = out["close"].iloc[399]
    assert np.isclose(out.at[d, "open"], prev) and out.at[d, "ohlc_repaired"]
    assert out.at[d, "low"] >= min(prev, out.at[d, "close"]) * 0.999
    assert {"fix_open", "fix_wick"} <= set(act.action)
    assert post[post.check.isin(["wick_outlier", "open_outlier"])].empty
