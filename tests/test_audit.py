import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.audit import audit_asset


def synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    v = np.exp(rng.normal(np.log(1e6), 0.3, n))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "adj_close": c,
                         "volume": v, "dividends": 0.0, "stock_splits": 0.0}, index=idx)


def scale_prices(df, start, factor, end=None, volume=None):
    sl = slice(df.index[start], df.index[end] if end is not None else None)
    df.loc[sl, ["open", "high", "low", "close", "adj_close"]] *= factor
    if volume:
        df.loc[sl, "volume"] *= volume
    return df


def checks(flags, severities=("error", "warning")):
    return flags[flags.severity.isin(severities)]


def test_clean_series_has_no_errors_or_warnings():
    for seed in range(5):
        f = audit_asset(synthetic(seed=seed), "X")
        assert checks(f).empty, f


def test_spike():
    df = scale_prices(synthetic(), 500, 3.0, end=500)
    f = checks(audit_asset(df, "X"))
    assert list(f.check) == ["spike"] and f.date.iloc[0] == df.index[500]


def test_unrecorded_split():
    df = scale_prices(synthetic(), 800, 0.5, volume=2)
    f = checks(audit_asset(df, "X"))
    assert list(f.check) == ["unrecorded_split"] and f.date.iloc[0] == df.index[800]


def test_split_like_move_without_volume_evidence():
    df = scale_prices(synthetic(), 800, 0.5)
    assert list(checks(audit_asset(df, "X")).check) == ["split_like_move"]


def test_recorded_split_not_adjusted():
    df = scale_prices(synthetic(), 1000, 0.25, volume=4)
    df.iloc[1000, df.columns.get_loc("stock_splits")] = 4.0
    assert list(checks(audit_asset(df, "X")).check) == ["split_not_adjusted"]


def test_recorded_split_adjusted_is_clean():
    df = synthetic()
    df.iloc[1000, df.columns.get_loc("stock_splits")] = 4.0
    assert checks(audit_asset(df, "X")).empty


@pytest.mark.parametrize("end,expected", [(None, "unit_shift"), (600, "unit_spike")])
def test_pence_pound(end, expected):
    df = scale_prices(synthetic(), 600, 0.01 if end is None else 100, end=end)
    assert list(checks(audit_asset(df, "X.L")).check) == [expected]


def test_ohlc_inconsistent():
    df = synthetic()
    df.iloc[300, df.columns.get_loc("high")] = df["close"].iloc[300] * 0.95
    f = checks(audit_asset(df, "X"))
    assert "ohlc_inconsistent" in set(f.check)
    assert f.loc[f.check == "ohlc_inconsistent", "severity"].iloc[0] == "error"


def test_stale_price():
    df = synthetic()
    df.iloc[100:108, :5] = df["close"].iloc[99]  # O, H, L, C, adj_close
    f = checks(audit_asset(df, "X"))
    assert list(f.check) == ["stale_price"] and f.value.iloc[0] == 9


def test_market_crash_is_not_flagged():
    df = synthetic()
    scale_prices(df, 700, np.exp(-0.3))
    market = pd.Series(0.0, index=df.index)
    market.iloc[700] = -0.3
    assert audit_asset(df, "X", market=market).query("check == 'large_move'").empty
    assert not audit_asset(df, "X").query("check == 'large_move'").empty


def test_calendar_checks():
    df = synthetic(300)
    cal = df.index
    sat = df.index[150] + pd.Timedelta(days=(5 - df.index[150].dayofweek) % 7)
    df = pd.concat([df.drop(df.index[200:215]), df.iloc[[150]].set_axis([sat])]).sort_index()
    f = audit_asset(df, "X", calendar=cal)
    assert {"weekend_bar", "gap", "missing_days"} <= set(f.check)


def test_adjustment_checks():
    df = synthetic()
    df["adj_close"] = df["close"]
    df.iloc[400:, df.columns.get_loc("adj_close")] *= 0.97  # salto sin dividendo
    df.iloc[900, df.columns.get_loc("dividends")] = df["close"].iloc[899] * 0.3
    assert {"adj_factor_jump", "large_dividend"} <= set(audit_asset(df, "X").check)


def test_market_event_downgrades_spikes():
    from agentic_trading.data.audit import mark_market_events
    d = pd.Timestamp("2020-03-18")
    flags = pd.DataFrame({"symbol": list("ABCDE"), "date": [d] * 4 + [d + pd.Timedelta(days=1)],
                          "check": ["spike"] * 5, "severity": ["error"] * 5, "value": 0.0,
                          "detail": ""})
    out = mark_market_events(flags, dict.fromkeys("ABCDE", "US"))
    assert list(out.severity) == ["info"] * 4 + ["error"]


def test_two_day_reversal_is_warning():
    df = scale_prices(synthetic(), 500, 3.0, end=501)
    f = checks(audit_asset(df, "X"))
    assert list(f.check) == ["spike"] and f.severity.iloc[0] == "warning"


def test_intraday_crash_is_not_a_split():
    # Caída del 50 % durante la sesión: abre en el nivel anterior y cierra a la mitad, con
    # volumen alto después. Un split no puede producirse a mitad de sesión: es un movimiento real.
    df = scale_prices(synthetic(), 800, 0.5, volume=2)
    d = df.index[800]
    df.loc[d, "open"] = df["close"].iloc[799]
    df.loc[d, "high"] = df["close"].iloc[799] * 1.01
    f = checks(audit_asset(df, "X"), ("error", "warning", "info"))
    assert f.query("check in ['unrecorded_split', 'split_like_move']").empty
    assert "large_move" in set(f.check)


def test_bad_dividend_detection():
    df = synthetic()
    c = df["close"]
    df.iloc[300, df.columns.get_loc("dividends")] = c.iloc[299] * 0.02      # normal
    df.iloc[600, df.columns.get_loc("dividends")] = c.iloc[599] * 0.30      # grande, sin caída
    df.iloc[900, df.columns.get_loc("dividends")] = c.iloc[899] * 5.00      # imposible
    f = audit_asset(df, "X").query("check == 'bad_dividend'")
    assert list(f.date) == [df.index[600], df.index[900]]


def test_special_dividend_with_drop_is_not_bad():
    df = synthetic()
    df.iloc[600, df.columns.get_loc("dividends")] = df["close"].iloc[599] * 0.30
    scale_prices(df, 600, 0.70)
    assert audit_asset(df, "X").query("check == 'bad_dividend'").empty


def test_coarse_prices_detection():
    df = synthetic()
    m = df.index.year == df.index[600].year
    grid = 0.05 * df.loc[m, "close"].median()
    df.loc[m, "close"] = (df.loc[m, "close"] / grid).round() * grid
    f = audit_asset(df, "X").query("check == 'coarse_prices'")
    assert len(f) == 1 and f.date.iloc[0].year == df.index[600].year


def test_dividend_on_market_rally_is_not_bad():
    # Dividendo real del 7 % en un día en que el mercado sube un 10 %: el valor sube, pero frente
    # al mercado cae lo que paga.
    df = synthetic()
    y = 0.07
    df.iloc[600, df.columns.get_loc("dividends")] = df["close"].iloc[599] * y
    scale_prices(df, 600, (1 - y) * np.exp(0.10))
    market = pd.Series(0.0, index=df.index)
    market.iloc[600] = 0.10
    assert not audit_asset(df, "X").query("check == 'bad_dividend'").empty
    assert audit_asset(df, "X", market=market).query("check == 'bad_dividend'").empty


def test_extreme_move_without_split_ratio():
    df = scale_prices(synthetic(), 700, 7.6)  # ×7,6: no casa con ningún ratio de split
    f = checks(audit_asset(df, "X"))
    assert "extreme_move" in set(f.check)


def test_corrupt_ohlc_year():
    df = synthetic()
    m = df.index.year == df.index[600].year
    df.loc[m, "open"] = df["close"].shift()[m] * 1.13       # como VOD.L 2003–2006
    df.loc[m, "high"] = np.maximum(df.loc[m, "open"], df.loc[m, "high"])
    f = audit_asset(df, "X").query("check == 'corrupt_ohlc'")
    assert len(f) == 1 and f.date.iloc[0].year == df.index[600].year
    assert audit_asset(synthetic(), "X").query("check == 'corrupt_ohlc'").empty
