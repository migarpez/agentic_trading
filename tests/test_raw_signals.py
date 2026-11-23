"""Causalidad de las señales de scripts/raw_signals.py (23-11-2026)."""

import numpy as np
import pandas as pd

from scripts.raw_signals import s1_volume, s2_season


def test_s1_is_the_holding_week_52_dates_back():
    days = pd.bdate_range("2001-01-01", "2004-12-31")
    wk = pd.DatetimeIndex([d for d in days if d.dayofweek == 4])          # viernes
    vol = np.full(len(days), 100.0)
    spike = wk[60] + pd.Timedelta(days=4)                                # martes tras w_60
    vol[days.get_loc(spike)] = 1000.0
    df = pd.DataFrame({"volume": vol, "valid": True}, index=days)
    s = s1_volume({"X": df}, wk)["X"]
    # la semana de tenencia de w_60 (w_60 < d ≤ w_61) aparece en w_112, ni antes ni después
    assert s[wk[112]] == np.log(10.0)
    assert (s.drop(wk[112]) == 0).all()
    assert s.index.min() > wk[51]


def test_s2_uses_only_previous_years_of_the_same_month():
    days = pd.bdate_range("2000-01-03", "2006-12-29")
    r = np.zeros(len(days))
    first_march = pd.Series(days, index=days)[days.month == 3].groupby(lambda d: d.year).min()
    r[days.isin(first_march)] = 0.05                     # X sube el primer día hábil de marzo
    close = pd.DataFrame({"X": 100 * np.exp(np.cumsum(r)), "Y": 100.0}, index=days)
    s = s2_season(close)
    mar = s.index[s.index.month == 3]
    # relativa: X − media = +0,025 cada marzo; con ≥ 3 años previos, desde marzo de 2003
    assert s.loc[mar[:3], "X"].isna().all()
    np.testing.assert_allclose(s.loc[mar[3:], "X"], 0.025)
    assert s.loc[s.index.month != 3, "X"].dropna().abs().max() < 1e-12
