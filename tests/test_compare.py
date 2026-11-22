import numpy as np
import pandas as pd
import pytest

from agentic_trading.model.compare import (
    cs_sums,
    ic_diff,
    pooled_rel_mse,
    rel_mse,
    train_subsample,
)


def test_cross_sectional_mse_ignores_daily_mean():
    rng = np.random.default_rng(0)
    d = np.repeat(np.arange(50), 30)
    y = rng.normal(size=len(d)) + np.repeat(rng.normal(size=50), 30)
    day_const = np.repeat(rng.normal(size=50), 30)            # μ igual para todos cada día
    assert rel_mse(cs_sums(d, day_const, y)) == pytest.approx(0.0, abs=1e-9)
    y_d = pd.Series(y).groupby(d).transform("mean").to_numpy()
    assert rel_mse(cs_sums(d, y - y_d + 5.0, y)) == pytest.approx(-100.0)   # perfecto entre activos
    halves = [cs_sums(d[:300], 0.1 * y[:300], y[:300]), cs_sums(d[300:], 0.1 * y[300:], y[300:])]
    assert pooled_rel_mse(halves) == pytest.approx(rel_mse(cs_sums(d, 0.1 * y, y)))


class _Data:
    def __init__(self, dates):
        self.samples = pd.DataFrame({"date": dates})


def test_train_subsample_takes_whole_dates():
    dates = np.repeat(pd.bdate_range("2010-01-01", periods=100), 7)
    data, f = _Data(dates), {"train": np.arange(len(dates))}
    sub = train_subsample(data, f, 3, n=100)
    assert 100 <= len(sub) < 107 and (pd.Series(dates[sub]).value_counts() == 7).all()
    np.testing.assert_array_equal(sub, train_subsample(data, f, 3, n=100))


def test_ic_diff_aligns_days():
    days = pd.bdate_range("2010-01-01", periods=500)
    rng = np.random.default_rng(1)
    a = pd.Series(0.03 + 0.01 * rng.normal(size=500), index=days)
    b = pd.Series(0.01 + 0.01 * rng.normal(size=400), index=days[100:])
    mean, t, n = ic_diff(a, b, 5)
    assert n == 400 and mean == pytest.approx(0.02, abs=0.003) and t > 10
