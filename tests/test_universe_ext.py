"""Universo ampliado (22-11-2026): exclusión del universo actual, equivalencias por nombre y
horquilla de Abdi–Ranaldo."""

import numpy as np
import pandas as pd

from agentic_trading.data.spread import abdi_ranaldo
from agentic_trading.data.universe import equivalent_groups, exclude_base


def test_exclude_base_by_symbol_alt_and_name():
    base = pd.DataFrame({"symbol": ["AAPL", "SHEL.L", "BTC-USD"],
                         "alt_symbols": [None, "SHELL.AS", None],
                         "name": ["Apple Inc.", "Shell plc", "Bitcoin"],
                         "asset_class": ["equity", "equity", "crypto"]})
    raw = pd.DataFrame({"symbol": ["AAPL", "SHELL.AS", "XYZ", "APPL.DE", "BTC"],
                        "name": ["Apple", "Shell", "Xyz Corp", "Apple Inc", "Bitcoin"]})
    kept, dropped = exclude_base(raw, base)
    assert set(dropped["symbol"]) == {"AAPL", "SHELL.AS", "APPL.DE"}
    assert set(kept["symbol"]) == {"XYZ", "BTC"}      # cripto no cuenta por nombre


def test_equivalent_groups_by_name_merges_and_respects_known_distinct():
    raw = pd.DataFrame({"symbol": ["UAA", "UA", "X1", "FBP", "FBNC"],
                        "name": ["Under Armour (Class A)", "Under Armour (Class C)", "Other",
                                 "First BanCorp (Puerto Rico)", "First Bancorp"],
                        "cik": [None, "1", "1", None, None]})
    assert equivalent_groups(raw) == [{"UA", "X1"}]
    groups = equivalent_groups(raw, by_name=True)
    assert groups == [{"UAA", "UA", "X1"}]          # CIK y nombre se unen; FBP/FBNC, distintas


def test_abdi_ranaldo_recovers_spread():
    rng = np.random.default_rng(0)
    n, k, spread = 2520, 40, 0.01
    steps = rng.normal(0, 0.012 / np.sqrt(k), (n, k)).cumsum(1)
    start = np.concatenate([[0.0], steps[:-1, -1].cumsum()])
    mid = start[:, None] + steps
    half = spread / 2
    close = mid[:, -1] + rng.choice([-half, half], n)
    df = pd.DataFrame({"open": np.exp(mid[:, 0]), "high": np.exp(mid.max(1) + half),
                       "low": np.exp(mid.min(1) - half), "close": np.exp(close),
                       "adj_factor": 0.9, "valid": True},
                      index=pd.bdate_range("2000-01-03", periods=n))
    s = abdi_ranaldo(df)
    assert len(s) > 100 and 0.007 < s.median() < 0.013
    df.loc[df.index[:30], "valid"] = False            # meses sin pares suficientes: fuera
    assert abdi_ranaldo(df).index.min() > s.index.min()
