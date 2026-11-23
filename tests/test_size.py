"""Escalar de tamaño del paso 2 de la ampliación (data/size.py, TrainConfig.size)."""

import json

import numpy as np
import pandas as pd

from tests.test_coarse import synthetic_data
from tests.test_mu_tree import MU_REF_CONFIG, rel1  # noqa: F401  (fixture)

SCALE = {"A": 1.0, "B": 10.0, "C": 10000.0}


def sized_data(tmp_path, scale=SCALE):
    """Dataset sintético en tmp_path/dataset con universo, limpios (adj_factor = 1) y tamaño.
    C·V en proporción a scale (A en USD, B en EUR, C en GBp: ×10.000 peniques = ×100 en libras)."""
    from agentic_trading.data.size import build_size
    data, frames = synthetic_data(tmp_path / "dataset")
    (tmp_path / "clean").mkdir()
    for s, df in frames.items():
        df = df.copy()
        df["adj_factor"] = 1.0
        df["volume"] = df["volume"] * scale[s] / df["close"]        # C·V = vol_original · scale
        df.to_parquet(tmp_path / "clean" / f"{s}.parquet")
    pd.DataFrame({"symbol": ["A", "B", "C"], "currency": ["USD", "EUR", "GBp"]}).to_parquet(
        tmp_path / "universe.parquet")
    idx = pd.bdate_range("2012-01-01", "2020-01-01")
    fx = pd.DataFrame({"USDEUR=X": 1.0, "GBPEUR=X": 1.0, "CHFEUR=X": 1.0}, index=idx)
    pct = build_size(tmp_path, fx)
    return data, pct


def test_traded_value_eur_units_and_zero_volume():
    from agentic_trading.data.size import traded_value_eur
    idx = pd.bdate_range("2015-01-01", periods=4)
    df = pd.DataFrame({"close": [200.0, 200, 200, 200], "adj_factor": [0.5, 0.5, 1, 1],
                       "volume": [10.0, 0, 10, 10]}, index=idx)
    fx = pd.DataFrame({"GBPEUR=X": [1.2, np.nan]}, index=idx[[1, 3]])
    v = traded_value_eur(df, "GBp", fx)
    # sin ajuste por dividendos (close / adj_factor), peniques → libras, último cambio conocido;
    # antes del primero, el primero
    np.testing.assert_allclose(v, [400 * 10 * 0.01 * 1.2, np.nan, 200 * 10 * 0.01 * 1.2,
                                   200 * 10 * 0.01 * 1.2])
    assert np.isnan(v[1])
    np.testing.assert_allclose(traded_value_eur(df, "EUR", fx)[2], 2000.0)


def test_build_size_orders_by_traded_value(tmp_path):
    data, pct = sized_data(tmp_path)
    assert pct.shape == (len(data.samples),) and pct.dtype == np.float32
    assert ((pct > 0) & (pct <= 1)).all()
    meta = json.loads((tmp_path / "dataset" / "size" / "meta.json").read_text())
    assert meta["n_samples"] == len(data.samples)
    np.testing.assert_array_equal(data.size, pct)
    # en las fechas con los tres activos, C (×100 en euros) arriba y A (×1) abajo
    smp = data.samples.assign(pct=pct)
    smp["sym"] = np.array(data.meta["symbols"])[smp["asset"].to_numpy()]
    full = smp.groupby("date")["sym"].transform("size") == 3
    top = smp[full].pivot(index="date", columns="sym", values="pct")
    assert (top["C"] > top["B"]).mean() > 0.95 and (top["B"] > top["A"]).mean() > 0.95


def test_batch_and_model_with_size(tmp_path, rel1):  # noqa: F811
    data, pct = sized_data(tmp_path)
    idx = np.arange(len(data.samples))
    b = data.batch(idx[:5], window=63, size=True)
    assert b["scalar"].shape == (5, 2)
    np.testing.assert_allclose(b["scalar"][:, 1], pct[:5])
    np.testing.assert_array_equal(b["scalar"][:, 0], data.batch(idx[:5], window=63)["scalar"][:, 0])
    cfg = rel1.TrainConfig(**{**MU_REF_CONFIG, "epochs": 1, "size": True})
    model, hist = rel1.train_model(data, idx, cfg, seed=3, log=lambda *_: None)
    assert model.size and np.isfinite([r["loss"] for r in hist]).all()
    p = rel1.predict(model, data, idx[::3])
    path = tmp_path / "m.pt"
    rel1.save_model(model, path, cfg, 3, hist)
    m2, _ = rel1.load_model(path)
    pd.testing.assert_frame_equal(rel1.predict(m2, data, idx[::3]), p)
