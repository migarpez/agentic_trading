"""Entrada en dos escalas (TrainConfig.coarse): velas agregadas y rama de la red.

La referencia bit a bit (fixtures/coarse_none_ref.npz) se generó con reference_outputs y el código
de antes de añadir coarse (8-11-2026, PyTorch 2.14.1 en CPU, esta máquina). En otra CPU u otra
versión de PyTorch puede cambiar en los últimos bits sin que el código haya cambiado.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REF = Path(__file__).parent / "fixtures" / "coarse_none_ref.npz"

# Configuraciones de la referencia: la de siempre (target absoluto, ventana de 252) y la adoptada
# (relativo, ventana de 63, λ = 3), en pequeño.
REF_CONFIGS = {
    "abs252": {"epochs": 1, "epoch_frac": 0.5, "batch": 128, "threads": 1},
    "rel63": {"epochs": 2, "epoch_frac": 0.5, "batch": 128, "threads": 1, "target": "rel",
              "window": 63, "lam": 3.0},
}


def synthetic_data(directory: Path, zero_volume: bool = False):
    """Dataset sintético de 3 activos. zero_volume: el activo A tiene días sueltos con volumen 0
    y una semana entera (posiciones 600–604) sin volumen."""
    from agentic_trading.data.dataset import WindowData, build_dataset
    from tests.test_dataset import clean_frame
    frames = {"A": clean_frame(800, 1), "B": clean_frame(700, 2), "C": clean_frame(750, 3)}
    if zero_volume:
        vol = frames["A"]["volume"].to_numpy().copy()
        vol[[300, 450, 451, 520, 590, 598]] = 0.0
        vol[600:605] = 0.0
        frames["A"]["volume"] = vol
    build_dataset(pd.DataFrame({"symbol": list(frames)}), frames.__getitem__, Path(directory))
    return WindowData(Path(directory)), frames


def reference_outputs(directory: Path) -> dict[str, np.ndarray]:
    """Lotes, historial de pérdidas, deriva, parámetros (nombres y formas) y predicciones de las
    configuraciones de REF_CONFIGS (sin coarse) con el dataset sintético. L_μ cuenta con cualquier
    número de muestras por fecha (REL_MIN_SAMPLES = 1), para que el target relativo entrene μ con
    solo 3 activos."""
    import agentic_trading.model.train as tr
    data, _ = synthetic_data(directory)
    idx = np.arange(len(data.samples))
    sel = idx[::180]                                   # 8 muestras de los 3 activos
    out = {"x252": data.batch(sel)["x"], "x63": data.batch(sel, window=63)["x"]}
    old, tr.REL_MIN_SAMPLES = tr.REL_MIN_SAMPLES, 1
    try:
        for name, kw in REF_CONFIGS.items():
            model, hist = tr.train_model(data, idx, tr.TrainConfig(**kw), seed=7,
                                         log=lambda *_: None)
            for key in ("loss", "l_mu", "l_sigma"):
                out[f"{name}_{key}"] = np.array([r[key] for r in hist])
            if getattr(model, "drift", None) is not None:
                out[f"{name}_drift"] = np.asarray(model.drift)
            out[f"{name}_params"] = np.array([f"{k}:{tuple(v.shape)}"
                                              for k, v in model.state_dict().items()])
            p = tr.predict(model, data, idx)
            for col in p.columns.drop("sample"):
                out[f"{name}_{col}"] = p[col].to_numpy()
    finally:
        tr.REL_MIN_SAMPLES = old
    return out


def manual_bars(df: pd.DataFrame, t: int, n_bars: int, n_ses: int) -> np.ndarray:
    """Velas agregadas de la ventana que termina en la posición t, a mano desde la serie."""
    w = df.iloc[t - 251:t + 1]
    sigma = np.std(np.diff(np.log(w["close"].to_numpy())), ddof=1)
    vol = w["volume"].to_numpy()
    med = np.median(vol[vol > 0])
    c_t = w["close"].iloc[-1]
    seg = df.iloc[t - n_bars * n_ses + 1:t + 1]
    out = []
    for g in range(n_bars):
        b = seg.iloc[g * n_ses:(g + 1) * n_ses]
        v = np.where(b["volume"] > 0, b["volume"], med).sum()
        out.append([np.log(b["open"].iloc[0] / c_t) / sigma, np.log(b["high"].max() / c_t) / sigma,
                    np.log(b["low"].min() / c_t) / sigma, np.log(b["close"].iloc[-1] / c_t) / sigma,
                    np.log(v / (n_ses * med))])
    return np.array(out)


def test_coarse_bars_match_manual_aggregation(tmp_path):
    from agentic_trading.data.dataset import COARSE
    data, frames = synthetic_data(tmp_path, zero_volume=True)
    smp = data.samples
    sym = data.meta["symbols"]
    offset = {s: data.meta["offsets"][s][0] for s in sym}
    pos = smp["row"].to_numpy() - np.array([offset[sym[a]] for a in smp["asset"]])
    a_rows = np.flatnonzero(smp["asset"].to_numpy() == 0)
    # muestras de A con ventanas que tocan los volúmenes 0, más la primera y la última de cada activo
    pick = [a_rows[np.searchsorted(pos[a_rows], t)] for t in (310, 452, 604, 606, 700)]
    pick += [i for a in range(3) for i in (np.flatnonzero(smp["asset"] == a)[[0, -1]])]
    idx = np.array(sorted(set(pick)))
    plain = data.batch(idx, window=63)
    assert "xc" not in plain
    assert COARSE == {"month": (12, 21), "week": (50, 5)}
    for kind, (n_bars, n_ses) in COARSE.items():
        b = data.batch(idx, window=63, coarse=kind)
        np.testing.assert_array_equal(b["x"], plain["x"])        # las diarias no cambian
        assert b["xc"].shape == (len(idx), n_bars, 5) and b["xc"].dtype == np.float32
        ref = np.stack([manual_bars(frames[sym[smp["asset"].iloc[i]]], pos[i], n_bars, n_ses)
                        for i in idx])
        np.testing.assert_allclose(b["xc"], ref, rtol=1e-5, atol=1e-5)
        assert (b["xc"][:, -1, 3] == 0).all()                     # el último cierre es C_t
    # la semana sin volumen (600–604) da v = 0 en la última vela semanal de la muestra de t = 604
    j = list(idx).index(a_rows[np.searchsorted(pos[a_rows], 604)])
    assert data.batch(idx, coarse="week")["xc"][j, -1, 4] == 0.0
    with pytest.raises(ValueError):
        data.batch(idx, coarse="day")


def test_coarse_none_is_bit_identical_to_before(tmp_path):
    ref = np.load(REF)
    out = reference_outputs(tmp_path)
    assert sorted(out) == sorted(ref.files)
    for k in ref.files:
        assert out[k].dtype == ref[k].dtype and np.array_equal(out[k], ref[k]), k


def test_coarse_branches():
    from agentic_trading.data.dataset import COARSE
    from agentic_trading.model.tcn import dilations_for
    from agentic_trading.model.train import TrainConfig, build_model

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    torch.manual_seed(0)
    x, s = torch.randn(4, 63, 5), torch.randn(4, 1)
    base = build_model(TrainConfig(window=63))
    assert base.coarse is None and build_model({"window": 63}).coarse is None   # checkpoints viejos
    with pytest.raises(ValueError):
        base(x, s, torch.randn(4, 12, 5))
    extra = {}
    for kind, (n_bars, _) in COARSE.items():
        m = build_model(TrainConfig(window=63, coarse=kind))
        xc = torch.randn(4, n_bars, 5)
        mu, logv = m(x, s, xc)
        assert mu.shape == logv.shape == (4, 3) and m.head[0].in_features == 32 + 32 + 1
        xc2 = xc.clone()
        xc2[:, 0] += 10                                 # la primera vela agregada cuenta
        assert not torch.allclose(m(x, s, xc2)[0], mu)
        x2 = x.clone()
        x2[:, -63] += 10                                # y las diarias siguen contando
        assert not torch.allclose(m(x2, s, xc)[0], mu)
        with pytest.raises(ValueError):
            m(x, s)
        extra[kind] = n_params(m) - n_params(base)
    # mes: MLP 60 → 64 → 32; semana: TCN de 5 capas; más las 32 entradas nuevas del MLP de salida
    assert extra["month"] == (60 * 64 + 64) + (64 * 32 + 32) + 32 * 64
    week = build_model(TrainConfig(window=63, coarse="week")).coarse_net
    assert dilations_for(50) == (1, 2, 4, 8, 16) and len(week.layers) + 1 == 5
    assert week.receptive_field >= 50
    assert extra["week"] == n_params(week) + 32 * 64
    z = torch.randn(8, 5, 50)
    torch.testing.assert_close(week.features_last(z), week.features_full(z), rtol=1e-5, atol=1e-5)
    # con la misma semilla, la rama diaria empieza igual que sin coarse
    torch.manual_seed(3)
    a = build_model(TrainConfig(window=63)).state_dict()
    torch.manual_seed(3)
    b = build_model(TrainConfig(window=63, coarse="month")).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if not k.startswith("head"))
    with pytest.raises(ValueError):
        build_model(TrainConfig(coarse="day"))


def test_coarse_training_and_checkpoint(tmp_path):
    from agentic_trading.data.dataset import COARSE
    from agentic_trading.model.train import (
        TrainConfig,
        load_model,
        predict,
        save_model,
        train_model,
    )
    data, _ = synthetic_data(tmp_path / "ds", zero_volume=True)
    idx = np.arange(len(data.samples))
    for kind in COARSE:
        cfg = TrainConfig(epochs=1, epoch_frac=0.5, batch=128, target="rel", window=63,
                          coarse=kind, threads=1)
        model, hist = train_model(data, idx, cfg, seed=0, log=lambda *_: None)
        p = predict(model, data, idx)
        assert np.isfinite(p.drop(columns="sample").to_numpy()).all()
        save_model(model, tmp_path / f"{kind}.pt", cfg, 0, hist)
        m2, ck = load_model(tmp_path / f"{kind}.pt")
        assert ck["config"]["coarse"] == kind and m2.coarse == kind and m2.window == 63
        pd.testing.assert_frame_equal(predict(m2, data, idx), p)
