"""Dataset combinado (universo actual + ampliado) y escalar de tamaño (22-11-2026, paso 2 de la
ampliación; logs/universe_ext2.md).

Combinado (build_combined): en otra carpeta (data/comb/), el universo actual y el ampliado juntos
(universe.parquet con la columna size_group: "grandes", o el grupo del ampliado,
universe_ext.group_of) y data/comb/clean/ con enlaces a los archivos limpios de cada uno. El
dataset se construye con el pipeline de siempre (scripts/build_dataset.py con AGENTIC_TRADING_DATA).
Los universos no se tocan.

Tamaño (build_size): por muestra, el percentil entre las muestras de su fecha de la mediana de
C·V en euros de las 252 velas de su ventana. C es el cierre sin el ajuste por dividendos (close /
adj_factor, el de Yahoo, ajustado por splits como el volumen); V > 0 solamente (como la mediana
del volumen del dataset). Cambio a euros con los pares de universe.FX_TO_EUR (el de la regla de
duplicados) de cada día, con el último cambio conocido; antes del primero que da Yahoo
(2003–2005), el primero (aproximación: solo mueve el orden entre divisas en 2000–2004). Percentil
= rango / n con empates en media, en (0, 1]. Va en <dataset>/size/pct.npy, alineado con
samples.parquet, con su meta.json (huella del dataset).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.data.universe import FX_TO_EUR, exchange_of

LARGE = "grandes"


def build_combined(base: Path, ext: Path, out: Path, groups: pd.Series) -> pd.DataFrame:
    """Universo y enlaces a los datos limpios del combinado en out. groups: grupo de cada
    símbolo del ampliado. Devuelve el universo combinado."""
    ub, ue = pd.read_parquet(base / "universe.parquet"), pd.read_parquet(ext / "universe.parquet")
    if set(ub["symbol"]) & set(ue["symbol"]):
        raise ValueError("símbolos en los dos universos")
    ub["size_group"] = np.where(ub["asset_class"] == "equity", LARGE, ub["asset_class"])
    ue["size_group"] = ue["symbol"].map(groups).to_numpy()
    uni = pd.concat([ub, ue], ignore_index=True)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    for src in (base, ext):
        for f in (src / "clean").glob("*.parquet"):
            if f.name.startswith("_"):
                continue
            link = out / "clean" / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(f.resolve(), link)
    uni.to_parquet(out / "universe.parquet", index=False)
    return uni


def fx_history(start: str = "1999-01-01", end: str = "2020-01-31") -> pd.DataFrame:
    """Cierres diarios de los pares de FX_TO_EUR (Yahoo), una columna por par."""
    import yfinance as yf
    pairs = sorted({p for p, _ in FX_TO_EUR.values() if p})
    d = yf.download(pairs, start=start, end=end, auto_adjust=False, progress=False)["Close"]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d[pairs]


def traded_value_eur(df: pd.DataFrame, currency: str, fx: pd.DataFrame) -> np.ndarray:
    """C·V en euros de cada vela de un activo limpio (NaN con V = 0)."""
    pair, unit = FX_TO_EUR[currency]
    close = df["close"].to_numpy(float) / df["adj_factor"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    rate = 1.0
    if pair:
        r = fx[pair].dropna()
        rate = r.reindex(df.index, method="ffill").fillna(r.iloc[0]).to_numpy(float)
    return np.where(vol > 0, close * vol * unit * rate, np.nan)


def build_size(data_dir: Path, fx: pd.DataFrame, window: int = 252) -> np.ndarray:
    """Percentil de tamaño por muestra del dataset de data_dir (ver el docstring)."""
    ds = data_dir / "dataset"
    meta = json.loads((ds / "meta.json").read_text())
    uni = pd.read_parquet(data_dir / "universe.parquet").set_index("symbol")
    med = np.full(meta["n_rows"], np.nan)
    for s, (off, n) in meta["offsets"].items():
        df = pd.read_parquet(data_dir / "clean" / f"{s}.parquet")
        if len(df) != n:
            raise ValueError(f"{s}: {len(df)} velas frente a {n} en el dataset")
        cur = uni.at[s, "currency"] if pd.notna(uni.at[s, "currency"]) else exchange_of(s)[1]
        tv = pd.Series(traded_value_eur(df, cur, fx))
        med[off:off + n] = tv.rolling(window, min_periods=1).median().to_numpy()
    smp = pd.read_parquet(ds / "samples.parquet", columns=["row", "date"])
    v = pd.Series(med[smp["row"].to_numpy()])
    pct = v.groupby(smp["date"].to_numpy()).rank(pct=True).fillna(0.5).to_numpy(np.float32)
    out = ds / "size"
    out.mkdir(exist_ok=True)
    np.save(out / "pct.npy", pct)
    (out / "meta.json").write_text(json.dumps({
        "n_samples": len(smp), "clean_fingerprint": meta.get("clean_fingerprint"),
        "window": window, "nan_median": int(np.isnan(v).sum())}, indent=1))
    return pct
