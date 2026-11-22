"""Vela del mercado por región, como entrada adicional del modelo (TrainConfig.market; prueba de
mu_tree, 22-11-2026).

- Región de cada activo (asset_regions): 0 = EE. UU., 1 = Europa. Cripto va con EE. UU.: la vela
  de EE. UU. de un día cierra antes que la de cripto (00:00 UTC), así que no usa nada posterior.
- Vela del mercado de una región y un día: la media, entre sus acciones con vela válida ese día
  y el anterior (el de cada acción), de log(O/C_anterior), log(H/C_anterior), log(L/C_anterior)
  y log(C/C_anterior). Solo cuentan los días con al menos MIN_STOCKS acciones así.
- Serie de precios (log) del mercado: el cierre es la suma acumulada de las medias de log(C/C_ant)
  y cada O, H, L, el cierre anterior del mercado más su media. Solo importan las diferencias
  dentro de una ventana, así que el origen da igual.
- Por fila del dataset (activo, fecha d): la vela de su región en d. Si su región no tiene vela
  ese día (fin de semana de cripto, festivo de su mercado), una vela plana en el último cierre
  del mercado (O = H = L = C); antes del primer día de la región, plana en el origen (0).
- Solo hasta el fin del desarrollo (31-12-2019): las filas posteriores quedan en NaN, para no
  abrir el test.

Las filas se alinean con las del dataset leyendo data/clean/<símbolo>.parquet (las mismas series
con las que se construyó; se comprueba la huella). Se guarda en <dataset>/market/ (m_log_open,
m_log_high, m_log_low, m_log_close .npy y meta.json). En la red, se normaliza en la ventana igual
que el activo: (log P_mercado − log C_mercado(t)) / σ_ventana del activo (WindowData.batch).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.data.dataset import SPLITS, asset_info, fingerprint

MIN_STOCKS = 20
REGION_CODES = {"US": 0, "crypto": 0, "Europa": 1}
MARKET_ARRAYS = ["m_log_open", "m_log_high", "m_log_low", "m_log_close"]


def asset_regions(symbols, universe: pd.DataFrame) -> np.ndarray:
    """Código de región (REGION_CODES) de cada activo, en el orden de symbols."""
    info = asset_info(list(symbols), universe)
    bad = ~info["region"].isin(list(REGION_CODES))
    if bad.any():
        raise ValueError(f"activos sin región: {info.loc[bad, 'symbol'].tolist()[:5]}")
    return info["region"].map(REGION_CODES).to_numpy(np.int64)


def market_rows(frames: list[pd.DataFrame], regions: np.ndarray, is_stock: np.ndarray,
                end=SPLITS["dev"][1], min_stocks: int = MIN_STOCKS) -> dict[str, np.ndarray]:
    """Arrays por fila (las filas de frames, concatenadas en orden) con el log-precio del mercado
    de la región de cada activo (ver el docstring del módulo). frames: series limpias (índice de
    fechas; open, high, low, close, valid) en el orden de los activos del dataset."""
    end = pd.Timestamp(end)
    parts = []
    for a, df in enumerate(frames):
        if not is_stock[a]:
            continue
        lc = np.log(df["close"].to_numpy(float))
        prev = np.r_[np.nan, lc[:-1]]
        ok = df["valid"].to_numpy(bool)
        ok = ok & np.r_[False, ok[:-1]] & (df.index <= end)
        parts.append(pd.DataFrame({
            "region": regions[a], "date": df.index[ok],
            **{k: np.log(df[k].to_numpy(float))[ok] - prev[ok] for k in ("open", "high", "low")},
            "close": lc[ok] - prev[ok]}))
    r = pd.concat(parts, ignore_index=True)
    g = r.groupby(["region", "date"])
    m = g[["open", "high", "low", "close"]].mean()
    m = m[g.size() >= min_stocks]
    out = {k: [] for k in MARKET_ARRAYS}
    cal = {}
    for reg in np.unique(regions):
        mr = m.loc[reg] if reg in m.index.get_level_values(0) else m.iloc[:0]
        c = np.cumsum(mr["close"].to_numpy())
        c_prev = np.r_[0.0, c[:-1]]
        cal[reg] = (mr.index.to_numpy(), c_prev + mr["open"].to_numpy(),
                    c_prev + mr["high"].to_numpy(), c_prev + mr["low"].to_numpy(), c)
    for a, df in enumerate(frames):
        dates, o, h, lo, c = cal[regions[a]]
        d = df.index.to_numpy()
        i = np.searchsorted(dates, d, side="right") - 1
        has = i >= 0
        ii = np.maximum(i, 0)
        same = has & (dates[ii] == d)
        flat = np.where(has, c[ii], 0.0)        # antes del primer día: el origen (0)
        after = d > end.to_datetime64()
        for k, v in zip(MARKET_ARRAYS, (o, h, lo, c)):
            x = np.where(same, v[ii], flat)
            x[after] = np.nan
            out[k].append(x)
    return {k: np.concatenate(v) for k, v in out.items()}


def build_market(data, clean_dir: Path, universe: pd.DataFrame, out_dir: Path | None = None,
                 end=SPLITS["dev"][1], min_stocks: int = MIN_STOCKS) -> Path:
    """Calcula y guarda la vela del mercado alineada con las filas del dataset de data."""
    out_dir = Path(out_dir or data.dir / "market")
    fp = fingerprint(Path(clean_dir))
    if data.meta.get("clean_fingerprint") not in (None, fp):
        raise ValueError("data/clean no es el de este dataset (huella distinta)")
    symbols = data.meta["symbols"]
    frames = []
    for s in symbols:
        df = pd.read_parquet(Path(clean_dir) / f"{s}.parquet")
        _, n = data.meta["offsets"][s]
        if len(df) != n:
            raise ValueError(f"{s}: {len(df)} velas en clean, {n} en el dataset")
        frames.append(df)
    info = asset_info(symbols, universe)
    arrs = market_rows(frames, asset_regions(symbols, universe),
                       (info["asset_class"] == "equity").to_numpy(), end, min_stocks)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in arrs.items():
        np.save(out_dir / f"{k}.npy", v.astype(np.float64))
    meta = {"clean_fingerprint": fp, "n_rows": int(data.meta["n_rows"]), "end": str(end),
            "min_stocks": min_stocks}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return out_dir
