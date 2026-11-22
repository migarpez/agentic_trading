"""Descarga de velas diarias desde Yahoo Finance (yfinance) a Parquet, un archivo por activo.

Se guarda todo lo que da Yahoo sin transformar: OHLC (ajustado por splits por Yahoo), Adj Close
(splits + dividendos), Volume, Dividends y Stock Splits. La elección de qué precios usa el
modelo se hace después, en el pipeline.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from agentic_trading.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw" / "yahoo"
START = "2000-01-01"
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


def path_for(symbol: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{symbol}.parquet"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=FIELDS)
    df = df[df["Close"].notna()]  # quita días alineados de otras bolsas
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    df.index.name = "date"
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download_batch(symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
    # end es exclusivo: se corta en ayer (UTC) para no guardar la vela de hoy a medio formar.
    end = datetime.now(timezone.utc).date().isoformat()
    data = yf.download(symbols, start=start, end=end, auto_adjust=False, actions=True,
                       group_by="ticker", threads=True, progress=False)
    out = {}
    for s in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            if s not in data.columns.get_level_values(0):
                continue
            df = data[s]
        else:
            df = data
        df = _clean(df)
        if len(df):
            out[s] = df
    return out


def download(symbols: list[str], start: str = START, raw_dir: Path = RAW_DIR,
             refresh: bool = False, batch_size: int = 50, retries: int = 3,
             pause: float = 2.0) -> pd.DataFrame:
    """Descarga los símbolos que falten (o todos con refresh) y devuelve un manifiesto."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in symbols if refresh or not path_for(s, raw_dir).exists()]
    failed: list[str] = []

    for attempt in range(1, retries + 1):
        todo, failed = pending, []
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            try:
                got = _download_batch(batch, start)
            except Exception as e:  # noqa: BLE001 - yfinance lanza de todo
                print(f"  lote {i // batch_size + 1}: error {e!r}")
                got = {}
            for s, df in got.items():
                df.to_parquet(path_for(s, raw_dir))
            failed += [s for s in batch if s not in got]
            print(f"intento {attempt} · {min(i + batch_size, len(todo))}/{len(todo)} "
                  f"· fallidos {len(failed)}")
            time.sleep(pause)
        if not failed:
            break
        pending = failed
        time.sleep(pause * 5 * attempt)

    return manifest(symbols, raw_dir)


def manifest(symbols: list[str], raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    for s in symbols:
        p = path_for(s, raw_dir)
        if p.exists():
            df = pd.read_parquet(p, columns=["close"])
            rows.append((s, "ok", len(df), df.index.min(), df.index.max(),
                         datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)))
        else:
            rows.append((s, "missing", 0, pd.NaT, pd.NaT, pd.NaT))
    return pd.DataFrame(rows, columns=["symbol", "status", "rows", "first_date", "last_date",
                                       "downloaded_at"])


def load_raw(symbol: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_parquet(path_for(symbol, raw_dir))


def coverage(man: pd.DataFrame, universe: pd.DataFrame, since: str = "2000-01-31") -> pd.DataFrame:
    """Cobertura por grupo: activos descargados y fracción con datos desde `since`."""
    m = man.merge(universe[["symbol", "country", "asset_class"]], on="symbol")
    m["group"] = m["country"].fillna(m["asset_class"])
    ok = m["status"] == "ok"
    return m.assign(ok=ok, from_start=ok & (m["first_date"] <= since)).groupby("group").agg(
        assets=("symbol", "size"), downloaded=("ok", "sum"), from_start=("from_start", "mean"),
        median_first=("first_date", "median"), last=("last_date", "max"))
