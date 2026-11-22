"""Descarga OHLCV diario de todo el universo a data/raw/yahoo/.

Uso: python scripts/download_data.py [--refresh] [--limit N]
"""

import argparse

import pandas as pd

from agentic_trading.config import DATA_DIR
from agentic_trading.data.download import RAW_DIR, coverage, download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="vuelve a descargar todo")
    ap.add_argument("--limit", type=int, help="solo los N primeros símbolos (pruebas)")
    args = ap.parse_args()

    path = DATA_DIR / "universe.parquet"
    if not path.exists():                  # el universo no se rehace solo: `make universe`
        raise SystemExit(f"Falta {path}: créalo con `make universe` (lo rehace desde Wikipedia).")
    universe = pd.read_parquet(path)
    symbols = universe["symbol"].tolist()[: args.limit]

    man = download(symbols, refresh=args.refresh)
    man.to_parquet(RAW_DIR / "_manifest.parquet", index=False)

    print(f"\n{(man.status == 'ok').sum()}/{len(man)} descargados\n")
    cov = coverage(man, universe)
    cov["from_start"] = cov["from_start"].map("{:.0%}".format)
    print(cov.to_string())
    if (man.status != "ok").any():
        print("\nFallidos:", man.loc[man.status != "ok", "symbol"].tolist())


if __name__ == "__main__":
    main()
