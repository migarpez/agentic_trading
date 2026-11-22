"""Construye el universo y lo guarda en data/universe.parquet (+ data/universe_listings.parquet).

Con --ext, el universo ampliado (medianas y pequeñas, universe.INDICES_EXT) sin las empresas del
universo de --exclude; solo con AGENTIC_TRADING_DATA apuntando a otra carpeta (`make ext`:
data/ext/), para no pisar el universo actual. Guarda también los componentes quitados por estar
ya en el universo actual (universe_excluded.parquet) y los leídos de cada índice
(universe_fetched.json).

Uso: python scripts/build_universe.py [--ext --exclude data/universe.parquet]
"""

import argparse
import json

import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.universe import INDICES_EXT, build_universe, possible_duplicates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", action="store_true", help="universo ampliado (medianas y pequeñas)")
    ap.add_argument("--exclude", help="universo cuyas empresas se quitan (con --ext)")
    args = ap.parse_args()

    if args.ext:
        if DATA_DIR.resolve() == (ROOT / "data").resolve():
            raise SystemExit("--ext pisaría data/universe.parquet: usa AGENTIC_TRADING_DATA "
                             "(p. ej. `make ext`).")
        if not args.exclude:
            raise SystemExit("--ext necesita --exclude (el universo actual).")
        uni, listings = build_universe(indices=INDICES_EXT, extra=[],
                                       exclude=pd.read_parquet(args.exclude), by_name=True)
    else:
        uni, listings = build_universe()
    info, uni.attrs = uni.attrs, {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "universe.parquet"
    uni.to_parquet(path, index=False)
    listings.to_parquet(DATA_DIR / "universe_listings.parquet", index=False)
    if args.ext:
        info["excluded"].to_parquet(DATA_DIR / "universe_excluded.parquet", index=False)
        (DATA_DIR / "universe_fetched.json").write_text(json.dumps(info["fetched"], indent=1))
        print(f"Leídos de Wikipedia: {info['fetched']}")
        print(f"Quitados por estar en el universo actual: {len(info['excluded'])}")

    print(f"Guardado {path} ({len(uni)} activos)\n")
    print(uni.groupby(["asset_class", "country"], dropna=False).size().to_string())

    if len(listings):
        print("\nMisma empresa en varios símbolos (se conserva el de mayor volumen, M EUR/día):")
        for _, g in listings.groupby("group"):
            g = g.sort_values("traded_value_eur", ascending=False)
            print("  " + "  vs  ".join(
                f"{'*' if k else ' '}{s} {v / 1e6:,.0f}"
                for s, v, k in zip(g.symbol, g.traded_value_eur, g.kept)))

    dups = possible_duplicates(uni)
    if len(dups):
        print("\nAVISO: posibles duplicados no declarados (revisar EQUIVALENT_LISTINGS "
              "o KNOWN_DISTINCT):")
        print(dups.to_string(index=False))


if __name__ == "__main__":
    main()
