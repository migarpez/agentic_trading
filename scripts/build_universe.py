"""Construye el universo y lo guarda en data/universe.parquet (+ data/universe_listings.parquet)."""

from agentic_trading.config import DATA_DIR
from agentic_trading.data.universe import build_universe, possible_duplicates


def main() -> None:
    uni, listings = build_universe()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "universe.parquet"
    uni.to_parquet(path, index=False)
    listings.to_parquet(DATA_DIR / "universe_listings.parquet", index=False)

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
