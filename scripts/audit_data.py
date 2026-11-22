"""Audita los datos descargados.

Salidas en data/audit/:
  flags.parquet    una fila por incidencia (símbolo, fecha, check, severidad, valor, detalle)
  summary.parquet  una fila por activo con el recuento por severidad y su estado
  plots/           un gráfico por incidencia de error o warning de precio (máx. --max-plots)

Uso: python scripts/audit_data.py [--max-plots N]
"""

import argparse
import shutil

import pandas as pd

from agentic_trading.config import DATA_DIR
from agentic_trading.data.audit import CHECKS, audit_universe
from agentic_trading.data.audit_plots import plot_flag
from agentic_trading.data.download import load_raw

PLOT_CHECKS = ["split_not_adjusted", "unit_spike", "unit_shift", "spike", "unrecorded_split",
               "split_like_move", "bad_dividend", "coarse_prices", "volume_shift",
               "wick_outlier", "open_outlier"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-plots", type=int, default=400)
    args = ap.parse_args()

    universe = pd.read_parquet(DATA_DIR / "universe.parquet")
    flags, summary = audit_universe(universe, load_raw)

    out = DATA_DIR / "audit"
    out.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(out / "flags.parquet", index=False)
    summary.to_parquet(out / "summary.parquet", index=False)

    # ---- informe
    pd.set_option("display.width", 160)
    print(f"Activos: {len(summary)} · estado: {summary.status.value_counts().to_dict()}\n")
    tab = flags.groupby(["check", "severity"]).agg(incidencias=("date", "size"),
                                                   activos=("symbol", "nunique"))
    tab = tab.reindex([(c, s) for c, (s, _) in CHECKS.items()] +
                      [i for i in tab.index if i[1] != CHECKS[i[0]][0]]).dropna()
    tab = tab.astype(int).reset_index()
    tab["descripción"] = tab["check"].map(lambda c: CHECKS[c][1])
    print(tab.to_string(index=False))

    print("\nActivos con más errores:")
    top = summary.sort_values(["error", "warning"], ascending=False).head(15)
    print(top[["symbol", "country", "rows", "error", "warning", "info"]].to_string(index=False))

    # ---- gráficos
    plots = out / "plots"
    shutil.rmtree(plots, ignore_errors=True)
    plots.mkdir()
    sel = flags[flags.check.isin(PLOT_CHECKS) & flags.severity.isin(["error", "warning"])]
    sel = sel.assign(o=sel.check.map(PLOT_CHECKS.index)).sort_values(["o", "symbol", "date"])
    for r in sel.head(args.max_plots).itertuples():
        name = f"{r.check}__{r.symbol}__{r.date.date()}.png"
        plot_flag(load_raw(r.symbol), r.symbol, r.date, r.check, r.detail, path=plots / name)
    print(f"\nGráficos: {min(len(sel), args.max_plots)}/{len(sel)} en {plots}")


if __name__ == "__main__":
    main()
