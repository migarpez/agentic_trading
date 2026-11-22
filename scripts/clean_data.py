"""Limpia los datos según la auditoría y vuelve a auditar el resultado.

Salidas en data/clean/:
  {símbolo}.parquet  open, high, low, close (ajustados), volume, adj_factor, valid,
                     ohlc_repaired
  _actions.parquet   una fila por acción aplicada (corte, reescalado, vela quitada, máscara...)
  _summary.parquet   una fila por activo: velas, velas no válidas, muestras utilizables
  _post_audit.parquet incidencias que quedan tras la limpieza

Uso: python scripts/clean_data.py [--ohlc-large repair|mask]
"""

import argparse
import shutil

import pandas as pd

from agentic_trading.config import DATA_DIR
from agentic_trading.data.clean import CleanParams, clean_universe
from agentic_trading.data.download import load_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlc-large", choices=["repair", "mask"], default="repair")
    args = ap.parse_args()
    params = CleanParams(ohlc_large=args.ohlc_large)

    universe = pd.read_parquet(DATA_DIR / "universe.parquet")
    flags = pd.read_parquet(DATA_DIR / "audit" / "flags.parquet")
    frames, actions, summary, post = clean_universe(universe, flags, load_raw, params)

    out = DATA_DIR / "clean"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for s, df in frames.items():
        df.to_parquet(out / f"{s}.parquet")
    actions.to_parquet(out / "_actions.parquet", index=False)
    summary.to_parquet(out / "_summary.parquet", index=False)

    # ---- informe
    pd.set_option("display.width", 160)
    print(f"Modo OHLC grave: {params.ohlc_large}\n")
    print("Acciones:")
    tab = actions.groupby(["action", "reason"]).agg(n=("date", "size"),
                                                    activos=("symbol", "nunique"))
    print(tab.to_string())
    tr = actions[actions.action == "truncate"]
    if len(tr):
        print("\nHistóricos cortados:")
        print(tr[["symbol", "date", "reason", "detail"]].to_string(index=False))

    tot = summary[["raw_rows", "clean_rows", "invalid_rows", "raw_samples",
                   "usable_samples"]].sum()
    n_rep = sum(int(df["ohlc_repaired"].sum()) for df in frames.values())
    print(f"\nVelas con OHLC grave reparado (ohlc_repaired): {n_rep:,}")
    print(f"Velas: {tot.raw_rows:,} -> {tot.clean_rows:,} "
          f"({tot.clean_rows / tot.raw_rows - 1:+.2%}); no válidas {tot.invalid_rows:,}")
    print(f"Muestras utilizables: {tot.raw_samples:,} -> {tot.usable_samples:,} "
          f"({tot.usable_samples / tot.raw_samples - 1:+.2%})")
    loss = (1 - summary.usable_samples / summary.raw_samples.where(summary.raw_samples > 0))
    worst = summary.assign(loss=loss).sort_values("loss", ascending=False).head(10)
    print("\nActivos que más muestras pierden:")
    print(worst[["symbol", "raw_samples", "usable_samples", "loss"]].to_string(
        index=False, formatters={"loss": "{:.0%}".format}))

    # ---- re-auditoría (la última pasada)
    print(f"\nPasadas: {actions['pass'].max() if len(actions) else 0}")
    post.to_parquet(out / "_post_audit.parquet", index=False)
    pre = flags[flags.severity == "error"].groupby("check").size().rename("antes")
    after = post[post.severity == "error"].groupby("check").size().rename("después")
    print("\nErrores antes y después de limpiar:")
    print(pd.concat([pre, after], axis=1).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
