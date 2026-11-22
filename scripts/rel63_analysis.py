"""Análisis sin entrenar de las predicciones fuera de muestra de 2000–2019 de una ejecución de
validación cruzada (por defecto rel63), a h = 5 y 10.

1. μ y la reversión de 1 mes, por bloque y en total:
   - correlación diaria (Spearman entre activos) entre μ_h y la reversión;
   - IC de μ quitándole la reversión: cada día, residuo (MCO) de los rangos de μ_h sobre los de la
     reversión, y su IC con y_h (metrics.rank_residual).
2. Sesgo de supervivencia (el universo son los componentes actuales: supervivientes):
   - diferencia diaria de IC entre μ y la reversión, año a año;
   - por bloque, IC de μ y de la reversión separando cada día los activos con caída fuerte en la
     ventana (cierre ≥ 30 % por debajo del máximo de los cierres de sus 252 velas) del resto.

Las t son de Newey-West con retardos = h. Uso:
python scripts/rel63_analysis.py --run rel63 [--out logs/rel63_analysis.md]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import H2, ic_diff, targets
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import baseline_features, daily_ic, nw_tstat, rank_residual

DROP = np.log(0.7)        # cierre ≤ 70 % del máximo de los cierres de la ventana


def stat(ic: pd.Series, h: int) -> str:
    m, t = nw_tstat(ic.to_numpy(), h)
    return f"{m:+.4f} ({t:.1f})" if np.isfinite(m) else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rel63")
    ap.add_argument("--out", default=None, help="por defecto logs/<run>_analysis.md")
    args = ap.parse_args()
    pr = DATA_DIR / "predictions" / args.run
    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    lc = np.asarray(data.arr["log_close"])
    peak = pd.Series(lc).rolling(data.window, min_periods=data.window).max().to_numpy()

    res = {}                                   # por bloque y h: series diarias
    for k, f in enumerate(folds):
        path = pr / f"{f['name']}.parquet"
        if not path.exists():
            continue
        p = pd.read_parquet(path)
        dates, rows, y = targets(data, p["sample"].to_numpy())
        rev = baseline_features(data, rows)["reversal"]
        drop = (lc[rows] - peak[rows]) <= DROP
        share = pd.Series(drop).groupby(dates).mean()
        res[k] = {"share": share}
        for h in H2:
            mu = p[f"mu_{h}"].to_numpy()
            r = {"corr": daily_ic(dates, mu, rev), "ic": daily_ic(dates, mu, y[h]),
                 "rev": daily_ic(dates, rev, y[h]),
                 "resid": daily_ic(dates, rank_residual(dates, mu, rev), y[h])}
            for g, m in (("drop", drop), ("rest", ~drop)):
                r[f"ic_{g}"] = daily_ic(dates[m], mu[m], y[h][m])
                r[f"rev_{g}"] = daily_ic(dates[m], rev[m], y[h][m])
            res[k][h] = r
    if not res:
        raise SystemExit(f"No hay predicciones en {pr}")

    def name(k):
        b = folds[k]["block"]
        return f"{folds[k]['name']} {b[0].year}–{b[1].year}"

    def pooled(h, key):
        return pd.concat([res[k][h][key] for k in res]).sort_index()

    L = [f"# Análisis de μ fuera de muestra: `{args.run}`", "",
         ("Predicciones fuera de muestra de 2000–2019 (ensemble de cada bloque). Entre paréntesis, "
         "t de Newey-West (retardos = h)."), ""]

    # 1. μ y la reversión
    L += ["## 1. μ y la reversión de 1 mes", "",
          ("Correlación: Spearman diaria entre μ_h y la reversión (media). IC residual: IC de los "
          "rangos de μ_h quitándoles, cada día, lo que explican los de la reversión (MCO)."), "",
          "| bloque | h | correlación μ–reversión | IC μ | IC residual | IC reversión |",
          "|---|---|---|---|---|---|"]
    for h in H2:
        for k in res:
            r = res[k][h]
            L.append(f"| {name(k)} | {h} | {r['corr'].mean():+.3f} | {stat(r['ic'], h)} | "
                     f"{stat(r['resid'], h)} | {stat(r['rev'], h)} |")
        L.append(f"| **total** | {h} | {pooled(h, 'corr').mean():+.3f} | "
                 f"{stat(pooled(h, 'ic'), h)} | {stat(pooled(h, 'resid'), h)} | "
                 f"{stat(pooled(h, 'rev'), h)} |")
    L.append("")

    # 2a. año a año
    L += ["## 2. Sesgo de supervivencia", "",
          "### Diferencia diaria de IC entre μ y la reversión, año a año", "",
          "| año | h = 5: diferencia (t) | h = 10: diferencia (t) | días |", "|---|---|---|---|"]
    diffs = {h: (pooled(h, "ic") - pooled(h, "rev")).dropna() for h in H2}
    years = sorted(set(diffs[5].index.year) | set(diffs[10].index.year))
    for yr in years:
        cells = [stat(diffs[h][diffs[h].index.year == yr], h) for h in H2]
        L.append(f"| {yr} | {cells[0]} | {cells[1]} | {(diffs[5].index.year == yr).sum()} |")
    for h in H2:
        mean, t, days = ic_diff(pooled(h, "ic"), pooled(h, "rev"), h)
        L.append(f"| **total h = {h}** | {mean:+.4f} ({t:.1f}) | | {days} |" if h == 5 else
                 f"| **total h = {h}** | | {mean:+.4f} ({t:.1f}) | {days} |")
    L.append("")

    # 2b. caída fuerte
    L += ["### Caída fuerte en la ventana frente al resto", "",
          ("Caída fuerte: cierre ≥ 30 % por debajo del máximo de los cierres de las 252 velas de la "
          "ventana. IC calculado cada día dentro de cada grupo (días con ≥ 20 activos en el grupo)."),
          "", ("| bloque | h | % con caída (media diaria) | IC μ caída | IC μ resto | "
          "IC reversión caída | IC reversión resto | días caída |"), "|---|---|---|---|---|---|---|---|"]
    for h in H2:
        for k in res:
            r = res[k][h]
            L.append(f"| {name(k)} | {h} | {100 * res[k]['share'].mean():.1f} % | "
                     f"{stat(r['ic_drop'], h)} | {stat(r['ic_rest'], h)} | "
                     f"{stat(r['rev_drop'], h)} | {stat(r['rev_rest'], h)} | {len(r['ic_drop'])} |")
        share = pd.concat([res[k]["share"] for k in res])
        L.append(f"| **total** | {h} | {100 * share.mean():.1f} % | "
                 f"{stat(pooled(h, 'ic_drop'), h)} | {stat(pooled(h, 'ic_rest'), h)} | "
                 f"{stat(pooled(h, 'rev_drop'), h)} | {stat(pooled(h, 'rev_rest'), h)} | "
                 f"{len(pooled(h, 'ic_drop'))} |")
    L.append("")

    out = Path(args.out) if args.out else ROOT / "logs" / f"{args.run}_analysis.md"
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nEscrito en {out}")


if __name__ == "__main__":
    main()
