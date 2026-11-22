"""Diagnóstico sin entrenar sobre mu_tree (22-11-2026): dónde cae la mejora de IC de E1, E2 y E4.

En las fechas semanales del agente (las de mu_compare.py), las acciones de cada fecha se reparten
en deciles de μ_5 (D1 el más bajo, D10 el más alto; decil = ⌈10·rango/n⌉). Por decil, el exceso de
apertura a apertura sobre la media de las acciones (la medida del top 20 de model/mu_metrics.py),
en pb por semana, con los 5 bloques juntos y por bloque. Para mu_E1, mu_E2, mu_E4 y final20
(referencia), la diferencia por pares con R0 (miembros 0 y 1 de final20) en cada decil, con su t
de NW (2 retardos). Resumen: pierna larga (media de D9 y D10) y pierna corta (media de D1 y D2).

Lectura fijada antes de verlo: si la mejora está sobre todo en D1–D2 (más negativos) y D9–D10 no
sube, estas opciones solo mejoran la pierna corta y no le sirven al agente (sin cortos).

Solo lee predicciones de 2000–2019. Escribe logs/mu_tree/deciles.md y deciles.json.

Uso: python scripts/mu_deciles.py
"""

import json

import numpy as np
import pandas as pd

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.model.mu_metrics import BLOCK_NAMES, OpenPrices, level, paired, weekly_frame
from scripts.mu_compare import BLOCK_LABEL, fmt, load_run

RUNS = {"R0": "final20:0,1", "E1 market": "mu_E1", "E2 target_rank": "mu_E2",
        "E4 tail_weight": "mu_E4", "final20 (20 miembros)": "final20"}
OUT = ROOT / "logs" / "mu_tree"


def decile_series(wf: pd.DataFrame, mu5: np.ndarray) -> pd.DataFrame:
    """Exceso semanal (pb) de cada decil de μ_5 sobre la media de las acciones de la fecha."""
    df = wf.assign(mu=mu5)
    g = df.groupby("date")
    pct = g["mu"].rank(method="first") / g["mu"].transform("size")
    df["dec"] = np.ceil(10 * pct).clip(1, 10).astype(int)
    allm = g["ret_oo"].mean()
    out = {f"D{d}": 1e4 * (df[df["dec"] == d].groupby("date")["ret_oo"].mean() - allm)
           for d in range(1, 11)}
    s = pd.DataFrame(out).dropna(how="all")
    s["larga"] = (s["D9"] + s["D10"]) / 2
    s["corta"] = (s["D1"] + s["D2"]) / 2
    return s


def main() -> None:
    data = WindowData(DATA_DIR / "dataset")
    preds = {k: load_run(v) for k, v in RUNS.items()}
    samples = preds["R0"]["sample"].to_numpy()
    for k, p in preds.items():
        if not np.array_equal(p["sample"].to_numpy(), samples):
            raise SystemExit(f"{k}: muestras distintas de R0")
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    equity = (info["asset_class"] == "equity").to_numpy()
    smp = data.samples.iloc[samples]
    dates = smp["date"]
    wk = weekly_dates(dates[dates.dt.dayofweek < 5])
    n_eq = smp[equity[smp["asset"].to_numpy()]].groupby("date").size()
    wk = wk[wk.isin(n_eq[n_eq >= 20].index)]
    prices = OpenPrices.from_clean(DATA_DIR / "clean",
                                   [data.meta["symbols"][a] for a in np.flatnonzero(equity)])
    wf = weekly_frame(data, samples, equity, wk, prices)
    pos = np.searchsorted(samples, wf["sample"].to_numpy())
    ser = {k: decile_series(wf, p["mu_5"].to_numpy()[pos]) for k, p in preds.items()}
    cols = [f"D{d}" for d in range(1, 11)] + ["larga", "corta"]
    res = {k: {c: {"level": level(s[c]),
                   **({"diff": paired(s[c], ser["R0"][c])} if k != "R0" else {})}
               for c in cols} for k, s in ser.items()}
    (OUT / "deciles.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))

    n = int(ser["R0"]["D10"].notna().sum())
    L = ["# Deciles de μ_5: dónde cae la mejora de IC (mu_tree, 22-11-2026)", "",
         f"Ver el docstring de scripts/mu_deciles.py. {n} semanas; exceso de apertura a apertura "
         "sobre la media de las acciones, pb por semana; diferencia por pares con R0, t de NW "
         "(2 retardos). Pierna larga = media de D9 y D10; corta = media de D1 y D2 (en la corta, "
         "una diferencia negativa es mejora).", "",
         "## Resumen, 5 bloques juntos", "",
         "| ejecución | larga (D9–D10) | − R0 (t) | corta (D1–D2) | − R0 (t) | D10 − R0 (t) | "
         "D1 − R0 (t) |", "|---|---|---|---|---|---|---|"]

    def dt(k, c, b=None):
        d = res[k][c]["diff"]
        x = d if b is None else d["blocks"][b]
        return f"{fmt(x['mean'])} ({fmt(x['t'], 2, False)})"

    for k in RUNS:
        r = res[k]
        if k == "R0":
            L.append(f"| R0 | {fmt(r['larga']['level']['mean'])} | — | "
                     f"{fmt(r['corta']['level']['mean'])} | — | — | — |")
        else:
            L.append(f"| {k} | {fmt(r['larga']['level']['mean'])} | {dt(k, 'larga')} | "
                     f"{fmt(r['corta']['level']['mean'])} | {dt(k, 'corta')} | "
                     f"{dt(k, 'D10')} | {dt(k, 'D1')} |")
    L += ["", "## Por decil, 5 bloques juntos: nivel (R0) y diferencia con R0 (t)", "",
          "| decil | R0 | " + " | ".join(k for k in RUNS if k != "R0") + " |",
          "|---|---|" + "---|" * (len(RUNS) - 1)]
    for c in cols[:10]:
        L.append(f"| {c} | {fmt(res['R0'][c]['level']['mean'])} | "
                 + " | ".join(dt(k, c) for k in RUNS if k != "R0") + " |")
    L += ["", "## Por bloque: larga y corta, diferencia con R0 (t)", "",
          "| bloque | R0 larga / corta | " + " | ".join(f"{k}: larga / corta" for k in RUNS
                                                        if k != "R0") + " |",
          "|---|---|" + "---|" * (len(RUNS) - 1)]
    for b in BLOCK_NAMES:
        lv = res["R0"]
        L.append(f"| {BLOCK_LABEL[b]} | {fmt(lv['larga']['level']['blocks'][b])} / "
                 f"{fmt(lv['corta']['level']['blocks'][b])} | "
                 + " | ".join(f"{dt(k, 'larga', b)} / {dt(k, 'corta', b)}"
                              for k in RUNS if k != "R0") + " |")
    (OUT / "deciles.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
