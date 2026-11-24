"""Diagnóstico de diversidad, sin entrenar (23-11-2026): ¿combinar modelos distintos ayuda más que
añadir semillas del mismo?

- Diverso (8 modelos): R0 (miembros 0 y 1 de final20), mu_E1, mu_E2 y mu_E4 (2 miembros cada
  una). El μ_h de cada ejecución (la media de sus 2 miembros) se estandariza entre activos cada
  día (z), porque E2 está en unidades de rango; el diverso es la media de los 4 z.
- Homogéneo (8 modelos): miembros 0–7 de final20 (media de sus μ_h; estandarizar no cambia
  rangos ni deciles).
- Medidas, diverso − homogéneo por pares, en los 5 bloques, por bloque y en 2012–19:
  - IC de selección (media de h = 5 y 10): diferencia diaria, t de NW con 10 retardos (con 5 y 10
    por horizonte).
  - Pierna larga D9–D10 de μ_5 (scripts/mu_deciles.decile_series, la de deciles.md): exceso
    semanal de apertura a apertura, t de NW con 2 retardos. También D1–D2 y el top 20.

Lectura fijada antes de verlo: si el diverso mejora la pierna larga en 2012–19 con t ≥ 2,
cambiaría qué modelo va al test (parar antes de preparar el test). Si no, se sigue con final20.

Solo lee predicciones de 2000–2019. Escribe logs/mu_tree/diversity.md y diversity.json.

Uso: python scripts/mu_diversity.py
"""

import json

import numpy as np
import pandas as pd

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.model.metrics import daily_ic, nw_tstat
from agentic_trading.model.mu_metrics import (
    BLOCK_NAMES,
    OpenPrices,
    block_of,
    tail_series,
    weekly_frame,
)
from scripts.mu_compare import BLOCK_LABEL, fmt, load_run
from scripts.mu_deciles import decile_series

DIVERSE = {"R0": "final20:0,1", "E1": "mu_E1", "E2": "mu_E2", "E4": "mu_E4"}
HOMOG = "final20:0,1,2,3,4,5,6,7"
OUT = ROOT / "logs" / "mu_tree"
PERIODS = [*BLOCK_NAMES, "2012–19", "total"]


def zday(dates: np.ndarray, x: np.ndarray) -> np.ndarray:
    g = pd.Series(x).groupby(dates)
    sd = g.transform("std").to_numpy()
    return (x - g.transform("mean").to_numpy()) / np.where(sd > 0, sd, np.nan)


def period_mask(idx, p: str) -> np.ndarray:
    blk = block_of(idx)
    if p == "total":
        return np.ones(len(blk), bool)
    if p == "2012–19":
        return np.isin(blk, ["cv3", "cv4"])
    return blk == p


def paired_periods(d: pd.Series, lags: int) -> dict:
    d = d.dropna().sort_index()
    out = {}
    for p in PERIODS:
        m, t = nw_tstat(d[period_mask(d.index, p)].to_numpy(), lags)
        out[p] = {"mean": float(m), "t": float(t)}
    return out


def level_periods(s: pd.Series) -> dict:
    s = s.dropna().sort_index()
    return {p: float(s[period_mask(s.index, p)].mean()) for p in PERIODS}


def main() -> None:
    data = WindowData(DATA_DIR / "dataset")
    runs = {k: load_run(v) for k, v in DIVERSE.items()}
    hom = load_run(HOMOG)
    samples = hom["sample"].to_numpy()
    for k, p in runs.items():
        if not np.array_equal(p["sample"].to_numpy(), samples):
            raise SystemExit(f"{k}: muestras distintas de final20")
    smp = data.samples.iloc[samples]
    dates = smp["date"].to_numpy()
    rows = smp["row"].to_numpy()
    div = {h: np.nanmean([zday(dates, p[f"mu_{h}"].to_numpy()) for p in runs.values()], axis=0)
           for h in (5, 10)}
    hmu = {h: hom[f"mu_{h}"].to_numpy() for h in (5, 10)}

    # IC
    ic = {}
    for h in (5, 10):
        y = np.asarray(data.arr[f"y_{h}"])[rows].astype(float)
        ic[h] = {"div": daily_ic(dates, div[h], y), "hom": daily_ic(dates, hmu[h], y)}
    res = {"ic": {}}
    for h in (5, 10):
        res["ic"][f"h{h}"] = {"div": level_periods(ic[h]["div"]),
                              "hom": level_periods(ic[h]["hom"]),
                              "diff": paired_periods(ic[h]["div"] - ic[h]["hom"], h)}
    d_ic = ((ic[5]["div"] - ic[5]["hom"]) + (ic[10]["div"] - ic[10]["hom"])) / 2
    res["ic"]["mean"] = {"div": level_periods((ic[5]["div"] + ic[10]["div"]) / 2),
                         "hom": level_periods((ic[5]["hom"] + ic[10]["hom"]) / 2),
                         "diff": paired_periods(d_ic, 10)}

    # Colas semanales (las de deciles.md)
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    equity = (info["asset_class"] == "equity").to_numpy()
    dts = smp["date"]
    wk = weekly_dates(dts[dts.dt.dayofweek < 5])
    n_eq = smp[equity[smp["asset"].to_numpy()]].groupby("date").size()
    wk = wk[wk.isin(n_eq[n_eq >= 20].index)]
    prices = OpenPrices.from_clean(DATA_DIR / "clean",
                                   [data.meta["symbols"][a] for a in np.flatnonzero(equity)])
    wf = weekly_frame(data, samples, equity, wk, prices)
    pos = np.searchsorted(samples, wf["sample"].to_numpy())
    sd, sh = decile_series(wf, div[5][pos]), decile_series(wf, hmu[5][pos])
    td, th = tail_series(wf, div[5][pos]), tail_series(wf, hmu[5][pos])
    res["tails"] = {}
    for name, a, b in (("larga", sd["larga"], sh["larga"]), ("corta", sd["corta"], sh["corta"]),
                       ("D10", sd["D10"], sh["D10"]), ("top20", td["top20_oo"], th["top20_oo"])):
        res["tails"][name] = {"div": level_periods(a), "hom": level_periods(b),
                              "diff": paired_periods(a - b, 2)}
    # correlación media diaria entre los μ_5 de las ejecuciones (diversidad efectiva)
    df = pd.DataFrame({k: p["mu_5"].to_numpy() for k, p in runs.items()} | {"d": dates})
    corr = df.groupby("d")[list(runs)].corr(method="spearman").groupby(level=1).mean()
    res["rank_corr"] = corr.loc[list(runs), list(runs)].round(3).to_dict()
    hm = load_run("final20:0,1")
    hm2 = load_run("final20:2,3")
    c2 = pd.DataFrame({"a": hm["mu_5"].to_numpy(), "b": hm2["mu_5"].to_numpy(), "d": dates}) \
        .groupby("d")[["a", "b"]].corr(method="spearman").xs("a", level=1)["b"].mean()
    res["rank_corr_seeds"] = float(c2)

    larga = res["tails"]["larga"]["diff"]["2012–19"]
    res["verdict"] = {"larga_2012_19": larga, "stop": bool(larga["mean"] > 0 and larga["t"] >= 2)}
    (OUT / "diversity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    lab = BLOCK_LABEL | {"2012–19": "**2012–19**", "total": "5 bloques"}
    n = int(sd["larga"].notna().sum())
    L = ["# Diversidad: modelos distintos frente a más semillas (23-11-2026)", "",
         (f"Ver el docstring de scripts/mu_diversity.py. Diverso: media del μ estandarizado por "
          f"día de R0, E1, E2 y E4 (2 miembros cada una). Homogéneo: miembros 0–7 de final20. "
          f"{n} semanas para las colas (pb por semana sobre la media de las acciones, t de NW con "
          "2 retardos); IC diario (t de NW con 10 retardos en la media de h = 5 y 10)."), "",
         (f"Veredicto (lectura fijada): pierna larga en 2012–19, diverso − homogéneo "
          f"{fmt(larga['mean'])} pb (t {fmt(larga['t'], 2, False)}): "
          f"**{'mejora con t ≥ 2: parar' if res['verdict']['stop'] else 'no mejora con t ≥ 2: se sigue con final20'}**."),
         "", ("| periodo | IC div. / hom. | dif. IC (t) | larga div. / hom. | dif. larga (t) | "
         "corta dif. (t) | top 20 dif. (t) |"), "|---|---|---|---|---|---|---|"]
    t = res["tails"]
    for p in PERIODS:
        im = res["ic"]["mean"]
        L.append(f"| {lab[p]} | {im['div'][p]:.4f} / {im['hom'][p]:.4f} | "
                 f"{fmt(im['diff'][p]['mean'], 4)} ({fmt(im['diff'][p]['t'], 2, False)}) | "
                 f"{fmt(t['larga']['div'][p])} / {fmt(t['larga']['hom'][p])} | "
                 f"{fmt(t['larga']['diff'][p]['mean'])} ({fmt(t['larga']['diff'][p]['t'], 2, False)}) | "
                 f"{fmt(t['corta']['diff'][p]['mean'])} ({fmt(t['corta']['diff'][p]['t'], 2, False)}) | "
                 f"{fmt(t['top20']['diff'][p]['mean'])} ({fmt(t['top20']['diff'][p]['t'], 2, False)}) |")
    L += ["", "IC por horizonte, diverso − homogéneo (t de NW, retardos = h):", "",
          "| periodo | h = 5 | h = 10 |", "|---|---|---|"]
    for p in PERIODS:
        L.append(f"| {lab[p]} | " + " | ".join(
            f"{fmt(res['ic'][f'h{h}']['diff'][p]['mean'], 4)} "
            f"({fmt(res['ic'][f'h{h}']['diff'][p]['t'], 2, False)})" for h in (5, 10)) + " |")
    L += ["", ("Correlación de Spearman media diaria entre los μ_5 de las ejecuciones (2 miembros "
          f"cada una); entre dos pares de semillas de final20 (0–1 y 2–3): {c2:.3f}."), "",
          "| | " + " | ".join(DIVERSE) + " |", "|---|" + "---|" * len(DIVERSE)]
    for a in DIVERSE:
        L.append(f"| {a} | " + " | ".join(f"{res['rank_corr'][b][a]:.3f}" for b in DIVERSE) + " |")
    (OUT / "diversity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
