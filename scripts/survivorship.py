"""Diagnóstico sin entrenar (23-11-2026): ¿la ventaja de μ en 2000–11 viene del sesgo de
supervivencia?

- Fecha de alta en el S&P 500 ("Date added") de la tabla actual de componentes de Wikipedia
  (data.universe.fetch_sp500_added), guardada con la fecha de descarga en data/sp500_added.parquet
  (se reutiliza si existe; --refresh la vuelve a bajar).
- Solo las acciones del S&P 500 del universo actual (índice SP500 en universe.parquet), por
  símbolo de Wikipedia (source_symbol). Grupos: antiguas, en el índice desde antes de 2000;
  tardías, alta en 2012 o después; intermedias, el resto. Las antiguas también son supervivientes,
  pero no se eligieron por haber crecido; las tardías, sí.
- Por grupo y por periodo (2000–11 y 2012–19), con las predicciones fuera de muestra de final20
  (20 miembros):
  - IC de selección dentro del grupo, cada día (model.metrics.daily_ic: Spearman con el target
    y_5, con ≥ 20 acciones del grupo ese día), de μ_5 y de la mezcla (calculada dentro del grupo:
    rangos de la reversión y el momentum entre las acciones del grupo), y su diferencia
    (μ − mezcla) con t de NW (5 retardos).
  - Diagnóstico: pierna larga (quintil alto dentro del grupo, D9–D10 de
    scripts/mu_deciles.decile_series) de μ y de la mezcla, exceso de apertura a apertura sobre la
    media del grupo, pb por semana, t de NW (2 retardos), en las fechas semanales de deciles.md.
  - Número de acciones por grupo y fecha (media, mínimo y máximo de los días con IC).

Lectura, fijada antes de verlo:
- Si en 2000–11 la ventaja de μ sobre la mezcla (diferencia de IC) en las antiguas es menos de la
  mitad que en las tardías, la ventaja de 2000–11 era sobre todo sesgo.
- Si es más de la mitad, era real y se ha ido.

Solo lee datos y predicciones (2000–2019). Escribe logs/survivorship.md y .json.
Uso: python scripts/survivorship.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.data.universe import fetch_sp500_added
from agentic_trading.model.metrics import baseline_features, baseline_mix, daily_ic, nw_tstat
from agentic_trading.model.mu_metrics import BLOCK_NAMES, OpenPrices, block_of, weekly_frame
from scripts.mu_compare import load_run
from scripts.mu_deciles import decile_series

OUT = ROOT / "logs"
REPORT = "survivorship"
CACHE = DATA_DIR / "sp500_added.parquet"
GROUPS = ["antiguas", "intermedias", "tardías"]
PERIODS = {"2000–11": ["cv0", "cv1", "cv2"], "2012–19": ["cv3", "cv4"]}
IC_LAGS, LEG_LAGS, MIN_N = 5, 2, 20


def log(msg: str) -> None:
    print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def group_of(date_added: pd.Series) -> pd.Series:
    g = pd.Series("intermedias", index=date_added.index, dtype=object)
    g[date_added < pd.Timestamp("2000-01-01")] = "antiguas"
    g[date_added >= pd.Timestamp("2012-01-01")] = "tardías"
    return g


def stats(s: pd.Series, lags: int) -> dict:
    s = s.dropna().sort_index()
    blk = block_of(s.index)
    out = {}
    for name, bl in {**PERIODS, **{b: [b] for b in BLOCK_NAMES}}.items():
        m, t = nw_tstat(s[np.isin(blk, bl)].to_numpy(), lags)
        out[name] = {"mean": m, "t": t, "n": int(np.isin(blk, bl).sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh or not CACHE.exists():
        fetch_sp500_added().to_parquet(CACHE, index=False)
    added = pd.read_parquet(CACHE)
    fetched = str(pd.Timestamp(added["fetched"].iloc[0]).date())

    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    sp = uni[(uni["asset_class"] == "equity") &
             uni["indices"].astype(str).str.contains("SP500")].copy()
    da = added.drop_duplicates("source_symbol").set_index("source_symbol")["date_added"]
    sp["date_added"] = sp["source_symbol"].map(da)
    missing = sorted(sp.loc[sp["date_added"].isna(), "symbol"])
    sp = sp.dropna(subset=["date_added"])
    sp["group"] = group_of(sp["date_added"]).to_numpy()
    grp = sp.set_index("symbol")["group"]
    log(f"S&P 500 del universo: {len(sp) + len(missing)}; sin fecha de alta: {missing}")

    data = WindowData(DATA_DIR / "dataset")
    pred = load_run("final20")
    samples = pred["sample"].to_numpy()
    smp = data.samples.iloc[samples]
    sym = np.asarray(data.meta["symbols"], object)[smp["asset"].to_numpy()]
    g = grp.reindex(sym).to_numpy(object)
    rows = smp["row"].to_numpy()
    dates = smp["date"].to_numpy()
    y5 = np.asarray(data.arr["y_5"])[rows].astype(float)
    base = baseline_features(data, rows)
    key = pd.Series(dates).astype(str).to_numpy() + "|" + pd.Series(g).astype(str).to_numpy()
    mix = baseline_mix(key, base["reversal"], base["momentum"])
    mu = pred["mu_5"].to_numpy()

    # semanas para la pierna larga, como en deciles.md
    equity = np.zeros(len(data.meta["symbols"]), bool)
    equity[[data.meta["symbols"].index(s) for s in grp.index if s in data.meta["symbols"]]] = True
    dd = smp["date"]
    wk = weekly_dates(dd[dd.dt.dayofweek < 5])
    prices = OpenPrices.from_clean(DATA_DIR / "clean", list(grp.index))
    wf = weekly_frame(data, samples, equity, wk, prices)
    pos = np.searchsorted(samples, wf["sample"].to_numpy())
    wf["mu"], wf["mix"] = mu[pos], mix[pos]
    wf["group"] = grp.reindex(np.asarray(data.meta["symbols"], object)[wf["asset"]]).to_numpy()

    res = {"fetched": fetched, "missing": missing,
           "n_assets": sp["group"].value_counts().to_dict(), "groups": {}}
    for name in GROUPS:
        sel = g == name
        ic_mu = daily_ic(dates[sel], mu[sel], y5[sel], MIN_N)
        ic_mix = daily_ic(dates[sel], mix[sel], y5[sel], MIN_N)
        diff = (ic_mu - ic_mix).dropna()
        n_day = pd.Series(dates[sel]).value_counts()
        n_day.index = pd.DatetimeIndex(n_day.index)
        n_day = n_day[n_day.index.isin(diff.index)]
        sub = wf[wf["group"] == name]
        sub = sub[sub.groupby("date")["mu"].transform("size") >= MIN_N]
        leg_mu = decile_series(sub, sub["mu"].to_numpy())["larga"]
        leg_mix = decile_series(sub, sub["mix"].to_numpy())["larga"]
        cnt = {}
        for p, bl in PERIODS.items():
            x = n_day[np.isin(block_of(n_day.index), bl)]
            cnt[p] = {"mean": float(x.mean()), "min": int(x.min()), "max": int(x.max()),
                      "days": len(x)} if len(x) else {"mean": 0.0, "min": 0, "max": 0, "days": 0}
        res["groups"][name] = {
            "ic_mu": stats(ic_mu, IC_LAGS), "ic_mix": stats(ic_mix, IC_LAGS),
            "ic_diff": stats(diff, IC_LAGS), "leg_mu": stats(leg_mu, LEG_LAGS),
            "leg_mix": stats(leg_mix, LEG_LAGS), "leg_diff": stats(leg_mu - leg_mix, LEG_LAGS),
            "count": cnt}
        log(f"{name} hecho")
    old = res["groups"]["antiguas"]["ic_diff"]["2000–11"]["mean"]
    late = res["groups"]["tardías"]["ic_diff"]["2000–11"]["mean"]
    res["verdict"] = {"old": old, "late": late, "ratio": old / late if late else float("nan"),
                      "bias": bool(late > 0 and old < 0.5 * late)}
    (OUT / f"{REPORT}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                   default=float))
    (OUT / f"{REPORT}.md").write_text("\n".join(markdown(res)) + "\n")
    log("hecho")


def f(x, nd=4, sign=True):
    return "—" if x is None or not np.isfinite(x) else f"{x:{'+' if sign else ''}.{nd}f}"


def mt(d, nd=4):
    return f"{f(d['mean'], nd)} ({f(d['t'], 2, False)})"


def markdown(res: dict) -> list[str]:
    v, G = res["verdict"], res["groups"]
    L = ["# ¿La ventaja de μ en 2000–11 viene del sesgo de supervivencia? (23-11-2026)", "",
         (f"Ver el docstring de scripts/survivorship.py. Fechas de alta de la tabla actual de "
          f"Wikipedia (descargada el {res['fetched']}). Acciones del S&P 500 del universo: "
          + ", ".join(f"{k} {res['n_assets'].get(k, 0)}" for k in GROUPS)
          + (f"; sin fecha de alta: {', '.join(res['missing'])}" if res["missing"] else "")
          + ". μ_5 de final20 (20 miembros); IC de selección diario dentro del grupo con y_5, t "
          "de NW (5 retardos); pierna larga (quintil alto) semanal, pb por semana, t de NW "
          "(2 retardos)."), "",
         "## Lectura (fijada antes de verlo)", ""]
    if not v["late"] > 0:
        L.append(f"- En 2000–11, μ no tiene ventaja sobre la mezcla en las tardías "
                 f"({f(v['late'])}): la regla no se puede aplicar tal como está escrita.")
    else:
        L.append(f"- Ventaja de μ sobre la mezcla (IC) en 2000–11: antiguas {f(v['old'])}, "
                 f"tardías {f(v['late'])}; antiguas / tardías = {v['ratio']:.2f}.")
        L.append("- Según la regla: " + (
            "**menos de la mitad**: la ventaja de 2000–11 era sobre todo sesgo de supervivencia."
            if v["bias"] else
            "**más de la mitad**: la ventaja de 2000–11 era real y se ha ido."))
    L += ["", "## IC de selección dentro del grupo (media diaria, t)", "",
          "| grupo | periodo | μ_5 | mezcla | μ − mezcla (t) | acciones por día (media, mín.–máx.) |",
          "|---|---|---|---|---|---|"]
    for name in GROUPS:
        r = G[name]
        for p in PERIODS:
            c = r["count"][p]
            L.append(f"| {name} | {p} | {mt(r['ic_mu'][p])} | {mt(r['ic_mix'][p])} | "
                     f"{mt(r['ic_diff'][p])} | {c['mean']:.0f} ({c['min']}–{c['max']}) |")
    L += ["", "## Pierna larga dentro del grupo (quintil alto, pb por semana)", "",
          "| grupo | periodo | μ_5 | mezcla | μ − mezcla (t) |", "|---|---|---|---|---|"]
    for name in GROUPS:
        r = G[name]
        for p in PERIODS:
            L.append(f"| {name} | {p} | {mt(r['leg_mu'][p], 1)} | {mt(r['leg_mix'][p], 1)} | "
                     f"{mt(r['leg_diff'][p], 1)} |")
    L += ["", "## Por bloque: μ − mezcla (IC y pierna larga)", "",
          "| grupo | " + " | ".join(BLOCK_NAMES) + " |", "|---|" + "---|" * len(BLOCK_NAMES)]
    for name in GROUPS:
        r = G[name]
        L.append(f"| {name} (IC) | " + " | ".join(mt(r["ic_diff"][b]) for b in BLOCK_NAMES)
                 + " |")
        L.append(f"| {name} (pierna larga) | " + " | ".join(
            mt(r["leg_diff"][b], 1) for b in BLOCK_NAMES) + " |")
    L += ["", "## Decisiones (opción conservadora ante la duda)", "",
          ("- Fechas de alta de la tabla de hoy, cruzadas por el símbolo de Wikipedia con el "
           "universo (congelado el 6-11-2026); una acción sin fecha queda fuera. La fecha es la "
           "de la última entrada: una empresa que salió y volvió cuenta por la última."),
          ("- Mezcla calculada dentro de cada grupo (rangos entre sus acciones), para que el IC "
           "dentro del grupo compare lo mismo."),
          ("- IC solo en los días con ≥ 20 acciones del grupo; pierna larga solo en las fechas "
           "semanales con ≥ 20."),
          ("- Si en 2000–11 μ no tiene ventaja en las tardías, la regla no es aplicable y se "
           "dice así.")]
    return L


if __name__ == "__main__":
    main()
