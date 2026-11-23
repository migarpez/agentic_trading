"""Diagnóstico sin entrenar (23-11-2026): ¿hay información en datos crudos que μ no ve? Desde 2012,
μ vale en la cola lo mismo que la mezcla (logs/mu_tree/decay.md y recency.md).

Universo actual, solo acciones, solo 2000–2019 (precios y volúmenes hasta el 31-12-2019; el test no
se abre). Las señales son solo para medir: no se dan al agente. Todas causales: en la fecha
semanal t usan solo datos hasta el cierre de t.

Fechas: las semanales de scripts/mu_deciles.py (baselines.weekly_dates con ≥ 20 acciones), con las
muestras de final20 (20 miembros). Semana de tenencia de la fecha w_j: las velas d con
w_j < d ≤ w_{j+1} (la medida es de apertura a apertura, de la vela siguiente a w_j a la siguiente
a w_{j+1}).

Señales:
- S1, anuncio de resultados: el máximo de log(V_d / mediana de V > 0 de las 252 velas válidas
  anteriores a d, con ≥ 126) en las velas de la semana de tenencia de hace 52 fechas semanales
  (w_{i−52} < d ≤ w_{i−51}). Por el calendario de cada acción; V = 0 no cuenta.
- S2, estacionalidad: la media de la rentabilidad relativa mensual de la acción (log-retorno del
  mes, del último cierre del mes anterior al último del mes, menos la media de las acciones con
  dato ese mes) en el mismo mes natural (el de t) de los años anteriores: hasta 10, mínimo 3. Con
  los datos desde enero de 2000, hay señal desde 2003.
- Vecinas: para cada acción, las 10 de su región (EE. UU. o Europa) con mayor correlación de los
  log-retornos diarios relativos (menos la media de la región ese día) en las 252 sesiones de la
  región que acaban en el último día del mes anterior a t (recalculadas cada mes); correlaciones
  por pares con ≥ 200 días válidos en común. Sesiones de la región: los días con ≥ 20 acciones de
  la región con vela válida; retorno diario, de la sesión anterior de la región a la del día (NaN
  si falta alguna de las dos velas).
  - S3a: la media del log-retorno de las vecinas en las últimas 21 sesiones de la región hasta t
    (cierres arrastrados hasta 5 sesiones), con ≥ 5 vecinas con dato.
  - S3b: S3a menos el de la propia acción en las mismas 21 sesiones.
  Retornos brutos (no relativos a la región), como en el enunciado; el orden es entre todas las
  acciones de la fecha.

Medidas por señal (deciles de scripts/mu_deciles.decile_series: exceso de apertura a apertura
sobre la media de las acciones, pb por semana; t de NW con 2 retardos), solo en las filas con la
señal (μ sola y la mezcla, en las mismas filas, para que los pares sean exactos):
- Pierna larga (D9–D10) de la señal sola.
- Lo que añade a μ: pierna larga del rango combinado (rango de μ_5 de final20 + rango de la señal,
  a partes iguales; rangos percentiles dentro de la fecha) menos la de μ sola, por pares por semana.
- Lo mismo frente a la mezcla (rango de la mezcla + rango de la señal, menos la mezcla).
- Correlación de Spearman media (por fecha) con μ_5 y con la mezcla; cobertura (parte de las
  filas acción-semana con la señal).
Por bloque y 2000–11 frente a 2012–19.

Regla, fijada antes de ver resultados:
- Una señal aporta si, en 2012–19, el rango combinado con μ mejora la pierna larga de μ sola con
  t ≥ 2, y la mejora es positiva en 2012–15 y en 2016–19.
- Si aporta, el siguiente paso es darle a la red los datos crudos correspondientes (las velas
  diarias de la misma época del año en años anteriores para S1 y S2, las de las vecinas para S3).
- 4 señales: comparación múltiple (con 4 pruebas, Bonferroni al 5 % pediría t ≥ 2,5).

Escribe logs/raw_signals.md y .json. Uso: python scripts/raw_signals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.agent.simulator import Market
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.model.metrics import baseline_features, baseline_mix, nw_tstat
from agentic_trading.model.mu_metrics import (
    BLOCK_NAMES,
    END,
    NW_LAGS_WEEKLY,
    OpenPrices,
    block_of,
    weekly_frame,
)
from scripts.mu_compare import BLOCK_LABEL, load_run
from scripts.mu_deciles import decile_series

OUT = ROOT / "logs"
REPORT = "raw_signals"
SIGNALS = {"S1": "anuncio de resultados (volumen de hace 52 semanas)",
           "S2": "estacionalidad (mismo mes de años anteriores)",
           "S3a": "vecinas: retorno a 21 sesiones",
           "S3b": "vecinas menos la propia acción"}
PERIODS = {"2000–11": ["cv0", "cv1", "cv2"], "2012–19": ["cv3", "cv4"]}
RULE = {"t_min": 2.0, "blocks_positive": ["cv3", "cv4"]}
N_NEIGH, MIN_DAYS, CORR_WIN, RET_WIN, MIN_NEIGH = 10, 200, 252, 21, 5


def log(msg: str) -> None:
    print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------------- señales

def s1_volume(frames: dict, wk: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """S1 por símbolo: Series índice de fecha semanal → señal."""
    out = {}
    for s, df in frames.items():
        d = df[df["valid"].astype(bool)]
        v = d["volume"].astype(float).where(d["volume"] > 0)
        med = v.rolling(252, min_periods=126).median().shift(1)
        rel = np.log(v / med).dropna()
        j = wk.searchsorted(rel.index, side="left")          # w_{j−1} < d ≤ w_j
        ok = j < len(wk)
        mx = pd.Series(rel.to_numpy()[ok]).groupby(j[ok]).max()   # semana de tenencia de w_{j−1}
        # en w_i: la semana de tenencia de w_{i−52}, que acaba en w_{i−51}: cubo j = i − 51
        i = mx.index.to_numpy() + 51
        keep = i < len(wk)
        out[s] = pd.Series(mx.to_numpy()[keep], index=wk[i[keep]])
    return out


def s2_season(close: pd.DataFrame) -> pd.DataFrame:
    """S2 por mes (índice: fin de mes) y símbolo: media de los años anteriores del mismo mes."""
    last = close.resample("ME").last()
    has = close.notna().resample("ME").sum() > 0
    r = np.log(last / last.shift(1)).where(has & has.shift(1, fill_value=False))
    rel = r.sub(r.mean(axis=1), axis=0)
    out = pd.DataFrame(np.nan, index=rel.index, columns=rel.columns)
    for m in range(1, 13):
        sub = rel[rel.index.month == m]
        out.loc[sub.index] = sub.shift(1).rolling(10, min_periods=3).mean().to_numpy()
    return out


def neighbours(close: pd.DataFrame, cols: list[str], wk: pd.DatetimeIndex
               ) -> tuple[dict, dict]:
    """S3a y S3b de una región: dict fecha semanal → Series por símbolo."""
    c = close[cols]
    days = c.index[c.notna().sum(axis=1) >= 20]
    c = c.loc[days]
    lc = np.log(c)
    r = lc - lc.shift(1)                                   # NaN si falta alguna de las dos
    rel = r.sub(r.mean(axis=1), axis=0)
    lcf = lc.ffill(limit=5)
    ret21 = lcf - lcf.shift(RET_WIN)
    month_end = pd.Series(np.arange(len(days)), index=days).groupby(days.to_period("M")).max()
    neigh_by_month = {}
    for per, e in month_end.items():
        if e + 1 < CORR_WIN:
            continue
        win = rel.iloc[e + 1 - CORR_WIN:e + 1]
        cm = win.corr(min_periods=MIN_DAYS).to_numpy()
        np.fill_diagonal(cm, np.nan)
        nb = {}
        for k in range(len(cols)):
            row = cm[k]
            ok = np.flatnonzero(np.isfinite(row))
            if len(ok) >= N_NEIGH:
                nb[k] = ok[np.argsort(-row[ok], kind="stable")[:N_NEIGH]]
        neigh_by_month[per] = nb
    a_out, b_out = {}, {}
    R = ret21.to_numpy()
    for t in wk:
        row = days.searchsorted(t, side="right") - 1
        prev = (t.to_period("M") - 1)
        if row < 0 or prev not in neigh_by_month:
            continue
        nb, x = neigh_by_month[prev], R[row]
        a = np.full(len(cols), np.nan)
        for k, idx in nb.items():
            v = x[idx]
            v = v[np.isfinite(v)]
            if len(v) >= MIN_NEIGH:
                a[k] = v.mean()
        a_out[t] = pd.Series(a, index=cols)
        b_out[t] = pd.Series(a - x, index=cols)
    return a_out, b_out


# ----------------------------------------------------------------------------- medidas

def pct_rank(df: pd.DataFrame, col: str) -> np.ndarray:
    return df.groupby("date")[col].rank(pct=True).to_numpy()


def period_stats(s: pd.Series) -> dict:
    s = s.dropna().sort_index()
    blk = block_of(s.index)
    out = {}
    for name, bl in {**{b: [b] for b in BLOCK_NAMES}, **PERIODS, "total": BLOCK_NAMES}.items():
        x = s[np.isin(blk, bl)].to_numpy()
        m, t = nw_tstat(x, NW_LAGS_WEEKLY)
        out[name] = {"mean": m, "t": t, "n": len(x)}
    return out


def spearman_by_date(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    ra = df.groupby("date")[a].rank()
    rb = df.groupby("date")[b].rank()
    return pd.DataFrame({"a": ra, "b": rb, "date": df["date"]}).groupby("date").apply(
        lambda g: g["a"].corr(g["b"]), include_groups=False)


def measure(wf: pd.DataFrame, sig: np.ndarray) -> dict:
    blk_all = block_of(wf["date"])
    cov = {p: float(np.isfinite(sig[np.isin(blk_all, bl)]).mean())
           for p, bl in {**PERIODS, "total": BLOCK_NAMES}.items()}
    sub = wf.assign(sig=sig)
    sub = sub[np.isfinite(sub["sig"])]
    n_day = sub.groupby("date")["sig"].transform("size")
    sub = sub[n_day >= 20].copy()                      # deciles con ≥ 20 acciones
    r_mu, r_mix, r_s = pct_rank(sub, "mu"), pct_rank(sub, "mix"), pct_rank(sub, "sig")
    leg = {k: decile_series(sub, v)["larga"] for k, v in {
        "señal": sub["sig"].to_numpy(), "μ": sub["mu"].to_numpy(), "μ + señal": r_mu + r_s,
        "mezcla": sub["mix"].to_numpy(), "mezcla + señal": r_mix + r_s}.items()}
    corr_mu = spearman_by_date(sub, "sig", "mu")
    corr_mix = spearman_by_date(sub, "sig", "mix")
    corr = {}
    for p, bl in {**PERIODS, "total": BLOCK_NAMES}.items():
        corr[p] = {"μ_5": float(corr_mu[np.isin(block_of(corr_mu.index), bl)].mean()),
                   "mezcla": float(corr_mix[np.isin(block_of(corr_mix.index), bl)].mean())}
    return {"coverage": cov, "corr": corr,
            "legs": {k: period_stats(v) for k, v in leg.items()},
            "adds_mu": period_stats(leg["μ + señal"] - leg["μ"]),
            "adds_mix": period_stats(leg["mezcla + señal"] - leg["mezcla"])}


def verdict(res: dict) -> dict:
    out = {}
    for s in SIGNALS:
        a = res[s]["adds_mu"]
        out[s] = bool(a["2012–19"]["t"] >= RULE["t_min"] and
                      all(a[b]["mean"] > 0 for b in RULE["blocks_positive"]))
    return out


# ----------------------------------------------------------------------------- informe

def f(x, nd=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:{'+' if sign else ''}.{nd}f}"


def mt(d):
    return f"{f(d['mean'])} ({f(d['t'], 2, False)})"


DECISIONS = [
    ("Medidas solo en las filas con la señal (μ y la mezcla, en las mismas filas), y solo en las "
     "fechas con ≥ 20 de esas filas: los pares son exactos, pero la μ sola de cada señal no es "
     "la de deciles.md con todas las acciones."),
    ("S1: la mediana de V es la de las 252 velas válidas anteriores a cada día (no la del año "
     "natural anterior), con ≥ 126; V = 0 no cuenta. Volumen de Yahoo, ajustado por splits."),
    ("S2: con datos desde enero de 2000, la señal empieza en 2003 (3 años previos); en 2000–11 "
     "cubre menos. Mes de la señal = el de la fecha t (la semana de tenencia puede caer en el "
     "siguiente). Precios ajustados por dividendos y empalmados en los cortes (Market)."),
    ("S3: vecinas recalculadas con la ventana que acaba el último día de la región del mes "
     "anterior a t (estrictamente antes de t). Retornos brutos a 21 sesiones de la región, como "
     "en el enunciado (no relativos a la región)."),
    ("Comparación múltiple: 4 señales; con Bonferroni al 5 %, t ≥ 2,5. La regla fijada usa t ≥ 2; "
     "se informa de las dos."),
]


def markdown(res: dict, n_weeks: int) -> list[str]:
    v = res["verdict"]
    L = ["# ¿Información en datos crudos que μ no ve? (23-11-2026)", "",
         (f"Ver el docstring de scripts/raw_signals.py. Universo actual, solo acciones, "
          f"2000–2019; {n_weeks} fechas semanales; μ_5 de final20 (20 miembros). Pierna larga "
          "(D9–D10): exceso de apertura a apertura sobre la media de las acciones, pb por semana; "
          "t de NW (2 retardos). Solo en las filas con la señal."), "",
         "## Lectura (regla fijada antes de verlo)", ""]
    for s, name in SIGNALS.items():
        a = res[s]["adds_mu"]
        L.append(f"- {s} ({name}): μ + señal − μ en 2012–19 {mt(a['2012–19'])}; 2012–15 "
                 f"{f(a['cv3']['mean'])}, 2016–19 {f(a['cv4']['mean'])}: "
                 f"**{'aporta' if v[s] else 'no aporta'}**"
                 + (" (con Bonferroni, t ≥ 2,5: "
                    f"{'también' if a['2012–19']['t'] >= 2.5 else 'no'})" if v[s] else "") + ".")
    if any(v.values()):
        L.append("- Según la regla, el siguiente paso con las que aportan sería darle a la red los "
                 "datos crudos correspondientes (no hecho; nada adoptado).")
    else:
        L.append("- Ninguna aporta según la regla: estas fuentes de información (calendario de "
                 "resultados, estacionalidad, vecinas) no explican lo que μ ha perdido desde 2012.")
    L += ["", "## Resumen, 2012–19 y 2000–11", "",
          ("| señal | cobertura 2000–11 / 2012–19 | corr. con μ_5 | corr. con la mezcla | "
           "larga de la señal 2000–11 | 2012–19 | μ + señal − μ 2000–11 | 2012–19 | "
           "mezcla + señal − mezcla 2000–11 | 2012–19 |"),
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SIGNALS:
        r = res[s]
        L.append(f"| {s} | {100 * r['coverage']['2000–11']:.0f} % / "
                 f"{100 * r['coverage']['2012–19']:.0f} % | {r['corr']['total']['μ_5']:.2f} | "
                 f"{r['corr']['total']['mezcla']:.2f} | "
                 f"{mt(r['legs']['señal']['2000–11'])} | {mt(r['legs']['señal']['2012–19'])} | "
                 f"{mt(r['adds_mu']['2000–11'])} | {mt(r['adds_mu']['2012–19'])} | "
                 f"{mt(r['adds_mix']['2000–11'])} | {mt(r['adds_mix']['2012–19'])} |")
    L += ["", "## Por bloque (media, pb por semana)", "",
          ("| señal | bloque | larga señal | larga μ | μ + señal − μ (t) | larga mezcla | "
           "mezcla + señal − mezcla (t) |"), "|---|---|---|---|---|---|---|"]
    for s in SIGNALS:
        r = res[s]
        for b in BLOCK_NAMES:
            L.append(f"| {s} | {BLOCK_LABEL[b]} | {f(r['legs']['señal'][b]['mean'])} | "
                     f"{f(r['legs']['μ'][b]['mean'])} | {mt(r['adds_mu'][b])} | "
                     f"{f(r['legs']['mezcla'][b]['mean'])} | {mt(r['adds_mix'][b])} |")
    L += ["", "## Correlación media con μ_5 y con la mezcla, por periodo", "",
          "| señal | 2000–11 μ_5 / mezcla | 2012–19 μ_5 / mezcla |", "|---|---|---|"]
    for s in SIGNALS:
        c = res[s]["corr"]
        L.append(f"| {s} | {c['2000–11']['μ_5']:.2f} / {c['2000–11']['mezcla']:.2f} | "
                 f"{c['2012–19']['μ_5']:.2f} / {c['2012–19']['mezcla']:.2f} |")
    L += ["", "## Decisiones (opción conservadora ante la duda)", ""] + [f"- {d}" for d in
                                                                      DECISIONS]
    return L


def main() -> None:
    data = WindowData(DATA_DIR / "dataset")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    pred = load_run("final20")
    samples = pred["sample"].to_numpy()
    info = asset_info(data.meta["symbols"], uni)
    equity = (info["asset_class"] == "equity").to_numpy()
    smp = data.samples.iloc[samples]
    dates = smp["date"]
    wk = weekly_dates(dates[dates.dt.dayofweek < 5])
    n_eq = smp[equity[smp["asset"].to_numpy()]].groupby("date").size()
    wk = pd.DatetimeIndex(sorted(wk[wk.isin(n_eq[n_eq >= 20].index)]))
    syms = [data.meta["symbols"][a] for a in np.flatnonzero(equity)]
    prices = OpenPrices.from_clean(DATA_DIR / "clean", syms)
    wf = weekly_frame(data, samples, equity, wk, prices)
    pos = np.searchsorted(samples, wf["sample"].to_numpy())
    wf["mu"] = pred["mu_5"].to_numpy()[pos]
    rows = data.samples["row"].to_numpy()[wf["sample"].to_numpy()]
    base = baseline_features(data, rows)
    wf["mix"] = baseline_mix(wf["date"].astype(str).to_numpy(), base["reversal"],
                             base["momentum"])
    wf["sym"] = np.asarray(data.meta["symbols"], object)[wf["asset"].to_numpy()]
    log(f"{len(wk)} fechas semanales, {len(wf)} filas")

    frames = {s: pd.read_parquet(DATA_DIR / "clean" / f"{s}.parquet").loc[:END] for s in syms}
    full = Market.from_frames(frames, end=END)
    close = pd.DataFrame(full.close, index=full.dates, columns=full.symbols)
    key = pd.MultiIndex.from_arrays([wf["date"], wf["sym"]])

    def lookup(by_date: dict) -> np.ndarray:
        st = pd.concat(by_date, names=["date", "sym"]) if by_date else pd.Series(dtype=float)
        return st.reindex(key).to_numpy(float)

    sig = {}
    s1 = s1_volume(frames, wk)
    s1d = {}
    for s, ser in s1.items():
        for t, x in ser.items():
            s1d.setdefault(t, {})[s] = x
    sig["S1"] = lookup({t: pd.Series(d) for t, d in s1d.items()})
    log("S1 hecha")
    s2 = s2_season(close)
    me = wf["date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    s2s = s2.stack()
    s2s.index = s2s.index.set_levels(s2s.index.levels[0].normalize(), level=0)
    sig["S2"] = s2s.reindex(pd.MultiIndex.from_arrays([me, wf["sym"]])).to_numpy(float)
    log("S2 hecha")
    region = uni.set_index("symbol")["country"].reindex(syms).eq("US")
    a_all, b_all = {}, {}
    for is_us in (True, False):
        cols = [s for s in syms if region[s] == is_us]
        a, b = neighbours(close, cols, wk)
        for t in a:
            a_all.setdefault(t, []).append(a[t])
            b_all.setdefault(t, []).append(b[t])
        log(f"vecinas {'EE. UU.' if is_us else 'Europa'} hechas")
    sig["S3a"] = lookup({t: pd.concat(v) for t, v in a_all.items()})
    sig["S3b"] = lookup({t: pd.concat(v) for t, v in b_all.items()})

    res = {}
    for s in SIGNALS:
        res[s] = measure(wf, sig[s])
        log(f"{s}: medida")
    res["verdict"] = verdict(res)
    (OUT / f"{REPORT}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                   default=float))
    (OUT / f"{REPORT}.md").write_text("\n".join(markdown(res, len(wk))) + "\n")
    log("hecho")


if __name__ == "__main__":
    main()
