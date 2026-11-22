"""Métricas de μ para mu_tree (22-11-2026): la cola que usa el agente y las de siempre.

Top 20 (la principal):
- En las fechas de baselines.weekly_dates (las de las decisiones semanales del agente), las 20
  acciones (sin cripto) de mayor μ_5.
- Su log-retorno medio entre la apertura del día siguiente a la fecha y la del día siguiente a la
  fecha semanal posterior (lo que cobra el agente: decide al cierre y opera en la apertura
  siguiente), menos el log-retorno medio de todas las acciones con predicción esa fecha en el
  mismo tramo, en pb. La apertura «siguiente» de cada acción es la de su próxima vela válida.
- Precios del simulador (agent.simulator.Market.from_frames): las velas no válidas no existen y
  el salto a través de ellas se empalma con retorno 0. Una acción sin vela entre las dos
  aperturas (la misma vela de entrada y salida) da 0; sin vela de salida antes del 31-12-2019
  (la última semana), no cuenta.
- Serie semanal; para dos ejecuciones, la diferencia por pares, con su media y su t de
  Newey-West (2 retardos).

Diagnóstico: la misma de cierre a cierre (y_5 · σ_ventana, la de scripts/rule_gap.py) y el decil
alto (el 10 % de mayor μ_5, redondeado hacia arriba) en vez del top 20, de apertura a apertura.

Además: IC de selección (media de h = 5 y 10 de la media diaria), QLIKE (media de h = 5, 10 y
20) y la diferencia diaria de IC con la mezcla fija de las líneas base (la de success_checks), con
su t de NW (retardos = h). Todo con los 5 bloques juntos y por bloque.

compare_rule: la regla de la fase 1 de mu_tree, fijada antes de ver resultados (ver RULE).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from agentic_trading.model.folds import CV_BLOCKS
from agentic_trading.model.metrics import (
    baseline_features,
    baseline_mix,
    daily_ic,
    nw_tstat,
    qlike,
    vs_mix,
)

H = (5, 10, 20)
TOP_N = 20
NW_LAGS_WEEKLY = 2
BLOCK_NAMES = [f"cv{k}" for k in range(len(CV_BLOCKS))]
END = pd.Timestamp("2019-12-31")

RULE = {
    "top20_t_min": 1.0,            # 1. top 20, 5 bloques juntos: media > 0 y t ≥ 1
    "top20_blocks_min": 3,         # 2. positiva en ≥ 3 de 5 bloques
    "top20_last_block": "cv4",     #    y no negativa en 2016–19
    "ic_drop_max": 0.002,          # 3. el IC medio no baja más de 0,002
    "qlike_worse_max": 0.01,       # 4. la QLIKE media no empeora más de un 1 %
}


def block_of(dates) -> np.ndarray:
    """Bloque de validación cruzada (cv0–cv4) de cada fecha; '' fuera de ellos."""
    d = pd.DatetimeIndex(dates)
    out = np.full(len(d), "", dtype=object)
    for name, (a, b) in zip(BLOCK_NAMES, CV_BLOCKS):
        out[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = name
    return out


# ----------------------------------------------------------------------------- precios

class OpenPrices:
    """Aperturas empalmadas (Market) y, por fecha, el índice de la próxima vela válida de cada
    acción (estrictamente después de la fecha)."""

    def __init__(self, market):
        self.market = market
        self.col = {s: j for j, s in enumerate(market.symbols)}
        T, N = market.open.shape
        ok = np.isfinite(market.open)
        nxt = np.full((T + 1, N), T, np.int64)          # nxt[t]: primera válida ≥ t
        for t in range(T - 1, -1, -1):
            nxt[t] = np.where(ok[t], t, nxt[t + 1])
        self.nxt = nxt
        self.log_open = np.log(np.where(ok, market.open, np.nan))

    @classmethod
    def from_clean(cls, clean_dir, symbols, start="1999-01-01", end=END):
        from agentic_trading.agent.simulator import Market
        frames = {s: pd.read_parquet(f"{clean_dir}/{s}.parquet") for s in symbols}
        return cls(Market.from_frames(frames, start=start, end=end))

    def entry_index(self, date, cols: np.ndarray) -> np.ndarray:
        """Índice de la primera vela válida después de date de cada columna (T si no hay)."""
        t = int(self.market.dates.searchsorted(pd.Timestamp(date), side="right"))
        return self.nxt[t, cols]

    def returns(self, d0, d1, cols: np.ndarray) -> np.ndarray:
        """Log-retorno de apertura (día siguiente a d0) a apertura (día siguiente a d1)."""
        T = self.log_open.shape[0]
        e0, e1 = self.entry_index(d0, cols), self.entry_index(d1, cols)
        ok = (e0 < T) & (e1 < T)
        out = np.full(len(cols), np.nan)
        out[ok] = (self.log_open[np.minimum(e1, T - 1), cols]
                   - self.log_open[np.minimum(e0, T - 1), cols])[ok]
        return out


def weekly_frame(data, samples: np.ndarray, equity: np.ndarray, weekly: pd.DatetimeIndex,
                 prices: OpenPrices) -> pd.DataFrame:
    """Muestras de las fechas semanales (solo acciones), con el retorno de apertura a apertura
    hasta la fecha semanal siguiente (ret_oo) y el de cierre a cierre a 5 velas (ret_cc), en
    log. Las mismas filas sirven para cualquier ejecución con esas muestras."""
    smp = data.samples.iloc[samples]
    df = pd.DataFrame({"sample": samples, "date": smp["date"].to_numpy(),
                       "asset": smp["asset"].to_numpy()})
    df = df[equity[df["asset"].to_numpy()] & df["date"].isin(weekly)].reset_index(drop=True)
    rows = data.samples["row"].to_numpy()[df["sample"].to_numpy()]
    df["ret_cc"] = (np.asarray(data.arr["y_5"])[rows].astype(float)
                    * np.asarray(data.arr["sigma"])[rows].astype(float))
    symbols = data.meta["symbols"]
    df["col"] = [prices.col.get(symbols[a], -1) for a in df["asset"]]
    if (df["col"] < 0).any():
        raise ValueError("acciones sin precios en el mercado del simulador")
    wk = pd.DatetimeIndex(sorted(weekly))
    nxt = dict(itertools.pairwise(wk))
    df["ret_oo"] = np.nan
    for d, g in df.groupby("date"):
        if d not in nxt:
            continue
        df.loc[g.index, "ret_oo"] = prices.returns(d, nxt[d], g["col"].to_numpy())
    return df


def tail_series(wf: pd.DataFrame, mu5: np.ndarray, n: int = TOP_N) -> pd.DataFrame:
    """Series semanales (pb) de una ejecución: top20_oo, top20_cc y dec_oo, cada una el retorno
    medio de la cola menos el de todas las acciones de la fecha. mu5: μ_5 de las filas de wf."""
    df = wf.assign(mu=mu5)
    rk = df.groupby("date")["mu"].rank(ascending=False, method="first")
    n_day = df.groupby("date")["mu"].transform("size")
    out = {}
    for name, col, sel in (("top20_oo", "ret_oo", rk <= n), ("top20_cc", "ret_cc", rk <= n),
                           ("dec_oo", "ret_oo", rk <= np.ceil(n_day / 10))):
        allm = df.groupby("date")[col].mean()
        top = df[sel].groupby("date")[col].mean()
        out[name] = 1e4 * (top - allm)
    return pd.DataFrame(out).dropna(how="all")


def paired(a: pd.Series, b: pd.Series, lags: int = NW_LAGS_WEEKLY) -> dict:
    """Diferencia por pares a − b de dos series semanales: total y por bloque."""
    d = (a - b).dropna().sort_index()
    m, t = nw_tstat(d.to_numpy(), lags)
    out = {"mean": m, "t": t, "n": len(d), "blocks": {}}
    blk = block_of(d.index)
    for name in BLOCK_NAMES:
        x = d[blk == name]
        bm, bt = nw_tstat(x.to_numpy(), lags)
        out["blocks"][name] = {"mean": bm, "t": bt, "n": len(x)}
    return out


def level(s: pd.Series, lags: int = NW_LAGS_WEEKLY) -> dict:
    """Media y t de NW de una serie semanal: total y por bloque."""
    m, t = nw_tstat(s.dropna().to_numpy(), lags)
    blk = block_of(s.index)
    return {"mean": m, "t": t,
            "blocks": {b: nw_tstat(s[blk == b].dropna().to_numpy(), lags)[0]
                       for b in BLOCK_NAMES}}


# ----------------------------------------------------------------------------- IC y QLIKE

class StdContext:
    """Lo que no depende de la ejecución para IC, QLIKE y mezcla: targets, máscaras, fechas,
    bloque y la mezcla de las líneas base de las muestras dadas."""

    def __init__(self, data, samples: np.ndarray):
        self.samples = np.asarray(samples)
        smp = data.samples.iloc[self.samples]
        self.rows = smp["row"].to_numpy()
        self.dates = smp["date"].to_numpy()
        self.block = block_of(self.dates)
        self.y = {h: np.asarray(data.arr[f"y_{h}"])[self.rows].astype(float) for h in H}
        self.v = {h: np.asarray(data.arr[f"v_{h}"])[self.rows].astype(float) for h in H}
        self.m = {h: smp[f"mask_sigma_{h}"].to_numpy() for h in H}
        base = baseline_features(data, self.rows)
        mix = baseline_mix(self.dates, base["reversal"], base["momentum"])
        self.ic_mix = {}
        for b in [None, *BLOCK_NAMES]:
            sel = slice(None) if b is None else self.block == b
            self.ic_mix[b] = {h: daily_ic(self.dates[sel], mix[sel], self.y[h][sel])
                              for h in (5, 10)}


def std_metrics(ctx: StdContext, pred: pd.DataFrame) -> dict:
    """IC, QLIKE y diferencia con la mezcla, total (None) y por bloque. pred alineado con
    ctx.samples."""
    if not np.array_equal(pred["sample"].to_numpy(), ctx.samples):
        raise ValueError("predicciones desalineadas con las muestras del contexto")
    out = {}
    for b in [None, *BLOCK_NAMES]:
        sel = np.ones(len(ctx.samples), bool) if b is None else ctx.block == b
        r = {}
        ics = []
        for h in (5, 10):
            ic = daily_ic(ctx.dates[sel], pred[f"mu_{h}"].to_numpy()[sel], ctx.y[h][sel])
            ics.append(ic.mean())
            r[f"IC_{h}"] = float(ic.mean())
            r.update({f"{k}_{h}": float(v) for k, v in
                      zip(("mix_diff", "mix_t"), vs_mix(ic, ctx.ic_mix[b][h], h).values())})
        r["IC"] = float(np.mean(ics))
        qs = []
        for h in H:
            m = ctx.m[h] & sel
            vhat = pred[f"sigma_{h}"].to_numpy()[m] / np.sqrt(h)
            q = float(qlike(ctx.v[h][m], vhat).mean())
            r[f"QLIKE_{h}"] = q
            qs.append(q)
        r["QLIKE"] = float(np.mean(qs))
        out["total" if b is None else b] = r
    return out


# ----------------------------------------------------------------------------- regla

def compare_rule(top20: dict, std_run: dict, std_base: dict, rule: dict = RULE) -> dict:
    """Las cuatro condiciones de la fase 1 de mu_tree. top20: paired(...) de top20_oo."""
    blocks = top20["blocks"]
    n_pos = sum(1 for b in BLOCK_NAMES if np.isfinite(blocks[b]["mean"]) and blocks[b]["mean"] > 0)
    last = blocks[rule["top20_last_block"]]["mean"]
    d_ic = std_run["total"]["IC"] - std_base["total"]["IC"]
    q_rel = std_run["total"]["QLIKE"] / std_base["total"]["QLIKE"] - 1
    c = {
        "1_top20_total": bool(top20["mean"] > 0 and top20["t"] >= rule["top20_t_min"]),
        "2_top20_blocks": bool(n_pos >= rule["top20_blocks_min"] and last >= 0),
        "3_ic": bool(d_ic >= -rule["ic_drop_max"]),
        "4_qlike": bool(q_rel <= rule["qlike_worse_max"]),
    }
    return {"conditions": c, "wins": all(c.values()), "n_blocks_positive": n_pos,
            "top20_last_block": last, "ic_diff": d_ic, "qlike_rel": q_rel}
