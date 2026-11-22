"""Línea base del agente (handoff §10 y §11) con las predicciones fuera de muestra de una ejecución
de validación cruzada (por defecto rel63, 2000–2019). No entrena nada.

Reglas fijas, decididas antes de ver los resultados. Rebalanceo semanal con μ_5 y σ_5 pasados a
retornos con σ_ventana (μ = μ_5·σ_ventana, σ = σ_5·σ_ventana):
A. Tipo Kelly: pesos ∝ max(μ, 0)/σ², normalizados para que sumen 1. Con la regla del 1 % del
   juego, se quedan las de mayor puntuación que, normalizadas entre ellas, pesan al menos un 1 %
   (baselines.kelly_weights).
B. Las 20 acciones de μ más alto, a partes iguales; una solo sale al caer fuera de las 60 primeras.
Solo acciones (sin cripto). Una acción sin predicción ese día no es elegible; si ya está en cartera,
se mantiene sin operar. A y B juegan con las reglas de §10 (agent/simulator.py): decisión al cierre
y operación en la apertura siguiente, sin apalancamiento ni cortos, posiciones < 1 % a 0, stops del
3 % del capital y costes c × importe operado.

Referencias (solo costes: no siguen la regla del 1 % ni los stops):
- comprar y mantener SPY (Yahoo, en data/raw/benchmarks/; ajustado por dividendos con el factor
  propio de §3);
- el universo a partes iguales (las acciones con predicción ese día), rebalanceado cada semana.

Decisión: una por semana, el último día laborable con predicciones de al menos el 80 % de las
acciones del día con más de esa semana. Para c = 0,1 % y 0,5 %: rentabilidad anual, volatilidad,
Sharpe, caída máxima y rotación anual, en total y por bloque.

Uso: python scripts/baseline.py [--run rel63] [--costs 0.001 0.005] [--out logs/baseline.md]
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.agent.baselines import (
    equal_weights,
    kelly_weights,
    top_n_weights,
    weekly_dates,
)
from agentic_trading.agent.simulator import (
    Market,
    Rules,
    performance,
    reference_rules,
    simulate,
)
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.clean import dividend_factor
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.data.download import download
from agentic_trading.model.folds import CV_BLOCKS

END = "2019-12-31"                      # desarrollo: el test (2020–) no se toca
BENCH_DIR = DATA_DIR / "raw" / "benchmarks"
NAMES = {"A": "A. Tipo Kelly", "B": "B. 20 de μ más alto", "EW": "Universo a partes iguales",
         "SPY": "SPY (comprar y mantener)"}
METRICS = ["rentabilidad anual", "volatilidad", "Sharpe", "caída máxima", "rotación anual"]


def predictions(data: WindowData, run: str) -> pd.DataFrame:
    """μ y σ a 5 días en retornos, por acción y fecha, fuera de muestra (los 5 bloques)."""
    paths = sorted((DATA_DIR / "predictions" / run).glob("cv[0-9].parquet"))
    if not paths:
        raise SystemExit(f"No hay predicciones de {run}")
    pr = pd.concat([pd.read_parquet(p, columns=["sample", "mu_5", "sigma_5"]) for p in paths])
    smp = data.samples.iloc[pr["sample"].to_numpy()]
    sig_v = np.asarray(data.arr["sigma"])[smp["row"].to_numpy()].astype(float)
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    asset = smp["asset"].to_numpy()
    df = pd.DataFrame({"date": smp["date"].to_numpy(), "symbol": info["symbol"].to_numpy()[asset],
                       "mu": pr["mu_5"].to_numpy() * sig_v,
                       "sigma": pr["sigma_5"].to_numpy() * sig_v, "sigma_v": sig_v})
    return df[info["asset_class"].to_numpy()[asset] == "equity"]


def spy_frame() -> pd.DataFrame:
    path = BENCH_DIR / "SPY.parquet"
    if not path.exists():
        download(["SPY"], raw_dir=BENCH_DIR)
    raw = pd.read_parquet(path)
    return raw[["open", "high", "low", "close"]].mul(dividend_factor(raw), axis=0)


def pct(x, nd=1, signed=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100 * x:+.{nd}f} %" if signed else f"{100 * x:.{nd}f} %"


def row(name, p):
    if not p:
        return f"| {name} | — | — | — | — | — |"
    return (f"| {name} | {pct(p['rentabilidad anual'])} | {pct(p['volatilidad'], signed=False)} | "
            f"{p['Sharpe']:.2f} | {pct(p['caída máxima'])} | {p['rotación anual']:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rel63")
    ap.add_argument("--costs", type=float, nargs="+", default=[0.001, 0.005])
    ap.add_argument("--out", default="logs/baseline.md")
    args = ap.parse_args()
    notes: list[str] = []

    data = WindowData(DATA_DIR / "dataset")
    pred = predictions(data, args.run)
    dates = weekly_dates(pred["date"])
    by_date = {d: g.set_index("symbol") for d, g in pred[pred["date"].isin(dates)].groupby("date")}
    symbols = sorted(pred["symbol"].unique())
    start = dates[0] - pd.Timedelta(days=10)
    clean = DATA_DIR / "clean"
    market = Market.from_frames({s: pd.read_parquet(clean / f"{s}.parquet") for s in symbols},
                                start=start, end=END)
    print(f"{len(dates)} decisiones ({dates[0].date()} – {dates[-1].date()}), "
          f"{len(symbols)} acciones, {len(market.dates)} días, {market.breaks} cortes empalmados",
          flush=True)

    logs: dict[tuple, list] = {}

    def policy(kind, key):
        log = logs.setdefault(key, [])

        def decide(state):
            p = by_date.get(state.date)
            if p is None:
                return pd.Series(dtype=float)
            if kind == "A":
                w = kelly_weights(p["mu"], p["sigma"])
            elif kind == "B":
                w = top_n_weights(p["mu"], state.weights.index)
            else:
                w = equal_weights(p["mu"])
            log.append((state.date, w[w > 0], p["sigma_v"]))
            return w
        return decide

    results: dict[tuple, object] = {}
    for c in args.costs:
        for kind in ("A", "B", "EW"):
            rules = Rules(cost=c) if kind in ("A", "B") else reference_rules(c)
            results[kind, c] = simulate(market, policy(kind, (kind, c)), dates, rules)
            print(f"  {kind} c={c}: valor final {results[kind, c].nav.iloc[-1]:.2f}", flush=True)
        try:
            spy = Market.from_frames({"SPY": spy_frame()}, start=start, end=END)
            first = spy.dates[spy.dates >= dates[0]][0]
            results["SPY", c] = simulate(spy, lambda s: pd.Series({"SPY": 1.0}), [first],
                                         reference_rules(c))
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"SPY (c = {c}): error\n```\n{traceback.format_exc()}```")

    first_day = dates[0]
    L = [f"# Línea base del agente: `{args.run}`, 2000–2019", "",
         ("Reglas fijas, decididas antes de ver los resultados (ver scripts/baseline.py): "
         "rebalanceo semanal con μ_5 y σ_5 en retornos (× σ_ventana). **A**: pesos ∝ max(μ, 0)/σ², "
         "que suman 1; con la regla del 1 %, se quedan las de mayor puntuación que, normalizadas "
         "entre ellas, pesan ≥ 1 %. **B**: las 20 de μ más alto a partes iguales; una sale solo al "
         "caer fuera de las 60 primeras. A y B, con las reglas del juego de §10 (operan en la "
         "apertura siguiente, sin apalancamiento ni cortos, posiciones < 1 % a 0, stops del 3 % "
         "del capital, costes c × importe operado). Referencias, solo con costes: el universo a "
         "partes iguales (las acciones con predicción) cada semana y SPY comprado y mantenido."), "",
         (f"Decisiones: {len(dates)} semanas, del {dates[0].date()} al {dates[-1].date()} (después "
         f"no hay predicciones: el target de 20 velas tiene que acabar en 2019); valoración hasta "
         f"el {END}. Acciones: {len(symbols)} (sin cripto). Rentabilidad total en la moneda de "
         "cada activo; Sharpe sin tipo de interés (la caja no rinde); rotación anual = (compras + "
         "ventas) / 2 / valor de la cartera, por año."), ""]
    for c in args.costs:
        L += [f"## c = {100 * c:.1f} %: total ({first_day.year}–2019)", "",
              ("| estrategia | rentabilidad anual | volatilidad | Sharpe | caída máxima | "
              "rotación anual |"), "|---|---|---|---|---|---|"]
        for kind, name in NAMES.items():
            if (kind, c) in results:
                L.append(row(name, performance(results[kind, c], first_day, END)))
        ew = performance(results["EW", c], first_day, END) if ("EW", c) in results else {}
        for kind in ("A", "B"):
            p = performance(results[kind, c], first_day, END)
            if p and ew:
                L.append(f"\n{NAMES[kind]} frente al universo a partes iguales: "
                         f"{pct(p['rentabilidad anual'] - ew['rentabilidad anual'])} de "
                         f"rentabilidad anual, Sharpe {p['Sharpe'] - ew['Sharpe']:+.2f}.")
        L += ["", f"### Por bloque, c = {100 * c:.1f} %", "",
              ("| bloque | estrategia | rentabilidad anual | volatilidad | Sharpe | caída máxima | "
              "rotación anual |"), "|---|---|---|---|---|---|---|"]
        for a, b in CV_BLOCKS:
            for kind, name in NAMES.items():
                if (kind, c) in results:
                    p = performance(results[kind, c], max(pd.Timestamp(a), first_day), b)
                    L.append(row(f"{a[:4]}–{b[:4]} | {name}", p))
        L.append("")

    # diagnóstico (con el primer coste)
    c0 = args.costs[0]
    L += [f"## Diagnóstico (c = {100 * c0:.1f} %)", "",
          ("| estrategia | posiciones (media) | peso máximo (media) | σ_ventana de la cartera / "
          "del universo | caja media | stops |"), "|---|---|---|---|---|---|"]
    for kind in ("A", "B", "EW"):
        res, lg = results[kind, c0], logs[kind, c0]
        npos = np.mean([len(w) for _, w, _ in lg])
        wmax = np.mean([w.max() if len(w) else 0 for _, w, _ in lg])
        tilt = np.mean([(w * sv.reindex(w.index)).sum() / w.sum() / sv.mean()
                        for _, w, sv in lg if len(w)])
        cash = float((res.cash / res.nav).loc[first_day:].mean())
        L.append(f"| {NAMES[kind]} | {npos:.0f} | {pct(wmax, signed=False)} | {tilt:.2f} | "
                 f"{pct(cash, signed=False)} | "
                 f"{len(res.stops)} |")
    L += ["", "## Notas", "",
          ("- La comparación justa es con el universo a partes iguales: tiene el mismo sesgo de "
          "supervivencia que las reglas (§3: componentes actuales). Batir a SPY en 2000–2019 con "
          "este universo estaría inflado."),
          ("- Elegibilidad: una acción solo tiene predicción si su ventana y su target (las 20 "
          "velas siguientes) son válidos, así que la elegibilidad usa algo de información futura "
          "(datos que se van a invalidar). Afecta igual a A, B y a los pesos iguales."),
          ("- Son predicciones fuera de muestra de desarrollo: el modelo de cada bloque ha visto "
          "datos posteriores (pega asumida, §9)."),
          ("- Precios ajustados por dividendos con el factor propio (§3): los stops se miden "
          "sobre precios ajustados."),
          (f"- Cortes: {market.breaks} saltos de precio a través de velas no válidas, empalmados "
           "con retorno 0 (`Market.from_frames`). Añadido tras ver la primera versión "
           "(logs/baseline_sin_empalme.md): BGEO.L se multiplica por 57 al cruzar un hueco de "
           "2011-02 a 2012-02 (antes cotizaba otra entidad) y B, que la tenía, ganaba un 371 % "
           "ese día."),
          ("- Inclinación por volatilidad (columna σ_ventana del diagnóstico): μ en retornos es "
           "(μ_rel + deriva)·σ_ventana, así que B, que ordena por μ, favorece las acciones más "
           "volátiles; A, que divide por σ², las menos volátiles.")]
    L += [f"- {n}" for n in notes]
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nEscrito en {out}")


if __name__ == "__main__":
    main()
