"""Línea base B rehecha (handoff §11): B ordenaba por μ en retornos ((μ_rel + deriva)·σ_ventana) y
por eso elegía las acciones más volátiles. Con las predicciones fuera de muestra de rel63
(2000–2019). No entrena nada.

- B: la de logs/baseline.md, para comparar. Las 20 de μ_5·σ_ventana más alto a partes iguales;
  una solo sale al caer fuera de las 60 primeras. Rebalanceo semanal.
- B2: igual que B, pero ordenando por μ_5 en unidades de σ_ventana (la salida del modelo).
- B3: igual que B2, pero cada 4 semanas y con μ_20.

Las tres, con las reglas del juego de §10 (agent/simulator.py). Mismas referencias que
scripts/baseline.py, solo con costes: el universo a partes iguales, rebalanceado cada semana, y
comprar y mantener SPY. Mismas decisiones semanales (baselines.weekly_dates); B3 decide en una de
cada 4. Para c = 0,1 % y 0,5 %: rentabilidad anual, volatilidad, Sharpe, caída máxima y rotación,
en total, por bloque y en 2016–19, y la σ_ventana media de la cartera frente a la del universo.

Uso: python scripts/baseline_b2.py [--run rel63] [--costs 0.001 0.005] [--out logs/baseline_b2.md]
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from baseline import END, pct, row, spy_frame  # scripts/baseline.py

from agentic_trading.agent.baselines import equal_weights, top_n_weights, weekly_dates
from agentic_trading.agent.simulator import Market, Rules, performance, reference_rules, simulate
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.model.folds import CV_BLOCKS

NAMES = {"B": "B. μ_5 en retornos, semanal", "B2": "B2. μ_5 en σ_ventana, semanal",
         "B3": "B3. μ_20 en σ_ventana, cada 4 semanas", "EW": "Universo a partes iguales",
         "SPY": "SPY (comprar y mantener)"}
SCORE = {"B": "mu5_ret", "B2": "mu5", "B3": "mu20"}
EVERY = {"B": 1, "B2": 1, "B3": 4, "EW": 1}
LAST = ("2016-01-01", "2019-12-31")


def predictions(data: WindowData, run: str) -> pd.DataFrame:
    """μ_5 y μ_20 en unidades de σ_ventana (salida del modelo), μ_5 en retornos y σ_ventana, por
    acción y fecha, fuera de muestra (los 5 bloques). Solo acciones."""
    paths = sorted((DATA_DIR / "predictions" / run).glob("cv[0-9].parquet"))
    if not paths:
        raise SystemExit(f"No hay predicciones de {run}")
    pr = pd.concat([pd.read_parquet(p, columns=["sample", "mu_5", "mu_20"]) for p in paths])
    smp = data.samples.iloc[pr["sample"].to_numpy()]
    sig_v = np.asarray(data.arr["sigma"])[smp["row"].to_numpy()].astype(float)
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    asset = smp["asset"].to_numpy()
    mu5, mu20 = pr["mu_5"].to_numpy(), pr["mu_20"].to_numpy()
    df = pd.DataFrame({"date": smp["date"].to_numpy(), "symbol": info["symbol"].to_numpy()[asset],
                       "mu5": mu5, "mu20": mu20, "mu5_ret": mu5 * sig_v, "sigma_v": sig_v})
    return df[info["asset_class"].to_numpy()[asset] == "equity"]


def tilt(log, a=None, b=None) -> float:
    """Media, en las decisiones entre a y b, de la σ_ventana media de la cartera (ponderada por el
    peso objetivo) dividida por la σ_ventana media del universo de ese día."""
    vals = [(w * sv.reindex(w.index)).sum() / w.sum() / sv.mean() for d, w, sv in log
            if len(w) and (a is None or d >= pd.Timestamp(a)) and (b is None or d <= pd.Timestamp(b))]
    return float(np.mean(vals)) if vals else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rel63")
    ap.add_argument("--costs", type=float, nargs="+", default=[0.001, 0.005])
    ap.add_argument("--out", default="logs/baseline_b2.md")
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
    print(f"{len(dates)} decisiones semanales ({dates[0].date()} – {dates[-1].date()}), "
          f"{len(symbols)} acciones, {len(market.dates)} días, {market.breaks} cortes empalmados",
          flush=True)

    logs: dict[tuple, list] = {}

    def policy(kind, key):
        log = logs.setdefault(key, [])

        def decide(state):
            p = by_date.get(state.date)
            if p is None:
                return pd.Series(dtype=float)
            if kind == "EW":
                w = equal_weights(p["mu5"])
            else:
                w = top_n_weights(p[SCORE[kind]], state.weights.index)
            log.append((state.date, w[w > 0], p["sigma_v"]))
            return w
        return decide

    results: dict[tuple, object] = {}
    for c in args.costs:
        for kind in ("B", "B2", "B3", "EW"):
            try:
                rules = reference_rules(c) if kind == "EW" else Rules(cost=c)
                results[kind, c] = simulate(market, policy(kind, (kind, c)),
                                            dates[::EVERY[kind]], rules)
                print(f"  {kind} c={c}: valor final {results[kind, c].nav.iloc[-1]:.2f}",
                      flush=True)
            except Exception:  # noqa: BLE001 — se apunta y se sigue
                notes.append(f"{kind} (c = {c}): error\n```\n{traceback.format_exc()}```")
        try:
            spy = Market.from_frames({"SPY": spy_frame()}, start=start, end=END)
            first = spy.dates[spy.dates >= dates[0]][0]
            results["SPY", c] = simulate(spy, lambda s: pd.Series({"SPY": 1.0}), [first],
                                         reference_rules(c))
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"SPY (c = {c}): error\n```\n{traceback.format_exc()}```")

    first_day = dates[0]

    def perf(kind, c, a=None, b=END):
        if (kind, c) not in results:
            return {}
        return performance(results[kind, c], max(pd.Timestamp(a), first_day) if a else first_day,
                           b)

    head = ("| estrategia | rentabilidad anual | volatilidad | Sharpe | caída máxima | "
            "rotación anual |")
    L = [f"# Línea base B rehecha: `{args.run}`, 2000–2019", "",
         ("B ordenaba por μ en retornos ((μ_rel + deriva)·σ_ventana) y por eso elegía las acciones "
          "más volátiles (logs/baseline.md). **B2**: igual que B (las 20 de μ más alto a partes "
          "iguales; una sale solo al caer fuera de las 60 primeras; rebalanceo semanal), pero "
          "ordenando por μ_5 en unidades de σ_ventana, la salida del modelo. **B3**: igual que B2, "
          "pero cada 4 semanas y con μ_20. Las tres, con las reglas del juego de §10. Referencias, "
          "solo con costes: el universo a partes iguales, rebalanceado cada semana, y SPY comprado "
          "y mantenido. B se repite para comparar."), "",
         (f"Decisiones: {len(dates)} semanas, del {dates[0].date()} al {dates[-1].date()}; B3, "
          f"{len(dates[::4])}. Valoración hasta el {END}. Acciones: {len(symbols)} (sin cripto). "
          "σ_ventana de la cartera / del universo: media, en las decisiones, de la σ_ventana media "
          "de la cartera objetivo (ponderada por peso) entre la del universo de ese día."), ""]

    L += ["## 2016–19 (el bloque de la decisión)", ""]
    for c in args.costs:
        L += [f"c = {100 * c:.1f} %:", "", head, "|---|---|---|---|---|---|"]
        L += [row(name, perf(kind, c, *LAST)) for kind, name in NAMES.items()]
        L.append("")
    L += [("| estrategia | σ_ventana cartera / universo, 2016–19 | diferencia con los pesos "
           "iguales, c = 0,1 %: rentabilidad / Sharpe | ídem, c = 0,5 % |"), "|---|---|---|---|"]
    for kind in ("B", "B2", "B3"):
        cells = []
        for c in args.costs:
            p, ew = perf(kind, c, *LAST), perf("EW", c, *LAST)
            cells.append(f"{pct(p['rentabilidad anual'] - ew['rentabilidad anual'])} / "
                         f"{p['Sharpe'] - ew['Sharpe']:+.2f}" if p and ew else "—")
        lg = logs.get((kind, args.costs[0]), [])
        L.append(f"| {NAMES[kind]} | {tilt(lg, *LAST):.2f} | " + " | ".join(cells) + " |")
    L.append("")

    for c in args.costs:
        L += [f"## c = {100 * c:.1f} %: total ({first_day.year}–2019)", "", head,
              "|---|---|---|---|---|---|"]
        L += [row(name, perf(kind, c)) for kind, name in NAMES.items()]
        ew = perf("EW", c)
        for kind in ("B", "B2", "B3"):
            p = perf(kind, c)
            if p and ew:
                L.append(f"\n{NAMES[kind]} frente al universo a partes iguales: "
                         f"{pct(p['rentabilidad anual'] - ew['rentabilidad anual'])} de "
                         f"rentabilidad anual, Sharpe {p['Sharpe'] - ew['Sharpe']:+.2f}.")
        L += ["", f"### Por bloque, c = {100 * c:.1f} %", "",
              ("| bloque | estrategia | rentabilidad anual | volatilidad | Sharpe | caída máxima | "
               "rotación anual |"), "|---|---|---|---|---|---|---|"]
        for a, b in CV_BLOCKS:
            for kind, name in NAMES.items():
                L.append(row(f"{a[:4]}–{b[:4]} | {name}", perf(kind, c, a, b)))
        L.append("")

    c0 = args.costs[0]
    L += [f"## Diagnóstico (c = {100 * c0:.1f} %)", "",
          ("| estrategia | decisiones | posiciones (media) | σ_ventana cartera / universo: total | "
           + " | ".join(f"{a[:4]}–{b[:4]}" for a, b in CV_BLOCKS) + " | caja media | stops |"),
          "|---|---|---|---|" + "---|" * len(CV_BLOCKS) + "---|---|"]
    for kind in ("B", "B2", "B3", "EW"):
        if (kind, c0) not in results:
            continue
        res, lg = results[kind, c0], logs[kind, c0]
        npos = np.mean([len(w) for _, w, _ in lg])
        cash = float((res.cash / res.nav).loc[first_day:].mean())
        L.append(f"| {NAMES[kind]} | {len(lg)} | {npos:.0f} | {tilt(lg):.2f} | "
                 + " | ".join(f"{tilt(lg, a, b):.2f}" for a, b in CV_BLOCKS)
                 + f" | {pct(cash, signed=False)} | {len(res.stops)} |")
    L += ["", "## Notas", "",
          ("- La comparación justa es con el universo a partes iguales: tiene el mismo sesgo de "
           "supervivencia que las reglas (§3). Batir a SPY en 2000–2019 con este universo estaría "
           "inflado."),
          ("- Predicciones fuera de muestra de desarrollo: el modelo de cada bloque, salvo el de "
           "2016–19, ha visto datos posteriores (pega asumida, §9)."),
          ("- μ_20 cubre 20 velas, más o menos las 4 semanas entre decisiones de B3. Los stops "
           "siguen mirándose cada día."),
          f"- Cortes: {market.breaks} saltos a través de velas no válidas, empalmados con retorno 0."]
    L += [f"- {n}" for n in notes]
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nEscrito en {out}")


if __name__ == "__main__":
    main()
