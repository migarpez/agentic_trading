"""Comprobación de costes antes de llevar el universo combinado al agente (23-11-2026;
logs/universe_ext2.md → logs/universe_ext3.md).

1. Modelo de costes por activo (agent/costs.py; Rules.cost_asset y Rules.buy_tax del simulador):
   c = la mitad de la horquilla de su grupo en 2012–19 (mínimo 0,1 %) en cada operación, y el
   impuesto de su país solo en las compras.
2. Con las predicciones de A del paso 2 (data/comb/predictions/ext_A, solo con el pasado), la regla
   fija del entorno del agente (agent.evaluation.rule_order: las 20 de mayor μ, a partes iguales,
   sale al caer fuera de las 60 primeras, stop en d_max, agent_rules) sobre el universo combinado
   entero (cripto incluida), como lo vería el agente: semanal con μ_5 y cada 4 semanas con μ_20.
   Frente a los pesos iguales del combinado (brutos, rebalanceo semanal, como en el paso 2) y la
   misma regla con la mezcla (rangos dentro de la fecha, en todo el combinado:
   data/comb/predictions/ext3_mix).
3. Lo mismo con todos los costes ×2 (horquilla e impuestos).
4. Por bloque (2012–15 y 2016–19; cada uno, un episodio que empieza en liquidez): rentabilidad,
   Sharpe, caída, rotación, coste pagado al año (Σ pagado / NAV al empezar cada paso, por año) y
   reparto medio de la cartera por grupo y por país (pesos al cierre de cada decisión, antes de
   operar; media de las decisiones).

Lectura, fijada antes de verlo:
- Con los costes del punto 1, la regla fija con μ debe batir a los pesos iguales del combinado en
  rentabilidad y en Sharpe en los dos bloques, con alguna de las dos frecuencias.
- Si además aguanta con costes ×2 (capital final no peor que los pesos iguales), se pasa al
  ensemble final con el universo combinado y después al agente.
- Si no pasa con los costes del punto 1, el universo combinado no sirve al agente tal como está.

Uso: AGENTIC_TRADING_DATA=data/comb python scripts/ext_costs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.agent.costs import BUY_TAX, MIN_COST, asset_costs, hq_country, tax_country
from agentic_trading.agent.data import build_panel
from agentic_trading.agent.env import TradingEnv
from agentic_trading.agent.evaluation import _perf, rule_order
from agentic_trading.agent.simulator import agent_rules
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.metrics import baseline_features, baseline_mix
from agentic_trading.model.mu_metrics import block_of
from scripts.ext_mu import BLOCKS, ew_perf, group_panel, load_pred
from scripts.mu_tree import log
from scripts.universe_ext import spread_by_month

COMB = DATA_DIR
BASE = ROOT / "data"
OUT = ROOT / "logs"
REPORT = "universe_ext3"
PRED = COMB / "predictions"
CRYPTO = "crypto"
FREQS = (("semanal μ_5", 1, 0), ("4 semanas μ_20", 4, 2))
SCALES = {"×1": 1.0, "×2": 2.0}
SIGNALS = {"A": "ext_A", "mezcla": "ext3_mix"}


def spreads() -> dict:
    """Horquilla media (pb) de 2012–19 por grupo: la del paso 1 y, para la cripto, la misma cuenta
    con BTC y ETH."""
    sp = {g: v["periods"]["2012–19"]["spread"] for g, v in
          json.loads((OUT / "universe_ext.json").read_text())["groups"].items()}
    m = spread_by_month(BASE, ["BTC-USD", "ETH-USD"])
    sp[CRYPTO] = float(m.loc["2012-01-01":"2019-12-31"].mean())
    return sp


def mix_all(data: WindowData) -> None:
    """Predicciones falsas con la mezcla (rangos dentro de la fecha, en todo el combinado) como
    μ_5, μ_10 y μ_20, en ext3_mix."""
    pa = load_pred(PRED / "ext_A")
    d = PRED / "ext3_mix"
    d.mkdir(exist_ok=True)
    smp = data.samples.iloc[pa["sample"].to_numpy()]
    base = baseline_features(data, smp["row"].to_numpy())
    mix = baseline_mix(smp["date"].astype(str).to_numpy(), base["reversal"], base["momentum"])
    out = pa.copy()
    for h in (5, 10, 20):
        out[f"mu_{h}"] = mix
    for b in BLOCKS:
        m = block_of(smp["date"]) == b
        out[m].to_parquet(d / f"{b}.parquet", index=False)
    (d / "config.json").write_text(json.dumps({"members": 0, "what": "mezcla (rangos, fecha)"}))


def run_detail(panel, start, end, rules, col: int, groups: np.ndarray,
               countries: np.ndarray) -> dict:
    """agent.evaluation.run_rule, más el coste pagado y el reparto medio por grupo y país."""
    k0, k1 = panel.k_range(start, end)
    env = TradingEnv(panel, rules, n_steps=k1 - 1 - k0)
    env.record = True
    obs = env.reset(k0)
    days, done, paid, W = [(panel.dec_t[k0], 1.0, 0.0)], False, 0.0, []
    while not done:
        W.append(env.pf.weights())
        nav0, paid0 = env.pf.nav, env.pf.paid
        obs, _, done, info = env.step_order(rule_order(obs, rules, col=col))
        paid += (env.pf.paid - paid0) / nav0
        days += info["days"]
    m = _perf(panel, days)
    m["coste anual"] = paid / m["años"]
    W = np.mean(W, 0)
    m["grupos"] = {g: float(W[groups == g].sum()) for g in pd.unique(groups)}
    m["países"] = {c: float(W[countries == c].sum()) for c in pd.unique(countries)}
    m["liquidez"] = 1 - float(W.sum())
    return m


def ew_detail(panel, start, end, groups, countries) -> dict:
    m = ew_perf(group_panel(panel, np.ones(len(panel.market.symbols), bool)), start, end)
    k0, k1 = panel.k_range(start, end)
    W = []
    for k in range(k0, k1 - 1):
        w = np.zeros(len(panel.market.symbols))
        w[panel.elig[k]] = 1 / len(panel.elig[k])
        W.append(w)
    W = np.mean(W, 0)
    m["coste anual"] = 0.0
    m["grupos"] = {g: float(W[groups == g].sum()) for g in pd.unique(groups)}
    m["países"] = {c: float(W[countries == c].sum()) for c in pd.unique(countries)}
    m["liquidez"] = 0.0
    return m


def evaluate() -> dict:
    data = WindowData(COMB / "dataset")
    uni = pd.read_parquet(COMB / "universe.parquet")
    sp = spreads()
    half = {g: max(v / 2 / 1e4, MIN_COST) for g, v in sp.items()}
    hq = hq_country(uni, COMB / "hq_country.json")
    countries = tax_country(uni, hq)
    groups = uni.set_index("symbol")["size_group"]
    log(f"horquillas (pb): {sp}; sedes en Irlanda: {sorted(hq.index[hq.eq('Ireland')])}")
    mix_all(data)
    panels = {k: build_panel(r, data, freq="weekly") for k, r in SIGNALS.items()}
    syms = list(panels["A"].market.symbols)
    assert list(panels["mezcla"].market.symbols) == syms
    g_arr = groups.reindex(syms).to_numpy(object)
    c_arr = countries.reindex(syms).to_numpy(object)
    res = {"spreads_bp": sp, "half_spread": half, "tax": BUY_TAX,
           "irish": sorted(hq.index[hq.eq("Ireland")]),
           "n_assets": {"grupos": pd.Series(g_arr).value_counts().to_dict(),
                        "países": pd.Series(c_arr).value_counts().to_dict()},
           "ew": {}, "runs": {}}
    for b, (_, a, z) in BLOCKS.items():
        res["ew"][b] = ew_detail(panels["A"], a, z, g_arr, c_arr)
    for sname, scale in SCALES.items():
        cost, tax = asset_costs(syms, groups, countries, half, scale)
        rules = agent_rules(0.0, cost_asset=cost, buy_tax=tax)
        for sig, p in panels.items():
            for fname, every, col in FREQS:
                gp = group_panel(p, np.ones(len(syms), bool), every)
                for b, (_, a, z) in BLOCKS.items():
                    res["runs"][f"{sig}|{fname}|{sname}|{b}"] = run_detail(gp, a, z, rules, col,
                                                                          g_arr, c_arr)
                log(f"{sig} {fname} costes {sname} hecho")
    res["verdict"] = verdict(res)
    (OUT / f"{REPORT}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                   default=float))
    return res


def beats(m: dict, ew: dict, sharpe: bool = True) -> bool:
    ok = m["rentabilidad anual"] > ew["rentabilidad anual"]
    return ok and (m["Sharpe"] > ew["Sharpe"] if sharpe else True)


def verdict(res: dict) -> dict:
    """Lectura fijada (docstring). "Aguanta ×2": capital final (rentabilidad anual del bloque) no
    peor que los pesos iguales en los dos bloques, con la misma frecuencia que pasa con ×1."""
    out = {"pasa": [], "aguanta": []}
    for fname, _, _ in FREQS:
        r = res["runs"]
        if all(beats(r[f"A|{fname}|×1|{b}"], res["ew"][b]) for b in BLOCKS):
            out["pasa"].append(fname)
            if all(r[f"A|{fname}|×2|{b}"]["rentabilidad anual"] >=
                   res["ew"][b]["rentabilidad anual"] for b in BLOCKS):
                out["aguanta"].append(fname)
    out["mezcla pasa"] = [f for f, _, _ in FREQS if all(
        beats(res["runs"][f"mezcla|{f}|×1|{b}"], res["ew"][b]) for b in BLOCKS)]
    return out


def pct(x, nd=1, sign=True):
    return f"{100 * x:{'+' if sign else ''}.{nd}f}%"


def fmt(m: dict) -> str:
    return (f"{pct(m['rentabilidad anual'])} / {m['Sharpe']:.2f} / "
            f"{pct(m['caída máxima'], 0, False)} / {m['rotación anual']:.1f} / "
            f"{pct(m['coste anual'], 2, False)}")


GROUP_ORDER = ["grandes", "medianas EE. UU.", "pequeñas EE. UU.", "medianas y pequeñas Europa",
               CRYPTO]
COUNTRY_ORDER = ["US", "GB", "IE", "FR", "DE", "NL", "IT", "ES", "CH", "cripto"]

DECISIONS = [
    ("Costes con la horquilla de Abdi–Ranaldo del grupo en 2012–19 (paso 1), la misma para todo el "
    "periodo y todos los activos del grupo; la cripto, con la de BTC y ETH (2014–19, desde que "
    "hay datos)."),
    ("Impuestos por el país de la bolsa, con el tipo más alto ante la duda y sin las exenciones por "
    "capitalización (Francia, España, Italia, Irlanda) ni por domicilio (Reino Unido): ver el "
    "docstring de agent/costs.py, con las fuentes. Francia, 0,5 % (el 0,4 % confirmado podría "
    "haber subido en la ley de 2026, que no pude confirmar). Suiza, el 0,15 % entero, solo en "
    "las compras."),
    ("Irlanda: las de Londres con sede en Irlanda según Yahoo pagan el 1 % en lugar del 0,5 %. Las "
    "irlandesas de EE. UU. (en DTC) no pagan."),
    ("Tipos de hoy aplicados a 2012–19 (antes eran menores o no existían: Francia desde 2012, "
    "Italia desde 2013, España desde 2021)."),
    ("Pesos iguales del combinado brutos (sin costes), con la cripto, rebalanceo en cada decisión "
    "semanal, como en el paso 2: la referencia más difícil de batir."),
    ("Aguantar ×2: la misma frecuencia que pasa con ×1, con capital final (rentabilidad del "
    "bloque) no peor que los pesos iguales en los dos bloques."),
    ("Mezcla: rangos de la reversión y el momentum dentro de cada fecha, en todo el combinado "
    "(no por grupo), como vería el agente; la misma en las dos frecuencias."),
    ("Sesgo de supervivencia: el de logs/universe_ext.md (componentes actuales); mayor en las "
    "pequeñas, infla los niveles."),
]


def markdown(res: dict) -> list[str]:
    v = res["verdict"]
    L = ["# Universo combinado con costes por activo: regla fija (23-11-2026)", "",
         ("Ver el docstring de scripts/ext_costs.py. Predicciones de A del paso 2 "
         "(logs/universe_ext2.md; solo con el pasado). La regla fija del entorno del agente "
         "sobre el combinado entero (las 20 de mayor μ de todos los grupos), frente a los pesos "
         "iguales del combinado (brutos) y la misma regla con la mezcla."), "",
         "## Lectura (fijada antes de verlo)", ""]
    pasa = ", ".join(v["pasa"]) or "ninguna frecuencia"
    L.append(f"- Con los costes del punto 1, la regla con μ bate a los pesos iguales en "
             f"rentabilidad y Sharpe en los dos bloques: **{'sí' if v['pasa'] else 'no'}** "
             f"({pasa}).")
    if v["pasa"]:
        ag = ", ".join(v["aguanta"]) or "ninguna"
        L.append(f"- Con costes ×2, capital final no peor que los pesos iguales: "
                 f"**{'sí' if v['aguanta'] else 'no'}** ({ag}).")
    if v["aguanta"]:
        L.append("- Según la regla: se pasaría al ensemble final con el universo combinado y "
                 "después al agente (no adoptado).")
    elif v["pasa"]:
        L.append("- Según la regla: pasa con los costes del punto 1, pero no aguanta ×2; no se "
                 "cumple la condición para pasar al ensemble final.")
    else:
        L.append("- Según la regla: el universo combinado no sirve al agente tal como está.")
    L.append(f"- Informativo: la regla con la mezcla bate a los pesos iguales con "
             f"{', '.join(v['mezcla pasa']) or 'ninguna frecuencia'}.")
    L += ["", "## Costes", "",
          "| grupo | horquilla 2012–19 (pb) | c por operación |", "|---|---|---|"]
    for g in GROUP_ORDER:
        L.append(f"| {g} | {res['spreads_bp'][g]:.0f} | {pct(res['half_spread'][g], 2, False)} |")
    L += ["", "Impuesto en las compras: " + "; ".join(
        f"{c} {pct(t, 2, False)}" for c, t in res["tax"].items()) + "; el resto, 0. "
          f"Sede en Irlanda (Londres): {', '.join(res['irish']) or 'ninguna'}.", "",
          "Activos con predicción: " + ", ".join(
              f"{g} {res['n_assets']['grupos'].get(g, 0)}" for g in GROUP_ORDER) + ".", "",
          "## Regla fija sobre el combinado", "",
          "Rentabilidad anual / Sharpe / caída máxima / rotación anual / coste pagado al año.", "",
          "| señal | frecuencia | costes | 2012–15 | 2016–19 |", "|---|---|---|---|---|",
          "| pesos iguales | semanal | brutos | " + " | ".join(fmt(res["ew"][b]) for b in BLOCKS)
          + " |"]
    for sig in SIGNALS:
        for fname, _, _ in FREQS:
            for sname in SCALES:
                L.append(f"| {sig} | {fname} | {sname} | " + " | ".join(
                    fmt(res["runs"][f"{sig}|{fname}|{sname}|{b}"]) for b in BLOCKS) + " |")
    for title, key, order in (("grupo", "grupos", GROUP_ORDER), ("país", "países",
                                                                 COUNTRY_ORDER)):
        L += ["", f"## Reparto medio de la cartera por {title} (costes ×1)", "",
              "| cartera | bloque | " + " | ".join(order) + " | liquidez |",
              "|---|---|" + "---|" * (len(order) + 1)]
        rows = [("pesos iguales", res["ew"])] + [
            (f"{sig}, {fname}", {b: res["runs"][f"{sig}|{fname}|×1|{b}"] for b in BLOCKS})
            for sig in SIGNALS for fname, _, _ in FREQS]
        for name, by_b in rows:
            for b, (lab, _, _) in BLOCKS.items():
                m = by_b[b]
                L.append(f"| {name} | {lab} | " + " | ".join(
                    pct(m[key].get(o, 0.0), 0, False) for o in order) +
                    f" | {pct(m['liquidez'], 0, False)} |")
    L += ["", "## Decisiones (opción conservadora ante la duda)", ""] + [f"- {d}" for d in
                                                                      DECISIONS]
    return L


def main() -> None:
    if COMB.resolve() == BASE.resolve():
        raise SystemExit("usar con AGENTIC_TRADING_DATA=data/comb")
    res = evaluate()
    (OUT / f"{REPORT}.md").write_text("\n".join(markdown(res)) + "\n")
    log("hecho")


if __name__ == "__main__":
    main()
