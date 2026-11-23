"""Última variante sobre el universo combinado (23-11-2026; logs/universe_ext3.md →
logs/universe_ext4.md): elegir por rendimiento esperado neto de costes.

- Predicciones: las de A del paso 2 (data/comb/predictions/ext_A; solo con el pasado).
- Puntuación de cada activo y día: (μ_rel,20 · σ_ventana − k) / (σ_20 · σ_ventana), con
  μ_rel,20 = μ_20 − deriva_20 (deriva del ensemble del bloque = la media de las de sus miembros,
  guardadas en checkpoints/ext_A/cv*/member*.pt, como μ es la media de los μ de los miembros);
  k = coste de ida y vuelta del activo con agent/costs.py: c de su grupo en la compra y en la
  venta, más el impuesto de su país en la compra (2c + impuesto); σ_20, la del modelo (en unidades
  de σ_ventana, como μ). σ_ventana, del dataset (como agent.data.load_predictions).
- La puntuación va como μ_20 en predicciones falsas (data/comb/predictions/ext4_net_x1 y _x2), y
  se usa la regla fija de universe_ext3 (scripts/ext_costs.py) cada 4 semanas con μ_20 (las 20 de
  mayor puntuación de todo el combinado, salen fuera de las 60 primeras). La estandarización del
  panel es la misma para todos los días (media y desviación típica fijas), así que no cambia el
  orden.
- Costes ×1 y ×2 (con ×2, también k ×2 en la puntuación). Frente a los pesos iguales del combinado
  (brutos) y a la regla de 4 semanas de universe_ext3 (A, μ_20, sin mirar costes), recalculada
  aquí igual. Por bloque: rentabilidad, Sharpe, caída, rotación, coste pagado y reparto por grupo
  y por país.

Lectura, fijada antes de verlo:
- Pasa si, con costes ×1, bate a los pesos iguales del combinado en rentabilidad y en Sharpe en
  los dos bloques.
- Robustez: con costes ×2, capital final no peor que los pesos iguales (en los dos bloques).
- Si no pasa, el universo combinado no sirve para operar y no se prueban más variantes sobre
  2012–19.

Uso: AGENTIC_TRADING_DATA=data/comb python scripts/ext_net.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.agent.costs import BUY_TAX, MIN_COST, asset_costs, hq_country, tax_country
from agentic_trading.agent.data import build_panel
from agentic_trading.agent.simulator import agent_rules
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.mu_metrics import block_of
from scripts.ext_costs import (
    COUNTRY_ORDER,
    GROUP_ORDER,
    SCALES,
    ew_detail,
    fmt,
    pct,
    run_detail,
    spreads,
)
from scripts.ext_mu import BLOCKS, group_panel, load_pred
from scripts.mu_tree import log

COMB = DATA_DIR
BASE = ROOT / "data"
OUT = ROOT / "logs"
REPORT = "universe_ext4"
PRED = COMB / "predictions"
CKPT = ROOT / "checkpoints" / "ext_A"
EVERY, COL = 4, 2                   # cada 4 semanas, con la columna de μ_20


def drift20() -> dict:
    """Deriva a 20 días del ensemble de cada bloque: media de las de sus miembros."""
    out = {}
    for b in BLOCKS:
        ds = [torch.load(p, weights_only=False)["drift"][2]
              for p in sorted((CKPT / b).glob("member*.pt"))]
        if len(ds) != 10:
            raise SystemExit(f"{b}: {len(ds)} miembros")
        out[b] = float(np.mean(ds))
    return out


def net_scores(data: WindowData, k_by_symbol: pd.Series, drift: dict) -> pd.DataFrame:
    """Predicciones de A con la puntuación neta en μ_5, μ_10 y μ_20."""
    pa = load_pred(PRED / "ext_A")
    smp = data.samples.iloc[pa["sample"].to_numpy()]
    sym = np.asarray(data.meta["symbols"], object)[smp["asset"].to_numpy()]
    sig_v = np.asarray(data.arr["sigma"])[smp["row"].to_numpy()].astype(float)
    blk = block_of(smp["date"])
    d = pd.Series(blk).map(drift).to_numpy(float)
    k = k_by_symbol.reindex(sym).to_numpy(float)
    if np.isnan(k).any() or np.isnan(d).any():
        raise ValueError("faltan costes o deriva")
    mu_rel = pa["mu_20"].to_numpy(float) - d
    score = (mu_rel * sig_v - k) / (pa["sigma_20"].to_numpy(float) * sig_v)
    out = pa.copy()
    for h in (5, 10, 20):
        out[f"mu_{h}"] = score
    out["block"] = blk
    return out


def write_pred(out: pd.DataFrame, name: str) -> None:
    d = PRED / name
    d.mkdir(exist_ok=True)
    for b in BLOCKS:
        out[out["block"] == b].drop(columns="block").to_parquet(d / f"{b}.parquet", index=False)
    (d / "config.json").write_text(json.dumps({"members": 0, "what": "puntuación neta (ext_net)"}))


def evaluate() -> dict:
    data = WindowData(COMB / "dataset")
    uni = pd.read_parquet(COMB / "universe.parquet")
    sp = spreads()
    half = {g: max(v / 2 / 1e4, MIN_COST) for g, v in sp.items()}
    hq = hq_country(uni, COMB / "hq_country.json")
    countries = tax_country(uni, hq)
    groups = uni.set_index("symbol")["size_group"]
    drift = drift20()
    log(f"deriva_20 por bloque: {drift}")
    syms_all = list(uni["symbol"])
    res = {"spreads_bp": sp, "half_spread": half, "tax": BUY_TAX, "drift_20": drift,
           "irish": sorted(hq.index[hq.eq("Ireland")]), "ew": {}, "runs": {}, "k": {}}
    pa_panel = build_panel("ext_A", data, freq="weekly")
    syms = list(pa_panel.market.symbols)
    g_arr = groups.reindex(syms).to_numpy(object)
    c_arr = countries.reindex(syms).to_numpy(object)
    for b, (_, a, z) in BLOCKS.items():
        res["ew"][b] = ew_detail(pa_panel, a, z, g_arr, c_arr)
    for sname, scale in SCALES.items():
        cost, tax = asset_costs(syms, groups, countries, half, scale)
        rules = agent_rules(0.0, cost_asset=cost, buy_tax=tax)
        c_all, t_all = asset_costs(syms_all, groups, countries, half, scale)
        k = pd.Series(2 * c_all + t_all, index=syms_all)
        res["k"][sname] = {g: float(k[groups.reindex(syms_all).eq(g).to_numpy()].mean())
                           for g in GROUP_ORDER}
        name = f"ext4_net_x{int(scale)}"
        write_pred(net_scores(data, k, drift), name)
        panels = {"neta": build_panel(name, data, freq="weekly"), "A (universe_ext3)": pa_panel}
        for sig, p in panels.items():
            assert list(p.market.symbols) == syms
            gp = group_panel(p, np.ones(len(syms), bool), EVERY)
            for b, (_, a, z) in BLOCKS.items():
                res["runs"][f"{sig}|{sname}|{b}"] = run_detail(gp, a, z, rules, COL, g_arr, c_arr)
            log(f"{sig} costes {sname} hecho")
    res["verdict"] = verdict(res)
    (OUT / f"{REPORT}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                   default=float))
    return res


def verdict(res: dict) -> dict:
    r, ew = res["runs"], res["ew"]
    pasa = all(r[f"neta|×1|{b}"]["rentabilidad anual"] > ew[b]["rentabilidad anual"] and
               r[f"neta|×1|{b}"]["Sharpe"] > ew[b]["Sharpe"] for b in BLOCKS)
    aguanta = all(r[f"neta|×2|{b}"]["rentabilidad anual"] >= ew[b]["rentabilidad anual"]
                  for b in BLOCKS)
    return {"pasa": pasa, "aguanta": aguanta}


DECISIONS = [
    ("Deriva_20 del ensemble = la media de las derivas de los 10 miembros del bloque (μ del "
     "ensemble es la media de los μ de los miembros)."),
    ("Costes, impuestos, Irlanda y pesos iguales: los de universe_ext3 (agent/costs.py y "
     "scripts/ext_costs.py), con el tipo más alto ante la duda y sin exenciones."),
    ("k en la puntuación: 2c + impuesto (ida y vuelta completa en cada compra), aunque la posición "
     "se mantenga varias decisiones: penaliza de más a las que se mantienen (la opción "
     "conservadora en el sentido de operar menos en lo caro)."),
    ("Robustez ×2: capital final (rentabilidad del bloque) no peor que los pesos iguales en los "
     "dos bloques."),
    ("Sesgo de supervivencia: el de logs/universe_ext.md (componentes actuales); mayor en las "
     "pequeñas, infla los niveles."),
]


def markdown(res: dict) -> list[str]:
    v = res["verdict"]
    L = ["# Universo combinado: elegir por rendimiento esperado neto de costes (23-11-2026)", "",
         ("Ver el docstring de scripts/ext_net.py. Predicciones de A del paso 2 (solo con el "
          "pasado). Puntuación (μ_rel,20 · σ_ventana − k) / (σ_20 · σ_ventana), con k el coste de "
          "ida y vuelta del activo; regla fija cada 4 semanas sobre el combinado entero (las 20 "
          "de mayor puntuación, salen fuera de las 60 primeras)."), "",
         "## Lectura (fijada antes de verlo)", "",
         (f"- Con costes ×1, bate a los pesos iguales del combinado en rentabilidad y Sharpe en "
          f"los dos bloques: **{'sí' if v['pasa'] else 'no'}**.")]
    if v["pasa"]:
        L.append(f"- Robustez, con costes ×2, capital final no peor que los pesos iguales: "
                 f"**{'sí' if v['aguanta'] else 'no'}**.")
    else:
        L.append("- Según la regla: el universo combinado no sirve para operar y no se prueban "
                 "más variantes sobre 2012–19.")
    L += ["", (f"Deriva_20 del ensemble: cv3 {res['drift_20']['cv3']:.4f}, "
          f"cv4 {res['drift_20']['cv4']:.4f}."), "",
          "k medio por grupo (ida y vuelta): " + "; ".join(
              f"{g} {pct(res['k']['×1'][g], 2, False)}" for g in GROUP_ORDER) + " (×1).", "",
          "## Regla cada 4 semanas sobre el combinado", "",
          "Rentabilidad anual / Sharpe / caída máxima / rotación anual / coste pagado al año.", "",
          "| señal | costes | 2012–15 | 2016–19 |", "|---|---|---|---|",
          "| pesos iguales | brutos | " + " | ".join(fmt(res["ew"][b]) for b in BLOCKS) + " |"]
    for sig in ("neta", "A (universe_ext3)"):
        for sname in SCALES:
            L.append(f"| {sig} | {sname} | " + " | ".join(
                fmt(res["runs"][f"{sig}|{sname}|{b}"]) for b in BLOCKS) + " |")
    for title, key, order in (("grupo", "grupos", GROUP_ORDER),
                              ("país", "países", COUNTRY_ORDER)):
        L += ["", f"## Reparto medio de la cartera por {title}", "",
              "| cartera | costes | bloque | " + " | ".join(order) + " | liquidez |",
              "|---|---|---|" + "---|" * (len(order) + 1)]
        rows = [("pesos iguales", "brutos", res["ew"])] + [
            (sig, sname, {b: res["runs"][f"{sig}|{sname}|{b}"] for b in BLOCKS})
            for sig in ("neta", "A (universe_ext3)") for sname in SCALES]
        for name, sname, by_b in rows:
            for b, (lab, _, _) in BLOCKS.items():
                m = by_b[b]
                L.append(f"| {name} | {sname} | {lab} | " + " | ".join(
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
