"""El test, de principio a fin (handoff, «Protocolo del test», fijado el 23-11-2026).

Es el único sitio del proyecto donde se pone allow_test=True, y solo con --mode test y
--open-test. El test se abre una sola vez y por orden explícita del usuario.

Modos:
- test: 2020-01-01 → último dato limpio. Walk-forward anual 2020–2026 con 20 miembros; agente,
  los checkpoints congelados de scripts/agent_final.py (data/agent/final_frozen.json, con su
  sha256). Salida: data/predictions/test_wf/, checkpoints/test_wf/, logs/test/.
- rehearsal (ensayo general): lo mismo sobre 2016–2019 «como si fuera el test», con
  walk-forward desde 2016, 2 miembros y el agente de agent_bc.py (bc_ppo_s*, entrenado con
  2000–15). allow_test=False: todo sale de desarrollo (targets ≤ 31-12-2019). Solo comprueba que
  todo funciona; no decide nada. Salida: data/predictions/rehearsal_wf/, logs/test_rehearsal/.

Fases (reanudables; cada una guarda su resultado y se salta si ya está):
1. predict: por año Y y miembro m, entrena con todas las muestras con target ≥ 20 velas antes del
   1-1-Y (folds.walk_forward_years) con la configuración de final20 (5 épocas, λ = 1,
   lr = 0,001, ventana 63 + velas mensuales, target relativo), semilla 100·Y + m, y predice las
   muestras de Y. 3 procesos × 4 hilos (fijo: el número de hilos cambia los últimos bits).
   wf<Y>_m<m>.parquet por miembro; al estar todos, wf<Y>_members.parquet y wf<Y>.parquet.
2. model: métricas de §9 (evaluate) con HAR ajustado en el entrenamiento de cada año, en total,
   por año y por región, y los criterios de success_checks (μ frente a la mezcla, t de NW > 2 a
   h = 5 o 10; σ frente a HAR en los tres horizontes).
3. systems: panel semanal (plan B) con las predicciones del walk-forward y la normalización del
   entrenamiento del agente. En el entorno del agente (reglas del juego de §10, agent_rules):
   - regla fija (principal): las 20 de mayor μ_5 a partes iguales, sale al caer de las 60
     primeras, stop en d_max; c = 0,1 % y 0,5 %, y c = 0,1 % más los impuestos a la compra de
     agent/costs.py (informativo);
   - regla tipo Kelly (§11.4; evaluation.kelly_order), c = 0,1 % y 0,5 %;
   - agente (secundario): 3 semillas × 20 semillas de muestreo (1000–1019) de la política
     estocástica, c = 0,1 % y 0,5 %; además, la determinista (informativa).
   Referencias con solo costes (simulate con reference_rules, como scripts/baseline.py): pesos
   iguales de los activos con predicción en cada decisión (acciones y cripto: el mismo universo
   que ve la regla), c = 0,1 % y 0,5 % (y con impuestos), y SPY comprado y mantenido.
4. report: criterios (abajo), tablas y por año, en <out>/report.md y report.json.
5. bias: comparación con y sin deslistadas en los dos últimos años (§3) con el plan gratuito de
   Massive. Necesita una clave (MASSIVE_API_KEY); sin ella, se apunta como pendiente.

Criterios (§10, para la regla fija y, aparte, para el agente con la media de sus 60 ejecuciones):
- principal: con c = 0,1 %, más capital final y más Sharpe que los pesos iguales;
- costes: con c = 0,5 %, capital final no menor que el de los pesos iguales con c = 0,5 %;
- riesgo: con c = 0,1 %, caída máxima no peor que la de los pesos iguales;
- línea base: con c = 0,1 %, más capital final y más Sharpe que la regla tipo Kelly (para la
  regla fija y para el agente).
Informativos: SPY, t de NW (2 retardos) de la diferencia semanal de log-retorno con los pesos
iguales (c = 0,1 %) y la regla con impuestos a la compra.

Periodo de las carteras: de la primera a la última fecha de decisión semanal con predicciones
(la última muestra con target completo, 20 velas antes del último dato limpio).

Uso: python scripts/test_run.py --mode rehearsal [--phase all|predict|model|systems|report]
     python scripts/test_run.py --mode test --open-test        (solo por orden del usuario)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.folds import walk_forward_years

MODES = {
    "test": {"years": None, "split": "test", "members": 20, "run": "test_wf", "allow_test": True,
             "agents": "final", "out": ROOT / "logs" / "test", "end": "2026-12-31"},
    "rehearsal": {"years": [2016, 2017, 2018, 2019], "split": "dev", "members": 2,
                  "run": "rehearsal_wf", "allow_test": False, "agents": "bc",
                  "out": ROOT / "logs" / "test_rehearsal", "end": "2019-12-31"},
}
WORKERS, THREADS = 3, 4
DEV_END = pd.Timestamp("2019-12-31")
COSTS = (0.001, 0.005)
SAMPLE_SEEDS = range(1000, 1020)
AGENT_SEEDS = (0, 1, 2)
FROZEN = DATA_DIR / "agent" / "final_frozen.json"
WF_PATTERN = "wf[0-9][0-9][0-9][0-9].parquet"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def open_data(mode: str) -> WindowData:
    return WindowData(DATA_DIR / "dataset", allow_test=MODES[mode]["allow_test"])


def folds(mode: str, data: WindowData) -> list[dict]:
    m = MODES[mode]
    out = walk_forward_years(data.samples, m["years"], m["split"])
    if mode == "rehearsal":         # nada pasa del 31-12-2019
        for f in out:
            for k in ("train", "predict"):
                assert (data.samples["target_end"].to_numpy()[f[k]] <= DEV_END).all()
    return [f for f in out if len(f["predict"])]


def train_config():
    from agentic_trading.model.train import TrainConfig
    cfg = json.loads((DATA_DIR / "predictions" / "final20" / "config.json").read_text())["train"]
    return TrainConfig(**{**cfg, "threads": THREADS})


# ----------------------------------------------------------------------------- 1. predict

def member_path(mode: str, year: int, m: int) -> Path:
    return DATA_DIR / "predictions" / MODES[mode]["run"] / f"wf{year}_m{m}.parquet"


def train_member(mode: str, job: tuple[int, int]) -> dict:
    from agentic_trading.model.train import predict, save_model, train_model
    from scripts.train_cv import atomic_parquet
    year, m = job
    data = open_data(mode)
    f = next(f for f in folds(mode, data) if f["name"] == f"wf{year}")
    cfg, seed = train_config(), 100 * year + m
    t0 = time.time()
    model, hist = train_model(data, f["train"], cfg, seed, log=lambda *a, **k: None)
    save_model(model, ROOT / "checkpoints" / MODES[mode]["run"] / f"wf{year}" / f"member{m}.pt",
               cfg, seed, hist)
    p = predict(model, data, f["predict"]).assign(member=m)
    atomic_parquet(p, member_path(mode, year, m))
    return {"year": year, "member": m, "seed": seed, "n_train": len(f["train"]),
            "n_predict": len(f["predict"]), "seconds": time.time() - t0}


def phase_predict(mode: str) -> None:
    from agentic_trading.model.train import ensemble
    from scripts.train_cv import atomic_parquet
    M = MODES[mode]
    pdir = DATA_DIR / "predictions" / M["run"]
    pdir.mkdir(parents=True, exist_ok=True)
    data = open_data(mode)
    fl = folds(mode, data)
    (pdir / "config.json").write_text(json.dumps({
        "train": train_config().__dict__, "members": M["members"], "mode": mode,
        "seeds": "100·año + miembro", "workers": WORKERS,
        "dataset": data.meta["clean_fingerprint"],
        "years": {f["name"]: {"train": len(f["train"]), "predict": len(f["predict"])}
                  for f in fl}}, indent=1))
    jobs = [(int(f["name"][2:]), m) for f in fl for m in range(M["members"])
            if not member_path(mode, int(f["name"][2:]), m).exists()]
    log(f"predict: {len(jobs)} miembros por entrenar ({WORKERS} procesos × {THREADS} hilos)")
    timing = M["out"] / "timing.jsonl"
    timing.parent.mkdir(parents=True, exist_ok=True)
    if jobs:
        with ProcessPoolExecutor(WORKERS) as ex:
            for r in ex.map(partial(train_member, mode), jobs):
                with open(timing, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                log(f"  wf{r['year']} miembro {r['member']}: {r['seconds']:.0f} s "
                    f"({r['n_train']:,} muestras de entrenamiento)")
    for f in fl:
        parts = [member_path(mode, int(f["name"][2:]), m) for m in range(M["members"])]
        if (pdir / f"{f['name']}.parquet").exists():
            continue
        preds = [pd.read_parquet(q) for q in parts]
        atomic_parquet(pd.concat(preds), pdir / f"{f['name']}_members.parquet")
        atomic_parquet(ensemble(preds), pdir / f"{f['name']}.parquet")
        log(f"  {f['name']}: ensemble de {len(preds)} miembros")


# ----------------------------------------------------------------------------- 2. model

def phase_model(mode: str) -> dict:
    from agentic_trading.data.dataset import asset_info
    from agentic_trading.model.metrics import (
        baseline_features,
        evaluate,
        fit_har,
        predict_har,
        success_checks,
    )
    from scripts.evaluate import targets
    M = MODES[mode]
    path = M["out"] / "model.json"
    if path.exists():
        return json.loads(path.read_text())
    data = open_data(mode)
    pdir = DATA_DIR / "predictions" / M["run"]
    parts, hars, names = [], [], []
    for f in folds(mode, data):
        p = pd.read_parquet(pdir / f"{f['name']}.parquet")
        tr_rows, tr_v, tr_m = targets(data, f["train"])
        rng = np.random.default_rng(0)
        sub = rng.choice(len(tr_rows), min(len(tr_rows), 500_000), replace=False)
        har = fit_har(baseline_features(data, tr_rows[sub])["har_x"], tr_v[sub], tr_m[sub])
        rows, _, _ = targets(data, p["sample"].to_numpy())
        parts.append(p)
        hars.append(predict_har(har, baseline_features(data, rows)["har_x"]))
        names.append(f["name"])
    pred, har = pd.concat(parts, ignore_index=True), np.concatenate(hars)
    rep = {"total": evaluate(data, pred, har)}
    lab = np.repeat(names, [len(p) for p in parts])
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    reg = info["region"].to_numpy()[data.samples["asset"].to_numpy()[pred["sample"].to_numpy()]]
    for name, mask in [(n, lab == n) for n in names] + [(g, reg == g) for g in ("US", "Europa")]:
        rep[name] = evaluate(data, pred[mask].reset_index(drop=True), har[mask])
    out = {"checks": success_checks(rep["total"]), "report": rep, "n": len(pred)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------- 3. systems

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def agent_spec(mode: str) -> tuple[dict, list[tuple[str, Path]]]:
    """Normalización del entrenamiento del agente y sus checkpoints (nombre, ruta)."""
    from agentic_trading.agent.data import build_panel
    ag = DATA_DIR / "agent"
    if MODES[mode]["agents"] == "final":
        fr = json.loads(FROZEN.read_text())
        cks = []
        for name, a in fr["agents"].items():
            p = ag / a["dir"] / a["ckpt"]
            if sha256(p) != a["sha256"]:
                raise SystemExit(f"{p}: el checkpoint no coincide con el congelado")
            cks.append((name, p))
        return fr["norm"], cks
    norm = build_panel("final20", freq="weekly", rank=True).norm      # la de 2000–15
    return norm, [(f"s{s}", ag / f"bc_ppo_s{s}" / "ckpt_sel.pt") for s in AGENT_SEEDS]


def make_panel(mode: str, norm: dict):
    from agentic_trading.agent.data import build_panel
    M = MODES[mode]
    return build_panel(M["run"], data=open_data(mode), freq="weekly", rank=True, end=M["end"],
                       pattern=WF_PATTERN, norm=norm)


def summary(res, a, b) -> dict:
    from agentic_trading.agent.simulator import performance
    p = {k: float(v) for k, v in performance(res, a, b).items()}
    nav = res.nav.loc[a:b]
    p["capital final"] = float(nav.iloc[-1] / nav.iloc[0])
    return p


def weekly_logret(res, dates) -> pd.Series:
    nav = res.nav.reindex(dates).ffill()
    return np.log(nav).diff().dropna()


def buy_tax(symbols) -> np.ndarray:
    from agentic_trading.agent.costs import BUY_TAX, hq_country, tax_country
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    cache = DATA_DIR / "comb" / "hq_country.json"
    countries = tax_country(uni, hq_country(uni, cache))
    return countries.reindex(list(symbols)).map(BUY_TAX).fillna(0.0).to_numpy(float)


def agent_runs(mode: str, norm: dict, item: tuple[str, str]) -> dict:
    """Las 20 semillas de muestreo (y la determinista) de un agente, con c = 0,1 % y 0,5 %."""
    import torch

    from agentic_trading.agent.ppo import panel_n_feat, run_policy
    from agentic_trading.agent.simulator import agent_rules
    from scripts.agent_eval import nets
    torch.set_num_threads(1)
    name, ck = item
    panel = make_panel(mode, norm)
    a, b = panel.dec_dates[0], panel.dec_dates[-1]
    fp = json.loads((Path(ck).parent / "fingerprint.json").read_text())
    cfg = fp["cfg"]
    net, _ = nets(cfg, ck, panel_n_feat(panel))
    out = {}
    for c in COSTS:
        rules = agent_rules(c)
        runs = []
        for s in SAMPLE_SEEDS:
            res, diag = run_policy(panel, net, a, b, rules, cfg["K"], explore=True, seed=int(s))
            runs.append({**summary(res, a, b), "posiciones": float(np.mean(diag["npos"])),
                         "liquidez": float(np.mean(diag["cash"]))})
        res, _ = run_policy(panel, net, a, b, rules, cfg["K"], explore=False)
        out[str(c)] = {"stochastic": runs, "deterministic": summary(res, a, b)}
    return {"name": name, **out}


def phase_systems(mode: str) -> dict:
    from agentic_trading.agent.evaluation import kelly_order, rule_order, run_orders
    from agentic_trading.agent.simulator import (
        Market,
        Rules,
        agent_rules,
        reference_rules,
        simulate,
    )
    from agentic_trading.model.metrics import nw_tstat
    from scripts.baseline import spy_frame
    M = MODES[mode]
    path = M["out"] / "systems.json"
    if path.exists():
        return json.loads(path.read_text())
    norm, cks = agent_spec(mode)
    t0 = time.time()
    panel = make_panel(mode, norm)
    a, b = panel.dec_dates[0], panel.dec_dates[-1]
    log(f"panel: {len(panel.dec_t)} decisiones ({a.date()} – {b.date()}), "
        f"{len(panel.market.symbols)} activos ({time.time() - t0:.0f} s)")
    syms = np.asarray(panel.market.symbols, dtype=object)
    tax = buy_tax(syms)
    by_date = {d: syms[panel.elig[k]] for k, d in enumerate(panel.dec_dates)}
    dec = panel.dec_dates[(panel.dec_dates >= a) & (panel.dec_dates <= b)]

    def ew_policy(state):
        s = by_date.get(state.date)
        return pd.Series(1.0 / len(s), index=s) if s is not None else pd.Series(dtype=float)

    res, out = {}, {"period": [str(a.date()), str(b.date())], "n_decisions": len(dec),
                    "n_assets": len(syms)}
    for c in COSTS:
        r = agent_rules(c)
        res[f"rule|{c}"] = run_orders(panel, a, b, r, lambda o, e, r=r: rule_order(o, r))
        res[f"kelly|{c}"] = run_orders(panel, a, b, r, lambda o, e, r=r: kelly_order(o, e, r))
        res[f"ew|{c}"] = simulate(panel.market, ew_policy, dec, reference_rules(c))
    r_tax = agent_rules(COSTS[0], buy_tax=tax)
    res["rule|tax"] = run_orders(panel, a, b, r_tax, lambda o, e: rule_order(o, r_tax))
    res["ew|tax"] = simulate(panel.market, ew_policy, dec,
                             Rules(cost=COSTS[0], min_weight=None, stop_loss=None, buy_tax=tax))
    spy = Market.from_frames({"SPY": spy_frame()}, start=a - pd.Timedelta(days=10), end=b)
    first = spy.dates[spy.dates >= a][0]
    res["spy"] = simulate(spy, lambda s: pd.Series({"SPY": 1.0}), [first],
                          reference_rules(COSTS[0]))
    out["systems"] = {k: summary(v, a, b) for k, v in res.items()}
    out["years"] = {k: {str(y): summary(res[k], max(a, pd.Timestamp(f"{y}-01-01")),
                                        min(b, pd.Timestamp(f"{y}-12-31")))
                        for y in range(a.year, b.year + 1)}
                    for k in ("rule|0.001", "kelly|0.001", "ew|0.001")}
    d = (weekly_logret(res["rule|0.001"], dec) - weekly_logret(res["ew|0.001"], dec)).dropna()
    m, t = nw_tstat(d.to_numpy(), 2)
    out["rule_vs_ew_weekly"] = {"mean_bp": float(1e4 * m), "t": float(t), "n": len(d)}
    log(f"reglas y referencias: {time.time() - t0:.0f} s")
    with ProcessPoolExecutor(len(cks)) as ex:
        out["agent"] = list(ex.map(partial(agent_runs, mode, norm),
                                   [(n, str(p)) for n, p in cks]))
    out["agent_checkpoints"] = {n: {"path": str(p), "sha256": sha256(Path(p))} for n, p in cks}
    log(f"agente: {time.time() - t0:.0f} s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------- 4. report

def agent_mean(sysr: dict, c: float) -> dict:
    rows = [r for ag in sysr["agent"] for r in ag[str(c)]["stochastic"]]
    keys = ("capital final", "rentabilidad anual", "Sharpe", "caída máxima", "rotación anual",
            "volatilidad")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n": len(rows)}


def criteria(x1: dict, x5: dict, S: dict) -> dict:
    ew1, ew5, k1 = S["ew|0.001"], S["ew|0.005"], S["kelly|0.001"]
    return {
        "principal": bool(x1["capital final"] > ew1["capital final"] and
                          x1["Sharpe"] > ew1["Sharpe"]),
        "costes": bool(x5["capital final"] >= ew5["capital final"]),
        "riesgo": bool(x1["caída máxima"] >= ew1["caída máxima"]),
        "kelly": bool(x1["capital final"] > k1["capital final"] and x1["Sharpe"] > k1["Sharpe"]),
    }


def pct(x, signed=True):
    return "—" if x is None or not np.isfinite(x) else (f"{100 * x:+.1f} %" if signed
                                                         else f"{100 * x:.1f} %")


def row(name: str, p: dict) -> str:
    return (f"| {name} | {p['capital final']:.3f} | {pct(p['rentabilidad anual'])} | "
            f"{pct(p['volatilidad'], False)} | {p['Sharpe']:.2f} | {pct(p['caída máxima'])} | "
            f"{p['rotación anual']:.1f} |")


def phase_report(mode: str, model: dict, sysr: dict) -> None:
    M = MODES[mode]
    S = sysr["systems"]
    crit_rule = criteria(S["rule|0.001"], S["rule|0.005"], S)
    ag1, ag5 = agent_mean(sysr, 0.001), agent_mean(sysr, 0.005)
    crit_agent = criteria(ag1, ag5, S)
    ck = model["checks"]
    yes = {True: "sí", False: "no"}
    title = ("Test (2020–)" if mode == "test" else
             "Ensayo general del test sobre 2016–2019 (no decide nada)")
    L = [f"# {title}", "",
         (f"Generado por scripts/test_run.py --mode {mode} el {time.strftime('%d-%m-%Y %H:%M')}. "
          f"Walk-forward anual con {M['members']} miembros (semilla 100·año + miembro). "
          f"Carteras del {sysr['period'][0]} al {sysr['period'][1]}, {sysr['n_decisions']} "
          f"decisiones semanales, {sysr['n_assets']} activos."), "",
         "## Criterios", "",
         "| criterio | cumple |", "|---|---|",
         f"| Modelo, μ frente a la mezcla (t NW > 2 a h = 5 o 10) | {yes[ck['μ']['ok']]} |",
         f"| Modelo, σ frente a HAR (QLIKE menor en h = 5, 10, 20) | {yes[ck['σ']['ok']]} |"]
    for who, cr in (("Regla fija (principal)", crit_rule), ("Agente (secundario)", crit_agent)):
        L += [(f"| {who}: c = 0,1 %, más capital final y más Sharpe que los pesos iguales | "
              f"{yes[cr['principal']]} |"),
              (f"| {who}: c = 0,5 %, capital final no peor que los pesos iguales | "
              f"{yes[cr['costes']]} |"),
              f"| {who}: caída máxima no peor que los pesos iguales | {yes[cr['riesgo']]} |",
              (f"| {who}: bate a la regla tipo Kelly (capital final y Sharpe) | "
              f"{yes[cr['kelly']]} |")]
    w = sysr["rule_vs_ew_weekly"]
    L += ["", (f"Informativo: diferencia semanal de log-retorno regla − pesos iguales (c = 0,1 %) "
               f"{w['mean_bp']:+.1f} pb, t de NW {w['t']:.2f} ({w['n']} semanas)."), "",
          "## Modelo", "",
          "| h | IC μ | IC mezcla | dif. (t NW) | QLIKE | QLIKE HAR |", "|---|---|---|---|---|---|"]
    for h in ("5", "10", "20"):
        r = model["report"]["total"][h]
        L.append(f"| {h} | {r['IC modelo']:.4f} | {r['IC mezcla']:.4f} | "
                 f"{r['dif. IC con la mezcla']:+.4f} ({r['t dif. IC mezcla']:.2f}) | "
                 f"{r['QLIKE modelo']:.4f} | {r['QLIKE HAR']:.4f} |")
    L += ["", "Por año y región (IC h = 5 · dif. con la mezcla (t) · QLIKE h = 5 frente a HAR):", "",
          "| tramo | IC h = 5 | dif. mezcla (t) | QLIKE / HAR |", "|---|---|---|---|"]
    for k, r in model["report"].items():
        if k == "total":
            continue
        r5 = r["5"]
        L.append(f"| {k} | {r5['IC modelo']:.4f} | {r5['dif. IC con la mezcla']:+.4f} "
                 f"({r5['t dif. IC mezcla']:.2f}) | {r5['QLIKE modelo']:.4f} / "
                 f"{r5['QLIKE HAR']:.4f} |")
    head = [("| cartera | capital final | rentab. anual | volatilidad | Sharpe | caída máx. | "
            "rotación |"), "|---|---|---|---|---|---|---|"]
    L += ["", "## Carteras", ""] + head
    names = {"rule": "Regla fija", "kelly": "Regla tipo Kelly", "ew": "Pesos iguales"}
    for c in ("0.001", "0.005"):
        for k in ("rule", "kelly", "ew"):
            L.append(row(f"{names[k]}, c = " + f"{float(c) * 100:.1f}".replace(".", ",") + " %", S[f"{k}|{c}"]))
    for c, x in (("0,1", ag1), ("0,5", ag5)):
        L.append(row(f"Agente (media de {x['n']}), c = {c} %", x))
    L += [row("Regla fija, c = 0,1 % + impuestos a la compra", S["rule|tax"]),
          row("Pesos iguales, c = 0,1 % + impuestos a la compra", S["ew|tax"]),
          row("SPY (comprar y mantener)", S["spy"]),
          "", ("Agente por semilla (media de las 20 semillas de muestreo; determinista aparte), "
          "c = 0,1 %:"), "", ("| semilla | capital final | Sharpe | caída máx. | rotación | "
          "determinista: capital / Sharpe |"), "|---|---|---|---|---|---|"]
    for ag in sysr["agent"]:
        st = ag["0.001"]["stochastic"]
        de = ag["0.001"]["deterministic"]
        L.append(f"| {ag['name']} | {np.mean([r['capital final'] for r in st]):.3f} | "
                 f"{np.mean([r['Sharpe'] for r in st]):.2f} | "
                 f"{pct(np.mean([r['caída máxima'] for r in st]))} | "
                 f"{np.mean([r['rotación anual'] for r in st]):.1f} | "
                 f"{de['capital final']:.3f} / {de['Sharpe']:.2f} |")
    L += ["", "Por año, c = 0,1 % (rentabilidad / Sharpe):", "",
          "| año | regla fija | Kelly | pesos iguales |", "|---|---|---|---|"]
    for y in sysr["years"]["rule|0.001"]:
        cells = [sysr["years"][k][y] for k in ("rule|0.001", "kelly|0.001", "ew|0.001")]
        L.append(f"| {y} | " + " | ".join(
            f"{pct(p.get('rentabilidad anual'))} / {p.get('Sharpe', float('nan')):.2f}"
            if p else "—" for p in cells) + " |")
    bias = M["out"] / "bias.json"
    L += ["", "## Sesgo de supervivencia (§3)", "",
          (json.loads(bias.read_text())["note"] if bias.exists() else "Sin hacer.")]
    (M["out"] / "report.md").write_text("\n".join(L) + "\n")
    (M["out"] / "report.json").write_text(json.dumps(
        {"criteria": {"model": {"μ": ck["μ"]["ok"], "σ": ck["σ"]["ok"]}, "rule": crit_rule,
                      "agent": crit_agent}, "agent_mean": {"0.001": ag1, "0.005": ag5}},
        indent=1, ensure_ascii=False))
    print("\n".join(L))


# ----------------------------------------------------------------------------- 5. bias

def phase_bias(mode: str) -> None:
    M = MODES[mode]
    key = os.environ.get("MASSIVE_API_KEY")
    note = ("Pendiente: la comparación con y sin deslistadas necesita una clave del plan "
            "gratuito de Massive (variable MASSIVE_API_KEY), que no está. No bloquea el resto.")
    if key and mode == "test":
        note = ("Clave de Massive presente, pero la descarga de deslistadas no está "
                "implementada todavía: pendiente.")
    M["out"].mkdir(parents=True, exist_ok=True)
    (M["out"] / "bias.json").write_text(json.dumps({"note": note}, ensure_ascii=False))
    log(note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--phase", default="all",
                    choices=["all", "predict", "model", "systems", "bias", "report"])
    ap.add_argument("--open-test", action="store_true",
                    help="obligatorio con --mode test: abre el test (una sola vez)")
    args = ap.parse_args()
    if args.mode == "test":
        if not args.open_test:
            raise SystemExit("El test es ciego: --mode test exige --open-test (orden del usuario).")
        if not FROZEN.exists():
            raise SystemExit(f"Falta {FROZEN}: congela antes el agente (scripts/agent_final.py).")
        stamp = MODES["test"]["out"] / "opened.json"
        stamp.parent.mkdir(parents=True, exist_ok=True)
        if not stamp.exists():
            stamp.write_text(json.dumps({"opened": time.strftime("%Y-%m-%d %H:%M:%S")}))
    M = MODES[args.mode]
    M["out"].mkdir(parents=True, exist_ok=True)
    ph = args.phase
    if ph in ("all", "predict"):
        phase_predict(args.mode)
    if ph in ("all", "model"):
        log("modelo: " + json.dumps({k: v["ok"] for k, v in phase_model(args.mode)["checks"]
                                     .items()}, ensure_ascii=False))
    if ph in ("all", "systems"):
        phase_systems(args.mode)
    if ph in ("all", "bias"):
        phase_bias(args.mode)
    if ph in ("all", "report"):
        phase_report(args.mode, phase_model(args.mode), phase_systems(args.mode))


if __name__ == "__main__":
    main()
