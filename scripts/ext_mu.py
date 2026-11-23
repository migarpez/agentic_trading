"""Paso 2 de la ampliación del universo (22-11-2026; logs/universe_ext.md): ¿μ entrenado con el
universo ampliado sirve en las empresas pequeñas, neto de costes?

Montaje, solo con el pasado (como scripts/mu_recency.py):
- Dataset combinado (universo actual + ampliado, data/comb/, scripts/build_comb.py). Se corre con
  AGENTIC_TRADING_DATA=data/comb: DATA_DIR es el combinado; el universo actual (data/) solo se lee.
- Bloque 2012–15 (cv3): entrenar con 2000–2011; bloque 2016–19 (cv4): con 2000–2015; hueco de 20
  velas (train_cv.py --train-years). Configuración de final20; 10 miembros por bloque con las
  semillas de los miembros 0–9 de final20 en ese bloque (1000·k + m).
- Variantes por pares (mismas semillas): A, tal cual; B, A más el escalar de tamaño
  (TrainConfig.size; data/size.py). En data/comb/predictions/ext_{A,B}/.
- Paralelismo: medido antes de entrenar (1 época del 2 %, como scripts/mu_smoke.py, con el dataset
  combinado) con varias combinaciones de procesos × hilos; se queda la de más épocas por segundo
  cuya memoria usada en el pico no pase del 80 % de la RAM. logs/universe_ext2/parallel.json.

Medidas en 2012–19, por grupo (grandes, medianas EE. UU., pequeñas EE. UU., medianas y pequeñas
Europa; size_group del universo combinado), con deciles dentro de cada grupo (scripts/mu_deciles.py,
en las fechas semanales de los grandes, con ≥ 20 acciones del grupo):
1. Pierna larga (D9–D10) de A y de B frente a la de la mezcla del grupo (rangos dentro del grupo),
   por pares con t de NW (2 retardos); B frente a A; en los grandes, A y B frente a P0
   (data/predictions/rec_P0, logs/mu_tree/recency.md: las mismas semillas con solo los grandes).
2. La regla fija del entorno del agente (agent.evaluation.rule_order: las 20 de mayor μ_5, a partes
   iguales, sale al caer fuera de las 60 primeras, stop en d_max, reglas de agent_rules) dentro de
   cada grupo, semanal; y la variante con μ_20 cada 4 semanas (una decisión de cada 4). Con la
   mezcla en lugar de μ (la misma regla). Coste c = la mitad de la horquilla de Abdi–Ranaldo del
   grupo en 2012–19 (logs/universe_ext.json), con un mínimo del 0,1 %, y c = 0,5 %. Frente a los
   pesos iguales del grupo, brutos (sin costes, como en agent_eval; rebalanceo en cada decisión
   semanal). Cada bloque es un episodio que empieza en liquidez. Rentabilidad, Sharpe, caída y
   rotación.
3. Correlación de Spearman entre acciones de μ_5 con el momentum de 12 meses, media de las fechas
   semanales, por grupo.

Reglas, fijadas antes de ver resultados:
- La ampliación sirve al agente si, en algún grupo, la regla fija con μ (A o B, cualquiera de las
  dos frecuencias), con el coste de su grupo, bate a los pesos iguales del grupo en rentabilidad
  y en Sharpe en los dos bloques.
- μ aporta sobre las reglas simples en un grupo si su pierna larga (A o B) bate a la de la mezcla
  con t ≥ 2 en 2012–19.
- B se queda si su pierna larga mejora a la de A con t ≥ 1 en las pequeñas de EE. UU. y en Europa.

Reanudable (miembros ya entrenados no se repiten; huella en data/comb/predictions/ext_*/
tree_fingerprint.json). Escribe logs/universe_ext2.md y .json; log en logs/universe_ext2/run.log.

Uso: AGENTIC_TRADING_DATA=data/comb setsid nohup .venv/bin/python -u scripts/ext_mu.py \
         > logs/universe_ext2/run.log 2>&1 &
     AGENTIC_TRADING_DATA=data/comb python scripts/ext_mu.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.agent.data import Panel, build_panel, equal_weight_logret
from agentic_trading.agent.evaluation import _perf, run_rule
from agentic_trading.agent.simulator import agent_rules
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info, dataset_id
from agentic_trading.data.size import LARGE
from agentic_trading.data.universe import EXT_EU_GROUP, EXT_US_GROUPS
from agentic_trading.model.metrics import baseline_features, baseline_mix, daily_rank, nw_tstat
from agentic_trading.model.mu_metrics import NW_LAGS_WEEKLY, OpenPrices, block_of, weekly_frame
from agentic_trading.model.train import ensemble
from scripts.mu_deciles import decile_series
from scripts.mu_smoke import BENCH
from scripts.mu_tree import CFG, PY, log, run

BASE = ROOT / "data"
COMB = DATA_DIR
OUT = ROOT / "logs"
LOGDIR = OUT / "universe_ext2"
PRED = COMB / "predictions"
FOLDS = {3: "2000-2011", 4: "2000-2015"}
BLOCKS = {"cv3": ("2012–15", "2012-01-01", "2015-12-31"),
          "cv4": ("2016–19", "2016-01-01", "2019-12-31")}
MEMBERS = 10
VARIANTS = {"A": [], "B": ["--size"]}
GROUPS = [LARGE, *EXT_US_GROUPS.values(), EXT_EU_GROUP]
SMALL = [EXT_US_GROUPS["SP600"], EXT_EU_GROUP]
MIN_STOCKS = 20
MIN_COST, HIGH_COST = 0.001, 0.005
PARALLEL = [(6, 2), (8, 1), (10, 1), (12, 1)]
MEM_MAX = 0.80
PREFIX, REPORT = "ext", "universe_ext2"
RULE = {"t_mu_vs_mix": 2.0, "t_b_vs_a": 1.0}


# ----------------------------------------------------------------------------- paralelismo

def bench_parallel() -> dict:
    """Épocas por segundo y pico de memoria de cada combinación (procesos × hilos)."""
    import psutil
    p = LOGDIR / "parallel.json"
    if p.exists():
        return json.loads(p.read_text())
    total = psutil.virtual_memory().total
    res = {}
    for n, t in PARALLEL:
        env = {**os.environ, "OMP_NUM_THREADS": str(t), "MKL_NUM_THREADS": str(t)}
        base_used = psutil.virtual_memory().used
        procs = [subprocess.Popen([PY, "-c", BENCH, str(t), "0.02", "train",
                                   json.dumps({"size": True})], cwd=ROOT,
                                  stdout=subprocess.PIPE, text=True, env=env) for _ in range(n)]
        peak, stop = [0], threading.Event()

        def watch(stop=stop, peak=peak, base_used=base_used):
            while not stop.is_set():
                peak[0] = max(peak[0], psutil.virtual_memory().used - base_used)
                time.sleep(1)
        th = threading.Thread(target=watch)
        th.start()
        out = [json.loads(q.communicate()[0].strip().splitlines()[-1]) for q in procs]
        stop.set()
        th.join()
        ep = float(np.mean([o["epoch_s"] for o in out]))
        res[f"{n}x{t}"] = {"workers": n, "threads": t, "epoch_s": ep, "epochs_per_s": n / ep,
                           "peak_used_gb": peak[0] / 1e9, "peak_share": peak[0] / total}
        log(f"paralelismo {n}×{t}: {ep:.0f} s por época, {n / ep:.3f} épocas/s, "
            f"pico {peak[0] / 1e9:.1f} GB")
    ok = {k: v for k, v in res.items() if v["peak_share"] <= MEM_MAX}
    best = max(ok, key=lambda k: ok[k]["epochs_per_s"])
    out = {"runs": res, "workers": res[best]["workers"], "threads": res[best]["threads"],
           "why": f"la de más épocas/s con el pico de memoria ≤ {MEM_MAX:.0%} de la RAM",
           "ram_gb": total / 1e9}
    p.write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- entrenamiento

def train_all(workers: int, threads: int) -> bool:
    data = WindowData(COMB / "dataset")
    jobs = []
    for v, flags in VARIANTS.items():
        name = f"ext_{v}"
        d = PRED / name
        d.mkdir(parents=True, exist_ok=True)
        fp = {"dataset": dataset_id(data.meta), "cfg": CFG, "flags": flags,
              "train_years": FOLDS, "members": MEMBERS, "seeds": "1000·k + m"}
        fp = json.loads(json.dumps(fp, default=str))
        p = d / "tree_fingerprint.json"
        if p.exists() and json.loads(p.read_text()) != fp:
            log(f"{name}: la huella no coincide; no se sobrescribe")
            return False
        p.write_text(json.dumps(fp, indent=1))
        jobs += [(name, flags, k, m) for m in range(MEMBERS) for k in FOLDS
                 if not (d / f"cv{k}_m{m}.parquet").exists()]
    log(f"{len(jobs)} entrenamientos pendientes ({workers} a la vez × {threads} hilos)")

    def job(j):
        name, flags, k, m = j
        cmd = [PY, "-u", str(ROOT / "scripts" / "train_cv.py"), "--run", name, "--folds", str(k),
               "--members", str(MEMBERS), "--member-ids", str(m), "--threads", str(threads),
               "--train-years", FOLDS[k], *CFG, *flags]
        out = LOGDIR / "train" / f"{name}_cv{k}_m{m}.log"
        env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
        return run(cmd, out, env) or run(cmd, out, env)

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(job, jobs))
    for v in VARIANTS:
        for k in FOLDS:
            d = PRED / f"ext_{v}"
            if not (d / f"cv{k}.parquet").exists() and all(
                    (d / f"cv{k}_m{m}.parquet").exists() for m in range(MEMBERS)):
                preds = [pd.read_parquet(d / f"cv{k}_m{m}.parquet") for m in range(MEMBERS)]
                pd.concat(preds).to_parquet(d / f"cv{k}_members.parquet", index=False)
                ensemble(preds).to_parquet(d / f"cv{k}.parquet", index=False)
    return all(ok)


# ----------------------------------------------------------------------------- evaluación

def load_pred(d: Path) -> pd.DataFrame:
    return (pd.concat([pd.read_parquet(d / f"{b}.parquet") for b in BLOCKS], ignore_index=True)
            .sort_values("sample").reset_index(drop=True))


def p0_on_comb(data: WindowData, samples: np.ndarray) -> np.ndarray:
    """μ_5 de P0 (universo actual) en las muestras del combinado (mismo símbolo y fecha)."""
    base = WindowData(BASE / "dataset")
    p0 = load_pred(BASE / "predictions" / "rec_P0")
    sb = base.samples.iloc[p0["sample"].to_numpy()]
    key_b = pd.Series(p0["mu_5"].to_numpy(), index=pd.MultiIndex.from_arrays(
        [np.asarray(base.meta["symbols"], object)[sb["asset"].to_numpy()], sb["date"].to_numpy()]))
    sc = data.samples.iloc[samples]
    idx = pd.MultiIndex.from_arrays([np.asarray(data.meta["symbols"], object)[sc["asset"]
                                                                              .to_numpy()],
                                     sc["date"].to_numpy()])
    return key_b.reindex(idx).to_numpy()


def leg_stats(s: pd.Series) -> dict:
    s = s.dropna()
    m, t = nw_tstat(s.to_numpy(), NW_LAGS_WEEKLY)
    blk = block_of(s.index)
    return {"mean": m, "t": t, "blocks": {b: dict(zip(("mean", "t"), nw_tstat(
        s[blk == b].to_numpy(), NW_LAGS_WEEKLY))) for b in BLOCKS}}


def week_dates() -> pd.DatetimeIndex:
    """Fechas semanales de los grandes en 2012–19 (las de logs/universe_ext.md)."""
    data = WindowData(BASE / "dataset")
    uni = pd.read_parquet(BASE / "universe.parquet")
    equity = (asset_info(data.meta["symbols"], uni)["asset_class"] == "equity").to_numpy()
    dev = data.samples.iloc[data.split("dev")]
    d = dev["date"]
    wk = weekly_dates(d[d.dt.dayofweek < 5])
    n_eq = dev[equity[dev["asset"].to_numpy()]].groupby("date").size()
    wk = wk[wk.isin(n_eq[n_eq >= MIN_STOCKS].index)]
    return wk[wk >= pd.Timestamp("2012-01-01")]


def legs_and_corr(data, pa, pb, groups_by_symbol) -> tuple[dict, pd.DataFrame]:
    samples = pa["sample"].to_numpy()
    assert np.array_equal(samples, pb["sample"].to_numpy())
    uni = pd.read_parquet(COMB / "universe.parquet")
    syms = data.meta["symbols"]
    equity = (asset_info(syms, uni)["asset_class"] == "equity").to_numpy()
    prices = OpenPrices.from_clean(COMB / "clean", [syms[a] for a in np.flatnonzero(equity)],
                                   start="2010-06-01")
    wf = weekly_frame(data, samples, equity, week_dates(), prices)
    pos = np.searchsorted(samples, wf["sample"].to_numpy())
    wf["A"], wf["B"] = pa["mu_5"].to_numpy()[pos], pb["mu_5"].to_numpy()[pos]
    wf["P0"] = p0_on_comb(data, wf["sample"].to_numpy())
    wf["group"] = groups_by_symbol.reindex(np.asarray(syms, object)[wf["asset"]]).to_numpy()
    base = baseline_features(data, data.samples["row"].to_numpy()[wf["sample"].to_numpy()])
    wf["rev"], wf["mom"] = base["reversal"], base["momentum"]
    res = {}
    for g in GROUPS:
        q = wf[wf["group"] == g]
        n = q.groupby("date").size()
        q = q[q["date"].isin(n[n >= MIN_STOCKS].index)].reset_index(drop=True)
        d = q["date"].to_numpy()
        sig = {"A": q["A"].to_numpy(), "B": q["B"].to_numpy(),
               "mezcla": baseline_mix(d, q["rev"].to_numpy(), q["mom"].to_numpy())}
        if g == LARGE:
            if q["P0"].isna().any():
                raise SystemExit(f"P0 sin predicción en {int(q['P0'].isna().sum())} filas")
            sig["P0"] = q["P0"].to_numpy()
        legs = {k: decile_series(q, x)["larga"] for k, x in sig.items()}
        r = {"stocks": int(q["symbol"].nunique()) if "symbol" in q else int(q["asset"].nunique()),
             "weeks": int(n[n >= MIN_STOCKS].size),
             "level": {k: leg_stats(s) for k, s in legs.items()},
             "diff": {f"{a} − {b}": leg_stats(legs[a] - legs[b])
                      for a, b in [("A", "mezcla"), ("B", "mezcla"), ("B", "A"),
                                   *([("A", "P0"), ("B", "P0")] if g == LARGE else [])]}}
        corr = {}
        for k in ("A", "B", "mezcla", *(["P0"] if g == LARGE else [])):
            rx, rm = daily_rank(d, sig[k]), daily_rank(d, q["mom"].to_numpy())
            c = pd.DataFrame({"d": d, "x": rx, "m": rm}).groupby("d").apply(
                lambda z: z["x"].corr(z["m"]), include_groups=False)
            blk = block_of(c.index)
            corr[k] = {"2012–19": float(c.mean()),
                       **{BLOCKS[b][0]: float(c[blk == b].mean()) for b in BLOCKS}}
        r["corr_mom"] = corr
        res[g] = r
    return res, wf


def mix_run(data, pa: pd.DataFrame, groups_by_symbol: pd.Series) -> None:
    """Predicciones falsas con la mezcla (rangos dentro de la fecha y el grupo) como μ_5 y μ_20,
    para correr la regla fija con la mezcla en el mismo entorno (data/comb/predictions/ext_mix)."""
    d = PRED / f"{PREFIX}_mix"
    d.mkdir(exist_ok=True)
    smp = data.samples.iloc[pa["sample"].to_numpy()]
    g = groups_by_symbol.reindex(np.asarray(data.meta["symbols"], object)[smp["asset"]
                                                                         .to_numpy()]).to_numpy()
    key = pd.Series(smp["date"].astype(str).to_numpy()) + "|" + pd.Series(g).astype(str)
    base = baseline_features(data, smp["row"].to_numpy())
    mix = baseline_mix(key.to_numpy(), base["reversal"], base["momentum"])
    out = pa.copy()
    for h in (5, 10, 20):
        out[f"mu_{h}"] = mix
    for b in BLOCKS:
        m = block_of(smp["date"]) == b
        out[m].to_parquet(d / f"{b}.parquet", index=False)
    (d / "config.json").write_text(json.dumps({"members": 0, "what": "mezcla (rangos)"}))


def group_panel(p: Panel, mask: np.ndarray, every: int = 1) -> Panel:
    """El panel con solo los activos de mask (columnas del mercado), y una decisión de cada
    `every`."""
    elig, feats, zs = [], [], []
    for e, f, z in zip(p.elig, p.feats, p.z_mu5):
        m = mask[e]
        elig.append(e[m])
        feats.append(f[m])
        zs.append(z[m])
    sl = slice(None, None, every)
    dec_t, elig, feats, zs = p.dec_t[sl], elig[sl], feats[sl], zs[sl]
    return Panel(p.market, dec_t, p.dec_dates[sl], elig, feats, zs, p.close_ffill, p.norm,
                 p.fingerprint, equal_weight_logret(p.close_ffill, dec_t, elig))


def ew_perf(p: Panel, start, end) -> dict:
    """Pesos iguales del grupo, brutos: rebalanceo en cada decisión; NAV diaria entre medias."""
    k0, k1 = p.k_range(start, end)
    days, nav = [(p.dec_t[k0], 1.0, 0.0)], 1.0
    for k in range(k0, k1 - 1):
        t0, t1, e = p.dec_t[k], p.dec_t[k + 1], p.elig[k]
        a = p.close_ffill[t0, e]
        for t in range(t0 + 1, t1 + 1):
            r = p.close_ffill[t, e] / a
            ok = np.isfinite(r)
            days.append((t, nav * (float(r[ok].mean()) if ok.any() else 1.0), 0.0))
        nav = days[-1][1]
    return _perf(p, days)


def rules_eval(groups_by_symbol: pd.Series, spreads: dict) -> dict:
    data = WindowData(COMB / "dataset")
    panels = {k: build_panel(f"{PREFIX}_{k}", data, freq="weekly") for k in ("A", "B", "mix")}
    syms = np.asarray(panels["A"].market.symbols, object)
    for p in panels.values():
        assert list(p.market.symbols) == list(syms)
    res = {}
    for g in GROUPS:
        mask = (groups_by_symbol.reindex(syms) == g).to_numpy()
        cost = max(spreads[g] / 2 / 1e4, MIN_COST)
        r = {"cost_group": cost, "runs": {}, "ew": {}}
        for b, (_, a, z) in BLOCKS.items():
            r["ew"][b] = ew_perf(group_panel(panels["A"], mask), a, z)
        for sig, pname in (("A", "A"), ("B", "B"), ("mezcla", "mix")):
            for freq, every, col in (("semanal μ_5", 1, 0), ("4 semanas μ_20", 4, 2)):
                gp = group_panel(panels[pname], mask, every)
                for cname, c in (("grupo", cost), ("0,5 %", HIGH_COST)):
                    for b, (_, a, z) in BLOCKS.items():
                        m = run_rule(gp, a, z, agent_rules(c), col=col)
                        r["runs"][f"{sig}|{freq}|{cname}|{b}"] = m
        res[g] = r
        log(f"regla fija: {g} hecho")
    return res


def evaluate() -> dict:
    data = WindowData(COMB / "dataset")
    uni = pd.read_parquet(COMB / "universe.parquet")
    groups = uni.set_index("symbol")["size_group"]
    spreads = {g: v["periods"]["2012–19"]["spread"] for g, v in
               json.loads((OUT / "universe_ext.json").read_text())["groups"].items()}
    pa, pb = load_pred(PRED / f"{PREFIX}_A"), load_pred(PRED / f"{PREFIX}_B")
    legs, _ = legs_and_corr(data, pa, pb, groups)
    log("piernas y correlaciones hechas")
    mix_run(data, pa, groups)
    rules = rules_eval(groups, spreads)
    res = {"legs": legs, "rules": rules, "spreads_bp": spreads}
    res["verdict"] = verdict(res)
    (OUT / f"{REPORT}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                       default=float))
    return res


def verdict(res: dict) -> dict:
    L, R = res["legs"], res["rules"]
    serves = []
    for g in GROUPS:
        ew = R[g]["ew"]
        for sig in ("A", "B"):
            for freq in ("semanal μ_5", "4 semanas μ_20"):
                ok = all(R[g]["runs"][f"{sig}|{freq}|grupo|{b}"]["rentabilidad anual"]
                         > ew[b]["rentabilidad anual"]
                         and R[g]["runs"][f"{sig}|{freq}|grupo|{b}"]["Sharpe"] > ew[b]["Sharpe"]
                         for b in BLOCKS)
                if ok:
                    serves.append(f"{g}: {sig}, {freq}")
    adds = [f"{g}: {v}" for g in GROUPS for v in ("A", "B")
            if L[g]["diff"][f"{v} − mezcla"]["mean"] > 0
            and L[g]["diff"][f"{v} − mezcla"]["t"] >= RULE["t_mu_vs_mix"]]
    b_keeps = all(L[g]["diff"]["B − A"]["mean"] > 0 and L[g]["diff"]["B − A"]["t"]
                  >= RULE["t_b_vs_a"] for g in SMALL)
    return {"sirve_al_agente": serves, "mu_aporta": adds, "B_se_queda": b_keeps}


# ----------------------------------------------------------------------------- informe

def f(x, nd=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def markdown(res: dict) -> list[str]:
    L, R, V = res["legs"], res["rules"], res["verdict"]
    out = ["# Universo ampliado, paso 2: μ entrenado con medianas y pequeñas (23-11-2026)", "",
           ("Ver el docstring de scripts/ext_mu.py. Solo con el pasado (2012–15 con 2000–11; "
            "2016–19 con 2000–15), 10 miembros, por pares A (tal cual) y B (más el escalar de "
            "tamaño). Piernas: exceso semanal de apertura a apertura sobre la media del grupo, pb "
            "por semana, deciles dentro del grupo; t de NW (2 retardos)."), "",
           "## Lectura (reglas fijadas antes de verlo)", "",
           ("- La ampliación sirve al agente (regla fija con μ, coste del grupo, bate a los pesos "
            "iguales del grupo en rentabilidad y Sharpe en los dos bloques): "
            + ("**sí**: " + "; ".join(V["sirve_al_agente"]) if V["sirve_al_agente"]
               else "**no**, en ningún grupo") + "."),
           ("- μ aporta sobre las reglas simples (pierna larga − mezcla, t ≥ 2 en 2012–19): "
            + ("**sí**: " + "; ".join(V["mu_aporta"]) if V["mu_aporta"]
               else "**no**, en ningún grupo") + "."),
           ("- B se queda (B − A en la pierna larga con t ≥ 1 en las pequeñas de EE. UU. y en "
            f"Europa): **{'sí' if V['B_se_queda'] else 'no'}**."), "",
           "## 1. Pierna larga (D9–D10), 2012–19", "",
           ("| grupo | acciones | A | B | mezcla | A − mezcla (t) | B − mezcla (t) | B − A (t) | "
           "A − P0 (t) | B − P0 (t) |"), "|---|---|---|---|---|---|---|---|---|---|"]

    def dt(d, k):
        x = d.get(k)
        return "—" if x is None else f"{f(x['mean'])} ({f(x['t'], 2, False)})"
    for g in GROUPS:
        lv, df = L[g]["level"], L[g]["diff"]
        out.append(f"| {g} | {L[g]['stocks']} | {f(lv['A']['mean'])} | {f(lv['B']['mean'])} | "
                   f"{f(lv['mezcla']['mean'])} | {dt(df, 'A − mezcla')} | "
                   f"{dt(df, 'B − mezcla')} | {dt(df, 'B − A')} | {dt(df, 'A − P0')} | "
                   f"{dt(df, 'B − P0')} |")
    out += ["", "Por bloque (media, pb por semana):", "",
            "| grupo | bloque | A | B | mezcla | A − mezcla | B − mezcla | B − A |",
            "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for b, (lab, *_) in BLOCKS.items():
            lv, df = L[g]["level"], L[g]["diff"]
            out.append(f"| {g} | {lab} | " + " | ".join(
                f(x["blocks"][b]["mean"]) for x in (lv["A"], lv["B"], lv["mezcla"],
                                                    df["A − mezcla"], df["B − mezcla"],
                                                    df["B − A"])) + " |")
    out += ["", "## 2. Regla fija dentro de cada grupo", "",
            ("Rentabilidad anual / Sharpe / caída máxima / rotación anual. Pesos iguales del "
             "grupo, brutos. Coste del grupo = máx(horquilla / 2, 0,1 %): "
             + ", ".join(f"{g} {R[g]['cost_group']:.2%}" for g in GROUPS) + "."), ""]
    for g in GROUPS:
        out += [f"### {g}", "", "| señal | frecuencia | coste | " + " | ".join(
            BLOCKS[b][0] for b in BLOCKS) + " |", "|---|---|---|---|---|"]
        out.append("| pesos iguales | — | 0 | " + " | ".join(
            fmt_perf(R[g]["ew"][b]) for b in BLOCKS) + " |")
        for sig in ("A", "B", "mezcla"):
            for freq in ("semanal μ_5", "4 semanas μ_20"):
                for cname in ("grupo", "0,5 %"):
                    lab = freq if sig != "mezcla" else freq.replace("μ_5", "").replace(
                        "μ_20", "").strip()
                    out.append(f"| {sig} | {lab} | {cname} | " + " | ".join(
                        fmt_perf(R[g]["runs"][f"{sig}|{freq}|{cname}|{b}"]) for b in BLOCKS)
                        + " |")
        out.append("")
    out += ["## 3. Correlación de Spearman con el momentum de 12 meses", "",
            "| grupo | señal | 2012–19 | 2012–15 | 2016–19 |", "|---|---|---|---|---|"]
    for g in GROUPS:
        for k, c in L[g]["corr_mom"].items():
            out.append(f"| {g} | {k} | {c['2012–19']:.2f} | {c['2012–15']:.2f} | "
                       f"{c['2016–19']:.2f} |")
    out += ["", *DECISIONS]
    return out


def fmt_perf(m: dict) -> str:
    return (f"{m['rentabilidad anual']:+.1%} / {m['Sharpe']:.2f} / {m['caída máxima']:.0%} / "
            f"{m['rotación anual']:.1f}")


DECISIONS = [
    "## Decisiones (opción conservadora ante la duda)", "",
    ("- Pesos iguales del grupo brutos (sin costes), como en agent_eval: la referencia más "
     "difícil de batir."),
    ("- La mezcla en la regla fija usa los rangos dentro de la fecha y del grupo; en la variante de "
     "4 semanas, la misma mezcla (no hay una mezcla a 20 días)."),
    ("- Tamaño (B): el cambio a euros de Yahoo empieza en 2003–2005; antes, el primer cambio "
     "conocido. Mediana de C·V solo con V > 0; sin ninguna, percentil 0,5."),
    ("- «En algún grupo» incluye a los grandes: la regla se aplica a los cuatro, pero un acierto "
     "solo en los grandes no diría que las pequeñas sirven."),
    ("- Paralelismo: la combinación de más épocas por segundo con el pico de memoria ≤ 80 % de la "
     "RAM (logs/universe_ext2/parallel.json)."),
    ("- Sesgo de supervivencia: el de logs/universe_ext.md (componentes actuales, más "
     "seleccionados en las pequeñas); infla los niveles y más el momentum."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--prefix", default="ext", help="predicciones <prefix>_A y _B (humo)")
    ap.add_argument("--out", default="universe_ext2", help="nombre del informe en logs/")
    args = ap.parse_args()
    global PREFIX, REPORT
    PREFIX, REPORT = args.prefix, args.out
    if COMB.resolve() == BASE.resolve():
        raise SystemExit("Corre con AGENTIC_TRADING_DATA=data/comb (el dataset combinado).")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        par = bench_parallel()
        log(f"paralelismo elegido: {par['workers']} × {par['threads']}")
        if not train_all(par["workers"], par["threads"]):
            log("algún entrenamiento falló; se evalúa solo si están todos")
    if not all((PRED / f"{PREFIX}_{v}" / f"{b}.parquet").exists() for v in VARIANTS for b in BLOCKS):
        log("faltan predicciones; no se evalúa")
        return
    res = evaluate()
    md = markdown(res)
    (OUT / f"{REPORT}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log("hecho")


if __name__ == "__main__":
    main()
