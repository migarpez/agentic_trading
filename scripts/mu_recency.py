"""Prueba de adaptación (22-11-2026): ¿el modelo deja de funcionar después de 2012 porque no se
adapta? (logs/mu_tree/decay.md: la ventaja de final20 sobre la mezcla está casi toda en 2000–11.)

Montaje, como en la operativa real, solo con el pasado:
- Bloque 2012–15 (cv3): entrenar con 2000–2011. Bloque 2016–19 (cv4): entrenar con 2000–2015.
  Mismo hueco de 20 velas (train_cv.py --train-years, folds.manual_fold).
- Configuración de final20; 10 miembros por bloque, semillas de los miembros 0–9 de final20 en
  ese bloque (1000·k + m).
- Variantes, por pares (mismas semillas): P0, todas las muestras con el mismo peso (control); P4,
  peso ∝ 2^(−antigüedad / 4 años) desde el final del entrenamiento, normalizado a media 1
  (TrainConfig.recency_halflife, en L_μ y L_σ); P2, semivida de 2 años.
- En data/predictions/rec_P{0,4,2}/ (cv3 y cv4). Nada abre el test; final20 no se toca.

Métricas en 2012–19 (los dos bloques juntos y por separado; medidas de logs/mu_tree/deciles.md:
exceso semanal de apertura a apertura sobre la media de las acciones, en pb):
- Principal: pierna larga (D9–D10), diferencia por pares con P0 y su t de NW (2 retardos).
- Además: top 20, pierna corta, IC (media de h = 5 y 10) y QLIKE (media de h = 5, 10 y 20); la
  pierna larga y la diferencia entre piernas frente a la mezcla de success_checks; y P0 frente a
  final20 (miembros 0–9, las mismas semillas, y los 20), como referencia.

Regla, fijada antes de ver resultados: una variante funciona si su pierna larga mejora a P0 con
t ≥ 2 en 2012–19, es positiva en los dos bloques y la QLIKE no empeora más de un 2 %. Si
funcionan las dos, se queda la de mayor mejora. Si funciona, el problema es de adaptación; si no
funciona ninguna, la ventaja de 2000–11 dependía de patrones que ya no están, o del sesgo, y
ningún ajuste del entrenamiento la recupera.

Reanudable (miembros ya entrenados no se repiten; huella en data/predictions/rec_*/
tree_fingerprint.json). Paralelismo de mu_tree (logs/mu_tree/parallel.json). Escribe
logs/mu_tree/recency.md y .json; log en logs/mu_tree/recency.log.

Uso: setsid nohup .venv/bin/python -u scripts/mu_recency.py > logs/mu_tree/recency.log 2>&1 &
     python scripts/mu_recency.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info, dataset_id
from agentic_trading.model.compare import load_members
from agentic_trading.model.metrics import (
    baseline_features,
    baseline_mix,
    daily_ic,
    qlike,
)
from agentic_trading.model.mu_metrics import (
    OpenPrices,
    block_of,
    level,
    paired,
    tail_series,
    weekly_frame,
)
from agentic_trading.model.train import ensemble
from scripts.mu_deciles import decile_series
from scripts.mu_tree import CFG, PY, log, run

OUT = ROOT / "logs" / "mu_tree"
PRED = DATA_DIR / "predictions"
FOLDS = {3: "2000-2011", 4: "2000-2015"}
BLOCKS = {"cv3": "2012–15", "cv4": "2016–19"}
MEMBERS = 10
VARIANTS = {"P0": [], "P4": ["--recency-halflife", "4"], "P2": ["--recency-halflife", "2"]}
RULE = {"t_min": 2.0, "qlike_worse_max": 0.02}
H = (5, 10, 20)


# ----------------------------------------------------------------------------- entrenamiento

def train_all(workers: int, threads: int) -> bool:
    data = WindowData(DATA_DIR / "dataset")
    jobs = []
    for v, flags in VARIANTS.items():
        name = f"rec_{v}"
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
        out = OUT / "train" / f"{name}_cv{k}_m{m}.log"
        env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
        return run(cmd, out, env) or run(cmd, out, env)

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(job, jobs))
    for v in VARIANTS:
        for k in FOLDS:
            d = PRED / f"rec_{v}"
            if not (d / f"cv{k}.parquet").exists() and all(
                    (d / f"cv{k}_m{m}.parquet").exists() for m in range(MEMBERS)):
                preds = [pd.read_parquet(d / f"cv{k}_m{m}.parquet") for m in range(MEMBERS)]
                pd.concat(preds).to_parquet(d / f"cv{k}_members.parquet", index=False)
                ensemble(preds).to_parquet(d / f"cv{k}.parquet", index=False)
    return all(ok)


# ----------------------------------------------------------------------------- evaluación

def load(name: str, members=None) -> pd.DataFrame:
    parts = []
    for b in BLOCKS:
        if members is None:
            parts.append(pd.read_parquet(PRED / name / f"{b}.parquet"))
        else:
            mem = load_members(PRED / name, b, members)
            assert len(mem) == len(members)
            parts.append(ensemble(list(mem.values())))
    return pd.concat(parts, ignore_index=True).sort_values("sample").reset_index(drop=True)


def std(data, pred: pd.DataFrame, mix: np.ndarray) -> dict:
    smp = data.samples.iloc[pred["sample"].to_numpy()]
    rows, dates = smp["row"].to_numpy(), smp["date"].to_numpy()
    blk = block_of(dates)
    out = {}
    for b in [None, *BLOCKS]:
        sel = np.ones(len(rows), bool) if b is None else blk == b
        ics, ics_mix, qs = [], [], []
        for h in H:
            y = np.asarray(data.arr[f"y_{h}"])[rows].astype(float)
            if h in (5, 10):
                ics.append(daily_ic(dates[sel], pred[f"mu_{h}"].to_numpy()[sel], y[sel]).mean())
                ics_mix.append(daily_ic(dates[sel], mix[sel], y[sel]).mean())
            v = np.asarray(data.arr[f"v_{h}"])[rows].astype(float)
            m = smp[f"mask_sigma_{h}"].to_numpy() & sel
            qs.append(qlike(v[m], pred[f"sigma_{h}"].to_numpy()[m] / np.sqrt(h)).mean())
        out["total" if b is None else b] = {"IC": float(np.mean(ics)),
                                            "IC mezcla": float(np.mean(ics_mix)),
                                            "QLIKE": float(np.mean(qs))}
    return out


def evaluate() -> dict:
    data = WindowData(DATA_DIR / "dataset")
    runs = {v: load(f"rec_{v}") for v in VARIANTS}
    runs["final20 (0–9)"] = load("final20", list(range(10)))
    runs["final20 (20)"] = load("final20")
    samples = runs["P0"]["sample"].to_numpy()
    for k, p in runs.items():
        if not np.array_equal(p["sample"].to_numpy(), samples):
            raise SystemExit(f"{k}: muestras distintas de P0")
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    equity = (info["asset_class"] == "equity").to_numpy()
    smp = data.samples.iloc[samples]
    dates = smp["date"]
    wk = weekly_dates(dates[dates.dt.dayofweek < 5])
    n_eq = smp[equity[smp["asset"].to_numpy()]].groupby("date").size()
    wk = wk[wk.isin(n_eq[n_eq >= 20].index)]
    prices = OpenPrices.from_clean(DATA_DIR / "clean",
                                   [data.meta["symbols"][a] for a in np.flatnonzero(equity)])
    wf = weekly_frame(data, samples, equity, wk, prices)
    pos = np.searchsorted(samples, wf["sample"].to_numpy())
    rows_w = data.samples["row"].to_numpy()[wf["sample"].to_numpy()]
    bw = baseline_features(data, rows_w)
    mix_w = baseline_mix(wf["date"].to_numpy(), bw["reversal"], bw["momentum"])
    ba = baseline_features(data, smp["row"].to_numpy())
    mix_all = baseline_mix(smp["date"].to_numpy(), ba["reversal"], ba["momentum"])

    def series(mu):
        s = decile_series(wf, mu)
        s["dif"] = s["larga"] - s["corta"]
        s["top20"] = tail_series(wf, mu)["top20_oo"]
        return s

    ser = {k: series(p["mu_5"].to_numpy()[pos]) for k, p in runs.items()}
    ser_mix = series(mix_w)

    def pr(a, b):
        x = paired(a, b)
        return {"mean": x["mean"], "t": x["t"], "n": x["n"],
                "blocks": {k: x["blocks"][k] for k in BLOCKS}}

    def lv(a):
        x = level(a)
        return {"mean": x["mean"], "blocks": {k: x["blocks"][k] for k in BLOCKS}}

    res = {"rule": RULE, "weeks": int(ser["P0"]["larga"].notna().sum()), "runs": {}}
    for k, s in ser.items():
        r = {c: {"level": lv(s[c])} for c in ("larga", "corta", "dif", "top20")}
        if k != "P0":
            for c in ("larga", "corta", "top20"):
                r[c]["vs P0"] = pr(s[c], ser["P0"][c])
        r["larga"]["vs mezcla"] = pr(s["larga"], ser_mix["larga"])
        r["dif"]["vs mezcla"] = pr(s["dif"], ser_mix["dif"])
        r["std"] = std(data, runs[k], mix_all)
        res["runs"][k] = r
    res["mezcla"] = {c: lv(ser_mix[c]) for c in ("larga", "corta", "dif", "top20")}
    for k in ("final20 (0–9)", "final20 (20)"):
        res["runs"]["P0"].setdefault("vs ref", {})[k] = {
            c: pr(ser["P0"][c], ser[k][c]) for c in ("larga", "corta", "top20")}
    # regla
    q0 = res["runs"]["P0"]["std"]["total"]["QLIKE"]
    verdicts = {}
    for v in ("P4", "P2"):
        d = res["runs"][v]["larga"]["vs P0"]
        q = res["runs"][v]["std"]["total"]["QLIKE"] / q0 - 1
        c = {"t": bool(d["mean"] > 0 and d["t"] >= RULE["t_min"]),
             "bloques": bool(all(d["blocks"][b]["mean"] > 0 for b in BLOCKS)),
             "qlike": bool(q <= RULE["qlike_worse_max"])}
        verdicts[v] = {"conditions": c, "works": all(c.values()), "mejora": d["mean"],
                       "qlike_rel": q}
    ok = [v for v in verdicts if verdicts[v]["works"]]
    best = max(ok, key=lambda v: verdicts[v]["mejora"]) if ok else None
    res["verdict"] = {"variants": verdicts, "best": best,
                      "lectura": ("el problema es de adaptación" if best else
                                  "ninguna funciona: la ventaja de 2000–11 dependía de patrones "
                                  "que ya no están, o del sesgo; ningún ajuste del entrenamiento "
                                  "la recupera")}
    return res


def f(x, nd=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ft(x):
    return f"{f(x['mean'])} ({f(x['t'], 2, False)})"


def markdown(r: dict) -> list[str]:
    v = r["verdict"]
    L = ["# Adaptación: pesos por antigüedad (2012–19, solo con el pasado; 22-11-2026)", "",
         ("Ver el docstring de scripts/mu_recency.py. Exceso semanal de apertura a apertura sobre "
          f"la media de las acciones, pb por semana; {r['weeks']} semanas; diferencias por pares "
          "con t de NW (2 retardos)."), ""]
    for k in ("P4", "P2"):
        x = v["variants"][k]
        c = x["conditions"]
        L.append(f"- {k}: {'**funciona**' if x['works'] else 'no funciona'} (larga − P0 "
                 f"{f(x['mejora'])} pb, t ≥ 2: {'sí' if c['t'] else 'no'}; positiva en los dos "
                 f"bloques: {'sí' if c['bloques'] else 'no'}; QLIKE {f(100 * x['qlike_rel'], 2)} "
                 f"%: {'sí' if c['qlike'] else 'no'}).")
    L += ["", f"**Lectura: {v['lectura']}" + (f" (se queda {v['best']})" if v["best"] else "")
          + ".**", "",
          "## Frente a P0", "",
          "| variante | bloque | larga − P0 (t) | top 20 − P0 (t) | corta − P0 (t) | IC | QLIKE |",
          "|---|---|---|---|---|---|---|"]
    for k in ("P0", "P4", "P2", "final20 (0–9)", "final20 (20)"):
        x = r["runs"][k]
        for b, lab in [("total", "2012–19"), *BLOCKS.items()]:
            def g(c, x=x, b=b, k=k):
                if k == "P0":
                    return "—"
                d = x[c]["vs P0"]
                return ft(d if b == "total" else d["blocks"][b])
            s = x["std"][b]
            L.append(f"| {k} | {lab} | {g('larga')} | {g('top20')} | {g('corta')} | "
                     f"{s['IC']:.4f} | {s['QLIKE']:.4f} |")
    L += ["", "## Niveles y frente a la mezcla", "",
          ("| ejecución | bloque | larga | corta | larga − corta | top 20 | larga − mezcla (t) | "
           "(larga − corta) − mezcla (t) | IC mezcla |"), "|---|---|---|---|---|---|---|---|---|"]
    for k, x in r["runs"].items():
        for b, lab in [("total", "2012–19"), *BLOCKS.items()]:
            def lvl(c, x=x, b=b):
                y = x[c]["level"]
                return f(y["mean"] if b == "total" else y["blocks"][b])

            def vm(c, x=x, b=b):
                d = x[c]["vs mezcla"]
                return ft(d if b == "total" else d["blocks"][b])
            L.append(f"| {k} | {lab} | {lvl('larga')} | {lvl('corta')} | {lvl('dif')} | "
                     f"{lvl('top20')} | {vm('larga')} | {vm('dif')} | "
                     f"{x['std'][b]['IC mezcla']:.4f} |")
    mz = r["mezcla"]
    L.append(f"| mezcla | 2012–19 | {f(mz['larga']['mean'])} | {f(mz['corta']['mean'])} | "
             f"{f(mz['dif']['mean'])} | {f(mz['top20']['mean'])} | — | — | — |")
    L += ["", "## P0 frente a final20 (referencia)", "",
          "| referencia | larga − ref. (t) | corta − ref. (t) | top 20 − ref. (t) |",
          "|---|---|---|---|"]
    for k, x in r["runs"]["P0"]["vs ref"].items():
        L.append(f"| {k} | {ft(x['larga'])} | {ft(x['corta'])} | {ft(x['top20'])} |")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    par = json.loads((OUT / "parallel.json").read_text())
    if not args.eval_only:
        log("Inicio de la prueba de adaptación")
        if not train_all(par["workers"], par["threads"]):
            log("FALLO: entrenamiento incompleto")
            sys.exit(1)
    log("Evaluación")
    res = evaluate()
    (OUT / "recency.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                 default=float))
    lines = markdown(res)
    (OUT / "recency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    log("Fin.")


if __name__ == "__main__":
    main()
