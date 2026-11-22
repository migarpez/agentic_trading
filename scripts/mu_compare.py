"""Compara una ejecución de mu_tree con su base y aplica la regla de la fase 1 (logs/mu_tree/
README.md; model/mu_metrics.py). Solo lee predicciones de 2000–2019 (fuera de muestra); el test
no se toca.

Ejecuciones: <run> (el ensemble de data/predictions/<run>/cv*.parquet) o <run>:0,1 (el ensemble
de esos miembros, de <bloque>_members.parquet; p. ej. final20:0,1, la base R0).

Escribe logs/mu_tree/<paso>.json y .md: top 20 (apertura a apertura; la principal), de cierre a
cierre y decil alto, por bloque y en total, con la diferencia por pares y su t de NW (2
retardos); IC, QLIKE y diferencia con la mezcla; las cuatro condiciones y el veredicto. Con
--full, además success_checks de data/predictions/<run>/metrics.json (evaluate.py) de las dos.

Modo agente (--agent; pasos F2 y F3): lee logs/agent/<nombre>_eval2.json (agent_eval.py) de
--names y de --ref y da, por periodo, cada ejecución, su regla fija y los pesos iguales, y la
diferencia por pares (mismas semillas de muestreo) con su referencia de la misma posición.

Uso: python scripts/mu_compare.py --step E1 --run mu_E1 --base final20:0,1
     python scripts/mu_compare.py --step F1 --run mu_final20 --base final20 --full
     python scripts/mu_compare.py --step F2 --agent --names mf_real_s0 ... --ref w_real_s0 ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.model.compare import load_members
from agentic_trading.model.metrics import success_checks
from agentic_trading.model.mu_metrics import (
    BLOCK_NAMES,
    RULE,
    OpenPrices,
    StdContext,
    compare_rule,
    level,
    paired,
    std_metrics,
    tail_series,
    weekly_frame,
)
from agentic_trading.model.train import ensemble

OUT = ROOT / "logs" / "mu_tree"
BLOCK_LABEL = dict(zip(BLOCK_NAMES, ["2000–03", "2004–07", "2008–11", "2012–15", "2016–19"]))


def parse_spec(spec: str) -> tuple[str, list[int] | None]:
    run, _, mem = spec.partition(":")
    return run, ([int(x) for x in mem.split(",")] if mem else None)


def load_run(spec: str) -> pd.DataFrame:
    """Ensemble fuera de muestra de los 5 bloques, ordenado por muestra."""
    run, members = parse_spec(spec)
    pdir = DATA_DIR / "predictions" / run
    parts = []
    for b in BLOCK_NAMES:
        if members is None:
            path = pdir / f"{b}.parquet"
            if not path.exists():
                raise SystemExit(f"falta {path}")
            parts.append(pd.read_parquet(path))
        else:
            mem = load_members(pdir, b, members)
            if len(mem) != len(members):
                raise SystemExit(f"{run} {b}: {len(mem)} miembros de {len(members)}")
            parts.append(ensemble(list(mem.values())))
    return pd.concat(parts, ignore_index=True).sort_values("sample").reset_index(drop=True)


def fmt(x, nd=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def compare(step: str, run: str, base: str, full: bool) -> dict:
    data = WindowData(DATA_DIR / "dataset")
    pr, pb = load_run(run), load_run(base)
    if not np.array_equal(pr["sample"].to_numpy(), pb["sample"].to_numpy()):
        raise SystemExit("las dos ejecuciones no tienen las mismas muestras")
    samples = pr["sample"].to_numpy()
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    equity = (info["asset_class"] == "equity").to_numpy()
    smp = data.samples.iloc[samples]
    dates = smp["date"]
    wk = weekly_dates(dates[dates.dt.dayofweek < 5])
    n_eq = smp[equity[smp["asset"].to_numpy()]].groupby("date").size()
    wk = wk[wk.isin(n_eq[n_eq >= 20].index)]
    syms = [data.meta["symbols"][a] for a in np.flatnonzero(equity)]
    prices = OpenPrices.from_clean(DATA_DIR / "clean", syms)
    wf = weekly_frame(data, samples, equity, wk, prices)
    pos = np.searchsorted(samples, wf["sample"].to_numpy())
    tr = tail_series(wf, pr["mu_5"].to_numpy()[pos])
    tb = tail_series(wf, pb["mu_5"].to_numpy()[pos])
    ctx = StdContext(data, samples)
    sr, sb = std_metrics(ctx, pr), std_metrics(ctx, pb)
    res = {"step": step, "run": run, "base": base, "rule": RULE,
           "weeks": int(tr["top20_oo"].notna().sum()),
           "tails": {k: {"diff": paired(tr[k], tb[k]), "run": level(tr[k]), "base": level(tb[k])}
                     for k in ("top20_oo", "top20_cc", "dec_oo")},
           "std": {"run": sr, "base": sb}}
    res["verdict"] = compare_rule(res["tails"]["top20_oo"]["diff"], sr, sb)
    if full:
        res["success_checks"] = {}
        for label, spec in (("run", run), ("base", base)):
            p = DATA_DIR / "predictions" / parse_spec(spec)[0] / "metrics.json"
            if p.exists() and parse_spec(spec)[1] is None:
                rep = json.loads(p.read_text())["total"]
                rep = {int(k) if k.isdigit() else k: v for k, v in rep.items()}
                res["success_checks"][label] = success_checks(rep)
    return res


def _yes(x) -> str:
    return "sí" if x else "no"


def _diff(t: dict, k: str, b: str) -> str:
    x = t[k]["diff"] if b == "total" else t[k]["diff"]["blocks"][b]
    return f"{fmt(x['mean'])} ({fmt(x['t'], 2, False)})"


def _level(t: dict, k: str, who: str, b: str):
    x = t[k][who]
    return x["mean"] if b == "total" else x["blocks"][b]


def markdown(r: dict) -> list[str]:
    v, t = r["verdict"], r["tails"]
    c = v["conditions"]
    verdict = (f"Veredicto: **{'gana' if v['wins'] else 'no gana'}** "
               f"(1 top 20 total: {_yes(c['1_top20_total'])} · 2 bloques: "
               f"{_yes(c['2_top20_blocks'])} ({v['n_blocks_positive']} de 5 positivos; "
               f"2016–19 {fmt(v['top20_last_block'])} pb) · 3 IC: {_yes(c['3_ic'])} "
               f"({fmt(v['ic_diff'], 4)}) · 4 QLIKE: {_yes(c['4_qlike'])} "
               f"({fmt(100 * v['qlike_rel'], 2)} %)).")
    head = ("| bloque | top 20 dif. (t) | top 20 ejec. / base | cierre a cierre dif. (t) | "
            "decil alto dif. (t) | IC ejec. / base | QLIKE ejec. / base | dif. mezcla h=5 (t) / "
            "h=10 (t), ejec. |")
    L = [f"# {r['step']}: {r['run']} frente a {r['base']}", "", verdict, "",
         (f"{r['weeks']} semanas. Colas en pb por semana sobre la media de las acciones; "
          "diferencia por pares con t de NW (2 retardos)."), "",
         head, "|---|---|---|---|---|---|---|---|"]
    for b in [*BLOCK_NAMES, "total"]:
        sr, sb = r["std"]["run"][b], r["std"]["base"][b]
        L.append(f"| {BLOCK_LABEL.get(b, 'total')} | {_diff(t, 'top20_oo', b)} | "
                 f"{fmt(_level(t, 'top20_oo', 'run', b))} / "
                 f"{fmt(_level(t, 'top20_oo', 'base', b))} | "
                 f"{_diff(t, 'top20_cc', b)} | {_diff(t, 'dec_oo', b)} | "
                 f"{sr['IC']:.4f} / {sb['IC']:.4f} | "
                 f"{sr['QLIKE']:.4f} / {sb['QLIKE']:.4f} | {fmt(sr['mix_diff_5'], 4)} "
                 f"({fmt(sr['mix_t_5'], 2, False)}) / {fmt(sr['mix_diff_10'], 4)} "
                 f"({fmt(sr['mix_t_10'], 2, False)}) |")
    if "success_checks" in r:
        L += ["", "success_checks (orientativo en desarrollo):", ""]
        for who, sc in r["success_checks"].items():
            mu = sc["μ"]["por horizonte"]
            hs = "; ".join(f"h={h}: dif. {fmt(x['dif.'], 4)}, t {fmt(x['t NW'], 2, False)}"
                           for h, x in mu.items())
            L.append(f"- {who}: μ {_yes(sc['μ']['ok'])} ({hs}) · σ {_yes(sc['σ']['ok'])}")
    return L


# ----------------------------------------------------------------------------- agente

def agent_compare(step: str, names: list[str], refs: list[str]) -> tuple[dict, list[str]]:
    def load(n):
        return json.loads((ROOT / "logs" / "agent" / f"{n}_eval2.json").read_text())
    R, F = [load(n) for n in names], [load(n) for n in refs]
    out = {"step": step, "names": names, "refs": refs, "periods": {}}
    L = [f"# {step}: agente", "",
         ("Política estocástica, 20 semillas de muestreo (media ± error típico); diferencia por "
          "pares con la referencia de la misma posición (mismas semillas), con su t."), ""]
    for per in R[0]["periods"]:
        rows = []
        head = ("| ejecución | pasos | rentab. | Sharpe | caída | rotación | "
                "− ref.: rentab. (t) | − ref.: Sharpe (t) | regla fija (sus pred.) | "
                "regla fija (ref.) | pesos iguales |")
        L += [f"## {per}", "", head, "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r, f in zip(R, F):
            p, q = r["periods"][per], f["periods"][per]
            diff = {}
            for k in ("rentabilidad anual", "Sharpe"):
                d = np.array([a[k] - b[k] for a, b in zip(p["learned_seeds"], q["learned_seeds"])])
                se = d.std(ddof=1) / np.sqrt(len(d))
                diff[k] = (float(d.mean()), float(se), float(d.mean() / se) if se > 0 else np.nan)
            ln = p["learned"]
            rows.append({"name": r["name"], "ref": f["name"], "learned": ln, "diff": diff,
                         "ref_learned": q["learned"], "rule": p["rule"], "rule_ref": q["rule"],
                         "ew_gross": p["ew_gross"]})
            L.append(f"| {r['name']} (ref. {f['name']}) | {r['steps']:,} | "
                     f"{100 * ln['rentabilidad anual'][0]:+.1f} % ± "
                     f"{100 * ln['rentabilidad anual'][1]:.1f} | {ln['Sharpe'][0]:.2f} | "
                     f"{100 * ln['caída máxima'][0]:.0f} % | {ln['rotación anual'][0]:.0f} | "
                     f"{100 * diff['rentabilidad anual'][0]:+.1f} "
                     f"({diff['rentabilidad anual'][2]:.1f}) | {diff['Sharpe'][0]:+.2f} "
                     f"({diff['Sharpe'][2]:.1f}) | "
                     f"{100 * p['rule']['rentabilidad anual']:+.1f} % / "
                     f"{p['rule']['Sharpe']:.2f} | "
                     f"{100 * q['rule']['rentabilidad anual']:+.1f} % / "
                     f"{q['rule']['Sharpe']:.2f} | {100 * p['ew_gross']:+.1f} % |")
        mr = np.mean([x["diff"]["rentabilidad anual"][0] for x in rows])
        ms = np.mean([x["diff"]["Sharpe"][0] for x in rows])
        L += ["", (f"Media de las {len(rows)} diferencias: rentab. {100 * mr:+.1f} puntos, "
                   f"Sharpe {ms:+.2f}."), ""]
        out["periods"][per] = {"rows": rows, "mean_diff_return": mr, "mean_diff_sharpe": ms}
    return out, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--run")
    ap.add_argument("--base")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--ref", nargs="+")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = ROOT / args.out_dir          # una ruta absoluta queda como está
    out.mkdir(parents=True, exist_ok=True)
    if args.agent:
        if not args.names or not args.ref or len(args.names) != len(args.ref):
            ap.error("--agent necesita --names y --ref de la misma longitud")
        res, lines = agent_compare(args.step, args.names, args.ref)
    else:
        if not args.run or not args.base:
            ap.error("hacen falta --run y --base")
        res = compare(args.step, args.run, args.base, args.full)
        lines = markdown(res)
    (out / f"{args.step}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                      default=float))
    (out / f"{args.step}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
