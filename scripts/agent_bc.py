"""El agente parte de la regla fija (23-11-2026; logs/agent/bc.md): clonado + crítico + PPO.

Universo actual, final20, configuración del plan B (semanal, γ = 0,8, episodios de 52, recompensa
relativa, ratio por componente, c = 0,1 %; la de w_real1000). El test no se abre.

1. clone: acciones de la regla fija del entorno en 2000–15 (agent/bc.py: episodios de 52 desde
   liquidez, uno cada 4 decisiones, más uno largo de 2000 a 2015), preentreno por máxima
   verosimilitud con las distribuciones de la política (pesos, mantener/operar y d) y 2 rondas de
   DAgger (estados que visita la política clonada, etiquetados con la regla). Comprobación: la
   clonada, determinista y estocástica (20 semillas), frente a la regla en 2000–15 y 2016–19:
   rentabilidad a ±2 puntos al año y rotación a ±25 % (la determinista y la media de la
   estocástica). Si no pasa, se repite con el rango de μ_5 del día como entrada
   (build_panel(rank=True)). Si tampoco, se para.
2. ppo: 3 semillas × 1.000 actualizaciones desde la clonada (scripts/train_agent.py --init), con el
   crítico ajustado antes (--critic-warmup 30) y la penalización KL hacia la clonada (coeficiente
   inicial KL_COEF, que baja hasta 0 en el primer 30 %). Checkpoints cada 100.
3. curve: en cada checkpoint (0, 100, …, 1.000), la política estocástica con 5 semillas en
   2000–15; se elige el de mayor Sharpe medio en 2000–15 (2016–19 no se mira).
4. eval: agent_eval.py con 20 semillas en el checkpoint elegido (inicial = la clonada), y el
   informe: por pares frente a la clonada, la regla fija, los pesos iguales y w_real1000.

Regla, fijada antes de ver resultados: va al test el agente (clonado + PPO) si en 2016–19, en la
media de las 3 semillas, iguala o supera a la regla fija en rentabilidad y en Sharpe, y no queda por
debajo de ella en 2012–15. Si no, va la regla fija, o la clonada si es mejor que la regla, y el
agente se informa como secundario.

Uso: python scripts/agent_bc.py clone | ppo | curve | eval | all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.agent.bc import collect_clone, collect_rule, train_bc
from agentic_trading.agent.data import TRAIN_END, build_panel
from agentic_trading.agent.evaluation import policy_runs, run_rule, summarize
from agentic_trading.agent.policy import AgentNet
from agentic_trading.agent.ppo import panel_n_feat
from agentic_trading.agent.simulator import agent_rules
from agentic_trading.config import DATA_DIR, ROOT

OUT = ROOT / "logs" / "agent"
AG = DATA_DIR / "agent"
COST, EPISODE, K = 0.001, 52, 20
SEEDS = (0, 1, 2)
UPDATES, KEEP, KL_COEF, KL_FRAC, CRITIC_WARMUP = 1000, 100, 1.0, 0.3, 30
CHECK = {"ret": 0.02, "rot": 0.25}
CHECK_PERIODS = (("2000–15", "2000-01-01", "2015-12-31"), ("2016–19", "2016-01-01", "2019-12-31"))
PLANB = ["--freq", "weekly", "--gamma", "0.8", "--reward", "relative", "--ratio", "component",
         "--trade-bias", "-2"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def clone_name(rank: bool) -> str:
    return "bc_clone_rank" if rank else "bc_clone"


def check_clone(panel, net, rules, periods=CHECK_PERIODS) -> dict:
    out = {"pass": True, "periods": {}}
    for label, a, b in periods:
        rule = run_rule(panel, a, b, rules)
        runs = policy_runs(panel, net, a, b, rules, K, range(1000, 1020))
        st = summarize(runs["stochastic"])
        det = runs["deterministic"]
        r = {"rule": rule, "det": det, "stoch": st}
        ok = True
        for name, ret, rot in (("det", det["rentabilidad anual"], det["rotación anual"]),
                               ("stoch", st["rentabilidad anual"][0], st["rotación anual"][0])):
            d_ret = ret - rule["rentabilidad anual"]
            d_rot = rot / rule["rotación anual"] - 1
            r[f"{name}_ok"] = bool(abs(d_ret) <= CHECK["ret"] and abs(d_rot) <= CHECK["rot"])
            ok &= r[f"{name}_ok"]
            log(f"  {label} {name}: rentab. {100 * ret:+.1f} % (regla "
                f"{100 * rule['rentabilidad anual']:+.1f} %), rotación {rot:.1f} (regla "
                f"{rule['rotación anual']:.1f}) → {'OK' if r[f'{name}_ok'] else 'NO'}")
        out["periods"][label] = r
        out["pass"] &= ok
    return out


def clone(rank: bool, train_end=TRAIN_END, name: str | None = None,
          periods=CHECK_PERIODS) -> dict:
    """Clonado y comprobación. train_end, name y periods: para el agente final
    (scripts/agent_final.py: 2000–2019, final_clone_rank)."""
    torch.set_num_threads(4)
    panel = build_panel("final20", freq="weekly", rank=rank, train_end=train_end)
    rules = agent_rules(COST)
    k_lo, k_hi = panel.k_range(panel.dec_dates[0], panel.train_end)
    k_hi = k_hi - 1 - EPISODE
    torch.manual_seed(0)
    net = AgentNet(n_feat=panel_n_feat(panel), trade_bias=-2.0)
    data = collect_rule(panel, rules, k_lo, k_hi, EPISODE)
    log(f"clonado (rank={rank}): {len(data)} estados de la regla")
    hist = train_bc(net, data, epochs=30, log=log)
    for rnd in range(2):
        net.eval()
        more = collect_clone(panel, net, rules, k_lo, k_hi, EPISODE, seed=rnd)
        data += more
        log(f"DAgger {rnd + 1}: +{len(more)} estados ({len(data)})")
        net.train()
        hist += train_bc(net, data, epochs=10, seed=rnd + 1, log=log)
    net.eval()
    name = name or clone_name(rank)
    d = AG / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "rank": rank, "history": hist}, d / "clone.pt")
    log("comprobación de la clonada")
    chk = check_clone(panel, net, rules, periods)
    chk["rank"], chk["n_states"] = rank, len(data)
    (OUT / f"{name}_check.json").write_text(json.dumps(chk, indent=1,
                                                                   ensure_ascii=False,
                                                                   default=float))
    return chk


def phase_clone() -> bool:
    for rank in (False, True):
        path = OUT / f"{clone_name(rank)}_check.json"
        chk = json.loads(path.read_text()) if path.exists() else clone(rank)
        log(f"clonada rank={rank}: {'pasa' if chk['pass'] else 'no pasa'}")
        if chk["pass"]:
            (AG / "bc_clone_chosen.json").write_text(json.dumps({"rank": rank}))
            return True
    return False


def chosen_rank() -> bool:
    return json.loads((AG / "bc_clone_chosen.json").read_text())["rank"]


def ppo_name(seed: int) -> str:
    return f"bc_ppo_s{seed}"


def train_one(seed: int, prefix: str = "bc_ppo", clone_dir: str | None = None,
              rank: bool | None = None, train_end: str | None = None) -> int:
    rank = chosen_rank() if rank is None else rank
    name = f"{prefix}_s{seed}"
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "train_agent.py"),
           "--name", name, "--seed", str(seed), "--updates", str(UPDATES),
           "--save-every", "20", "--keep-every", str(KEEP), *PLANB,
           "--init", str(AG / (clone_dir or clone_name(rank)) / "clone.pt"), "--kl-coef",
           str(KL_COEF), "--kl-frac", str(KL_FRAC), "--critic-warmup", str(CRITIC_WARMUP)]
    if rank:
        cmd.append("--rank")
    if train_end:
        cmd += ["--train-end", train_end]
    with open(OUT / f"{name}.log", "a") as fh:
        return subprocess.run(cmd, check=False, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def phase_ppo() -> bool:
    with ProcessPoolExecutor(len(SEEDS)) as ex:
        codes = list(ex.map(train_one, SEEDS))
    log(f"PPO: códigos de salida {codes}")
    return all(c == 0 for c in codes)


CURVE_SEEDS = range(500, 505)


def curve_one(seed: int, prefix: str = "bc_ppo", start: str = "2000-01-01",
              end: str = "2015-12-31") -> dict:
    """Curva de evaluación en el entrenamiento (2000–15; el agente final, 2000–19) de una
    semilla de PPO: Sharpe y rentabilidad medios de la estocástica (5 semillas de muestreo) en
    cada checkpoint."""
    from scripts.agent_eval import nets, panel_for
    torch.set_num_threads(1)
    d = AG / f"{prefix}_s{seed}"
    fp = json.loads((d / "fingerprint.json").read_text())
    panel = panel_for(fp)
    rules = agent_rules(COST)
    out = {}
    for u in range(0, UPDATES + 1, KEEP):
        net, _ = nets(fp["cfg"], d / f"ckpt_u{u}.pt", panel_n_feat(panel))
        runs = policy_runs(panel, net, start, end, rules, K, CURVE_SEEDS)
        st = summarize(runs["stochastic"])
        out[u] = {"Sharpe": st["Sharpe"][0], "rentabilidad anual": st["rentabilidad anual"][0],
                  "rotación anual": st["rotación anual"][0]}
        log(f"curva s{seed} u{u}: Sharpe {out[u]['Sharpe']:.2f} · rentab. "
            f"{100 * out[u]['rentabilidad anual']:+.1f} %")
    best = max(out, key=lambda u: out[u]["Sharpe"])
    import shutil
    shutil.copy(d / f"ckpt_u{best}.pt", d / "ckpt_sel.pt")
    return {"seed": seed, "curve": out, "best": best}


def phase_curve() -> None:
    with ProcessPoolExecutor(len(SEEDS)) as ex:
        res = list(ex.map(curve_one, SEEDS))
    (OUT / "bc_curve.json").write_text(json.dumps(res, indent=1, default=float))


def phase_eval() -> None:
    names = [ppo_name(s) for s in SEEDS]
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "agent_eval.py"), "--names",
                    *names, "--seeds", "20", "--ckpt", "ckpt_sel.pt"], check=True, cwd=ROOT)
    report()


def pair(a: np.ndarray, b) -> tuple[float, float]:
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return float(d.mean()), float(d.mean() / se) if se > 0 else float("nan")


def report() -> None:
    ev = {s: json.loads((OUT / f"{ppo_name(s)}_eval2.json").read_text()) for s in SEEDS}
    wr = {s: json.loads((OUT / f"w_real1000_s{s}_eval2.json").read_text()) for s in SEEDS}
    curve = json.loads((OUT / "bc_curve.json").read_text())
    chk_rank = chosen_rank()
    chk = json.loads((OUT / f"{clone_name(chk_rank)}_check.json").read_text())
    keys = ("rentabilidad anual", "Sharpe", "caída máxima", "rotación anual")
    res = {"periods": {}}
    for label in ("2012–15", "2016–19"):
        P = {}
        for s in SEEDS:
            p = ev[s]["periods"][label]
            ag = {k: np.array([r[k] for r in p["learned_seeds"]]) for k in keys}
            cl = {k: np.array([r[k] for r in p["init_seeds"]]) for k in keys}
            w = {k: np.array([r[k] for r in wr[s]["periods"][label]["learned_seeds"]])
                 for k in keys}
            P[s] = {"agent": ag, "clone": cl, "w_real1000": w, "rule": p["rule"],
                    "ew": p["ew_gross"], "det": p["learned_det"], "clone_det": p["init_det"]}
        res["periods"][label] = P
    L = ["# El agente parte de la regla fija: clonado + crítico + PPO (23-11-2026)", "",
         ("Ver el docstring de scripts/agent_bc.py. Universo actual, final20, plan B (semanal, "
          f"γ = 0,8, episodios de 52, recompensa relativa, ratio por componente, c = 0,1 %). "
          f"Clonada {'con' if chk_rank else 'sin'} el rango de μ_5 como entrada; PPO, 3 "
          f"semillas × {UPDATES} actualizaciones, KL inicial {KL_COEF} hasta 0 en el primer "
          f"{int(100 * KL_FRAC)} %, crítico ajustado {CRITIC_WARMUP} iteraciones antes; "
          "checkpoint elegido por el Sharpe estocástico en 2000–15 (5 semillas). Evaluación: "
          "20 semillas de muestreo; coste pagado al año = 2 × rotación × c."), ""]
    # lectura
    def mean_over(label, who, k):
        return float(np.mean([res["periods"][label][s][who][k].mean() for s in SEEDS]))
    rule = {lab: ev[0]["periods"][lab]["rule"] for lab in ("2012–15", "2016–19")}
    ag_ret = {lab: mean_over(lab, "agent", "rentabilidad anual") for lab in rule}
    ag_sh = {lab: mean_over(lab, "agent", "Sharpe") for lab in rule}
    cl_ret = {lab: mean_over(lab, "clone", "rentabilidad anual") for lab in rule}
    cl_sh = {lab: mean_over(lab, "clone", "Sharpe") for lab in rule}
    goes = (ag_ret["2016–19"] >= rule["2016–19"]["rentabilidad anual"] and
            ag_sh["2016–19"] >= rule["2016–19"]["Sharpe"] and
            ag_ret["2012–15"] >= rule["2012–15"]["rentabilidad anual"] and
            ag_sh["2012–15"] >= rule["2012–15"]["Sharpe"])
    clone_better = (cl_ret["2016–19"] > rule["2016–19"]["rentabilidad anual"] and
                    cl_sh["2016–19"] > rule["2016–19"]["Sharpe"])
    pick = "el agente (clonado + PPO)" if goes else (
        "la política clonada" if clone_better else "la regla fija")
    L += ["## Lectura (regla fijada antes de verlo)", "",
          (f"- 2016–19, media de las 3 semillas: agente {100 * ag_ret['2016–19']:+.1f} % / "
           f"Sharpe {ag_sh['2016–19']:.2f}; regla fija "
           f"{100 * rule['2016–19']['rentabilidad anual']:+.1f} % / "
           f"{rule['2016–19']['Sharpe']:.2f}; clonada {100 * cl_ret['2016–19']:+.1f} % / "
           f"{cl_sh['2016–19']:.2f}."),
          (f"- 2012–15: agente {100 * ag_ret['2012–15']:+.1f} % / {ag_sh['2012–15']:.2f}; regla "
           f"{100 * rule['2012–15']['rentabilidad anual']:+.1f} % / "
           f"{rule['2012–15']['Sharpe']:.2f}."),
          f"- Según la regla, iría al test **{pick}** (no adoptado)."
          + ("" if goes else " El agente se informa como secundario."), "",
          "## Comprobación de la clonada (frente a la regla)", "",
          "| periodo | regla: rentab. / rotación | determinista | estocástica (20) | pasa |",
          "|---|---|---|---|---|"]
    for lab, r in chk["periods"].items():
        L.append(f"| {lab} | {100 * r['rule']['rentabilidad anual']:+.1f} % / "
                 f"{r['rule']['rotación anual']:.1f} | {100 * r['det']['rentabilidad anual']:+.1f}"
                 f" % / {r['det']['rotación anual']:.1f} | "
                 f"{100 * r['stoch']['rentabilidad anual'][0]:+.1f} % / "
                 f"{r['stoch']['rotación anual'][0]:.1f} | "
                 f"{'sí' if r['det_ok'] and r['stoch_ok'] else 'no'} |")
    L += ["", "## Curva en 2000–15 (Sharpe estocástico, 5 semillas) y checkpoint elegido", "",
          "| semilla | " + " | ".join(str(u) for u in range(0, UPDATES + 1, KEEP)) +
          " | elegido |", "|---|" + "---|" * (UPDATES // KEEP + 2)]
    for c in curve:
        L.append(f"| {c['seed']} | " + " | ".join(
            f"{c['curve'][str(u)]['Sharpe']:.2f}" for u in range(0, UPDATES + 1, KEEP))
            + f" | {c['best']} |")
    for lab in ("2012–15", "2016–19"):
        L += ["", f"## {lab}", "",
              ("Rentabilidad / Sharpe / caída / rotación / coste pagado al año (media de 20 "
               "semillas de muestreo)."), "",
              "| cartera | semilla | rentab. | Sharpe | caída | rotación | coste |",
              "|---|---|---|---|---|---|---|"]
        r0 = rule[lab]
        L.append(f"| regla fija | — | {100 * r0['rentabilidad anual']:+.1f} % | "
                 f"{r0['Sharpe']:.2f} | {100 * r0['caída máxima']:.0f} % | "
                 f"{r0['rotación anual']:.1f} | {100 * 2 * r0['rotación anual'] * COST:.2f} % |")
        L.append(f"| pesos iguales (bruto) | — | {100 * ev[0]['periods'][lab]['ew_gross']:+.1f} % "
                 "| — | — | — | 0 |")
        for who in ("clone", "agent", "w_real1000"):
            for s in SEEDS:
                x = res["periods"][lab][s][who]
                L.append(f"| {who} | {s} | {100 * x['rentabilidad anual'].mean():+.1f} % | "
                         f"{x['Sharpe'].mean():.2f} | {100 * x['caída máxima'].mean():.0f} % | "
                         f"{x['rotación anual'].mean():.1f} | "
                         f"{100 * 2 * x['rotación anual'].mean() * COST:.2f} % |")
        for s in SEEDS:
            d = res["periods"][lab][s]["det"]
            L.append(f"| agente determinista | {s} | {100 * d['rentabilidad anual']:+.1f} % | "
                     f"{d['Sharpe']:.2f} | {100 * d['caída máxima']:.0f} % | "
                     f"{d['rotación anual']:.1f} | {100 * 2 * d['rotación anual'] * COST:.2f} % |")
        L += ["", (f"Por pares (agente − referencia, media de las semillas de muestreo; t entre "
              f"semillas), {lab}:"), "",
              ("| semilla | − clonada: rentab. (t) / Sharpe (t) | − regla: rentab. (t) / Sharpe (t) "
              "| − pesos iguales: rentab. (t) | − w_real1000: rentab. (t) / Sharpe (t) |"),
              "|---|---|---|---|---|"]
        for s in SEEDS:
            P = res["periods"][lab][s]
            a = P["agent"]
            c1 = pair(a["rentabilidad anual"], P["clone"]["rentabilidad anual"])
            c2 = pair(a["Sharpe"], P["clone"]["Sharpe"])
            r1 = pair(a["rentabilidad anual"], P["rule"]["rentabilidad anual"])
            r2 = pair(a["Sharpe"], P["rule"]["Sharpe"])
            e1 = pair(a["rentabilidad anual"], P["ew"])
            w1 = pair(a["rentabilidad anual"], P["w_real1000"]["rentabilidad anual"])
            w2 = pair(a["Sharpe"], P["w_real1000"]["Sharpe"])
            L.append(f"| {s} | {100 * c1[0]:+.1f} ({c1[1]:.1f}) / {c2[0]:+.2f} ({c2[1]:.1f}) | "
                     f"{100 * r1[0]:+.1f} ({r1[1]:.1f}) / {r2[0]:+.2f} ({r2[1]:.1f}) | "
                     f"{100 * e1[0]:+.1f} ({e1[1]:.1f}) | {100 * w1[0]:+.1f} ({w1[1]:.1f}) / "
                     f"{w2[0]:+.2f} ({w2[1]:.1f}) |")
    L += ["", "## Decisiones (opción conservadora ante la duda)", "",
          ("- Clonado: la regla rebalancea a 1/20 las que se quedan, así que todas las posiciones "
           "«operan»; puntuaciones κ·z(μ_5) con κ = 0,05 (pesos casi iguales) y − 10 para las que "
           "se venden; d / d_max = 0,98 (la beta no llega a 1); desviación típica del ruido de las "
           "puntuaciones fija en 0,1 durante el clonado (si se aprende, se va a 0 y PPO no "
           "explora). Dos rondas de DAgger."),
          ("- «Reproducir la regla»: rentabilidad a ±2 puntos al año y rotación a ±25 %, la "
           "determinista y la media de la estocástica, en 2000–15 y 2016–19."),
          (f"- KL: coeficiente inicial {KL_COEF} (el enunciado no lo fija; un ancla fuerte es lo "
           "conservador), KL analítica por componente (la del ratio por componente)."),
          ("- Dónde parar: el checkpoint de mayor Sharpe estocástico medio en 2000–15 (5 "
           "semillas), incluido el 0 (la clonada con el crítico ajustado); 2016–19 no se mira."),
          ("- La regla de selección pide igualar o superar a la regla fija en rentabilidad y "
           "Sharpe en 2016–19, y no quedar por debajo en ninguna de las dos en 2012–15."),
          ("- Las comparaciones con la regla fija y los pesos iguales usan su valor único "
           "(determinista); el t es entre las 20 semillas de muestreo del agente."),
          ("- w_real1000: las mismas semillas de muestreo (1000–1019) de su evaluación, "
           "logs/agent/w_real1000_s*_eval2.json.")]
    (OUT / "bc.md").write_text("\n".join(L) + "\n")
    log("informe escrito")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["clone", "ppo", "curve", "eval", "all"])
    args = ap.parse_args()
    if args.phase in ("clone", "all") and not phase_clone():
        log("la clonada no reproduce la regla: se para")
        return
    if args.phase in ("ppo", "all") and not phase_ppo():
        return
    if args.phase in ("curve", "all"):
        phase_curve()
    if args.phase in ("eval", "all"):
        phase_eval()


if __name__ == "__main__":
    main()
