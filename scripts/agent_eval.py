"""Evaluación de ejecuciones de train_agent.py con el protocolo de agent/evaluation.py (handoff
§10, 21-11-2026): política estocástica con --seeds semillas de muestreo frente a la política
inicial congelada (misma semilla de red, sin entrenar), la determinista, la regla fija (las 20 de
mayor μ_5) y los pesos iguales brutos. En 2012–15 (dentro del entrenamiento) y 2016–19
(validación). El test (2020–) no se toca.

El panel se reconstruye con los parámetros de data/agent/<name>/fingerprint.json. Escribe
logs/agent/<name>_eval2.json y, con --summary, una tabla de todas las ejecuciones dadas en ese
fichero markdown.

Uso: python scripts/agent_eval.py --names w_real_s0 w_real_s1 --seeds 20 --summary logs/agent/planb.md
"""

import argparse
import json
import time

import torch

from agentic_trading.agent.data import VALID_START, build_panel
from agentic_trading.agent.evaluation import (
    ew_annual,
    paired,
    policy_runs,
    run_rule,
    summarize,
)
from agentic_trading.agent.policy import AgentNet
from agentic_trading.agent.ppo import panel_n_feat
from agentic_trading.agent.simulator import agent_rules
from agentic_trading.config import DATA_DIR, ROOT

PERIODS = (("2012–15", "2012-01-01", "2015-12-31"), ("2016–19", str(VALID_START.date()),
                                                     "2019-12-31"))
_PANELS: dict = {}


def panel_for(fp: dict):
    p = fp["panel"]
    key = (p["run"], p.get("freq", "daily"), p.get("signal"), p.get("signal_seed", 0),
           p.get("prior", "mu5"), p.get("rank", False), p.get("train_end"))
    if key not in _PANELS:
        _PANELS[key] = build_panel(key[0], freq=key[1], signal=key[2], signal_seed=key[3],
                                   prior=key[4], rank=key[5],
                                   **({"train_end": key[6]} if key[6] else {}))
    return _PANELS[key]


def nets(cfg: dict, ckpt, n_feat: int = 11) -> tuple[AgentNet, AgentNet]:
    """Red entrenada y política inicial: la red con la misma semilla antes de entrenar o, si el
    checkpoint la trae ("init_net", arranque desde la política clonada), esa."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])            # igual que Trainer: la red inicial
    kw = {"n_feat": n_feat, "hidden": cfg["hidden"], "emb": cfg["emb"],
          "trade_bias": cfg["trade_bias"]}
    init, net = AgentNet(**kw), AgentNet(**kw)
    ck = torch.load(ckpt, weights_only=False)
    net.load_state_dict(ck["net"])
    if "init_net" in ck:
        init.load_state_dict(ck["init_net"])
    return net.eval(), init.eval()


def evaluate(name: str, seeds, ckpt_name: str = "ckpt.pt") -> dict:
    run_dir = DATA_DIR / "agent" / name
    fp = json.loads((run_dir / "fingerprint.json").read_text())
    cfg = fp["cfg"]
    panel = panel_for(fp)
    rules = agent_rules(cfg["cost"])
    net, init = nets(cfg, run_dir / ckpt_name, panel_n_feat(panel))
    steps = torch.load(run_dir / ckpt_name, weights_only=False)["steps"]
    out = {"name": name, "steps": steps, "panel": fp["panel"], "cfg": cfg, "periods": {}}
    for label, a, b in PERIODS:
        L = policy_runs(panel, net, a, b, rules, cfg["K"], seeds)
        I = policy_runs(panel, init, a, b, rules, cfg["K"], seeds)
        out["periods"][label] = {
            "learned": summarize(L["stochastic"]), "init": summarize(I["stochastic"]),
            "learned_det": L["deterministic"], "init_det": I["deterministic"],
            "diff": {k: paired(L["stochastic"], I["stochastic"], k)
                     for k in ("rentabilidad anual", "Sharpe", "rotación anual")},
            "rule": run_rule(panel, a, b, rules), "ew_gross": ew_annual(panel, a, b),
            "learned_seeds": L["stochastic"], "init_seeds": I["stochastic"]}
    return out


def pct(x):
    return f"{100 * x:+.1f} %"


def table(results: list[dict]) -> list[str]:
    L = []
    for label, *_ in PERIODS:
        L += [f"### {label}", "",
              ("| ejecución | pasos | aprendida: rentab. / Sharpe / caída / rotación | inicial: "
               "rentab. / Sharpe / rotación | aprendida − inicial: rentab. (t) / Sharpe (t) | "
               "determinista: rentab. / rotación | regla fija: rentab. / Sharpe / rotación | "
               "pesos iguales (bruto) |"), "|---|---|---|---|---|---|---|---|"]
        for r in results:
            p = r["periods"][label]
            ln, it, df, rl = p["learned"], p["init"], p["diff"], p["rule"]
            L.append(
                f"| {r['name']} | {r['steps']:,} | {pct(ln['rentabilidad anual'][0])} ± "
                f"{100 * ln['rentabilidad anual'][1]:.1f} / {ln['Sharpe'][0]:.2f} / "
                f"{100 * ln['caída máxima'][0]:.0f} % / {ln['rotación anual'][0]:.0f} | "
                f"{pct(it['rentabilidad anual'][0])} / {it['Sharpe'][0]:.2f} / "
                f"{it['rotación anual'][0]:.0f} | "
                f"{100 * df['rentabilidad anual'][0]:+.1f} ({df['rentabilidad anual'][2]:.1f}) / "
                f"{df['Sharpe'][0]:+.2f} ({df['Sharpe'][2]:.1f}) | "
                f"{pct(p['learned_det']['rentabilidad anual'])} / "
                f"{p['learned_det']['rotación anual']:.0f} | "
                f"{pct(rl['rentabilidad anual'])} / {rl['Sharpe']:.2f} / "
                f"{rl['rotación anual']:.0f} | {pct(p['ew_gross'])} |")
        L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--ckpt", default="ckpt.pt", help="checkpoint de cada ejecución")
    args = ap.parse_args()
    log_dir = ROOT / "logs" / "agent"
    results = []
    for name in args.names:
        t0 = time.time()
        r = evaluate(name, range(1000, 1000 + args.seeds), args.ckpt)
        (log_dir / f"{name}_eval2.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        results.append(r)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    lines = table(results)
    print("\n".join(lines))
    if args.summary:
        (ROOT / args.summary).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
