"""Entrena el agente de RL (handoff §10) con PPO en 2000–2015, con las predicciones de final20.

Una ejecución = un γ y una semilla. Reanudable: guarda en data/agent/<name>/ el checkpoint
(ckpt.pt, cada --save-every actualizaciones y al acabar) y la huella (fingerprint.json: dataset,
predicciones y configuración); si la huella no coincide, no reanuda. Log por actualización en
logs/agent/<name>.jsonl.

Para cuando se cumple lo primero de --updates o --minutes. Al acabar, si --eval, corre la política
determinista en 2012–15 (entrenamiento) y 2016–19 (validación) y escribe el comportamiento en
logs/agent/<name>_eval.json. El test (2020–) no se toca.

Plan B (21-11-2026): --freq weekly (una decisión por semana; --episode por defecto 52). Prueba de
capacidad: --signal ρ sustituye μ por una señal sintética con IC ≈ ρ (agent/data.py), y
--prior none quita μ_5 de la puntuación inicial. La evaluación con muchas semillas y frente a la
política inicial está en scripts/agent_eval.py.

Uso: python scripts/train_agent.py --name smoke --gamma 0.95 --seed 0 --minutes 30 --eval
"""

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from agentic_trading.agent.data import VALID_START, build_panel
from agentic_trading.agent.ppo import PPOConfig, Trainer, run_policy, write_jsonl
from agentic_trading.agent.simulator import agent_rules, performance
from agentic_trading.config import DATA_DIR, ROOT


def check_fingerprint(run_dir, fp: dict) -> bool:
    """Guarda la huella o comprueba que coincide. Devuelve True si hay checkpoint que reanudar."""
    path = run_dir / "fingerprint.json"
    fp = json.loads(json.dumps(fp, default=str))
    if path.exists():
        if json.loads(path.read_text()) != fp:
            raise SystemExit(f"{run_dir}: la huella (dataset, predicciones o configuración) no "
                             "coincide. No se reanuda: usa otro --name.")
        return (run_dir / "ckpt.pt").exists()
    if (run_dir / "ckpt.pt").exists():
        raise SystemExit(f"{run_dir}: checkpoint sin huella. No se reanuda: usa otro --name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, indent=1))
    return False


def behaviour(res, diag, start, end) -> dict:
    p = performance(res, start, end)
    wd = np.array(diag["wd"]) if diag["wd"] else np.array([np.nan])
    return {**{k: float(v) for k, v in p.items()}, "posiciones (media)": float(np.mean(diag["npos"])),
            "liquidez (media)": float(np.mean(diag["cash"])),
            "operados por día": float(np.mean(diag["n_trade"])),
            "w·d p10/p50/p90": [float(np.nanpercentile(wd, q)) for q in (10, 50, 90)],
            "stops": int(res.n_stops)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=10**9)
    ap.add_argument("--minutes", type=float, default=10**9)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--run", default="final20")
    ap.add_argument("--reward", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--ratio", choices=["joint", "component"], default="joint")
    ap.add_argument("--trade-bias", type=float, default=0.0)
    ap.add_argument("--freq", choices=["daily", "weekly"], default="daily")
    ap.add_argument("--episode", type=int, default=None,
                    help="decisiones por episodio (252 diario, 52 semanal)")
    ap.add_argument("--signal", type=float, default=None)
    ap.add_argument("--signal-seed", type=int, default=0)
    ap.add_argument("--prior", choices=["mu5", "none"], default="mu5")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    cfg = PPOConfig(gamma=args.gamma, seed=args.seed, n_envs=args.n_envs, rollout=args.rollout,
                    reward=args.reward, ratio=args.ratio, trade_bias=args.trade_bias,
                    episode=args.episode or (52 if args.freq == "weekly" else 252))
    run_dir = DATA_DIR / "agent" / args.name
    log_dir = ROOT / "logs" / "agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = build_panel(args.run, freq=args.freq, signal=args.signal,
                        signal_seed=args.signal_seed, prior=args.prior)
    print(f"panel: {len(panel.dec_t)} días de decisión, {len(panel.market.symbols)} activos "
          f"({time.time() - t0:.0f} s)", flush=True)
    resume = check_fingerprint(run_dir, {"panel": panel.fingerprint, "cfg": asdict(cfg)})
    tr = Trainer(panel, cfg)
    if resume:
        tr.load(run_dir / "ckpt.pt")
        print(f"reanuda en la actualización {tr.updates}", flush=True)
    log = log_dir / f"{args.name}.jsonl"
    start = time.time()
    while tr.updates < args.updates and (time.time() - start) / 60 < args.minutes:
        row = tr.train_step()
        write_jsonl(log, row)
        if tr.updates % args.save_every == 0:
            tr.save(run_dir / "ckpt.pt")
        sps = (cfg.n_envs * cfg.rollout) / (row["t_collect"] + row["t_update"])
        print(f"upd {row['update']:4d} · pasos {row['steps']:7d} · r {row['reward_mean']:+.5f} · "
              f"r−EW {row['reward_rel_mean']:+.5f} · "
              f"rot {row['turnover_day']:.3f} · pos {row['npos']:.1f} · liq {row['cash']:.2f} · "
              f"Lv {row['loss_v']:.2f} · H {row['entropy']:.1f} · {sps:.0f} pasos/s", flush=True)
    tr.save(run_dir / "ckpt.pt")

    if args.eval:
        rules = agent_rules(cfg.cost)
        out = {}
        for label, a, b in (("2012–15", "2012-01-01", "2015-12-31"),
                            ("2016–19", VALID_START, "2019-12-31")):
            res, diag = run_policy(panel, tr.net, a, b, rules, cfg.K)
            out[label] = behaviour(res, diag, None, None)
        (log_dir / f"{args.name}_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
