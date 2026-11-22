"""Plan B del agente y prueba de capacidad (handoff §10, §11.7; 21-11-2026).

Todas: recompensa relativa, ratio por componente, p(operar) inicial 0,12 (trade_bias −2),
c = 0,1 %, --updates actualizaciones cada una (400: 204.800 pasos; el mismo presupuesto para
todas, fijado antes de lanzar), en paralelo (un hilo por proceso). Semanales con γ = 0,8 (≈ 5
semanas, del orden de los 20 días de γ = 0,95 de las pruebas de humo diarias) y episodios de 52
decisiones.

- w_real_s{0,1,2}: plan B con μ de final20.
- w_sig10_none_s{0,1} y w_sig05_none_s{0,1}: señal sintética con ρ = 0,10 y 0,05 en vez de μ, sin
  μ_5 en la puntuación inicial: ¿aprende a usar una señal que sabemos que existe?
- w_sig10_mu5_s0: ρ = 0,10 con la señal en la puntuación inicial: ¿mejora lo que ya hace la
  regla de partida?
- d_sig10_none_s0: como w_sig10_none_s0, pero con decisiones diarias (γ = 0,95, episodios de
  252): ¿el problema de las pruebas de humo era la frecuencia?

Después, scripts/agent_eval.py con 20 semillas → logs/agent/planb.md.

Uso: python scripts/planb.py [--updates 400]
"""

import argparse
import subprocess
import sys

from agentic_trading.config import ROOT

COMMON = ["--reward", "relative", "--ratio", "component", "--trade-bias", "-2"]
WEEKLY = ["--freq", "weekly", "--gamma", "0.8"]
RUNS = {
    **{f"w_real_s{s}": WEEKLY + ["--seed", str(s)] for s in range(3)},
    **{f"w_sig10_none_s{s}": WEEKLY + ["--seed", str(s), "--signal", "0.10", "--prior", "none"]
       for s in range(2)},
    **{f"w_sig05_none_s{s}": WEEKLY + ["--seed", str(s), "--signal", "0.05", "--prior", "none"]
       for s in range(2)},
    "w_sig10_mu5_s0": WEEKLY + ["--seed", "0", "--signal", "0.10"],
    "d_sig10_none_s0": ["--freq", "daily", "--gamma", "0.95", "--seed", "0", "--signal", "0.10",
                        "--prior", "none"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    logs = ROOT / "logs" / "agent"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for name, extra in RUNS.items():
        cmd = [sys.executable, str(ROOT / "scripts" / "train_agent.py"), "--name", name,
               "--updates", str(args.updates), *COMMON, *extra]
        procs.append((name, subprocess.Popen(cmd, stdout=open(logs / f"{name}.log", "a"),  # noqa: SIM115
                                             stderr=subprocess.STDOUT,
                                             env={"OMP_NUM_THREADS": "1", **_env()})))
    bad = [n for n, p in procs if p.wait() != 0]
    if bad:
        print("fallaron:", bad, flush=True)
    ok = [n for n, _ in procs if n not in bad]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "agent_eval.py"), "--names", *ok,
                    "--seeds", str(args.seeds), "--summary", "logs/agent/planb.md"], check=True)


def _env() -> dict:
    import os
    return dict(os.environ)


if __name__ == "__main__":
    main()
