"""Agente secundario para el test (23-11-2026; handoff, «Protocolo del test»): clonado + PPO con
2000–2019 entero, con la misma configuración que scripts/agent_bc.py, y checkpoints congelados.

Mismo procedimiento que agent_bc.py, con el fin del entrenamiento en el 31-12-2019 (episodios,
clonado y normalización de las variables con 2000–2019; build_panel(train_end=...)):

1. clone: la regla fija clonada con el rango de μ_5 como entrada (rank=True, la que se eligió en
   agent_bc), 2 rondas de DAgger, en data/agent/final_clone_rank/. Comprobación frente a la regla
   en 2000–19 (solo se informa; la configuración ya está fijada).
2. ppo: 3 semillas (0, 1, 2) × 1.000 actualizaciones desde la clonada (crítico ajustado 30
   iteraciones antes, KL inicial 1 que baja a 0 en el primer 30 %), en data/agent/final_ppo_s*/.
3. curve: el checkpoint de mayor Sharpe estocástico medio (5 semillas de muestreo) en 2000–19,
   el periodo de entrenamiento (en agent_bc, 2000–15), copiado a ckpt_sel.pt.
4. freeze: sha256 de ckpt_sel.pt de cada semilla y la normalización de las variables del panel
   de entrenamiento, en data/agent/final_frozen.json; ckpt_sel.pt pasa a solo lectura.
   scripts/test_run.py se niega a usar un checkpoint que no coincida con esa huella.

El test no se toca: las predicciones son las de final20 (2000–2019).

Uso: python scripts/agent_final.py clone | ppo | curve | freeze | all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trading.agent.data import build_panel
from scripts.agent_bc import AG, OUT, SEEDS, clone, curve_one, log, train_one

TRAIN_END = "2019-12-31"
CLONE = "final_clone_rank"
PREFIX = "final_ppo"
FROZEN = AG / "final_frozen.json"
PERIODS = (("2000–19", "2000-01-01", TRAIN_END),)


def phase_clone() -> None:
    path = OUT / f"{CLONE}_check.json"
    if path.exists():
        log("clonada ya hecha")
        return
    chk = clone(True, train_end=TRAIN_END, name=CLONE, periods=PERIODS)
    log(f"clonada: {'reproduce' if chk['pass'] else 'no reproduce'} la regla en 2000–19 "
        "(solo informativo)")


def phase_ppo() -> bool:
    fn = partial(train_one, prefix=PREFIX, clone_dir=CLONE, rank=True, train_end=TRAIN_END)
    with ProcessPoolExecutor(len(SEEDS)) as ex:
        codes = list(ex.map(fn, SEEDS))
    log(f"PPO: códigos de salida {codes}")
    return all(c == 0 for c in codes)


def phase_curve() -> None:
    fn = partial(curve_one, prefix=PREFIX, start="2000-01-01", end=TRAIN_END)
    with ProcessPoolExecutor(len(SEEDS)) as ex:
        res = list(ex.map(fn, SEEDS))
    (OUT / "final_curve.json").write_text(json.dumps(res, indent=1, default=float))
    for r in res:
        log(f"semilla {r['seed']}: checkpoint elegido {r['best']}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_freeze() -> None:
    panel = build_panel("final20", freq="weekly", rank=True, train_end=TRAIN_END)
    curve = {r["seed"]: r["best"] for r in json.loads((OUT / "final_curve.json").read_text())}
    agents = {}
    for s in SEEDS:
        d = AG / f"{PREFIX}_s{s}"
        ck = d / "ckpt_sel.pt"
        fp = json.loads((d / "fingerprint.json").read_text())
        if fp["panel"] != json.loads(json.dumps(panel.fingerprint, default=str)):
            raise SystemExit(f"{d}: la huella del panel no coincide con la de 2000–19")
        agents[f"s{s}"] = {"dir": str(d.relative_to(AG)), "ckpt": "ckpt_sel.pt",
                           "sha256": sha256(ck), "update": curve[s], "cfg": fp["cfg"]}
        os.chmod(ck, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    FROZEN.write_text(json.dumps({"train_end": TRAIN_END, "panel": panel.fingerprint,
                                  "norm": panel.norm, "agents": agents}, indent=1, default=str))
    log(f"congelado en {FROZEN}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["clone", "ppo", "curve", "freeze", "all"])
    args = ap.parse_args()
    if args.phase in ("clone", "all"):
        phase_clone()
    if args.phase in ("ppo", "all") and not phase_ppo():
        return
    if args.phase in ("curve", "all"):
        phase_curve()
    if args.phase in ("freeze", "all"):
        phase_freeze()


if __name__ == "__main__":
    main()
