"""Ensemble final con la configuración cerrada (9-11-2026), sin abrir el test: target relativo,
ventana de 63, velas mensuales, MSE, λ = 1, lr = 0,001, 5 épocas, 20 % por época; 20 miembros por
bloque (semillas 1000·k + m, m = 0–19). Son las predicciones fuera de muestra de 2000–2019 con las
que entrenará el agente.

a. train_cv.py --run final20 (antes, comprueba que quedan al menos 5 GB libres).
b. evaluate.py --run final20 > logs/final20_eval.log.
c. decision_test.py --run final20 → logs/final20_decision.md: la decisión con el ensemble de los 10
   primeros miembros (como se fijó el 8-11-2026), también con 20 (informativo), y el IC de los k
   primeros con k = 1, 2, 5, 10 y 20.
d. final_check.py → logs/final.md (solo lee predicciones).

Si un paso falla, se apunta en el log y se sigue. No sobrescribe runs existentes: si final20 ya
existe, no se entrena ni se evalúa (c y d solo leen y se ejecutan igual).

Uso: nohup python -u scripts/final.py > logs/final.log 2>&1 &
     python scripts/final.py --smoke   (prueba de humo: 1 bloque, 2 miembros, 1 época, 1 %;
                                        la reproducibilidad, frente a smoke_month_lam1)
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime

from agentic_trading.config import DATA_DIR, ROOT

PY = sys.executable
RUN, REPRO, BASE = "final20", "month_lam1", "rel63"
CFG = ["--lr", "0.001", "--lam", "1", "--target", "rel", "--window", "63", "--coarse", "month"]
FULL = ["--members", "20", "--epochs", "5", "--epoch-frac", "0.2"]
SMOKE = ["--folds", "3", "--members", "2", "--epochs", "1", "--epoch-frac", "0.01"]
DECISION = ["--decision-members", "10", "--also", "20", "--ks", "1", "2", "5", "10", "20"]
MIN_FREE_GB = 5


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run(cmd: list[str], out=None) -> bool:
    log("$ " + " ".join(cmd[1:]) + (f" > {out.relative_to(ROOT)}" if out else ""))
    t0 = time.time()
    try:
        if out:
            with open(out, "w") as fh:
                rc = subprocess.run(cmd, check=False, cwd=ROOT, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
        else:
            rc = subprocess.run(cmd, check=False, cwd=ROOT).returncode
    except Exception as e:                      # noqa: BLE001 — se apunta y se sigue
        log(f"FALLO al lanzar: {e!r}")
        return False
    log(f"{'OK' if rc == 0 else f'FALLO (código {rc})'} · {(time.time() - t0) / 60:.1f} min")
    return rc == 0


def script(name: str, *args: str, out: str | None = None) -> bool:
    return run([PY, "-u", f"scripts/{name}", *args], out=ROOT / out if out else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    prefix = "smoke_" if args.smoke else ""
    name, repro = f"{prefix}{RUN}", f"{prefix}{REPRO}"
    opts = (SMOKE if args.smoke else FULL) + CFG
    log(f"Inicio ({'humo' if args.smoke else 'ensemble final'}): {name}")
    taken = [str(p / name) for p in (DATA_DIR / "predictions", ROOT / "checkpoints")
             if (p / name).exists()]
    free = shutil.disk_usage(DATA_DIR).free / 1e9
    log(f"Espacio libre: {free:.0f} GB")
    if taken:
        log(f"FALLO: ya existe {', '.join(taken)}; no se sobrescribe: no se entrena ni se evalúa.")
    elif free < MIN_FREE_GB:
        log(f"FALLO: menos de {MIN_FREE_GB} GB libres; no se entrena.")
    else:
        log(f"== a. entrenamiento: {name}")
        script("train_cv.py", "--run", name, *opts)
        log("== b. evaluación")
        script("evaluate.py", "--run", name, out=f"logs/{name}_eval.log")
    log("== c. prueba de decisión")
    script("decision_test.py", "--run", name, *DECISION, "--out", f"logs/{name}_decision.md",
           out=f"logs/{name}_decision.log")
    log("== d. comprobaciones")
    script("final_check.py", "--run", name, "--repro", repro, "--base", BASE,
           "--out", f"logs/{prefix}final.md", out=f"logs/{prefix}final_check.log")
    log("Fin.")


if __name__ == "__main__":
    main()
