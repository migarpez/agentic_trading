"""Trabajo del día (8-11-2026), en orden y sin supervisión. Si un paso falla, se apunta en el log
y se sigue con el siguiente.

Configuración adoptada: target relativo, ventana de 63, MSE, λ = 3, lr = 0,001, 5 épocas, 20 % de
las muestras por época; semillas 1000·k + m.

a. rel63: ensemble de 10 en los 5 bloques (train_cv.py) y evaluate.py > logs/rel63_eval.log.
b. Prueba de decisión (decision_test.py > logs/rel63_decision.md).
c. Análisis sin entrenar (rel63_analysis.py > logs/rel63_analysis.md).
d. rel63_e10: igual que a con 2 miembros y 10 épocas; evaluate.py > logs/rel63_e10_eval.log y
   comparación por pares con los miembros 0 y 1 de rel63 (epochs_compare.py >
   logs/rel63_e10_compare.md).

No sobrescribe runs existentes.

Uso: nohup python -u scripts/day.py > logs/day.log 2>&1 &
     python scripts/day.py --smoke   (prueba de humo: 1 bloque, 1 miembro, 1 época, 1 %)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

from agentic_trading.config import DATA_DIR, ROOT

PY = sys.executable
CFG = ["--lr", "0.001", "--lam", "3", "--target", "rel", "--window", "63"]
FULL = {"a": ["--members", "10", "--epochs", "5", "--epoch-frac", "0.2"],
        "d": ["--members", "2", "--epochs", "10", "--epoch-frac", "0.2"]}
SMOKE = ["--folds", "4", "--members", "1", "--epochs", "1", "--epoch-frac", "0.01"]


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
    a, d = f"{prefix}rel63", f"{prefix}rel63_e10"
    opts = {k: (SMOKE if args.smoke else FULL[k]) + CFG for k in "ad"}

    if not args.smoke:
        taken = [n for n in (a, d) for p in (DATA_DIR / "predictions", ROOT / "checkpoints")
                 if (p / n).exists()]
        if taken:
            log(f"Ya existen {sorted(set(taken))}: no se sobrescriben. Fin.")
            sys.exit(1)

    log(f"Inicio ({'humo' if args.smoke else 'día'}): a = {a}, d = {d}")
    log(f"== a: {a} {' '.join(opts['a'])}")
    script("train_cv.py", "--run", a, *opts["a"])
    script("evaluate.py", "--run", a, out=f"logs/{a}_eval.log")

    log("== b: prueba de decisión")
    script("decision_test.py", "--run", a, "--out", f"logs/{a}_decision.md")

    log("== c: análisis sin entrenar")
    script("rel63_analysis.py", "--run", a, "--out", f"logs/{a}_analysis.md")

    log(f"== d: {d} {' '.join(opts['d'])}")
    script("train_cv.py", "--run", d, *opts["d"])
    script("evaluate.py", "--run", d, out=f"logs/{d}_eval.log")
    script("epochs_compare.py", "--run", d, "--base", a, "--members", "0", "1",
           "--out", f"logs/{d}_compare.md")
    log("Fin.")


if __name__ == "__main__":
    main()
