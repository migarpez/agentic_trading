"""Prueba de entrada en dos escalas (8-11-2026). La ventana de 63 velas no ve el momentum de 12
meses, que en 2012–15 fue la mejor regla simple. Se añade, junto a las 63 velas diarias, el año
entero con menos resolución (TrainConfig.coarse):

a. coarse_month: 12 velas de 21 sesiones, con un MLP.
b. coarse_week: 50 velas de 5 sesiones, con una TCN.

Cada una con la configuración adoptada (target relativo, ventana de 63, λ = 3, lr = 0,001,
5 épocas, 20 % por época) y 2 miembros por bloque, con las semillas de los miembros 0 y 1 de
rel63; después, evaluate.py de cada una. Al final, coarse_compare.py > logs/coarse.md: por pares
con los miembros 0 y 1 de rel63 y, para σ, también con night_rel (ventana de 252).

Regla fijada antes de ver los resultados: una opción se adopta si sube el IC medio, gana en al
menos 3 de los 5 bloques y la QLIKE no empeora. Si las dos lo cumplen, se adopta la de mayor IC;
si la diferencia es menor de 0,001, la mensual, que es más simple.

Si un paso falla, se apunta en el log y se sigue. No sobrescribe runs existentes: si una ya
existe, se salta.

Uso: nohup python -u scripts/coarse.py > logs/coarse.log 2>&1 &
     python scripts/coarse.py --smoke   (prueba de humo: 1 bloque, 1 miembro, 1 época, 1 %)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

from agentic_trading.config import DATA_DIR, ROOT

PY = sys.executable
KINDS = ("month", "week")
CFG = ["--lr", "0.001", "--lam", "3", "--target", "rel", "--window", "63"]
FULL = ["--members", "2", "--epochs", "5", "--epoch-frac", "0.2"]
SMOKE = ["--folds", "3", "--members", "1", "--epochs", "1", "--epoch-frac", "0.01"]


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
    opts = (SMOKE if args.smoke else FULL) + CFG
    names = {k: f"{prefix}coarse_{k}" for k in KINDS}
    log(f"Inicio ({'humo' if args.smoke else 'prueba'}): {', '.join(names.values())}")
    for kind, name in names.items():
        log(f"== {name}")
        taken = [str(p / name) for p in (DATA_DIR / "predictions", ROOT / "checkpoints")
                 if (p / name).exists()]
        if taken:
            log(f"FALLO: ya existe {', '.join(taken)}; no se sobrescribe. Se salta.")
            continue
        script("train_cv.py", "--run", name, *opts, "--coarse", kind)
        script("evaluate.py", "--run", name, out=f"logs/{name}_eval.log")
    log("== comparación")
    script("coarse_compare.py", "--runs", *names.values(), "--out", f"logs/{prefix}coarse.md")
    log("Fin.")


if __name__ == "__main__":
    main()
