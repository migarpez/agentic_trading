"""Prueba de λ con la configuración adoptada (9-11-2026): target relativo, ventana de 63, velas
mensuales, MSE, lr = 0,001, 5 épocas, 20 % por época. El ajuste de λ de §6 se hizo con el target
absoluto; con el relativo, L_μ baja de ~3,0 a ~2,15, así que λ = 3 pesa más que cuando se eligió.

Entrena month_lam1 (λ = 1) y month_lam9 (λ = 9), 2 miembros por bloque con las semillas de
coarse_month, y pasa evaluate.py a cada una. La referencia, λ = 3, es coarse_month (miembros 0 y
1): no se reentrena. Al final, lam_compare.py > logs/lam.md.

Regla fijada antes de ver los resultados (referencia: λ = 3; manda el IC, porque la QLIKE mejora
siempre al subir λ):
a. Se descarta el λ cuya QLIKE sea más de un 2 % peor que la de λ = 3.
b. De los que quedan, se adopta el que supere el IC medio de λ = 3 en al menos 0,001 y le gane en
   IC en al menos 3 de los 5 bloques. Si lo cumplen los dos, el de mayor IC; si se llevan menos de
   0,001, el de menor QLIKE.
c. Si ninguno lo cumple, el de menor QLIKE entre los que no bajan el IC medio más de 0,001
   (λ = 3 incluido).

Si un paso falla, se apunta en el log y se sigue. No sobrescribe runs existentes: si una ya
existe, se salta y se apunta.

Uso: nohup python -u scripts/lam.py > logs/lam.log 2>&1 &
     python scripts/lam.py --smoke   (prueba de humo: 1 bloque, 1 miembro, 1 época, 1 %; la
                                      referencia es smoke_coarse_month, con esa misma prueba)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

from agentic_trading.config import DATA_DIR, ROOT

PY = sys.executable
LAMS = {"month_lam1": "1", "month_lam9": "9"}
REF = "coarse_month"
CFG = ["--target", "rel", "--window", "63", "--coarse", "month", "--lr", "0.001"]
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
    names = {f"{prefix}{n}": lam for n, lam in LAMS.items()}
    ref = f"{prefix}{REF}"
    log(f"Inicio ({'humo' if args.smoke else 'prueba'}): {', '.join(names)}; referencia λ = 3: "
        f"{ref}")
    if not (DATA_DIR / "predictions" / ref).exists():
        log(f"FALLO: no existe la referencia {ref}; se entrena igual y la comparación lo dirá.")
    for name, lam in names.items():
        log(f"== {name} (λ = {lam})")
        taken = [str(p / name) for p in (DATA_DIR / "predictions", ROOT / "checkpoints")
                 if (p / name).exists()]
        if taken:
            log(f"FALLO: ya existe {', '.join(taken)}; no se sobrescribe. Se salta.")
            continue
        script("train_cv.py", "--run", name, "--lam", lam, *opts)
        script("evaluate.py", "--run", name, out=f"logs/{name}_eval.log")
    log("== comparación")
    script("lam_compare.py", "--runs", *names, "--ref", ref, "--out", f"logs/{prefix}lam.md")
    log("Fin.")


if __name__ == "__main__":
    main()
