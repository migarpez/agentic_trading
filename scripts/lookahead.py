"""Prueba de mirar el futuro (8-11-2026): ¿infla la validación cruzada los resultados? En 2016–19,
el único bloque cuyo modelo solo ha visto el pasado, μ no bate a la reversión; en los demás, sí.

Dos ensembles que predicen 2012–15 (cv3), con la configuración adoptada (target relativo, ventana
de 63, λ = 3, lr = 0,001, 5 épocas, 20 % por época, 10 miembros) y las semillas de cv3 de rel63
(1000·3 + m):

- F (solo pasado): entrena con 2000–2011.
- P (futuro en lugar de pasado): entrena con 2000–2007 y 2016–2019, submuestreado al azar
  (semilla 0) hasta tantas muestras como F.

Después, evaluate.py de cada una y lookahead_compare.py > logs/lookahead.md (F, P y cv3 de rel63).
Regla fijada antes de ver los resultados: la validación cruzada infla si la diferencia diaria de
IC P − F es positiva con t de NW > 2 (retardos = h) a h = 5 o 10.

Si un paso falla, se apunta en el log y se sigue. No sobrescribe runs existentes.

Uso: nohup python -u scripts/lookahead.py > logs/lookahead.log 2>&1 &
     python scripts/lookahead.py --smoke   (prueba de humo: 1 miembro, 1 época, 1 %)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.folds import manual_fold, parse_years

PY = sys.executable
K = 3
F_YEARS, P_YEARS = ["2000-2011"], ["2000-2007", "2016-2019"]
CFG = ["--folds", str(K), "--lr", "0.001", "--lam", "3", "--target", "rel", "--window", "63"]
FULL = ["--members", "10", "--epochs", "5", "--epoch-frac", "0.2"]
SMOKE = ["--members", "1", "--epochs", "1", "--epoch-frac", "0.01"]


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
    f_run, p_run = f"{prefix}lookahead_F", f"{prefix}lookahead_P"
    opts = (SMOKE if args.smoke else FULL) + CFG
    taken = [n for n in (f_run, p_run) for p in (DATA_DIR / "predictions", ROOT / "checkpoints")
             if (p / n).exists()]
    if taken:
        log(f"Ya existen {sorted(set(taken))}: no se sobrescriben. Fin.")
        sys.exit(1)

    samples = WindowData(DATA_DIR / "dataset").samples
    n_f = len(manual_fold(samples, K, [parse_years(x) for x in F_YEARS])["train"])
    n_p = len(manual_fold(samples, K, [parse_years(x) for x in P_YEARS])["train"])
    log(f"Inicio ({'humo' if args.smoke else 'prueba'}): F = {f_run} ({' '.join(F_YEARS)}, "
        f"{n_f:,} muestras), P = {p_run} ({' '.join(P_YEARS)}, {n_p:,} → {n_f:,})")

    log(f"== F: {f_run}")
    script("train_cv.py", "--run", f_run, "--train-years", *F_YEARS, *opts)
    script("evaluate.py", "--run", f_run, out=f"logs/{f_run}_eval.log")

    log(f"== P: {p_run}")
    if n_p < n_f:
        log(f"FALLO: P tiene menos muestras que F ({n_p:,} < {n_f:,}); no se submuestrea.")
        p_opts = []
    else:
        p_opts = ["--train-n", str(n_f), "--subsample-seed", "0"]
    script("train_cv.py", "--run", p_run, "--train-years", *P_YEARS, *p_opts, *opts)
    script("evaluate.py", "--run", p_run, out=f"logs/{p_run}_eval.log")

    log("== comparación")
    script("lookahead_compare.py", "--f", f_run, "--p", p_run, "--cv", "rel63",
           "--out", f"logs/{prefix}lookahead.md")
    log("Fin.")


if __name__ == "__main__":
    main()
