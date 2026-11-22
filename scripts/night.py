"""Pruebas de la noche sobre μ (8-11-2026), en orden y sin supervisión. Si una falla, se apunta
en el log y se sigue con la siguiente.

a. night_rel: target relativo (--target rel).
b. night_rel_huber: target relativo y Huber (--target rel --huber 3).
c. night_w63: la ganadora de a y b con --window 63. Gana la de mayor IC medio (media de h = 5 y
   10) en los bloques que tengan las dos; si la diferencia es menor de 0,001, se queda la a.

Todas con λ = 3, lr = 0,001, 5 épocas, 20 % por época, 2 miembros y los 5 bloques (mismas semillas
que el ajuste). Cada una: train_cv.py y después evaluate.py > logs/<nombre>_eval.log. Al final,
night_summary.py escribe logs/night_summary.md. No sobrescribe runs existentes.

Uso: nohup python -u scripts/night.py > logs/night.log 2>&1 &
     python scripts/night.py --smoke   (prueba de humo: 1 bloque, 1 miembro, 1 época, 1 %)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.tuning import fold_scores

PY = sys.executable
FULL = ["--members", "2", "--epochs", "5", "--epoch-frac", "0.2", "--lr", "0.001", "--lam", "3"]
SMOKE = ["--folds", "4", "--members", "1", "--epochs", "1", "--epoch-frac", "0.01",
         "--lr", "0.001", "--lam", "3"]
OPTS = {"a": ["--target", "rel"], "b": ["--target", "rel", "--huber", "3"]}
MIN_GAIN = 0.001


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


def test(name: str, opts: list[str], base: list[str]) -> bool:
    ok = run([PY, "-u", "scripts/train_cv.py", "--run", name, *base, *opts])
    ok_eval = run([PY, "-u", "scripts/evaluate.py", "--run", name],
                  out=ROOT / "logs" / f"{name}_eval.log")
    return ok and ok_eval


def block_ics(data, name: str) -> dict:
    pr = DATA_DIR / "predictions" / name
    return {p.stem: fold_scores(data, pd.read_parquet(p))["ic"]
            for p in sorted(pr.glob("cv[0-9].parquet"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    prefix, base = ("smoke_", SMOKE) if args.smoke else ("night_", FULL)
    names = {"a": f"{prefix}rel", "b": f"{prefix}rel_huber", "c": f"{prefix}w63"}
    summary_out = f"logs/{prefix}summary.md"

    if not args.smoke:
        taken = [n for n in names.values() for d in (DATA_DIR / "predictions", ROOT / "checkpoints")
                 if (d / n).exists()]
        if taken:
            log(f"Ya existen {sorted(set(taken))}: no se sobrescriben. Fin.")
            sys.exit(1)

    log(f"Inicio ({'humo' if args.smoke else 'noche'}): {names}")
    for lab in "ab":
        log(f"== {lab}: {names[lab]} {' '.join(OPTS[lab])}")
        test(names[lab], OPTS[lab], base)

    # ganadora de a y b, en los bloques que tengan las dos
    data = WindowData(DATA_DIR / "dataset")
    ics = {lab: block_ics(data, names[lab]) for lab in "ab"}
    common = sorted(set(ics["a"]) & set(ics["b"]))
    mean = {lab: float(np.mean([ics[lab][k] for k in common])) if common else None for lab in "ab"}
    if common:
        win = "b" if mean["b"] - mean["a"] >= MIN_GAIN else "a"
        why = (f"IC medio en {len(common)} bloques: a {mean['a']:.4f}, b {mean['b']:.4f} "
               f"(b gana solo con ≥ {MIN_GAIN}): {win}")
    else:
        win = "b" if ics["b"] and not ics["a"] else "a"
        why = f"sin bloques comunes (a: {sorted(ics['a'])}, b: {sorted(ics['b'])}): {win}"
    log(f"Base de c: {why}")
    (ROOT / "logs" / f"{prefix}choice.json").write_text(json.dumps(
        {"base": win, "base_run": names[win], "ic_a": ics["a"], "ic_b": ics["b"],
         "mean_common": mean, "blocks_common": common}, indent=1))

    log(f"== c: {names['c']} {' '.join(OPTS[win])} --window 63")
    test(names["c"], [*OPTS[win], "--window", "63"], base)

    log("== resumen")
    run([PY, "-u", "scripts/night_summary.py", "--runs", *names.values(), "--out", summary_out])
    log("Fin.")


if __name__ == "__main__":
    main()
