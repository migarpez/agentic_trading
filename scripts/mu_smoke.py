"""Pruebas de humo de mu_tree (fase 0) y medida del paralelismo.

1. Humo de cada opción (base, market, target_rank, xsec, tail_weight y todas juntas): train_cv.py
   con 1 miembro, el bloque cv4, 1 época y el 2 % por época (--member-ids, el camino del
   orquestador), en data/predictions/smoke_mu_<opción>/. Comprueba que corre, que las
   predicciones no tienen NaN y cuánto tarda (época y total).
2. Paralelismo: N entrenamientos a la vez con T hilos cada uno (N × T ≤ 12), con la configuración
   base, 1 época del 2 %; el rendimiento es N / (segundos por época). Solo se mide la época (sin
   cargar datos ni predecir).
3. Predicción de un bloque entero con 1, 2 y 8 hilos.

Escribe logs/mu_tree/smoke.json y smoke.md. No toca final20.

Uso: python scripts/mu_smoke.py [--skip-options] [--skip-parallel]
"""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT

PY = sys.executable
OUT = ROOT / "logs" / "mu_tree"
BASE = ["--lr", "0.001", "--lam", "1", "--target", "rel", "--window", "63", "--coarse", "month"]
OPTIONS = {"base": [], "market": ["--market"], "target_rank": ["--target-rank"],
           "xsec": ["--xsec"], "tail_weight": ["--tail-weight", "3"],
           "all": ["--market", "--target-rank", "--xsec", "--tail-weight", "3"]}
FRAC = 0.02
PARALLEL = [(1, 8), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)]

BENCH = r"""
import json, sys, time
import numpy as np, torch
from agentic_trading.config import DATA_DIR
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.train import TrainConfig, train_model, predict
threads, frac, mode, opts = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
data = WindowData(DATA_DIR / "dataset")
f = cv_folds(data.samples)[4]
cfg = TrainConfig(epochs=1, epoch_frac=frac, lr=1e-3, lam=1.0, target="rel", window=63,
                  coarse="month", threads=threads, **opts)
if mode == "train":
    _, hist = train_model(data, f["train"], cfg, 4000, log=lambda *_: None)
    print(json.dumps({"epoch_s": hist[0]["seconds"]}))
else:
    model, _ = train_model(data, f["train"], TrainConfig(**{**cfg.__dict__, "epoch_frac": 0.001}),
                           4000, log=lambda *_: None)
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    predict(model, data, f["predict"])
    print(json.dumps({"predict_s": time.perf_counter() - t0, "n": len(f["predict"])}))
"""


def smoke_options() -> dict:
    res = {}
    for name, extra in OPTIONS.items():
        run = f"smoke_mu_{name}"
        cmd = [PY, "-u", str(ROOT / "scripts" / "train_cv.py"), "--run", run, "--folds", "4",
               "--members", "1", "--member-ids", "0", "--epochs", "1", "--epoch-frac", str(FRAC),
               "--threads", "8", *BASE, *extra]
        t0 = time.time()
        p = subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True)
        r = {"ok": p.returncode == 0, "total_s": time.time() - t0, "log": p.stdout[-1500:],
             "err": p.stderr[-1500:]}
        if r["ok"]:
            pr = pd.read_parquet(DATA_DIR / "predictions" / run / "cv4.parquet")
            r["nan"] = int(pr.drop(columns="sample").isna().to_numpy().sum())
            r["n_pred"] = len(pr)
            import torch
            ck = torch.load(ROOT / "checkpoints" / run / "cv4" / "member0.pt", weights_only=False)
            r["epoch_s"] = ck["history"][0]["seconds"]
            r["loss"] = ck["history"][0]["loss"]
            r["n_params"] = int(sum(v.numel() for k, v in ck["state"].items() if k != "scale"))
        res[name] = r
        print(name, {k: v for k, v in r.items() if k not in ("log", "err")}, flush=True)
    return res


def bench(n: int, threads: int, frac: float = FRAC, opts: dict | None = None) -> list[dict]:
    env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    import os
    procs = [subprocess.Popen([PY, "-c", BENCH, str(threads), str(frac), "train",
                               json.dumps(opts or {})], cwd=ROOT, stdout=subprocess.PIPE,
                              text=True, env={**os.environ, **env}) for _ in range(n)]
    return [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-options", action="store_true")
    ap.add_argument("--skip-parallel", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "smoke.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if not args.skip_options:
        res["options"] = smoke_options()
        path.write_text(json.dumps(res, indent=1))
    if not args.skip_parallel:
        par = {}
        for n, t in PARALLEL:
            r = bench(n, t)
            s = float(np.mean([x["epoch_s"] for x in r]))
            par[f"{n}x{t}"] = {"n": n, "threads": t, "epoch_s": s, "throughput": n / s}
            print(f"{n} × {t} hilos: {s:.1f} s por época (2 %) · {n / s:.3f} épocas/s", flush=True)
        res["parallel"] = par
        import os
        pred = {}
        for t in (1, 2, 8):
            p = subprocess.run([PY, "-c", BENCH, str(t), str(FRAC), "predict", "{}"], check=False,
                               cwd=ROOT,
                               capture_output=True, text=True,
                               env={**os.environ, "OMP_NUM_THREADS": str(t)})
            pred[t] = json.loads(p.stdout.strip().splitlines()[-1])
            print(f"predicción con {t} hilos: {pred[t]}", flush=True)
        res["predict"] = pred
        path.write_text(json.dumps(res, indent=1))
    L = ["# Humo de mu_tree", "", f"1 miembro, cv4, 1 época, {100 * FRAC:g} % por época, 8 hilos.",
         "", "| opción | corre | NaN | parámetros | época (s) | total (s) | L |",
         "|---|---|---|---|---|---|---|"]
    for k, r in res.get("options", {}).items():
        L.append(f"| {k} | {'sí' if r['ok'] else 'NO'} | {r.get('nan', '—')} | "
                 f"{r.get('n_params', '—')} | {r.get('epoch_s', float('nan')):.1f} | "
                 f"{r['total_s']:.0f} | {r.get('loss', float('nan')):.4f} |")
    if "parallel" in res:
        L += ["", "Paralelismo (base, época del 2 %):", "",
              "| procesos × hilos | s por época | épocas/s |", "|---|---|---|"]
        for k, r in res["parallel"].items():
            L.append(f"| {k} | {r['epoch_s']:.1f} | {r['throughput']:.3f} |")
        L += ["", "Predicción de cv4: " + ", ".join(
            f"{t} hilos {r['predict_s']:.0f} s" for t, r in res["predict"].items())]
    (OUT / "smoke.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
