"""Comprobación de mu_tree (fase 0): sin las opciones nuevas, el código da bit a bit lo de antes.

Reentrena un miembro de final20 (por defecto, el 0 de cv4: semilla 4000) con su configuración
(la del checkpoint, 8 hilos; también al predecir: con otro número de hilos cambian
los últimos bits) y compara sus predicciones con las guardadas en
data/predictions/final20/<bloque>_members.parquet. También predice con el checkpoint guardado.
No escribe nada en final20. Resultado en logs/mu_tree/repro.json.

Uso: python scripts/mu_repro.py [--fold 4] [--member 0]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.train import TrainConfig, load_model, predict, train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--member", type=int, default=0)
    args = ap.parse_args()
    data = WindowData(DATA_DIR / "dataset")
    f = cv_folds(data.samples)[args.fold]
    ck = ROOT / "checkpoints" / "final20" / f["name"] / f"member{args.member}.pt"
    ref = pd.read_parquet(DATA_DIR / "predictions" / "final20" / f"{f['name']}_members.parquet")
    ref = ref[ref["member"] == args.member].drop(columns="member").reset_index(drop=True)
    cols = [c for c in ref.columns if c != "sample"]
    out = {"fold": f["name"], "member": args.member}

    model, saved = load_model(ck)
    torch.set_num_threads(saved["config"]["threads"])     # el resultado depende de los hilos
    p = predict(model, data, f["predict"])
    out["checkpoint_identical"] = bool(all(np.array_equal(p[c].to_numpy(), ref[c].to_numpy())
                                           for c in cols + ["sample"]))
    print("predicción con el checkpoint:", out["checkpoint_identical"], flush=True)

    cfg = TrainConfig(**saved["config"])
    t0 = time.time()
    model, hist = train_model(data, f["train"], cfg, saved["seed"])
    out["train_minutes"] = (time.time() - t0) / 60
    p = predict(model, data, f["predict"])
    out["retrain_identical"] = bool(all(np.array_equal(p[c].to_numpy(), ref[c].to_numpy())
                                        for c in cols + ["sample"]))
    out["max_abs_diff"] = float(max(np.abs(p[c].to_numpy() - ref[c].to_numpy()).max()
                                    for c in cols))
    out["history_identical"] = bool(np.array_equal([r["loss"] for r in hist],
                                                   [r["loss"] for r in saved["history"]]))
    print(json.dumps(out, indent=1), flush=True)
    (ROOT / "logs" / "mu_tree" / "repro.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
