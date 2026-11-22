"""Ajuste de λ y de la tasa de aprendizaje con los bloques de validación cruzada (ver
model/tuning.py para la comparación pareada y la regla de elección).

Reanudable: cada (configuración, bloque) terminado se guarda en data/predictions/tune/results.parquet
y no se repite. Guarda la huella del dataset (dataset.json) y no reanuda si el dataset ha cambiado.
Si el λ elegido cae en un extremo, amplía la rejilla y sigue.

Uso: python scripts/tune.py [--lams 0.3 1 3] [--lrs 0.001 0.003] [--members 2] [--epochs 5]
"""

import argparse
import json

import pandas as pd

from agentic_trading.config import DATA_DIR
from agentic_trading.data.dataset import WindowData, dataset_id
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.train import TrainConfig, ensemble, predict, train_model
from agentic_trading.model.tuning import check_dataset, extend_lambda, fold_scores, select


def key(lam, lr):
    return f"lam={lam:g} lr={lr:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--members", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--epoch-frac", type=float, default=0.2)
    ap.add_argument("--max-extensions", type=int, default=2)
    ap.add_argument("--name", default="tune", help="carpeta de resultados (otra para ensayos)")
    ap.add_argument("--folds", type=int, nargs="*", help="solo estos bloques (ensayos)")
    args = ap.parse_args()

    out = DATA_DIR / "predictions" / args.name
    data = WindowData(DATA_DIR / "dataset")
    check_dataset(out, dataset_id(data.meta))      # no reanuda con otro dataset
    res_path = out / "results.parquet"
    results = pd.read_parquet(res_path) if res_path.exists() else pd.DataFrame()
    folds = cv_folds(data.samples)
    if args.folds:
        folds = [folds[k] for k in args.folds]

    def run(lam, lr):
        nonlocal results
        cfg = TrainConfig(epochs=args.epochs, epoch_frac=args.epoch_frac, lr=lr, lam=lam)
        for f in folds:
            k = int(f["name"][2:])                  # semilla ligada al bloque, no al orden
            done = (len(results) and ((results.config == key(lam, lr))
                                      & (results.fold == f["name"])).any())
            if done:
                continue
            print(f"\n== {key(lam, lr)} · {f['name']} {f['block']}", flush=True)
            preds = []
            for m in range(args.members):
                model, _ = train_model(data, f["train"], cfg, seed=1000 * k + m)
                preds.append(predict(model, data, f["predict"]))
            s = fold_scores(data, ensemble(preds))
            print(f"  IC {s['ic']:+.4f} · QLIKE {s['qlike']:.4f}", flush=True)
            row = pd.DataFrame([{"config": key(lam, lr), "lam": lam, "lr": lr,
                                 "fold": f["name"], **s}])
            results = pd.concat([results, row], ignore_index=True)
            results.to_parquet(res_path, index=False)

    lams = list(args.lams)
    for lam in lams:
        for lr in args.lrs:
            run(lam, lr)
    extensions = 0
    while True:
        sel = select(results)
        chosen = results.loc[results.config == sel["config"]].iloc[0]
        print("\n" + sel["table"].round(4).to_string())
        print(f"Elegida: {sel['config']} ({sel['reason']})", flush=True)
        new = extend_lambda(lams, chosen["lam"])
        if new is None:
            break
        if extensions >= args.max_extensions:
            print("λ sigue en el extremo, pero se alcanzó el máximo de ampliaciones", flush=True)
            break
        print(f"λ en el extremo de la rejilla: se amplía con λ = {new:g}", flush=True)
        lams.append(new)
        run(new, chosen["lr"])
        extensions += 1
    (out / "choice.json").write_text(json.dumps(
        {"config": sel["config"], "lam": float(chosen["lam"]), "lr": float(chosen["lr"]),
         "reason": sel["reason"], "members": args.members, "epochs": args.epochs,
         "epoch_frac": args.epoch_frac}, indent=1))


if __name__ == "__main__":
    main()
