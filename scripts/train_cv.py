"""Validación cruzada por bloques (desarrollo 2000–2019): para cada bloque, entrena un ensemble
con los otros cuatro y predice ese bloque fuera de muestra.

Salidas:
  checkpoints/<run>/<bloque>/member<k>.pt      pesos, configuración e historial de cada miembro
  data/predictions/<run>/<bloque>_members.parquet   μ_h y log v_h de cada miembro
  data/predictions/<run>/<bloque>.parquet      salidas del ensemble (μ_h, σ_h, s_h)

Años de entrenamiento a mano (--train-years): de las muestras de entrenamiento de cada bloque
elegido, solo las de esos años, con el mismo hueco de 20 velas alrededor del bloque predicho
(folds.manual_fold); --train-n las submuestra al azar, con semilla fija (--subsample-seed), hasta
ese número. Mismos nombres de bloque y mismas semillas que la validación cruzada normal.

Velas agregadas (--coarse month|week): la red ve además el año entero con menos resolución
(TrainConfig.coarse; data.dataset.COARSE). Sin la opción, como siempre.

Opciones de mu_tree (22-11-2026; TrainConfig): --market (vela del mercado de la región; antes,
data.market.build_market), --target-rank (L_μ sobre el rango gaussianizado de la fecha), --xsec
(resumen aprendido del universo, lotes por fecha) y --tail-weight w (peso del decil alto de la
fecha en L_μ). Sin ellas, como siempre. Prueba de adaptación: --recency-halflife H (peso
2^(−antigüedad / H años) de cada muestra en la pérdida).

En paralelo (mu_tree): --member-ids entrena solo esos miembros (semillas 1000·k + m) y guarda sus
predicciones en <bloque>_m<m>.parquet; cuando están todos los de --members, junta
<bloque>_members.parquet y <bloque>.parquet. --threads: hilos de PyTorch (los resultados cambian
en los últimos bits con otro número de hilos).

Uso: python scripts/train_cv.py --run base [--folds 0 1] [--members 10] [--epochs 10] ...
     [--target abs|rel] [--huber δ] [--window 252] [--coarse month|week]
     [--train-years 2000-2007 2016-2019 [--train-n N] [--subsample-seed 0]]
     [--market] [--target-rank] [--xsec] [--tail-weight 3] [--threads 8] [--member-ids 0 1]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.folds import cv_folds, manual_fold, parse_years
from agentic_trading.model.metrics import daily_ic
from agentic_trading.model.train import TrainConfig, ensemble, predict, save_model, train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, nargs="*", help="bloques a entrenar (por defecto todos)")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--epoch-frac", type=float, default=TrainConfig.epoch_frac)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--lam", type=float, default=TrainConfig.lam)
    ap.add_argument("--batch", type=int, default=TrainConfig.batch)
    ap.add_argument("--target", choices=["abs", "rel"], default=TrainConfig.target,
                    help="target de μ: absoluto o relativo a la media de la fecha")
    ap.add_argument("--huber", type=float, default=TrainConfig.huber,
                    help="δ de la pérdida de Huber para μ (umbral δ·√h); sin él, MSE")
    ap.add_argument("--window", type=int, default=TrainConfig.window,
                    help="velas que ve la red (las últimas de la ventana de 252)")
    ap.add_argument("--coarse", choices=["month", "week"], default=TrainConfig.coarse,
                    help="además, el año entero en velas agregadas: 12 de 21 sesiones o 50 de 5")
    ap.add_argument("--train-years", nargs="+", metavar="AAAA-AAAA",
                    help="años de entrenamiento fijados a mano (p. ej. 2000-2007 2016-2019)")
    ap.add_argument("--train-n", type=int,
                    help="con --train-years: submuestra al azar hasta este número de muestras")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--market", action="store_true", help="vela del mercado de la región")
    ap.add_argument("--target-rank", action="store_true",
                    help="L_μ sobre el rango gaussianizado del target en la fecha")
    ap.add_argument("--xsec", action="store_true", help="resumen aprendido del universo")
    ap.add_argument("--tail-weight", type=float, default=TrainConfig.tail_weight,
                    help="peso del decil alto de la fecha en L_μ")
    ap.add_argument("--recency-halflife", type=float, default=TrainConfig.recency_halflife,
                    help="semivida (años) del peso por antigüedad de las muestras")
    ap.add_argument("--threads", type=int, default=TrainConfig.threads)
    ap.add_argument("--member-ids", type=int, nargs="+",
                    help="solo estos miembros (de 0 a --members − 1), en ficheros aparte")
    args = ap.parse_args()
    if args.member_ids and not all(0 <= m < args.members for m in args.member_ids):
        ap.error("--member-ids debe estar entre 0 y --members − 1")
    if args.train_n is not None and not args.train_years:
        ap.error("--train-n solo vale con --train-years")
    cfg = TrainConfig(epochs=args.epochs, epoch_frac=args.epoch_frac, lr=args.lr, lam=args.lam,
                      batch=args.batch, target=args.target, huber=args.huber, window=args.window,
                      coarse=args.coarse, threads=args.threads, market=args.market,
                      target_rank=args.target_rank, xsec=args.xsec, tail_weight=args.tail_weight,
                      recency_halflife=args.recency_halflife)

    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    chosen = args.folds if args.folds else range(len(folds))
    manual = None
    if args.train_years:
        years = [parse_years(x) for x in args.train_years]
        folds = {k: manual_fold(data.samples, k, years, args.train_n, args.subsample_seed)
                 for k in chosen}
        manual = {"train_years": [list(r) for r in years], "train_n": args.train_n,
                  "subsample_seed": args.subsample_seed if args.train_n is not None else None}
    ck, pr = ROOT / "checkpoints" / args.run, DATA_DIR / "predictions" / args.run
    pr.mkdir(parents=True, exist_ok=True)
    (pr / "config.json").write_text(json.dumps({"train": cfg.__dict__, "members": args.members,
                                                "dataset": data.meta["clean_fingerprint"],
                                                "manual_fold": manual}))
    for k in chosen:
        f = folds[k]
        print(f"\n== {f['name']} {f['block']} · entrenamiento {len(f['train']):,} · "
              f"predicción {len(f['predict']):,}"
              + (f" · velas agregadas: {args.coarse}" if args.coarse else "")
              + (f" · años de entrenamiento {args.train_years}" if manual else "")
              + (f", submuestra de {args.train_n:,} (semilla {args.subsample_seed})"
                 if args.train_n is not None else ""))
        preds = []
        for m in (args.member_ids or range(args.members)):
            seed = 1000 * k + m
            print(f" miembro {m + 1}/{args.members} (semilla {seed})", flush=True)
            model, hist = train_model(data, f["train"], cfg, seed)
            save_model(model, ck / f["name"] / f"member{m}.pt", cfg, seed, hist)
            p = predict(model, data, f["predict"]).assign(member=m)
            if args.member_ids:
                atomic_parquet(p, pr / f"{f['name']}_m{m}.parquet")
            preds.append(p)
        if args.member_ids:
            parts = [pr / f"{f['name']}_m{m}.parquet" for m in range(args.members)]
            if not all(q.exists() for q in parts):
                continue
            preds = [pd.read_parquet(q) for q in parts]
            atomic_parquet(pd.concat(preds), pr / f"{f['name']}_members.parquet")
            ens = ensemble(preds)
            atomic_parquet(ens, pr / f"{f['name']}.parquet")
        else:
            pd.concat(preds).to_parquet(pr / f"{f['name']}_members.parquet", index=False)
            ens = ensemble(preds)
            ens.to_parquet(pr / f"{f['name']}.parquet", index=False)

        y = np.stack([data.arr[f"y_{h}"][data.samples["row"].to_numpy()[f["predict"]]]
                      for h in (5, 10, 20)], 1)
        smp = data.samples.iloc[f["predict"]]
        for j, h in enumerate((5, 10, 20)):
            ic = daily_ic(smp["date"].to_numpy(), ens[f"mu_{h}"].to_numpy(), y[:, j])
            print(f"  h={h}: IC de selección medio {ic.mean():+.4f} ({len(ic)} días)")


def atomic_parquet(df: pd.DataFrame, path) -> None:
    """Escribe y renombra: otro proceso nunca ve un fichero a medias."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
