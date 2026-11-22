"""Prueba de épocas: una ejecución con más épocas (por defecto rel63_e10: 2 miembros, 10 épocas)
frente a los mismos miembros de su base (rel63, 5 épocas: miembros 0 y 1, mismas semillas), por
pares y bloque a bloque.

Para cada bloque y cada una: IC (media de h = 5 y 10), QLIKE (media de los tres horizontes) y MSE
de μ frente a μ = 0 entre activos a h = 5 y 10, fuera de muestra y en entrenamiento (con los
checkpoints, sobre la misma submuestra fija de ~200.000 muestras de entrenamiento por bloque que
night_summary), y L_μ de entrenamiento por época.

Uso: python scripts/epochs_compare.py --run rel63_e10 --base rel63 [--members 0 1]
     [--out logs/rel63_e10_compare.md]
"""

import argparse
import traceback
from pathlib import Path

import numpy as np

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import (
    H2,
    insample_stats,
    load_members,
    oos_stats,
    pooled_rel_mse,
    rel_mse,
    train_subsample,
)
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.train import ensemble


def pct(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{nd}f} %"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rel63_e10")
    ap.add_argument("--base", default="rel63")
    ap.add_argument("--members", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=None, help="por defecto logs/<run>_compare.md")
    args = ap.parse_args()
    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    labels = {"base": args.base, "run": args.run}
    stats, notes, used = {lab: {} for lab in labels}, [], {}

    for k, f in enumerate(folds):
        paths = {lab: DATA_DIR / "predictions" / r for lab, r in labels.items()}
        if not all((p / f"{f['name']}_members.parquet").exists() for p in paths.values()):
            continue
        try:
            mem = {lab: load_members(p, f["name"], args.members) for lab, p in paths.items()}
            common = sorted(set(mem["base"]) & set(mem["run"]))   # por pares: mismas semillas
            used[k] = common
            sub = train_subsample(data, f, k)
            for lab, r in labels.items():
                s = oos_stats(data, ensemble([mem[lab][m] for m in common]))
                s.update(insample_stats(data, [ROOT / "checkpoints" / r / f["name"] /
                                               f"member{m}.pt" for m in common], sub))
                stats[lab][k] = s
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"{f['name']}: error\n```\n{traceback.format_exc()}```")

    blocks = [k for k in stats["base"] if k in stats["run"]]
    b, r = stats["base"], stats["run"]
    n_ep = {lab: len(stats[lab][blocks[0]]["l_mu_epochs"]) if blocks else None for lab in labels}
    L = [(f"# Prueba de épocas: `{args.run}` ({n_ep['run']} épocas) frente a `{args.base}` "
         f"({n_ep['base']} épocas)"), "",
         ("Por pares: los mismos miembros (mismas semillas 1000·k + m) en las dos; ensemble de esos "
         "miembros. MSE entre activos: MSE de μ frente a μ = 0 restando a μ y a y su media diaria "
         "(negativo = mejor que 0)."), ""]
    if not blocks:
        L += ["Sin bloques comunes: no hay comparación.", ""]
    else:
        L.append(f"Miembros comparados: {sorted({m for k in blocks for m in used[k]})}.")
        L.append("")
        L += ["## Por bloque", "",
              ("| bloque | IC base | IC más épocas | QLIKE base | QLIKE más épocas | "
              "MSE ea fuera h=5 base | más | h=10 base | más | MSE ea entren. h=5 base | más | "
              "h=10 base | más |"), "|---|" + "---|" * 12]
        for k in blocks:
            cells = [f"{b[k]['ic']:.4f}", f"{r[k]['ic']:.4f}", f"{b[k]['qlike']:.4f}",
                     f"{r[k]['qlike']:.4f}"]
            for key in ("oos", "ins"):
                for h in H2:
                    cells += [pct(rel_mse(b[k][f"{key}_{h}"])), pct(rel_mse(r[k][f"{key}_{h}"]))]
            L.append(f"| {folds[k]['name']} | " + " | ".join(cells) + " |")
        L.append("")
        wins = sum(r[k]["ic"] > b[k]["ic"] for k in blocks)
        q_b, q_r = np.mean([b[k]["qlike"] for k in blocks]), np.mean([r[k]["qlike"] for k in blocks])
        L += ["## Resumen", "",
              "| | base | más épocas |", "|---|---|---|",
              (f"| IC medio | {np.mean([b[k]['ic'] for k in blocks]):.4f} | "
              f"{np.mean([r[k]['ic'] for k in blocks]):.4f} (gana en {wins}/{len(blocks)}) |"),
              f"| QLIKE media | {q_b:.4f} | {q_r:.4f} ({100 * (q_r / q_b - 1):+.1f} %) |"]
        for key, what in (("oos", "fuera de muestra"), ("ins", "entrenamiento")):
            for h in H2:
                L.append(f"| MSE entre activos {what}, h = {h} (bloques juntos) | "
                         f"{pct(pooled_rel_mse([b[k][f'{key}_{h}'] for k in blocks]))} | "
                         f"{pct(pooled_rel_mse([r[k][f'{key}_{h}'] for k in blocks]))} |")
        L.append(f"| s por época | {np.mean([b[k]['sec_epoch'] for k in blocks]):.0f} | "
                 f"{np.mean([r[k]['sec_epoch'] for k in blocks]):.0f} |")
        L.append("")
        L += ["## L_μ de entrenamiento por época (media de los miembros y de los bloques)", ""]
        for lab, name in (("base", args.base), ("run", args.run)):
            curve = np.mean([stats[lab][k]["l_mu_epochs"] for k in blocks], 0)
            L.append(f"- `{name}`: " + " · ".join(f"{x:.4f}" for x in curve) +
                     f" (de la primera a la última: {100 * (curve[-1] / curve[0] - 1):+.1f} %)")
        L.append("")
    L += ["## Notas", ""]
    L += [f"- {n}" for n in notes] or ["- Sin incidencias."]

    out = Path(args.out) if args.out else ROOT / "logs" / f"{args.run}_compare.md"
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nEscrito en {out}")


if __name__ == "__main__":
    main()
