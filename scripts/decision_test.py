"""Prueba de decisión para μ con el ensemble de una ejecución de validación cruzada (por defecto
rel63: 10 miembros, target relativo, ventana de 63).

Regla (fijada antes de ver los resultados): μ funciona si en 2016–19 (cv4), a h = 5 o 10, la
diferencia diaria de IC entre el ensemble completo y la reversión de 1 mes es positiva con t de
Newey-West > 2 (retardos = h). También se da por bloque y con los 5 bloques juntos, pero la
decisión es la de 2016–19.

Incluye el IC (media de h = 5 y 10, como en el ajuste) del ensemble de los k primeros miembros
(k = 1, 2, 5 y 10) en cada bloque.

Opciones (sin ellas, lo de arriba): --decision-members N toma la decisión con el ensemble de los N
primeros miembros (la regla del 8-11-2026 es con 10); --also k da también, a título informativo,
la decisión con los k primeros (p. ej. 20); --ks cambia los k de la tabla del IC.

Uso: python scripts/decision_test.py --run rel63 [--decision-fold 4] [--out logs/rel63_decision.md]
     [--decision-members 10] [--also 20] [--ks 1 2 5 10 20]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import H2, ic_diff, load_members, targets
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import baseline_features, daily_ic, nw_tstat
from agentic_trading.model.train import ensemble
from agentic_trading.model.tuning import fold_scores

KS = (1, 2, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rel63")
    ap.add_argument("--decision-fold", type=int, default=4, help="bloque de la decisión (2016–19)")
    ap.add_argument("--out", default=None, help="por defecto logs/<run>_decision.md")
    ap.add_argument("--decision-members", type=int, default=None,
                    help="decisión con el ensemble de los N primeros miembros (por defecto, todos)")
    ap.add_argument("--also", type=int, nargs="*", default=[],
                    help="también, informativo, la decisión con los k primeros miembros")
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS),
                    help="k de la tabla del IC de los k primeros miembros")
    args = ap.parse_args()
    ks = tuple(args.ks)
    pr = DATA_DIR / "predictions" / args.run
    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)

    blocks, ks_ic, n_members, n_dec, also = {}, {}, {}, {}, {a: {} for a in args.also}
    for k, f in enumerate(folds):
        if not (pr / f"{f['name']}_members.parquet").exists():
            continue
        mem = list(load_members(pr, f["name"]).values())
        n_members[k] = len(mem)
        ks_ic[k] = {kk: fold_scores(data, ensemble(mem[:kk]))["ic"] for kk in ks if kk <= len(mem)}
        n_dec[k] = len(mem) if args.decision_members is None else min(args.decision_members,
                                                                        len(mem))
        full = ensemble(mem[:n_dec[k]])
        dates, rows, y = targets(data, full["sample"].to_numpy())
        rev = baseline_features(data, rows)["reversal"]
        blocks[k] = {h: {"model": daily_ic(dates, full[f"mu_{h}"].to_numpy(), y[h]),
                         "rev": daily_ic(dates, rev, y[h])} for h in H2}
        for a in args.also:
            if a <= len(mem):
                e = ensemble(mem[:a])
                also[a][k] = {h: {"model": daily_ic(dates, e[f"mu_{h}"].to_numpy(), y[h]),
                                  "rev": blocks[k][h]["rev"]} for h in H2}
    if not blocks:
        raise SystemExit(f"No hay predicciones en {pr}")

    def row(name, ics_m, ics_r, h):
        m_ic, m_t = nw_tstat(ics_m.to_numpy(), h)
        mean, t, days = ic_diff(ics_m, ics_r, h)
        return (f"| {name} | {h} | {m_ic:.4f} ({m_t:.1f}) | {ics_r.mean():.4f} | {mean:+.4f} | "
                f"{t:+.2f} | {days} |"), mean, t

    dk = args.decision_fold
    which = ("el ensemble completo" if args.decision_members is None else
             f"el ensemble de los {args.decision_members} primeros miembros")
    L = [f"# Prueba de decisión para μ: `{args.run}`", "",
         ("Regla (fijada antes de ver los resultados): μ funciona si en 2016–19, a h = 5 o 10, la "
         f"diferencia diaria de IC entre {which} y la reversión de 1 mes es positiva "
         "con t de Newey-West > 2 (retardos = h)."), ""]
    verdict = None
    if dk in blocks:
        verdict = False
        L += [(f"## Decisión: {folds[dk]['name']} {folds[dk]['block'][0].year}–"
              f"{folds[dk]['block'][1].year} (ensemble de {n_dec[dk]})"), "",
              "| bloque | h | IC modelo (t) | IC reversión | diferencia | t NW | días |",
              "|---|---|---|---|---|---|---|"]
        for h in H2:
            line, mean, t = row(folds[dk]["name"], blocks[dk][h]["model"], blocks[dk][h]["rev"], h)
            verdict |= bool(mean > 0 and t > 2)
            L.append(line)
        L += ["", (f"**Veredicto: μ {'FUNCIONA' if verdict else 'NO FUNCIONA'}** "
              f"(regla: diferencia > 0 con t > 2 a h = 5 o 10 en {folds[dk]['name']})."), ""]
        if args.decision_members is not None and n_dec[dk] < args.decision_members:
            L += [f"Aviso: solo hay {n_members[dk]} miembros; la decisión usa {n_dec[dk]}.", ""]
        for a, res in also.items():
            if dk not in res:
                L += [f"También con {a} miembros: no hay tantos ({n_members[dk]}).", ""]
                continue
            L += [f"### También: ensemble de los {a} primeros miembros (informativo)", "",
                  "| bloque | h | IC modelo (t) | IC reversión | diferencia | t NW | días |",
                  "|---|---|---|---|---|---|---|"]
            ok = False
            for h in H2:
                line, mean, t = row(folds[dk]["name"], res[dk][h]["model"], res[dk][h]["rev"], h)
                ok |= bool(mean > 0 and t > 2)
                L.append(line)
            L += ["", f"Con {a} miembros, la regla daría: μ {'FUNCIONA' if ok else 'NO FUNCIONA'}.",
                  ""]
    else:
        L += [f"Falta el bloque de la decisión ({folds[dk]['name']}): sin veredicto.", ""]

    L += ["## Por bloque y con los 5 bloques juntos (informativo)", "",
          "| bloque | h | IC modelo (t) | IC reversión | diferencia | t NW | días |",
          "|---|---|---|---|---|---|---|"]
    for h in H2:
        for k in blocks:
            f = folds[k]
            name = f"{f['name']} {f['block'][0].year}–{f['block'][1].year}"
            L.append(row(name, blocks[k][h]["model"], blocks[k][h]["rev"], h)[0])
        pooled_m = pd.concat([blocks[k][h]["model"] for k in blocks])
        pooled_r = pd.concat([blocks[k][h]["rev"] for k in blocks])
        L.append(row(f"**{len(blocks)} bloques juntos**", pooled_m, pooled_r, h)[0])
    L.append("")

    L += ["## IC del ensemble de los k primeros miembros (media de h = 5 y 10)", "",
          "| bloque | " + " | ".join(f"k = {kk}" for kk in ks) + " |",
          "|---|" + "---|" * len(ks)]
    for k in blocks:
        L.append(f"| {folds[k]['name']} | " + " | ".join(
            f"{ks_ic[k][kk]:.4f}" if kk in ks_ic[k] else "—" for kk in ks) + " |")
    L.append("| media | " + " | ".join(
        f"{np.mean([ks_ic[k][kk] for k in blocks]):.4f}"
        if all(kk in ks_ic[k] for k in blocks) else "—" for kk in ks) + " |")
    L.append("")

    out = Path(args.out) if args.out else ROOT / "logs" / f"{args.run}_decision.md"
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nEscrito en {out}")


if __name__ == "__main__":
    main()
