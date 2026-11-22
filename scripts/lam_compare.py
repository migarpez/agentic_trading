"""Prueba de λ: compara cada ejecución (por defecto month_lam1 y month_lam9) por pares con los
mismos miembros de la referencia λ = 3 (coarse_month, miembros 0 y 1: mismas semillas), bloque a
bloque. Reutiliza compare.py y coarse_compare.py (cell, epoch_seconds) sin cambiarlos.

- IC (media de h = 5 y 10) por bloque y medio, y en cuántos bloques gana a λ = 3.
- QLIKE (media de h = 5, 10 y 20).
- Diferencia diaria de IC con la mejor línea base de cada bloque (momentum de 12 meses o reversión
  de 1 mes, la de mayor IC medio a ese horizonte), con su t de Newey-West (retardos = h).
- Diferencia diaria de IC con λ = 3, con los 5 bloques juntos (t NW).
- Correlación diaria de μ_5 con el momentum y con la reversión.
- L_μ y L_σ de la última época y segundos por época (de los checkpoints).
- Comprobación: las configuraciones solo difieren en λ.

Regla fijada antes de ver los resultados (referencia: λ = 3; manda el IC, porque la QLIKE mejora
siempre al subir λ):
a. Se descarta el λ cuya QLIKE sea más de un 2 % peor que la de λ = 3.
b. De los que quedan, se adopta el que supere el IC medio de λ = 3 en al menos 0,001 y le gane en
   IC en al menos 3 de los 5 bloques. Si lo cumplen los dos, el de mayor IC; si se llevan menos de
   0,001, el de menor QLIKE.
c. Si ninguno lo cumple, el de menor QLIKE entre los que no bajan el IC medio más de 0,001
   (λ = 3 incluido).

Uso: python scripts/lam_compare.py [--runs month_lam1 month_lam9] [--ref coarse_month]
     [--members 0 1] [--out logs/lam.md]
"""

import argparse
import json
import traceback

import numpy as np
import pandas as pd
import torch
from coarse_compare import cell, epoch_seconds

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import H2, ic_diff, load_members, targets
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import HORIZONS, baseline_features, daily_ic
from agentic_trading.model.train import ensemble
from agentic_trading.model.tuning import fold_scores

Q_MAX, IC_MIN, MIN_WINS = 0.02, 0.001, 3


def last_epoch_losses(run: str, fold: str, members) -> tuple[float, float]:
    """L_μ y L_σ de la última época (media de los miembros), del historial de los checkpoints."""
    l_mu, l_sig = [], []
    for m in members:
        ck = torch.load(ROOT / "checkpoints" / run / fold / f"member{m}.pt", weights_only=False)
        l_mu.append(ck["history"][-1]["l_mu"])
        l_sig.append(ck["history"][-1]["l_sigma"])
    return float(np.mean(l_mu)), float(np.mean(l_sig))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["month_lam1", "month_lam9"])
    ap.add_argument("--ref", default="coarse_month")
    ap.add_argument("--members", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=str(ROOT / "logs" / "lam.md"))
    args = ap.parse_args()
    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    pred = DATA_DIR / "predictions"
    notes, cfg_ok = [], True

    cfgs = {}
    for r in [args.ref, *args.runs]:
        try:
            cfgs[r] = json.loads((pred / r / "config.json").read_text())
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"{r}: sin config.json legible")
            cfgs[r] = {"train": {}}
    lam = {r: cfgs[r]["train"].get("lam") for r in cfgs}
    for r in args.runs:                       # solo pueden diferir en λ
        tr, ref = cfgs[r]["train"], cfgs[args.ref]["train"]
        diff = {k: (tr.get(k), ref.get(k)) for k in sorted(set(tr) | set(ref))
                if tr.get(k) != ref.get(k) and k != "lam"}
        for key in ("members", "dataset", "manual_fold"):
            if cfgs[r].get(key) != cfgs[args.ref].get(key):
                diff[key] = (cfgs[r].get(key), cfgs[args.ref].get(key))
        if diff or lam[r] == lam[args.ref]:
            cfg_ok = False
            notes.append(f"FALLO: {r} no difiere de {args.ref} solo en λ: {diff or 'mismo λ'}")
    have = {r: any((pred / r / f"{f['name']}_members.parquet").exists() for f in folds)
            for r in cfgs}
    for r, ok in have.items():
        if not ok:
            notes.append(f"FALLO: {r} no tiene predicciones; queda fuera de la comparación")
    labs = sorted((r for r in cfgs if have[r]), key=lambda r: (lam[r] is None, lam[r] or 0))
    runs = [r for r in args.runs if r in labs]                     # λ de menor a mayor en labs
    disp = {r: (f"λ = {lam[r]:g}" if lam[r] is not None else r)
            + (" (referencia)" if r == args.ref else "") for r in labs}

    res = {r: {} for r in labs}
    base_ic, used = {}, {}
    for k, f in enumerate(folds):
        name = f["name"]
        if not all((pred / r / f"{name}_members.parquet").exists() for r in labs):
            continue
        try:
            mem = {r: load_members(pred / r, name, args.members) for r in labs}
            common = sorted(set.intersection(*(set(m) for m in mem.values())))
            used[k] = common
            ens = {r: ensemble([m[i] for i in common]).sort_values("sample").reset_index(drop=True)
                   for r, m in mem.items()}
            idx = ens[args.ref]["sample"].to_numpy()
            for r, e in ens.items():
                assert np.array_equal(e["sample"].to_numpy(), idx), f"{r}: otras muestras"
            dates, rows, y = targets(data, idx, H2)
            bf = baseline_features(data, rows)
            base_ic[k] = {h: {"momentum": daily_ic(dates, bf["momentum"], y[h]),
                              "reversión": daily_ic(dates, bf["reversal"], y[h])} for h in H2}
            for r, e in ens.items():
                s = fold_scores(data, e)
                s["icd"] = {h: daily_ic(dates, e[f"mu_{h}"].to_numpy(), y[h]) for h in H2}
                mu5 = e["mu_5"].to_numpy()
                s["corr_mom"] = float(daily_ic(dates, mu5, bf["momentum"]).mean())
                s["corr_rev"] = float(daily_ic(dates, mu5, bf["reversal"]).mean())
                try:
                    s["sec"] = epoch_seconds(r, name, common)
                    s["l_mu"], s["l_sig"] = last_epoch_losses(r, name, common)
                except Exception:  # noqa: BLE001
                    s["sec"] = s["l_mu"] = s["l_sig"] = np.nan
                    notes.append(f"{r} {name}: sin checkpoints legibles")
                res[r][k] = s
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"{name}: error\n```\n{traceback.format_exc()}```")
            for v in res.values():
                v.pop(k, None)

    blocks = [k for k in range(len(folds)) if all(k in res[r] for r in labs)]
    bname = {k: f"{folds[k]['block'][0].year}–{folds[k]['block'][1].year}" for k in range(len(folds))}
    hl = {k: f"**{b}**" if b in ("2012–2015", "2016–2019") else b for k, b in bname.items()}
    L = ["# Prueba de λ con la configuración adoptada", "",
         ("Regla (fijada antes de ver los resultados; referencia λ = 3, y manda el IC porque la "
          "QLIKE mejora siempre al subir λ): a) se descarta el λ cuya QLIKE (media de h = 5, 10 y "
          f"20) sea más de un {100 * Q_MAX:.0f} % peor que la de λ = 3; b) de los que quedan, se "
          f"adopta el que supere el IC medio (media de h = 5 y 10) de λ = 3 en al menos {IC_MIN} "
          f"y le gane en IC en al menos {MIN_WINS} de los 5 bloques; si lo cumplen los dos, el de "
          f"mayor IC, y si se llevan menos de {IC_MIN}, el de menor QLIKE; c) si ninguno lo "
          f"cumple, el de menor QLIKE entre los que no bajan el IC medio más de {IC_MIN} (λ = 3 "
          "incluido)."), "",
         (f"Por pares: miembros {sorted({m for k in blocks for m in used[k]})} de cada ejecución "
          f"(mismas semillas 1000·k + m). Referencia: `{args.ref}`."), ""]
    t = cfgs[args.ref]["train"]
    L += [(f"Configuración: target {t.get('target')}, ventana de {t.get('window')}, velas "
           f"agregadas {t.get('coarse')}, lr = {t.get('lr')}, {t.get('epochs')} épocas, "
           f"{100 * t.get('epoch_frac', 0):.0f} % por época. Ejecuciones: "
           + ", ".join(f"`{r}` (λ = {lam[r]:g})" if lam[r] is not None else f"`{r}`"
                       for r in labs) + ". "
           + ("Solo difieren en λ (config.json)." if cfg_ok else
              "**No solo difieren en λ: ver Notas.**")), ""]

    if not blocks:
        L += ["Sin bloques comunes: no hay comparación.", ""]
    else:
        ic = {r: np.mean([res[r][k]["ic"] for k in blocks]) for r in labs}
        q = {r: np.mean([res[r][k]["qlike"] for k in blocks]) for r in labs}
        wins = {r: sum(res[r][k]["ic"] > res[args.ref][k]["ic"] for k in blocks)
                for r in runs}

        L += ["## IC (media de h = 5 y 10)", "",
              "| bloque | " + " | ".join(disp[r] for r in labs) + " |",
              "|---|" + "---|" * len(labs)]
        for k in blocks:
            L.append(f"| {hl[k]} | " + " | ".join(
                f"{res[r][k]['ic']:.4f}" + ("" if r == args.ref else
                                            f" ({res[r][k]['ic'] - res[args.ref][k]['ic']:+.4f})")
                for r in labs) + " |")
        L.append("| media | " + " | ".join(
            f"{ic[r]:.4f}" + ("" if r == args.ref else
                              f" ({ic[r] - ic[args.ref]:+.4f}; gana en {wins[r]}/{len(blocks)})")
            for r in labs) + " |")
        L += ["", "Entre paréntesis, la diferencia con λ = 3.", ""]

        L += ["## QLIKE", "", "Media de h = 5, 10 y 20, por bloque.", "",
              "| bloque | " + " | ".join(disp[r] for r in labs) + " |",
              "|---|" + "---|" * len(labs)]
        for k in blocks:
            L.append(f"| {hl[k]} | " + " | ".join(f"{res[r][k]['qlike']:.4f}" for r in labs) + " |")
        L.append("| media | " + " | ".join(
            f"{q[r]:.4f}" + ("" if r == args.ref else f" ({100 * (q[r] / q[args.ref] - 1):+.1f} %)")
            for r in labs) + " |")
        L += ["", "Por horizonte (media de los bloques):", "",
              "| h | " + " | ".join(disp[r] for r in labs) + " |", "|---|" + "---|" * len(labs)]
        for h in HORIZONS:
            qh = {r: np.mean([res[r][k][f"qlike_{h}"] for k in blocks]) for r in labs}
            L.append(f"| {h} | " + " | ".join(
                f"{qh[r]:.4f}" + ("" if r == args.ref else
                                  f" ({100 * (qh[r] / qh[args.ref] - 1):+.1f} %)") for r in labs)
                + " |")
        L.append("")

        L += ["## Frente a la mejor línea base de cada bloque", "",
              ("Diferencia diaria de IC con la mejor línea base del bloque a ese horizonte (la de "
               "mayor IC medio), con su t de Newey-West (retardos = h)."), ""]
        for h in H2:
            L += [f"h = {h}:", "",
                  "| bloque | mejor línea base (IC) | " + " | ".join(disp[r] for r in labs) + " |",
                  "|---|---|" + "---|" * len(labs)]
            for k in blocks:
                best = max(base_ic[k][h], key=lambda b, k=k, h=h: base_ic[k][h][b].mean())
                L.append(f"| {hl[k]} | {best} ({base_ic[k][h][best].mean():.4f}) | " + " | ".join(
                    cell(*ic_diff(res[r][k]["icd"][h], base_ic[k][h][best], h)[:2]) for r in labs)
                    + " |")
            pooled = {b: pd.concat([base_ic[k][h][b] for k in blocks]) for b in base_ic[blocks[0]][h]}
            best = max(pooled, key=lambda b: pooled[b].mean())
            L += [f"| {len(blocks)} bloques juntos | {best} ({pooled[best].mean():.4f}) | "
                  + " | ".join(cell(*ic_diff(pd.concat([res[r][k]["icd"][h] for k in blocks]),
                                             pooled[best], h)[:2]) for r in labs) + " |", ""]

        L += ["## Diferencia diaria de IC con λ = 3", "",
              f"Con los {len(blocks)} bloques juntos (t NW):", "",
              "| | h = 5 | h = 10 |", "|---|---|---|"]
        for r in runs:
            L.append(f"| {disp[r]} | " + " | ".join(
                cell(*ic_diff(pd.concat([res[r][k]["icd"][h] for k in blocks]),
                              pd.concat([res[args.ref][k]["icd"][h] for k in blocks]), h)[:2])
                for h in H2) + " |")
        L.append("")

        L += ["## Correlación de μ_5 con las líneas base", "",
              "Spearman diaria (media) de μ_5 con el momentum de 12 meses / con la reversión de 1 mes.",
              "", "| bloque | " + " | ".join(disp[r] for r in labs) + " |",
              "|---|" + "---|" * len(labs)]
        for k in blocks:
            L.append(f"| {hl[k]} | " + " | ".join(
                f"{res[r][k]['corr_mom']:+.2f} / {res[r][k]['corr_rev']:+.2f}" for r in labs) + " |")
        L.append("| media | " + " | ".join(
            f"{np.mean([res[r][k]['corr_mom'] for k in blocks]):+.2f} / "
            f"{np.mean([res[r][k]['corr_rev'] for k in blocks]):+.2f}" for r in labs) + " |")
        L.append("")

        L += ["## Entrenamiento", "",
              "Media de los miembros y de los bloques, de los checkpoints.", "",
              "| | " + " | ".join(disp[r] for r in labs) + " |", "|---|" + "---|" * len(labs)]
        for key, what, fmt in (("l_mu", "L_μ, última época", "{:.4f}"),
                               ("l_sig", "L_σ, última época", "{:.4f}"),
                               ("sec", "segundos por época", "{:.0f}")):
            L.append(f"| {what} | " + " | ".join(
                fmt.format(np.nanmean([res[r][k][key] for k in blocks])) for r in labs) + " |")
        L += ["", ("L = L_μ + λ·L_σ. Los segundos son de ejecuciones de momentos distintos: la "
                   "comparación es aproximada."), ""]

        L += ["## Veredicto", ""]
        d_q = {r: q[r] / q[args.ref] - 1 for r in runs}
        d_ic = {r: ic[r] - ic[args.ref] for r in runs}
        kept = [r for r in runs if d_q[r] <= Q_MAX]
        for r in runs:
            L.append(f"- a) {disp[r]}: QLIKE {100 * d_q[r]:+.2f} % frente a λ = 3 → "
                     + ("sigue." if r in kept else f"se descarta (más de un {100 * Q_MAX:.0f} % peor)."))
        passed = [r for r in kept if d_ic[r] >= IC_MIN and wins[r] >= MIN_WINS]
        for r in kept:
            L.append(f"- b) {disp[r]}: IC medio {d_ic[r]:+.4f}, gana en {wins[r]} de "
                     f"{len(blocks)} bloques → " + ("cumple." if r in passed else "no cumple."))
        if len(blocks) < len(folds):
            L += ["", f"**Regla no aplicable: solo {len(blocks)} de {len(folds)} bloques.**"]
        elif not cfg_ok:
            L += ["", "**Regla no aplicable: las configuraciones no solo difieren en λ (Notas).**"]
        else:
            if len(passed) == 1:
                pick, why = passed[0], "b), el único que cumple"
            elif len(passed) > 1:
                top = sorted(passed, key=lambda r: ic[r], reverse=True)
                if ic[top[0]] - ic[top[1]] < IC_MIN:
                    pick = min(top[:2], key=lambda r: q[r])
                    why = f"b), cumplen los dos y se llevan menos de {IC_MIN}: el de menor QLIKE"
                else:
                    pick, why = top[0], "b), cumplen los dos: el de mayor IC"
            else:
                cand = [args.ref] + [r for r in kept if d_ic[r] >= -IC_MIN]
                pick = min(cand, key=lambda r: q[r])
                why = ("c), ninguno cumple b): el de menor QLIKE entre los que no bajan el IC "
                       f"medio más de {IC_MIN} (" + ", ".join(disp[r] for r in cand) + ")")
            L += ["", f"**Veredicto: {disp[pick].replace(' (referencia)', '')} ({why}).**"]
        L.append("")
    L += ["## Notas", ""] + ([f"- {n}" for n in notes] or ["- Sin incidencias."])
    text = "\n".join(L) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"Escrito en {args.out}")


if __name__ == "__main__":
    main()
