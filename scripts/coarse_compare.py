"""Prueba de entrada en dos escalas: compara cada ejecución con velas agregadas (por defecto
coarse_month y coarse_week) por pares con los mismos miembros de la referencia (rel63, miembros 0
y 1: mismas semillas, ventana de 63 sin velas agregadas), bloque a bloque. Para σ, también
night_rel (ventana de 252, miembros 0 y 1) como referencia aproximada.

- IC (media de h = 5 y 10), por bloque y medio, y en cuántos bloques gana a la referencia.
- Por bloque, diferencia diaria de IC con la mejor línea base del bloque (momentum de 12 meses o
  reversión de 1 mes, la de mayor IC medio a ese horizonte), con su t de Newey-West (retardos = h).
- QLIKE (media de h = 5, 10 y 20) y segundos por época (de los checkpoints).
- Extra, fuera de la regla: diferencia diaria de IC con la referencia (por pares) y correlación
  diaria de μ_5 con el momentum y con la reversión.

Regla fijada antes de ver los resultados: una opción se adopta si sube el IC medio, gana en al
menos 3 de los 5 bloques y la QLIKE no empeora. Si las dos lo cumplen, se adopta la de mayor IC;
si la diferencia es menor de 0,001, la mensual, que es más simple.

Uso: python scripts/coarse_compare.py [--runs coarse_month coarse_week] [--base rel63]
     [--members 0 1] [--sigma-ref night_rel] [--out logs/coarse.md]
"""

import argparse
import json
import traceback

import numpy as np
import pandas as pd
import torch

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import H2, ic_diff, load_members, targets
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import HORIZONS, baseline_features, daily_ic
from agentic_trading.model.train import ensemble
from agentic_trading.model.tuning import fold_scores

KIND = {"month": "mensual", "week": "semanal"}
MIN_WINS, IC_TIE = 3, 0.001


def epoch_seconds(run: str, fold: str, members) -> float:
    secs = []
    for m in members:
        ck = torch.load(ROOT / "checkpoints" / run / fold / f"member{m}.pt", weights_only=False)
        secs += [r["seconds"] for r in ck["history"]]
    return float(np.mean(secs))


def cell(d: float, t: float) -> str:
    return f"{d:+.4f} ({t:.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["coarse_month", "coarse_week"])
    ap.add_argument("--base", default="rel63")
    ap.add_argument("--members", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--sigma-ref", default="night_rel")
    ap.add_argument("--out", default=str(ROOT / "logs" / "coarse.md"))
    args = ap.parse_args()
    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    pred = DATA_DIR / "predictions"
    runs = {"ref": args.base, **{r: r for r in args.runs}, "sig": args.sigma_ref}
    main_labs = ["ref", *args.runs]
    notes = []

    cfgs, kind = {}, {}
    for lab, r in runs.items():
        try:
            cfgs[lab] = json.loads((pred / r / "config.json").read_text())
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"{r}: sin config.json legible")
            cfgs[lab] = {"train": {}}
    for r in args.runs:
        kind[r] = cfgs[r]["train"].get("coarse")
        tr, ref = cfgs[r]["train"], cfgs["ref"]["train"]
        diff = {k: (tr.get(k), ref.get(k)) for k in sorted(set(tr) | set(ref))
                if tr.get(k) != ref.get(k) and k != "coarse"}
        if diff:
            notes.append(f"{r}: configuración distinta de la de {args.base} (además de coarse): "
                         f"{diff}")
        if cfgs[r].get("dataset") != cfgs["ref"].get("dataset"):
            notes.append(f"{r}: dataset distinto del de {args.base}")

    res = {lab: {} for lab in runs}
    base_ic, used = {}, {}
    for k, f in enumerate(folds):
        name = f["name"]
        have = {lab: (pred / r / f"{name}_members.parquet").exists() for lab, r in runs.items()}
        if not all(have[lab] for lab in main_labs):
            continue
        try:
            mem = {lab: load_members(pred / r, name, args.members)
                   for lab, r in runs.items() if have[lab]}
            common = sorted(set.intersection(*(set(mem[lab]) for lab in main_labs)))
            used[k] = common
            ens = {lab: ensemble([m[i] for i in (common if lab != "sig" else sorted(m))])
                   .sort_values("sample").reset_index(drop=True) for lab, m in mem.items()}
            idx = ens["ref"]["sample"].to_numpy()
            for lab, e in ens.items():
                assert np.array_equal(e["sample"].to_numpy(), idx), f"{runs[lab]}: otras muestras"
            dates, rows, y = targets(data, idx, H2)
            bf = baseline_features(data, rows)
            base_ic[k] = {h: {"momentum": daily_ic(dates, bf["momentum"], y[h]),
                              "reversión": daily_ic(dates, bf["reversal"], y[h])} for h in H2}
            for lab, e in ens.items():
                r = fold_scores(data, e)
                r["icd"] = {h: daily_ic(dates, e[f"mu_{h}"].to_numpy(), y[h]) for h in H2}
                mu5 = e["mu_5"].to_numpy()
                r["corr_mom"] = float(daily_ic(dates, mu5, bf["momentum"]).mean())
                r["corr_rev"] = float(daily_ic(dates, mu5, bf["reversal"]).mean())
                mem_lab = common if lab != "sig" else sorted(mem[lab])
                try:
                    r["sec"] = epoch_seconds(runs[lab], name, mem_lab)
                except Exception:  # noqa: BLE001
                    r["sec"] = np.nan
                    notes.append(f"{runs[lab]} {name}: sin checkpoints legibles")
                res[lab][k] = r
            if not have["sig"]:
                notes.append(f"{args.sigma_ref}: falta {name}")
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"{name}: error\n```\n{traceback.format_exc()}```")
            for r in res.values():
                r.pop(k, None)

    blocks = [k for k in res["ref"] if all(k in res[r] for r in args.runs)]
    bname = {k: f"{folds[k]['block'][0].year}–{folds[k]['block'][1].year}" for k in range(len(folds))}
    hl = {k: f"**{bname[k]}**" if bname[k] == "2012–2015" else bname[k] for k in bname}
    disp = {"ref": f"{args.base} (ventana de 63)", "sig": f"{args.sigma_ref} (ventana de 252)",
            **{r: f"{r} ({KIND.get(kind[r], kind[r])})" for r in args.runs}}

    L = ["# Prueba de entrada en dos escalas: velas agregadas del año entero", "",
         ("Regla (fijada antes de ver los resultados): una opción se adopta si sube el IC medio "
          f"(media de h = 5 y 10), gana en al menos {MIN_WINS} de los 5 bloques y la QLIKE (media "
          "de h = 5, 10 y 20) no empeora. Si las dos lo cumplen, la de mayor IC; si la diferencia "
          f"es menor de {IC_TIE}, la mensual, que es más simple."), "",
         (f"Por pares: miembros {sorted({m for k in blocks for m in used[k]})} de cada ejecución "
          "(mismas semillas 1000·k + m), ensemble de esos miembros. Referencia: "
          f"`{args.base}`. Para σ, también `{args.sigma_ref}` (miembros 0 y 1), como referencia "
          "aproximada."), ""]
    t = cfgs["ref"]["train"]
    if t:
        L += [(f"Configuración de la referencia (las diferencias de las demás, además de coarse, "
               f"en Notas): target {t.get('target')}, ventana de {t.get('window')}, "
               f"λ = {t.get('lam')}, lr = {t.get('lr')}, {t.get('epochs')} épocas, "
               f"{100 * t.get('epoch_frac', 0):.0f} % por época."), ""]
    if not blocks:
        L += ["Sin bloques comunes: no hay comparación.", ""]
    else:
        ic = {lab: np.mean([res[lab][k]["ic"] for k in blocks]) for lab in res if
              all(k in res[lab] for k in blocks)}
        q = {lab: np.mean([res[lab][k]["qlike"] for k in blocks]) for lab in ic}
        wins = {r: sum(res[r][k]["ic"] > res["ref"][k]["ic"] for k in blocks) for r in args.runs}

        L += ["## IC (media de h = 5 y 10)", "",
              "| bloque | " + " | ".join(disp[lab] for lab in main_labs) + " |",
              "|---|" + "---|" * len(main_labs)]
        for k in blocks:
            cells = [f"{res['ref'][k]['ic']:.4f}"]
            cells += [f"{res[r][k]['ic']:.4f} ({res[r][k]['ic'] - res['ref'][k]['ic']:+.4f})"
                      for r in args.runs]
            L.append(f"| {hl[k]} | " + " | ".join(cells) + " |")
        L.append(f"| media | {ic['ref']:.4f} | " + " | ".join(
            f"{ic[r]:.4f} ({ic[r] - ic['ref']:+.4f}; gana en {wins[r]}/{len(blocks)})"
            for r in args.runs) + " |")
        L += ["", "Entre paréntesis, la diferencia con la referencia.", ""]

        L += ["## Frente a la mejor línea base de cada bloque", "",
              ("Diferencia diaria de IC con la mejor línea base del bloque a ese horizonte (la de "
               "mayor IC medio), con su t de Newey-West (retardos = h)."), ""]
        for h in H2:
            L += [f"h = {h}:", "",
                  "| bloque | mejor línea base (IC) | " + " | ".join(disp[lab] for lab in main_labs)
                  + " |", "|---|---|" + "---|" * len(main_labs)]
            for k in blocks:
                best = max(base_ic[k][h], key=lambda b, k=k, h=h: base_ic[k][h][b].mean())
                cells = [cell(*ic_diff(res[lab][k]["icd"][h], base_ic[k][h][best], h)[:2])
                         for lab in main_labs]
                L.append(f"| {hl[k]} | {best} ({base_ic[k][h][best].mean():.4f}) | "
                         + " | ".join(cells) + " |")
            pooled = {b: pd.concat([base_ic[k][h][b] for k in blocks]) for b in base_ic[blocks[0]][h]}
            best = max(pooled, key=lambda b: pooled[b].mean())
            cells = [cell(*ic_diff(pd.concat([res[lab][k]["icd"][h] for k in blocks]),
                                   pooled[best], h)[:2]) for lab in main_labs]
            L += [f"| {len(blocks)} bloques juntos | {best} ({pooled[best].mean():.4f}) | "
                  + " | ".join(cells) + " |", ""]

        sig_ok = "sig" in ic
        q_labs = main_labs + (["sig"] if sig_ok else [])
        L += ["## QLIKE", "", "Media de h = 5, 10 y 20, por bloque.", "",
              "| bloque | " + " | ".join(disp[lab] for lab in q_labs) + " |",
              "|---|" + "---|" * len(q_labs)]
        for k in blocks:
            L.append(f"| {hl[k]} | " + " | ".join(f"{res[lab][k]['qlike']:.4f}" for lab in q_labs)
                     + " |")
        L.append("| media | " + " | ".join(
            f"{q[lab]:.4f}" + ("" if lab == "ref" else f" ({100 * (q[lab] / q['ref'] - 1):+.1f} %)")
            for lab in q_labs) + " |")
        L += ["", "Por horizonte (media de los bloques):", "",
              "| h | " + " | ".join(disp[lab] for lab in q_labs) + " |",
              "|---|" + "---|" * len(q_labs)]
        for h in HORIZONS:
            qh = {lab: np.mean([res[lab][k][f"qlike_{h}"] for k in blocks]) for lab in q_labs}
            L.append(f"| {h} | " + " | ".join(
                f"{qh[lab]:.4f}" + ("" if lab == "ref" else
                                    f" ({100 * (qh[lab] / qh['ref'] - 1):+.1f} %)")
                for lab in q_labs) + " |")
        L.append("")

        L += ["## Tiempo por época", "",
              "| | " + " | ".join(disp[lab] for lab in q_labs) + " |",
              "|---|" + "---|" * len(q_labs),
              "| segundos por época (media de miembros y bloques) | " + " | ".join(
                  f"{np.nanmean([res[lab][k]['sec'] for k in blocks]):.0f}" for lab in q_labs)
              + " |", "",
              ("De los checkpoints; las ejecuciones son de momentos distintos, así que la "
               "comparación es aproximada."), ""]

        L += ["## Veredicto", ""]
        ok = {}
        for r in args.runs:
            d_ic, d_q = ic[r] - ic["ref"], 100 * (q[r] / q["ref"] - 1)
            ok[r] = d_ic > 0 and wins[r] >= MIN_WINS and q[r] <= q["ref"]
            L.append(f"- `{r}`: IC medio {d_ic:+.4f}, gana en {wins[r]} de {len(blocks)} bloques, "
                     f"QLIKE {d_q:+.2f} % → {'cumple' if ok[r] else 'no cumple'}.")
        L.append("")
        if len(blocks) < len(folds):
            L.append(f"**Regla no aplicable: solo {len(blocks)} de {len(folds)} bloques.**")
        else:
            passed = [r for r in args.runs if ok[r]]
            if not passed:
                verdict = "no se adopta ninguna"
            elif len(passed) == 1:
                verdict = f"se adopta `{passed[0]}`"
            else:
                top = max(passed, key=lambda r: ic[r])
                month = [r for r in passed if kind[r] == "month"]
                gap = abs(ic[passed[0]] - ic[passed[1]])
                pick = month[0] if gap < IC_TIE and month else top
                verdict = (f"se adopta `{pick}` (cumplen las dos; diferencia de IC {gap:.4f}"
                           + (f" < {IC_TIE}: la mensual)" if gap < IC_TIE else ")"))
            L.append(f"**Veredicto: {verdict}.**")
        L.append("")

        L += ["## Extra (no forma parte de la regla)", "",
              "Diferencia diaria de IC con la referencia, por pares (t NW):", "",
              "| bloque | " + " | ".join(f"{r}, h = {h}" for r in args.runs for h in H2) + " |",
              "|---|" + "---|" * (len(args.runs) * len(H2))]
        for k in blocks:
            L.append(f"| {hl[k]} | " + " | ".join(
                cell(*ic_diff(res[r][k]["icd"][h], res["ref"][k]["icd"][h], h)[:2])
                for r in args.runs for h in H2) + " |")
        L.append("| " + f"{len(blocks)} bloques juntos | " + " | ".join(
            cell(*ic_diff(pd.concat([res[r][k]["icd"][h] for k in blocks]),
                          pd.concat([res["ref"][k]["icd"][h] for k in blocks]), h)[:2])
            for r in args.runs for h in H2) + " |")
        L += ["", ("Correlación diaria (Spearman) de μ_5 con el momentum de 12 meses / con la "
                   "reversión de 1 mes:"), "",
              "| bloque | " + " | ".join(disp[lab] for lab in q_labs) + " |",
              "|---|" + "---|" * len(q_labs)]
        for k in blocks:
            L.append(f"| {hl[k]} | " + " | ".join(
                f"{res[lab][k]['corr_mom']:+.2f} / {res[lab][k]['corr_rev']:+.2f}"
                for lab in q_labs) + " |")
        L.append("| media | " + " | ".join(
            f"{np.mean([res[lab][k]['corr_mom'] for k in blocks]):+.2f} / "
            f"{np.mean([res[lab][k]['corr_rev'] for k in blocks]):+.2f}" for lab in q_labs) + " |")
        L += ["", "IC por horizonte (media de los bloques):", "",
              "| h | " + " | ".join(disp[lab] for lab in q_labs) + " |",
              "|---|" + "---|" * len(q_labs)]
        for h in H2:
            L.append(f"| {h} | " + " | ".join(
                f"{np.mean([res[lab][k][f'ic_{h}'] for k in blocks]):.4f}" for lab in q_labs)
                + " |")
        L.append("")
    L += ["## Notas", ""] + ([f"- {n}" for n in notes] or ["- Sin incidencias."])
    text = "\n".join(L) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"Escrito en {args.out}")


if __name__ == "__main__":
    main()
