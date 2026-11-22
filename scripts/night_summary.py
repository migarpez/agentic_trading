"""Resumen de las pruebas de la noche sobre μ (scripts/night.py) frente a la referencia del ajuste.

Referencia: la configuración "lam=3 lr=0.001" de data/predictions/tune/results.parquet (target
absoluto, MSE, ventana de 252; mismas semillas, así que la comparación es por pares). El ajuste no
guardó predicciones ni modelos; para el MSE en 2016–19 se usa ens10, cuyos miembros 0 y 1 son la
referencia (misma semilla y configuración; reproducen su IC).

Por prueba:
- IC (media de h = 5 y 10) por bloque y medio, y en cuántos bloques gana a la referencia.
- QLIKE media (de los tres horizontes) frente a la de la referencia.
- MSE de μ frente a μ = 0 entre activos (restando la media diaria a μ y a y), a h = 5 y 10, fuera
  de muestra y en entrenamiento (con los checkpoints, sobre una submuestra fija de ~200.000 muestras
  de entrenamiento por bloque: fechas completas al azar, semilla = nº de bloque).
- Diferencia diaria de IC con la reversión de 1 mes, con los 5 bloques juntos, con t de
  Newey-West (retardos = h).
- La prueba de ventana corta frente a su base, bloque a bloque: IC, QLIKE, MSE entre activos y
  tiempo por época.

Criterio (fijado antes de ver los resultados): el target relativo arregla la memoria si
1) el MSE entre activos fuera de muestra es < 0 a h = 5 y 10 (5 bloques juntos),
2) el IC gana a la referencia en al menos 4 de los 5 bloques y
3) la QLIKE media no empeora más de un 2 %.

Uso: python scripts/night_summary.py [--runs night_rel night_rel_huber night_w63]
     [--out logs/night_summary.md]
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import (
    H2,
    insample_stats,
    oos_stats,
    targets,
    train_subsample,
)
from agentic_trading.model.compare import rel_mse as rel
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import baseline_features, daily_ic, nw_tstat
from agentic_trading.model.train import ensemble

REF_CONFIG = "lam=3 lr=0.001"
REF_CV4_RUN, REF_CV4_MEMBERS = "ens10", (0, 1)


def fmt(x, nd=4, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f} %" if pct else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=3, default=["night_rel", "night_rel_huber", "night_w63"],
                    metavar=("A", "B", "C"))
    ap.add_argument("--out", default=str(ROOT / "logs" / "night_summary.md"))
    args = ap.parse_args()
    notes: list[str] = []

    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    names = [f["name"] for f in folds]
    subs = {k: train_subsample(data, f, k) for k, f in enumerate(folds)}

    tune = pd.read_parquet(DATA_DIR / "predictions" / "tune" / "results.parquet")
    ref = tune[tune.config == REF_CONFIG].set_index("fold")
    rev_ic = {}                                  # IC diario de la reversión por bloque y h
    for k, f in enumerate(folds):
        dates, rows, y = targets(data, f["predict"])
        r = baseline_features(data, rows)["reversal"]
        rev_ic[k] = {h: daily_ic(dates, r, y[h]) for h in H2}

    # --- pruebas
    runs = {}
    for label, run in zip("abc", args.runs):
        pr, ck = DATA_DIR / "predictions" / run, ROOT / "checkpoints" / run
        cfg_path = pr / "config.json"
        info = {"run": run, "blocks": {},
                "config": json.loads(cfg_path.read_text())["train"] if cfg_path.exists() else None}
        for k, f in enumerate(folds):
            p = pr / f"{f['name']}.parquet"
            if not p.exists():
                notes.append(f"{label} ({run}): falta {f['name']} (no terminó o falló)")
                continue
            try:
                b = oos_stats(data, pd.read_parquet(p))
                members = sorted((ck / f["name"]).glob("member*.pt"))
                if members:
                    b.update(insample_stats(data, members, subs[k]))
                else:
                    notes.append(f"{label} ({run}): sin checkpoints en {f['name']}")
                info["blocks"][k] = b
            except Exception:  # noqa: BLE001 — se apunta y se sigue
                notes.append(f"{label} ({run}) {f['name']}: error\n```\n{traceback.format_exc()}```")
        runs[label] = info

    # --- referencia en 2016–19 con ens10 (miembros 0 y 1)
    ref4 = None
    try:
        mem = pd.read_parquet(DATA_DIR / "predictions" / REF_CV4_RUN / "cv4_members.parquet")
        preds = [mem[mem.member == m].drop(columns="member").reset_index(drop=True)
                 for m in REF_CV4_MEMBERS]
        ref4 = oos_stats(data, ensemble(preds))
        ref4.update(insample_stats(data, [ROOT / "checkpoints" / REF_CV4_RUN / "cv4" /
                                          f"member{m}.pt" for m in REF_CV4_MEMBERS], subs[4]))
        if abs(ref4["ic"] - ref.loc["cv4", "ic"]) > 5e-4:
            notes.append(f"ens10 m0–1 no reproduce la referencia en cv4: IC {ref4['ic']:.4f} "
                         f"frente a {ref.loc['cv4', 'ic']:.4f}")
    except Exception:  # noqa: BLE001 — se apunta y se sigue
        notes.append(f"referencia cv4 (ens10): error\n```\n{traceback.format_exc()}```")

    # --- base de c
    base = None
    cfg_c = runs["c"]["config"]
    if cfg_c:
        for lab in "ab":
            cfg = runs[lab]["config"]
            if cfg and all(cfg.get(key) == cfg_c.get(key) for key in ("target", "huber", "lam",
                                                                      "lr", "epochs")):
                base = lab
                break

    # --- informe
    L = ["# Pruebas de la noche sobre μ", ""]
    L.append("Común: λ = 3, lr = 0,001, 5 épocas, 20 % por época, 2 miembros, semillas 1000·k + m "
             "(por pares con el ajuste). Referencia: ajuste `lam=3 lr=0.001` (target absoluto, MSE, "
             "ventana 252).")
    L.append("")
    for lab, info in runs.items():
        c = info["config"]
        desc = (f"target {c['target']}, huber {c['huber']}, ventana {c['window']}" if c
                else "sin config.json")
        L.append(f"- **{lab}** = `{info['run']}`: {desc}")
    if base:
        L.append(f"- Base de c: **{base}**")
    L.append("")

    def ic_row(lab):
        b = runs[lab]["blocks"]
        return {k: b[k]["ic"] for k in b}

    def pooled(lab, key):
        b = runs[lab]["blocks"]
        s = [b[k][key] for k in b if key in b[k]]
        return rel((sum(x[0] for x in s), sum(x[1] for x in s))) if s else None

    # veredicto
    L += ["## Veredicto (criterio fijado antes de ver los resultados)", "",
          ("| prueba | MSE entre activos fuera, h = 5 | h = 10 | IC gana a la ref. | "
          "QLIKE frente a la ref. | ¿arregla la memoria? |"), "|---|---|---|---|---|---|"]
    for lab in runs:
        b = runs[lab]["blocks"]
        if not b:
            L.append(f"| {lab} | — | — | — | — | sin resultados |")
            continue
        m5, m10 = pooled(lab, "oos_5"), pooled(lab, "oos_10")
        wins = sum(b[k]["ic"] > ref.loc[names[k], "ic"] for k in b)
        q = np.mean([b[k]["qlike"] for k in b])
        q_ref = ref.loc[[names[k] for k in b], "qlike"].mean()
        dq = 100 * (q / q_ref - 1)
        ok = (m5 < 0 and m10 < 0, wins >= 4, dq <= 2.0)
        L.append(f"| {lab} | {fmt(m5, 3, True)} {'✓' if m5 < 0 else '✗'} | "
                 f"{fmt(m10, 3, True)} {'✓' if m10 < 0 else '✗'} | "
                 f"{wins}/{len(b)} {'✓' if ok[1] else '✗'} | {dq:+.1f} % {'✓' if ok[2] else '✗'} | "
                 f"**{'sí' if all(ok) else 'no'}** |")
    L.append("")

    # IC
    L += ["## IC de selección (media de h = 5 y 10)", "",
          "| | " + " | ".join(names) + " | media | gana a la ref. |",
          "|---|" + "---|" * (len(names) + 2)]
    L.append("| referencia | " + " | ".join(fmt(ref.loc[n, "ic"]) for n in names) +
             f" | {fmt(ref['ic'].mean())} | |")
    for lab in runs:
        r = ic_row(lab)
        wins = sum(r[k] > ref.loc[names[k], "ic"] for k in r)
        L.append(f"| {lab} | " + " | ".join(fmt(r.get(k)) for k in range(len(names))) +
                 f" | {fmt(np.mean(list(r.values())) if r else None)} | {wins}/{len(r)} |")
    L.append("")

    # QLIKE
    L += ["## QLIKE (media de los tres horizontes)", "",
          "| | " + " | ".join(names) + " | media | frente a la ref. |",
          "|---|" + "---|" * (len(names) + 2)]
    L.append("| referencia | " + " | ".join(fmt(ref.loc[n, "qlike"]) for n in names) +
             f" | {fmt(ref['qlike'].mean())} | |")
    for lab in runs:
        b = runs[lab]["blocks"]
        if not b:
            continue
        q = np.mean([b[k]["qlike"] for k in b])
        q_ref = ref.loc[[names[k] for k in b], "qlike"].mean()
        L.append(f"| {lab} | " + " | ".join(fmt(b[k]["qlike"]) if k in b else "—"
                                          for k in range(len(names))) +
                 f" | {fmt(q)} | {100 * (q / q_ref - 1):+.1f} % |")
    L.append("")

    # MSE entre activos
    for h in H2:
        L += [f"## MSE de μ frente a μ = 0 entre activos, h = {h} (%)", "",
              "| | " + " | ".join(names) + " | 5 bloques juntos |",
              "|---|" + "---|" * (len(names) + 1)]
        for lab in runs:
            b = runs[lab]["blocks"]
            for key, what in ((f"oos_{h}", "fuera de muestra"), (f"ins_{h}", "entrenamiento")):
                L.append(f"| {lab} {what} | " + " | ".join(
                    fmt(rel(b[k][key]), 3, True) if k in b and key in b[k] else "—"
                    for k in range(len(names))) + f" | {fmt(pooled(lab, key), 3, True)} |")
        if ref4:
            for key, what in ((f"oos_{h}", "fuera de muestra"), (f"ins_{h}", "entrenamiento")):
                L.append(f"| referencia {what} (solo cv4, ens10 m0–1) | " +
                         " | ".join("—" for _ in names[:-1]) +
                         f" | {fmt(rel(ref4[key]), 3, True)} | — |")
        L.append("")
    if ref4 is not None or any(runs[lab]["blocks"] for lab in runs):
        n_sub = {k: len(s) for k, s in subs.items()}
        L.append(f"Entrenamiento: submuestra fija de fechas completas, {min(n_sub.values()):,}–"
                 f"{max(n_sub.values()):,} muestras por bloque.")
        L.append("")

    # diferencia con la reversión
    L += ["## Diferencia diaria de IC con la reversión de 1 mes (5 bloques juntos)", "",
          "| prueba | h | IC modelo | IC reversión | diferencia | t NW | días |",
          "|---|---|---|---|---|---|---|"]
    for lab in runs:
        b = runs[lab]["blocks"]
        if not b:
            continue
        for h in H2:
            icm = pd.concat([b[k][f"icd_{h}"] for k in b])
            icr = pd.concat([rev_ic[k][h] for k in b])
            diff = (icm - icr).dropna().sort_index()
            mean, t = nw_tstat(diff.to_numpy(), h)
            L.append(f"| {lab} | {h} | {icm.mean():.4f} | {icr.mean():.4f} | {mean:+.4f} | "
                     f"{t:+.2f} | {len(diff)} |")
    L.append("")

    # c frente a su base
    if base and runs["c"]["blocks"]:
        bb, bc = runs[base]["blocks"], runs["c"]["blocks"]
        L += [f"## c (ventana 63) frente a su base ({base}), bloque a bloque", "",
              ("| bloque | IC base | IC c | QLIKE base | QLIKE c | MSE ea h=5 base | c | "
              "MSE ea h=10 base | c | s/época base | c |"), "|---|" + "---|" * 10]
        for k in range(len(names)):
            if k not in bb or k not in bc:
                continue
            x, z = bb[k], bc[k]
            cells = [fmt(x["ic"]), fmt(z["ic"]), fmt(x["qlike"]), fmt(z["qlike"]),
                     fmt(rel(x["oos_5"]), 3, True), fmt(rel(z["oos_5"]), 3, True),
                     fmt(rel(x["oos_10"]), 3, True), fmt(rel(z["oos_10"]), 3, True),
                     fmt(x.get("sec_epoch"), 0), fmt(z.get("sec_epoch"), 0)]
            L.append(f"| {names[k]} | " + " | ".join(cells) + " |")
        L.append("")
    if ref4 and "sec_epoch" in ref4:
        L.append(f"Tiempo por época de la referencia (ens10 m0–1, cv4): {ref4['sec_epoch']:.0f} s.")
        L.append("")

    L += ["## Notas", ""]
    L += [f"- {n}" for n in notes] or ["- Sin incidencias."]
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nEscrito en {out}")


if __name__ == "__main__":
    main()
