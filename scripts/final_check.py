"""Comprobaciones del ensemble final (9-11-2026): final20, 20 miembros con la configuración cerrada.
Solo lee predicciones; la deriva de entrenamiento se recalcula con el dataset (y se contrasta con
la del checkpoint del miembro 0).

1. Reproducibilidad: los miembros 0 y 1 de final20 deben coincidir con los de month_lam1 (misma
   configuración y semillas).
2. Los cambios de hoy con 10 miembros: final20 (miembros 0–9) frente a rel63 (10), por bloque y con
   los 5 juntos: IC (media de h = 5 y 10), diferencia diaria de IC a h = 5 y 10 (t NW) y QLIKE.
   Lectura fijada de antemano: los cambios se confirman para μ si la diferencia con los 5 bloques
   juntos es positiva a h = 5 y 10; si es negativa con t < −2 en alguna, aviso (motivo para revisar
   las velas mensuales).
3. La mezcla (mix_check.compute y render) para final20 con 20 miembros, con la misma lectura que
   entonces.
4. Confianza, con 20 miembros, a h = 5, 10 y 20, por bloque y con los 5 juntos: IC dentro de cada
   quintil diario de s_h, y acierto de signo relativo (μ_h − deriva_h frente a y_h − media del día)
   por quintil diario de s_h (Q1: menor s_h, más confianza). Solo días con al menos 100 muestras
   (20 por quintil). Sin regla: sirve para proponer el criterio de confianza del test.

Uso: python scripts/final_check.py [--run final20] [--repro month_lam1] [--base rel63]
     [--out logs/final.md]
"""

import argparse
import json
import traceback

import numpy as np
import pandas as pd
import torch
from coarse_compare import cell
from mix_check import compute as mix_compute
from mix_check import describe
from mix_check import render as mix_render

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import H2, ic_diff, load_members, targets
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import HORIZONS, daily_ic, nw_tstat
from agentic_trading.model.train import ensemble
from agentic_trading.model.tuning import fold_scores

N_Q, MIN_Q = 5, 20                      # quintiles; muestras por quintil y día
TEN = list(range(10))
PRED_COLS = [f"{p}_{h}" for h in HORIZONS for p in ("mu", "logv")]


def daily_quintile(dates, x, n: int = N_Q) -> np.ndarray:
    """Quintil de x entre activos, cada día (0: los menores)."""
    r = pd.Series(x).groupby(dates).rank(pct=True).to_numpy()
    return np.minimum((r * n).astype(int), n - 1)


def fmt(m, t) -> str:
    return f"{m:+.4f} ({t:.2f})" if np.isfinite(m) else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final20")
    ap.add_argument("--repro", default="month_lam1")
    ap.add_argument("--base", default="rel63")
    ap.add_argument("--out", default=str(ROOT / "logs" / "final.md"))
    args = ap.parse_args()
    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    pred = DATA_DIR / "predictions"
    bname = {k: f"{f['block'][0].year}–{f['block'][1].year}" for k, f in enumerate(folds)}
    have = [k for k, f in enumerate(folds) if (pred / args.run / f"{f['name']}.parquet").exists()]
    notes = []
    cfg = json.loads((pred / args.run / "config.json").read_text())
    L = [f"# Ensemble final: `{args.run}`", "",
         (f"{cfg['members']} miembros; target {cfg['train']['target']}, ventana de "
          f"{cfg['train']['window']}, velas agregadas {cfg['train']['coarse']}, λ = "
          f"{cfg['train']['lam']}, lr = {cfg['train']['lr']}, {cfg['train']['epochs']} épocas, "
          f"{100 * cfg['train']['epoch_frac']:.0f} % por época. Bloques: "
          + ", ".join(bname[k] for k in have) + "."), ""]

    # ------------------------------------------------------------- 1. reproducibilidad
    L += ["## Reproducibilidad", "",
          (f"Miembros 0 y 1 de `{args.run}` frente a los de `{args.repro}` (misma configuración y "
           "semillas): μ_h y log v_h de cada muestra, bit a bit."), "",
          "| bloque | miembros | bit a bit | máx. diferencia |", "|---|---|---|---|"]
    repro_ok = []
    for k in have:
        name = folds[k]["name"]
        try:
            if not (pred / args.repro / f"{name}_members.parquet").exists():
                L.append(f"| {bname[k]} | — | sin `{args.repro}` | — |")
                continue
            a = load_members(pred / args.run, name, [0, 1])
            b = load_members(pred / args.repro, name, [0, 1])
            for m in sorted(set(a) & set(b)):
                same = np.array_equal(a[m]["sample"], b[m]["sample"]) and all(
                    np.array_equal(a[m][c].to_numpy(), b[m][c].to_numpy()) for c in PRED_COLS)
                diff = max(float(np.abs(a[m][c].to_numpy() - b[m][c].to_numpy()).max())
                           for c in PRED_COLS) if len(a[m]) == len(b[m]) else np.inf
                repro_ok.append(same)
                L.append(f"| {bname[k]} | {m} | {'sí' if same else 'NO'} | {diff:.1e} |")
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"reproducibilidad {name}: error\n```\n{traceback.format_exc()}```")
    L += ["", ("**Coinciden bit a bit.**" if repro_ok and all(repro_ok) else
               "**No coinciden bit a bit: ver la tabla.**" if repro_ok else
               "**Sin miembros comunes: no se comprueba.**"), ""]

    # ------------------------------------------------------------- 2. cambios con 10 miembros
    L += [(f"## Los cambios de hoy con 10 miembros: `{args.run}` (miembros 0–9) frente a "
           f"`{args.base}` (10)"), "",
          ("Lectura (fijada de antemano): los cambios se confirman para μ si la diferencia diaria "
           "de IC con los 5 bloques juntos es positiva a h = 5 y 10; si es negativa con t < −2 en "
           "alguna, aviso (motivo para revisar las velas mensuales). IC: media de h = 5 y 10; "
           "QLIKE: media de h = 5, 10 y 20."), "",
          ("| bloque | IC base | IC final (0–9) | diferencia diaria h = 5 (t) | h = 10 (t) | "
           "QLIKE base | QLIKE final (0–9) |"), "|---|---|---|---|---|---|---|"]
    chg, n_mem = {}, {}
    for k in have:
        name = folds[k]["name"]
        try:
            new = load_members(pred / args.run, name, TEN)
            n_mem[k] = len(new)
            e_new = ensemble(list(new.values())).sort_values("sample").reset_index(drop=True)
            e_old = (ensemble(list(load_members(pred / args.base, name).values()))
                     .sort_values("sample").reset_index(drop=True))
            idx = e_new["sample"].to_numpy()
            assert np.array_equal(e_old["sample"].to_numpy(), idx), "otras muestras"
            dates, _, y = targets(data, idx, H2)
            r = {"new": fold_scores(data, e_new), "old": fold_scores(data, e_old)}
            r["icd"] = {lab: {h: daily_ic(dates, e[f"mu_{h}"].to_numpy(), y[h]) for h in H2}
                        for lab, e in (("new", e_new), ("old", e_old))}
            chg[k] = r
            cells = [cell(*ic_diff(r["icd"]["new"][h], r["icd"]["old"][h], h)[:2]) for h in H2]
            L.append(f"| {bname[k]} | {r['old']['ic']:.4f} | {r['new']['ic']:.4f} | "
                     + " | ".join(cells) + f" | {r['old']['qlike']:.4f} | {r['new']['qlike']:.4f} |")
        except Exception:  # noqa: BLE001
            notes.append(f"cambios {name}: error\n```\n{traceback.format_exc()}```")
    reading = None
    if chg:
        ks = list(chg)
        pooled = {h: ic_diff(pd.concat([chg[k]["icd"]["new"][h] for k in ks]),
                             pd.concat([chg[k]["icd"]["old"][h] for k in ks]), h) for h in H2}
        ic_o, ic_n = (np.mean([chg[k][lab]["ic"] for k in ks]) for lab in ("old", "new"))
        q_o, q_n = (np.mean([chg[k][lab]["qlike"] for k in ks]) for lab in ("old", "new"))
        L.append(f"| **{len(ks)} bloques juntos** | {ic_o:.4f} | {ic_n:.4f} | "
                 + " | ".join(cell(*pooled[h][:2]) for h in H2)
                 + f" | {q_o:.4f} | {q_n:.4f} ({100 * (q_n / q_o - 1):+.1f} %) |")
        warn = [h for h in H2 if pooled[h][0] < 0 and pooled[h][1] < -2]
        if warn:
            reading = (f"AVISO: la diferencia es negativa con t < −2 a h = "
                       f"{', '.join(map(str, warn))}: motivo para revisar las velas mensuales")
        elif all(pooled[h][0] > 0 for h in H2):
            reading = "los cambios se confirman para μ (diferencia positiva a h = 5 y 10)"
        else:
            reading = ("los cambios no se confirman para μ (la diferencia no es positiva en los dos "
                       "horizontes), sin llegar al aviso (ninguna con t < −2)")
        L += ["", (f"IC y QLIKE de la última fila: media de los bloques; diferencias: series "
                   f"diarias de los {len(ks)} bloques juntos."), ""]
        if any(n < len(TEN) for n in n_mem.values()):
            notes.append(f"{args.run}: menos de 10 miembros en algún bloque ({n_mem}); la "
                         "comparación usa los que hay")
        if len(ks) < len(folds):
            reading += f" (orientativo: solo {len(ks)} de {len(folds)} bloques)"
        L += [f"**Lectura: {reading}.**", ""]

    # ------------------------------------------------------------- 3. la mezcla
    try:
        runs = {args.run: None}
        label, n_exp = describe(args.run, None, cfg["members"])
        L += mix_render(mix_compute(data, folds, runs, {args.run: n_exp}), folds, runs,
                        {args.run: label}, level="##")
        L.append("")
    except Exception:  # noqa: BLE001
        notes.append(f"mezcla: error\n```\n{traceback.format_exc()}```")

    # ------------------------------------------------------------- 4. confianza
    conf = {h: {} for h in HORIZONS}
    drift_chk = []
    for k in have:
        f = folds[k]
        try:
            e = pd.read_parquet(pred / args.run / f"{f['name']}.parquet")
            e = e.sort_values("sample").reset_index(drop=True)
            idx = e["sample"].to_numpy()
            dates, _, y = targets(data, idx, HORIZONS)
            rows_tr = data.samples["row"].to_numpy()[f["train"]]
            drift = np.array([np.asarray(data.arr[f"y_{h}"])[rows_tr].astype(np.float64).mean()
                              for h in HORIZONS])
            ck = ROOT / "checkpoints" / args.run / f["name"] / "member0.pt"
            if ck.exists():
                stored = np.asarray(torch.load(ck, weights_only=False)["drift"])
                drift_chk.append(float(np.abs(stored - drift).max()))
            n_day = pd.Series(dates).groupby(dates).transform("size").to_numpy()
            keep = n_day >= N_Q * MIN_Q
            d = dates[keep]
            for j, h in enumerate(HORIZONS):
                mu, s, yy = (e[f"mu_{h}"].to_numpy()[keep], e[f"s_{h}"].to_numpy()[keep],
                             y[h][keep])
                q = daily_quintile(d, s)
                y_rel = yy - pd.Series(yy).groupby(d).transform("mean").to_numpy()
                hit = (np.sign(mu - drift[j]) == np.sign(y_rel)).astype(float)
                conf[h][k] = {
                    "ic": [daily_ic(d[q == i], mu[q == i], yy[q == i]) for i in range(N_Q)],
                    "hit": [pd.Series(hit[q == i]).groupby(d[q == i]).mean() for i in range(N_Q)],
                    "n_hit": [(hit[q == i].sum(), (q == i).sum()) for i in range(N_Q)],
                    "ic_all": daily_ic(d, mu, yy), "hit_all": (hit.sum(), len(hit))}
        except Exception:  # noqa: BLE001
            notes.append(f"confianza {f['name']}: error\n```\n{traceback.format_exc()}```")
    if any(conf[h] for h in HORIZONS):
        L += [f"## Confianza ({cfg['members']} miembros)", "",
              ("Quintiles diarios de s_h (Q1: menor s_h, más confianza; Q5: menos). Solo días con al "
               f"menos {N_Q * MIN_Q} muestras. Entre paréntesis, t de Newey-West (retardos = h) de "
               "la diferencia diaria Q1 − Q5. Sin regla: sirve para proponer el criterio de "
               "confianza del test."), "",
              "### IC dentro de cada quintil diario de s_h", "",
              "| h | bloque | todos | Q1 | Q2 | Q3 | Q4 | Q5 | Q1 − Q5 (t) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for h in HORIZONS:
            ks = list(conf[h])
            for k in ks + ["total"]:
                if k == "total":
                    c = {"ic": [pd.concat([conf[h][kk]["ic"][i] for kk in ks]) for i in range(N_Q)],
                         "ic_all": pd.concat([conf[h][kk]["ic_all"] for kk in ks])}
                    lab = f"**{len(ks)} bloques juntos**"
                else:
                    c, lab = conf[h][k], bname[k]
                m, t, _ = ic_diff(c["ic"][0], c["ic"][-1], h)
                L.append(f"| {h} | {lab} | {c['ic_all'].mean():.4f} | "
                         + " | ".join(f"{s.mean():.4f}" for s in c["ic"]) + f" | {fmt(m, t)} |")
        L += ["", "### Acierto de signo relativo por quintil diario de s_h", "",
              ("Signo de μ_h − deriva_h (la media de y_h en el entrenamiento del bloque) frente al "
               "de y_h − media del día."), "",
              "| h | bloque | todos | Q1 | Q2 | Q3 | Q4 | Q5 | Q1 − Q5 (t) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for h in HORIZONS:
            ks = list(conf[h])
            for k in ks + ["total"]:
                if k == "total":
                    n_hit = [tuple(np.sum([conf[h][kk]["n_hit"][i] for kk in ks], 0))
                             for i in range(N_Q)]
                    hit_all = tuple(np.sum([conf[h][kk]["hit_all"] for kk in ks], 0))
                    series = [pd.concat([conf[h][kk]["hit"][i] for kk in ks]) for i in range(N_Q)]
                    lab = f"**{len(ks)} bloques juntos**"
                else:
                    n_hit, hit_all = conf[h][k]["n_hit"], conf[h][k]["hit_all"]
                    series, lab = conf[h][k]["hit"], bname[k]
                diff = (series[0] - series[-1]).dropna().sort_index()
                m, t = nw_tstat(diff.to_numpy(), h)
                L.append(f"| {h} | {lab} | {hit_all[0] / hit_all[1]:.4f} | "
                         + " | ".join(f"{a / b:.4f}" for a, b in n_hit) + f" | {fmt(m, t)} |")
        L.append("")
    if drift_chk:
        notes.append(f"Deriva recalculada con el dataset frente a la del checkpoint (miembro 0): "
                     f"máxima diferencia {max(drift_chk):.1e}.")
    L += ["## Notas", ""] + ([f"- {n}" for n in notes] or ["- Sin incidencias."])
    text = "\n".join(L) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"Escrito en {args.out}")


if __name__ == "__main__":
    main()
