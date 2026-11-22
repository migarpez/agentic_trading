"""Prueba de mirar el futuro: compara, en 2012–15 (cv3), tres ensembles con las mismas semillas:

- F: entrena solo con el pasado (2000–2011).
- P: cambia el vecino del pasado (2008–11) por el del futuro (2016–19): 2000–2007 y 2016–2019,
  submuestreado al azar hasta tantas muestras como F.
- CV: cv3 de la validación cruzada normal (rel63), que ve los dos vecinos.

Para cada uno: IC a h = 5 y 10 (t NW), diferencia diaria de IC con la reversión de 1 mes y QLIKE.
Diferencia diaria de IC P − F con su t de Newey-West (retardos = h); también CV − F y CV − P, y la
diferencia con el momentum de 12 meses (la otra línea base), que no forman parte de la regla.

Regla (fijada antes de ver los resultados): la validación cruzada infla si la diferencia P − F es
positiva con t > 2 a h = 5 o 10.

Uso: python scripts/lookahead_compare.py [--f lookahead_F] [--p lookahead_P] [--cv rel63]
     [--out logs/lookahead.md]
"""

import argparse
import json
import traceback

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import H2, ic_diff, load_members, targets
from agentic_trading.model.folds import cv_folds, manual_fold
from agentic_trading.model.metrics import HORIZONS, baseline_features, daily_ic, nw_tstat, qlike
from agentic_trading.model.train import ensemble
from agentic_trading.model.tuning import fold_scores

K = 3                                   # cv3: 2012–15
DESC = {"F": "solo pasado", "P": "futuro en vez de pasado", "CV": "los dos vecinos"}


def fmt_n(n) -> str:
    return f"{n:,}".replace(",", ".")


def train_info(data, run: str) -> tuple[str, str]:
    """Años y número de muestras de entrenamiento de cv3 en una ejecución (de su config.json)."""
    cfg = json.loads((DATA_DIR / "predictions" / run / "config.json").read_text())
    man = cfg.get("manual_fold")
    if not man:
        f = cv_folds(data.samples)[K]
        return "2000–2011 y 2016–2019", fmt_n(len(f["train"]))
    years = [tuple(r) for r in man["train_years"]]
    full = len(manual_fold(data.samples, K, years)["train"])
    txt = " y ".join(f"{a}–{b}" for a, b in years)
    if man.get("train_n") is None:
        return txt, fmt_n(full)
    return txt, (f"{fmt_n(man['train_n'])} (al azar de {fmt_n(full)}, "
                 f"semilla {man['subsample_seed']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", default="lookahead_F")
    ap.add_argument("--p", default="lookahead_P")
    ap.add_argument("--cv", default="rel63")
    ap.add_argument("--out", default=str(ROOT / "logs" / "lookahead.md"))
    args = ap.parse_args()
    runs = {"F": args.f, "P": args.p, "CV": args.cv}
    data = WindowData(DATA_DIR / "dataset")
    fold = cv_folds(data.samples)[K]
    name, (a, b) = fold["name"], fold["block"]
    notes = []

    cfgs = {lab: json.loads((DATA_DIR / "predictions" / r / "config.json").read_text())
            for lab, r in runs.items()}
    for lab in ("F", "P"):                     # misma configuración y dataset que CV
        tr, ref = cfgs[lab]["train"], cfgs["CV"]["train"]
        diff = {k: (tr.get(k), ref.get(k)) for k in sorted(set(tr) | set(ref))
                if tr.get(k) != ref.get(k)}
        if diff:
            notes.append(f"{lab}: configuración distinta de la de {runs['CV']}: {diff}")
        if cfgs[lab].get("dataset") != cfgs["CV"].get("dataset"):
            notes.append(f"{lab}: dataset distinto del de {runs['CV']}")
    t = cfgs["F"]["train"]
    mem = {lab: load_members(DATA_DIR / "predictions" / r, name) for lab, r in runs.items()}
    ens = {lab: ensemble(list(m.values())).sort_values("sample").reset_index(drop=True)
           for lab, m in mem.items()}
    idx = ens["F"]["sample"].to_numpy()
    assert np.array_equal(idx, np.sort(fold["predict"])), "F no predice las muestras de cv3"
    for lab in ("P", "CV"):
        assert np.array_equal(ens[lab]["sample"].to_numpy(), idx), f"{lab}: otras muestras"

    dates, rows, y = targets(data, idx, HORIZONS)
    smp = data.samples.iloc[idx]
    base = baseline_features(data, rows)
    rev = base["reversal"]
    mom_ic = {h: daily_ic(dates, base["momentum"], y[h]) for h in H2}
    ics = {lab: {h: daily_ic(dates, e[f"mu_{h}"].to_numpy(), y[h]) for h in H2}
           for lab, e in ens.items()}
    rev_ic = {h: daily_ic(dates, rev, y[h]) for h in H2}
    ql = {}
    for lab, e in ens.items():
        for h in HORIZONS:
            v = np.asarray(data.arr[f"v_{h}"])[rows].astype(float)
            m = smp[f"mask_sigma_{h}"].to_numpy()
            ql[lab, h] = float(qlike(v[m], e[f"sigma_{h}"].to_numpy()[m] / np.sqrt(h)).mean())

    L = [f"# Prueba de mirar el futuro: {a.year}–{b.year} ({name})", "",
         ("Regla (fijada antes de ver los resultados): la validación cruzada infla si la diferencia "
          "diaria de IC P − F es positiva con t de Newey-West > 2 (retardos = h) a h = 5 o 10."), "",
         (f"Configuración de F (las diferencias con las otras, en Notas): target {t['target']}, "
          f"ventana de {t['window']}, {'MSE' if t.get('huber') is None else 'Huber'}, "
          f"λ = {t['lam']}, lr = {t['lr']}, {t['epochs']} épocas, {100 * t['epoch_frac']:.0f} % "
          "por época. Mismas semillas (1000·3 + m) y mismo hueco de 20 velas alrededor del "
          "bloque predicho."), "",
         "| | ejecución | entrena con | muestras de entrenamiento | miembros |",
         "|---|---|---|---|---|"]
    for lab, r in runs.items():
        try:
            yrs, n = train_info(data, r)
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            yrs, n = "?", "?"
            notes.append(f"{lab}: sin config.json legible\n```\n{traceback.format_exc()}```")
        L.append(f"| {lab} ({DESC[lab]}) | {r} | {yrs} | {n} | {len(mem[lab])} |")
    L += ["", f"Muestras predichas: {fmt_n(len(idx))} ({len(np.unique(dates))} días).", ""]

    L += ["## IC y QLIKE", "",
          ("| | h | IC (t NW) | IC reversión | diferencia diaria con la reversión (t NW) | "
           "QLIKE |"), "|---|---|---|---|---|---|"]
    for lab in runs:
        for h in H2:
            m_ic, m_t = nw_tstat(ics[lab][h].to_numpy(), h)
            d, t, _ = ic_diff(ics[lab][h], rev_ic[h], h)
            L.append(f"| {lab} | {h} | {m_ic:.4f} ({m_t:.1f}) | {rev_ic[h].mean():.4f} | "
                     f"{d:+.4f} ({t:.2f}) | {ql[lab, h]:.4f} |")
    L += ["", "QLIKE a h = 20: " + ", ".join(f"{lab} {ql[lab, 20]:.4f}" for lab in runs) + ".", ""]

    L += ["## Diferencia diaria de IC", "",
          "| | h = 5 (t NW) | h = 10 (t NW) | días |", "|---|---|---|---|"]
    res = {}
    for x, z in (("P", "F"), ("CV", "F"), ("CV", "P")):
        cells = []
        for h in H2:
            d, t, n = ic_diff(ics[x][h], ics[z][h], h)
            res[x, z, h] = (d, t)
            cells.append(f"{d:+.4f} ({t:.2f})")
        tag = " (regla)" if (x, z) == ("P", "F") else ""
        L.append(f"| {x} − {z}{tag} | " + " | ".join(cells) + f" | {n} |")
    L.append("")
    hit = [h for h in H2 if res["P", "F", h][0] > 0 and res["P", "F", h][1] > 2]
    verdict = "la validación cruzada infla" if hit else "la validación cruzada no infla"
    L += [f"**Veredicto: {verdict}** (regla: P − F > 0 con t > 2 a h = 5 o 10"
          + (f"; se cumple a h = {', '.join(map(str, hit))})." if hit else "; no se cumple)."), ""]

    L += ["## Extra: frente al momentum de 12 meses (no forma parte de la regla)", "",
          ("IC medio de las líneas base: "
           + "; ".join(f"h = {h}: momentum {mom_ic[h].mean():.4f}, reversión {rev_ic[h].mean():.4f}"
                       for h in H2) + "."), "",
          "| | h = 5: diferencia con el momentum (t NW) | h = 10 (t NW) |", "|---|---|---|"]
    for lab in runs:
        cells = []
        for h in H2:
            d, t, _ = ic_diff(ics[lab][h], mom_ic[h], h)
            cells.append(f"{d:+.4f} ({t:.2f})")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## P − F por año", "", "| año | h = 5 (t NW) | h = 10 (t NW) |", "|---|---|---|"]
    years = pd.DatetimeIndex(ics["F"][5].index).year
    for yr in sorted(set(years)):
        cells = []
        for h in H2:
            dd = (ics["P"][h] - ics["F"][h]).dropna()
            dd = dd[pd.DatetimeIndex(dd.index).year == yr]
            mean, t = nw_tstat(dd.sort_index().to_numpy(), h)
            cells.append(f"{mean:+.4f} ({t:.2f})")
        L.append(f"| {yr} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## IC por miembro (media de h = 5 y 10)", "",
          "| | media | sd | mín | máx | ensemble |", "|---|---|---|---|---|---|"]
    for lab in runs:
        per = np.array([fold_scores(data, ensemble([p]))["ic"] for p in mem[lab].values()])
        L.append(f"| {lab} | {per.mean():.4f} | {per.std(ddof=1) if len(per) > 1 else 0:.4f} | "
                 f"{per.min():.4f} | {per.max():.4f} | {fold_scores(data, ens[lab])['ic']:.4f} |")
    L += ["", "## Notas", ""] + ([f"- {n}" for n in notes] or ["- Sin incidencias."])
    text = "\n".join(L) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"Escrito en {args.out}")


if __name__ == "__main__":
    main()
