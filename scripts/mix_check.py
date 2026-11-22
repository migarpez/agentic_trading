"""Comprobación sin entrenar (9-11-2026): μ frente a una mezcla fija de las dos líneas base.

Hoy la vara de μ es la mejor línea base de cada bloque, elegida a posteriori; en real no se sabe
cuál de las dos va a funcionar. Se compara μ con una mezcla fija, decidida de antemano: cada día,
la media a partes iguales de los rangos entre activos de la reversión de 1 mes y del momentum de
12 meses (metrics.baseline_mix, la misma que usa success_checks), con los mismos días y muestras
que el IC de μ.

Para coarse_month (miembros 0 y 1) y rel63 (ensemble de 10), a h = 5 y 10, por bloque y con los 5
juntos: IC de μ, de la mezcla y de cada regla (t de Newey-West, retardos = h); diferencia diaria
de IC μ − mezcla, con su t; e IC residual de μ frente a las dos reglas: IC de los rangos de μ
quitándoles, cada día, lo que explican los de las dos (MCO con constante).

Lectura fijada antes de ver los resultados (diagnóstico; no cambia nada): μ aporta algo más que
las dos reglas juntas si bate a la mezcla (diferencia > 0 con t > 2), a h = 5 o 10, con los 5
bloques juntos y en 2016–19.

Solo lee predicciones: no entrena ni toca ninguna ejecución.

--runs elige otras ejecuciones y miembros (run, todos sus miembros, o run:miembros, p. ej.
coarse_month:0,1 o final20:0-9). Sin la opción, las dos de arriba. final_check.py reutiliza
compute y render.

Uso: python scripts/mix_check.py [--runs final20] [--out logs/mix.md]
"""

import argparse
import json
import traceback

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.compare import H2, ic_diff, load_members, targets
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import (
    baseline_features,
    baseline_mix,
    daily_ic,
    daily_rank,
    nw_tstat,
)
from agentic_trading.model.train import ensemble

RUNS = {"coarse_month": [0, 1], "rel63": None}          # None: todos los miembros
LABEL = {"coarse_month": "coarse_month (miembros 0 y 1)", "rel63": "rel63 (ensemble de 10)"}
N_MEMBERS = {"coarse_month": 2, "rel63": 10}
DECISION = 4                                             # cv4: 2016–19
T_MIN = 2.0
MIN_DAY = 20                                             # muestras por día que exige daily_ic


class TwoRuleResidual:
    """MCO diaria, con constante, de los rangos de un predictor sobre los de las dos reglas; se
    preparan una vez por bloque (rangos centrados y sus sumas por día)."""

    def __init__(self, dates, a, b):
        self.dates = dates
        df = pd.DataFrame({"d": dates, "a": daily_rank(dates, a), "b": daily_rank(dates, b)})
        g = df.groupby("d")
        self.ca = (df["a"] - g["a"].transform("mean")).to_numpy()
        self.cb = (df["b"] - g["b"].transform("mean")).to_numpy()
        s = pd.DataFrame({"aa": self.ca ** 2, "bb": self.cb ** 2, "ab": self.ca * self.cb})
        s = s.groupby(dates).transform("sum")
        self.saa, self.sbb, self.sab = (s[k].to_numpy() for k in ("aa", "bb", "ab"))
        self.det = self.saa * self.sbb - self.sab ** 2
        self.ok = self.det > 1e-9 * self.saa * self.sbb       # días con las dos reglas no colineales

    def residual(self, x) -> np.ndarray:
        rx = pd.Series(daily_rank(self.dates, x))
        cx = (rx - rx.groupby(self.dates).transform("mean")).to_numpy()
        s = pd.DataFrame({"ax": self.ca * cx, "bx": self.cb * cx}).groupby(self.dates)
        s = s.transform("sum")
        sax, sbx = s["ax"].to_numpy(), s["bx"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = np.where(self.ok, (self.sbb * sax - self.sab * sbx) / self.det, 0.0)
            b2 = np.where(self.ok, (self.saa * sbx - self.sab * sax) / self.det, 0.0)
        return cx - b1 * self.ca - b2 * self.cb


def lstsq_check(dates, x, a, b, resid, n_days: int = 5, seed: int = 0) -> float:
    """Máxima diferencia entre el residuo y np.linalg.lstsq, día a día, en n_days días al azar de
    los que cuentan en el IC (al menos MIN_DAY muestras; con menos, el ajuste no está
    determinado)."""
    d_all, n = np.unique(dates, return_counts=True)
    days = d_all[n >= MIN_DAY]
    worst = 0.0
    for d in np.random.default_rng(seed).choice(days, size=min(n_days, len(days)), replace=False):
        m = dates == d
        X = np.column_stack([np.ones(m.sum()), daily_rank(dates[m], a[m]),
                             daily_rank(dates[m], b[m])])
        yy = daily_rank(dates[m], x[m])
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        worst = max(worst, float(np.abs(yy - X @ coef - resid[m]).max()))
    return worst


def stat(ic: pd.Series, h: int) -> str:
    m, t = nw_tstat(ic.to_numpy(), h)
    return f"{m:+.4f} ({t:.1f})" if np.isfinite(m) else "—"


def parse_runs(specs: list[str]) -> dict[str, list[int] | None]:
    """'final20' (todos los miembros), 'coarse_month:0,1' o 'final20:0-9'."""
    runs = {}
    for sp in specs:
        run, _, mem = sp.partition(":")
        members = None
        if mem:
            members = []
            for part in mem.split(","):
                a, _, b = part.partition("-")
                members += list(range(int(a), int(b or a) + 1))
        runs[run] = members
    return runs


def describe(run: str, members: list[int] | None, n_total: int) -> tuple[str, int]:
    """Etiqueta y número de miembros esperado de una ejecución."""
    if members is None:
        return f"{run} (ensemble de {n_total})", n_total
    if len(members) == 2:
        txt = f"miembros {members[0]} y {members[1]}"
    elif members == list(range(members[0], members[-1] + 1)):
        txt = f"miembros {members[0]}–{members[-1]}"
    else:
        txt = "miembros " + ", ".join(map(str, members))
    return f"{run} ({txt})", len(members)


def compute(data, folds, runs: dict, expected: dict) -> dict:
    """Series diarias de IC por ejecución, bloque y horizonte (μ, mezcla, reglas, residual y
    correlación μ–mezcla), con las notas y comprobaciones."""
    S = {run: {} for run in runs}           # S[run][k][h] = {serie: IC diario}
    notes, checks, degenerate = [], [], 0
    first = next(iter(runs))
    for k, f in enumerate(folds):
        try:
            ens = {}
            for run, members in runs.items():
                mem = load_members(DATA_DIR / "predictions" / run, f["name"], members)
                if len(mem) != expected[run]:
                    notes.append(f"{run} {f['name']}: {len(mem)} miembros (se esperaban "
                                 f"{expected[run]})")
                ens[run] = ensemble(list(mem.values())).sort_values("sample").reset_index(drop=True)
            idx = ens[first]["sample"].to_numpy()
            for run, e in ens.items():
                assert np.array_equal(e["sample"].to_numpy(), idx), f"{run}: otras muestras"
            dates, rows, y = targets(data, idx, H2)
            bf = baseline_features(data, rows)
            rev, mom = bf["reversal"], bf["momentum"]
            mix = baseline_mix(dates, rev, mom)
            two = TwoRuleResidual(dates, rev, mom)
            n_day = pd.Series(dates).groupby(dates).transform("size").to_numpy()
            degenerate += len(np.unique(dates[(~two.ok) & (n_day >= MIN_DAY)]))
            base = {h: {"mezcla": daily_ic(dates, mix, y[h]), "reversión": daily_ic(dates, rev, y[h]),
                        "momentum": daily_ic(dates, mom, y[h])} for h in H2}
            for run, e in ens.items():
                S[run][k] = {}
                for h in H2:
                    mu = e[f"mu_{h}"].to_numpy()
                    res = two.residual(mu)
                    if h == 5:
                        checks.append(lstsq_check(dates, mu, rev, mom, res, seed=k))
                    S[run][k][h] = {"μ": daily_ic(dates, mu, y[h]),
                                    "residual": daily_ic(dates, res, y[h]),
                                    "corr": daily_ic(dates, mu, mix), **base[h]}
        except Exception:  # noqa: BLE001 — se apunta y se sigue
            notes.append(f"{f['name']}: error\n```\n{traceback.format_exc()}```")
            for r in S.values():
                r.pop(k, None)
    return {"S": S, "notes": notes, "checks": checks, "degenerate": degenerate}


def render(res: dict, folds, runs: dict, labels: dict, level: str = "#") -> list[str]:
    """Informe en Markdown (level: nivel de los títulos; "##" para meterlo en otro informe)."""
    S, notes = res["S"], list(res["notes"])
    blocks = [k for k in range(len(folds)) if all(k in S[run] for run in runs)]
    name = {k: f"{folds[k]['block'][0].year}–{folds[k]['block'][1].year}" for k in range(len(folds))}
    h1, h2 = level, level + "#"

    def pooled(run, h, key):
        return pd.concat([S[run][k][h][key] for k in blocks]).sort_index()

    L = [f"{h1} μ frente a una mezcla fija de las dos líneas base", "",
         ("Mezcla, decidida de antemano: cada día, la media a partes iguales de los rangos entre "
          "activos de la reversión de 1 mes y del momentum de 12 meses, con los mismos días y "
          "muestras que el IC de μ. IC residual: IC de los rangos de μ quitándoles, cada día, lo "
          "que explican los de las dos reglas (MCO con constante). Entre paréntesis, t de "
          "Newey-West (retardos = h)."), "",
         ("Lectura (fijada antes de ver los resultados; diagnóstico, no cambia nada): μ aporta "
          "algo más que las dos reglas juntas si bate a la mezcla (diferencia diaria > 0 con "
          f"t > {T_MIN:.0f}), a h = 5 o 10, con los 5 bloques juntos y en 2016–19."), ""]
    if not blocks:
        L += ["Sin bloques: no hay comparación.", ""]
    else:
        L += [f"{h2} Lectura", "",
              ("| | h | μ − mezcla, 5 bloques juntos (t) | μ − mezcla, 2016–19 (t) | "
               "IC residual, 5 bloques juntos | IC residual, 2016–19 |"),
              "|---|---|---|---|---|---|"]
        verdict = {}
        for run in runs:
            hits = {}
            for h in H2:
                dp, tp, _ = ic_diff(pooled(run, h, "μ"), pooled(run, h, "mezcla"), h)
                cells = [f"{dp:+.4f} ({tp:.2f})"]
                ok = [dp > 0 and tp > T_MIN]
                if DECISION in blocks:
                    dd, td, _ = ic_diff(S[run][DECISION][h]["μ"], S[run][DECISION][h]["mezcla"], h)
                    cells.append(f"{dd:+.4f} ({td:.2f})")
                    ok.append(dd > 0 and td > T_MIN)
                else:
                    cells.append("—")
                    ok.append(False)
                cells += [stat(pooled(run, h, "residual"), h),
                          stat(S[run][DECISION][h]["residual"], h) if DECISION in blocks else "—"]
                hits[h] = ok
                L.append(f"| {labels[run]} | {h} | " + " | ".join(cells) + " |")
            same = [h for h in H2 if all(hits[h])]
            each = any(hits[h][0] for h in H2) and any(hits[h][1] for h in H2)
            if len(blocks) < len(folds):
                verdict[run] = f"no aplicable (solo {len(blocks)} de {len(folds)} bloques)"
            elif same:
                verdict[run] = f"sí (a h = {', '.join(map(str, same))})"
            elif each:
                verdict[run] = ("solo con horizontes distintos para los 5 bloques juntos y para "
                                "2016–19")
            else:
                verdict[run] = "no"
        L.append("")
        for run in runs:
            L.append(f"- **{labels[run]}: ¿aporta algo más que las dos reglas juntas? "
                     f"{verdict[run]}.**")
        L.append("")

        for run in runs:
            L += [f"{h2} {labels[run]}", "",
                  ("| bloque | h | IC μ | IC mezcla | IC reversión | IC momentum | μ − mezcla (t) | "
                   "IC residual de μ | correlación μ–mezcla |"),
                  "|---|---|---|---|---|---|---|---|---|"]
            for h in H2:
                for k in blocks + ["total"]:
                    if k == "total":
                        r = {key: pooled(run, h, key) for key in S[run][blocks[0]][h]}
                        label = "**5 bloques juntos**" if len(blocks) == 5 else "juntos"
                    else:
                        r = S[run][k][h]
                        label = f"**{name[k]}**" if k == DECISION else name[k]
                    d, t, _ = ic_diff(r["μ"], r["mezcla"], h)
                    L.append(f"| {label} | {h} | {stat(r['μ'], h)} | {stat(r['mezcla'], h)} | "
                             f"{stat(r['reversión'], h)} | {stat(r['momentum'], h)} | "
                             f"{d:+.4f} ({t:.2f}) | {stat(r['residual'], h)} | "
                             f"{r['corr'].mean():+.2f} |")
            L.append("")
        L += [("Correlación μ–mezcla: Spearman diaria entre μ_h y la mezcla (media)."
               + (" La mezcla y las reglas son las mismas para las dos ejecuciones (mismas "
                  "muestras)." if len(runs) == 2 else "")), ""]
    L += [f"{h2} Notas", ""]
    if res["checks"]:
        notes.append(f"Residuo frente a np.linalg.lstsq, en 5 días al azar por bloque (de los "
                     f"que cuentan en el IC, con al menos {MIN_DAY} muestras): máxima diferencia "
                     f"{max(res['checks']):.1e}.")
    notes.append(f"Días con al menos {MIN_DAY} muestras en que las dos reglas son colineales "
                 f"(sin residualizar): {res['degenerate']}.")
    L += [f"- {n}" for n in notes]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run o run:miembros (p. ej. final20 o coarse_month:0,1); por defecto, "
                         "coarse_month:0,1 y rel63")
    ap.add_argument("--out", default=str(ROOT / "logs" / "mix.md"))
    args = ap.parse_args()
    data = WindowData(DATA_DIR / "dataset")
    folds = cv_folds(data.samples)
    if args.runs is None:
        runs, labels, expected = RUNS, LABEL, N_MEMBERS
    else:
        runs, labels, expected = parse_runs(args.runs), {}, {}
        for run, members in runs.items():
            cfg = json.loads((DATA_DIR / "predictions" / run / "config.json").read_text())
            labels[run], expected[run] = describe(run, members, cfg["members"])
    L = render(compute(data, folds, runs, expected), folds, runs, labels)
    text = "\n".join(L) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"Escrito en {args.out}")


if __name__ == "__main__":
    main()
