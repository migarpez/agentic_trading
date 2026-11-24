"""Bloques de validación cruzada (desarrollo 2000–2019) y años del walk-forward (test).

Hueco de 20 velas a cada lado del bloque que se predice: una muestra de entrenamiento se descarta
si su intervalo [t, t+20] toca el bloque ampliado [inicio − 20, fin + 20] (en días hábiles). Así
ningún target de entrenamiento llega a menos de 20 velas del bloque y el entrenamiento se
reanuda 20 velas después de que terminen los targets del bloque.

manual_fold: un bloque con los años de entrenamiento fijados a mano (prueba de mirar el futuro,
scripts/lookahead.py), con el mismo hueco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_trading.data.dataset import SPLITS, split_mask

CV_BLOCKS = [
    ("2000-01-01", "2003-12-31"),
    ("2004-01-01", "2007-12-31"),
    ("2008-01-01", "2011-12-31"),
    ("2012-01-01", "2015-12-31"),
    ("2016-01-01", "2019-12-31"),
]
GAP = pd.offsets.BDay(20)


def cv_folds(samples: pd.DataFrame, blocks=CV_BLOCKS) -> list[dict]:
    """Por bloque: índices de entrenamiento (resto de desarrollo, con hueco) y de predicción."""
    dev = split_mask(samples, "dev")
    date, end = samples["date"], samples["target_end"]
    folds = []
    for k, (a, b) in enumerate(blocks):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        pred = dev & (date >= a).to_numpy() & (date <= b).to_numpy()
        touch = (end >= a - GAP).to_numpy() & (date <= b + GAP).to_numpy()
        train = dev & ~touch
        folds.append({"name": f"cv{k}", "block": (a.date(), b.date()),
                      "train": np.flatnonzero(train), "predict": np.flatnonzero(pred)})
    return folds


def parse_years(text: str) -> tuple[int, int]:
    """'2000-2011' → (2000, 2011); '2005' → (2005, 2005)."""
    a, _, b = text.partition("-")
    return int(a), int(b or a)


def manual_fold(samples: pd.DataFrame, k: int, train_years: list[tuple[int, int]],
                n: int | None = None, seed: int = 0, blocks=CV_BLOCKS) -> dict:
    """Bloque k de cv_folds con los años de entrenamiento fijados a mano: de sus muestras de
    entrenamiento (con el mismo hueco de 20 velas alrededor del bloque predicho), solo las de
    fecha dentro de train_years (rangos de años, ambos incluidos). n: submuestra al azar, con
    semilla fija, hasta n muestras. Mismo nombre (cvK) y mismas muestras de predicción."""
    f = cv_folds(samples, blocks)[k]
    a, b = f["block"]
    for lo, hi in train_years:
        if lo > hi or (lo <= b.year and hi >= a.year):
            raise ValueError(f"años de entrenamiento {lo}–{hi}: vacíos o dentro del bloque "
                             f"predicho ({a.year}–{b.year})")
    year = samples["date"].dt.year.to_numpy()[f["train"]]
    keep = np.zeros(len(year), bool)
    for lo, hi in train_years:
        keep |= (year >= lo) & (year <= hi)
    train = f["train"][keep]
    if n is not None:
        if n > len(train):
            raise ValueError(f"n = {n:,} es mayor que las {len(train):,} muestras de entrenamiento")
        train = np.sort(np.random.default_rng(seed).choice(train, size=n, replace=False))
    return {**f, "train": train, "train_years": [list(r) for r in train_years],
            "train_n": n, "subsample_seed": seed if n is not None else None}


def walk_forward_years(samples: pd.DataFrame, years=None, split: str = "test") -> list[dict]:
    """Test: para cada año, entrenar con todo lo anterior (targets ≥ 20 velas antes del 1 de
    enero) y predecir ese año. Solo se usa al abrir el test.

    years y split (23-11-2026, ensayo general de scripts/test_run.py): los años que se predicen y
    el bloque de donde salen sus muestras (p. ej. 2016–2019 de "dev"); por defecto, todos los
    años del test."""
    start, end = (pd.Timestamp(x) for x in SPLITS[split])
    test = split_mask(samples, split)
    out = []
    for year in (years if years is not None else range(start.year, end.year + 1)):
        a = pd.Timestamp(f"{year}-01-01")
        b = min(pd.Timestamp(f"{year}-12-31"), end)
        train = (samples["target_end"] < a - GAP).to_numpy()
        pred = test & (samples["date"] >= a).to_numpy() & (samples["date"] <= b).to_numpy()
        out.append({"name": f"wf{year}", "train": np.flatnonzero(train),
                    "predict": np.flatnonzero(pred)})
    return out
