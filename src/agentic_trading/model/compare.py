"""Comparaciones entre ejecuciones de validación cruzada (pruebas sobre μ).

- MSE de μ frente a μ = 0 entre activos (restando a μ y a y su media diaria), fuera de muestra y
  en entrenamiento (con los checkpoints, sobre una submuestra fija de fechas de entrenamiento).
- Ensembles de los k primeros miembros, a partir de <bloque>_members.parquet.
- Diferencia diaria de IC entre dos predictores, con su t de Newey-West.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.model.metrics import daily_ic, nw_tstat
from agentic_trading.model.train import ensemble, load_model, predict
from agentic_trading.model.tuning import fold_scores

H2 = (5, 10)
N_TRAIN = 200_000


def targets(data, idx, horizons=H2):
    """Fechas, filas y y_h de las muestras idx."""
    smp = data.samples.iloc[idx]
    rows = smp["row"].to_numpy()
    return smp["date"].to_numpy(), rows, {h: np.asarray(data.arr[f"y_{h}"])[rows].astype(float)
                                          for h in horizons}


def cs_sums(dates, mu, y) -> tuple[float, float]:
    """Σ e² y Σ y² entre activos: μ e y menos su media diaria; e = y − μ."""
    mu_d = pd.Series(mu).groupby(dates).transform("mean").to_numpy()
    y_d = pd.Series(y).groupby(dates).transform("mean").to_numpy()
    e0 = y - y_d
    e = e0 - (mu - mu_d)
    return float(e @ e), float(e0 @ e0)


def rel_mse(sums) -> float:
    """MSE relativo a μ = 0, en %."""
    return 100 * (sums[0] / sums[1] - 1)


def pooled_rel_mse(sums: list) -> float | None:
    """MSE relativo a μ = 0 (%) de varios bloques juntos."""
    return rel_mse((sum(s[0] for s in sums), sum(s[1] for s in sums))) if sums else None


def train_subsample(data, f: dict, k: int, n: int = N_TRAIN) -> np.ndarray:
    """Fechas completas de entrenamiento del bloque f al azar (semilla k) hasta llegar a n
    muestras."""
    idx = f["train"]
    dates = data.samples["date"].to_numpy()[idx]
    _, inv, counts = np.unique(dates, return_inverse=True, return_counts=True)
    order = np.random.default_rng(k).permutation(len(counts))
    take = order[:np.searchsorted(np.cumsum(counts[order]), n) + 1]
    return idx[np.isin(inv, take)]


def load_members(pred_dir: Path, fold: str, members=None) -> dict[int, pd.DataFrame]:
    """Predicciones de cada miembro ({nº de miembro: DataFrame}) de <pred_dir>/<fold>_members."""
    mem = pd.read_parquet(Path(pred_dir) / f"{fold}_members.parquet")
    out = {int(m): g.drop(columns="member").reset_index(drop=True)
           for m, g in mem.groupby("member", sort=True)}
    if members is not None:
        out = {m: out[m] for m in members if m in out}
    idx = next(iter(out.values()))["sample"].to_numpy()
    assert all((p["sample"].to_numpy() == idx).all() for p in out.values()), "miembros desalineados"
    return out


def oos_stats(data, ens: pd.DataFrame) -> dict:
    """IC y QLIKE (fold_scores), sumas del MSE entre activos e IC diario, fuera de muestra."""
    dates, _, y = targets(data, ens["sample"].to_numpy())
    r = fold_scores(data, ens)
    for h in H2:
        mu = ens[f"mu_{h}"].to_numpy()
        r[f"oos_{h}"] = cs_sums(dates, mu, y[h])
        r[f"icd_{h}"] = daily_ic(dates, mu, y[h])
    return r


def insample_stats(data, paths, sub) -> dict:
    """Con los checkpoints de los miembros (paths): sumas del MSE entre activos del ensemble en
    la submuestra de entrenamiento sub, segundos por época y L_μ por época (media de miembros)."""
    preds, secs, l_mu = [], [], []
    for p in paths:
        model, ck = load_model(p)
        preds.append(predict(model, data, sub))
        secs += [rec["seconds"] for rec in ck["history"]]
        l_mu.append([rec["l_mu"] for rec in ck["history"]])
    e = ensemble(preds)
    dates, _, y = targets(data, sub)
    out = {f"ins_{h}": cs_sums(dates, e[f"mu_{h}"].to_numpy(), y[h]) for h in H2}
    out["sec_epoch"] = float(np.mean(secs))
    out["n_train_sub"] = len(sub)
    out["l_mu_epochs"] = np.mean(l_mu, 0).tolist()
    return out


def ic_diff(ic_a: pd.Series, ic_b: pd.Series, lags: int) -> tuple[float, float, int]:
    """Media y t de Newey-West de la diferencia diaria de IC (a − b), y número de días."""
    diff = (ic_a - ic_b).dropna().sort_index()
    mean, t = nw_tstat(diff.to_numpy(), lags)
    return mean, t, len(diff)
