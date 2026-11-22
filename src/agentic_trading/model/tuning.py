"""Ajuste de hiperparámetros con los bloques de validación cruzada.

Comparación pareada: en todas las configuraciones, el miembro m del bloque k usa la misma semilla
(1000·k + m), y con ella la misma inicialización, el mismo remuestreo por meses y las mismas
submuestras por época. Solo cambia la configuración.

Regla de elección (por bloque: IC = media del IC de selección a 5 y 10 días; QLIKE = media en
los tres horizontes):
- Referencia: la configuración con menor QLIKE media.
- Se elige por IC la de mayor IC medio solo si su IC supera al de la referencia en al menos 4 de
  los 5 bloques y su QLIKE media no es más de un 2 % peor. Si no, se queda la referencia.
- Si el λ elegido está en un extremo de la rejilla, se amplía en esa dirección (×3) y se repite.

Reanudación: los resultados guardan la huella del dataset (check_dataset); si el dataset ha
cambiado, el ajuste no se reanuda.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.model.metrics import daily_ic, qlike

HORIZONS = (5, 10, 20)


def fold_scores(data, ens: pd.DataFrame) -> dict:
    """IC (media de h=5 y h=10) y QLIKE (media de los tres horizontes) de un bloque."""
    smp = data.samples.iloc[ens["sample"].to_numpy()]
    rows, dates = smp["row"].to_numpy(), smp["date"].to_numpy()
    out = {}
    for h in HORIZONS:
        y = np.asarray(data.arr[f"y_{h}"])[rows].astype(float)
        v = np.asarray(data.arr[f"v_{h}"])[rows].astype(float)
        m = smp[f"mask_sigma_{h}"].to_numpy()
        out[f"ic_{h}"] = float(daily_ic(dates, ens[f"mu_{h}"].to_numpy(), y).mean())
        vhat = ens[f"sigma_{h}"].to_numpy() / np.sqrt(h)
        out[f"qlike_{h}"] = float(qlike(v[m], vhat[m]).mean())
    out["ic"] = (out["ic_5"] + out["ic_10"]) / 2
    out["qlike"] = float(np.mean([out[f"qlike_{h}"] for h in HORIZONS]))
    return out


def select(results: pd.DataFrame, min_wins: int = 4, max_qlike_loss: float = 0.02) -> dict:
    """results: una fila por (config, fold) con columnas ic y qlike. Devuelve la elección."""
    per = results.groupby("config")[["ic", "qlike"]].mean()
    ref = per["qlike"].idxmin()
    cand = per["ic"].idxmax()
    pivot = results.pivot(index="fold", columns="config", values="ic")
    wins = int((pivot[cand] > pivot[ref]).sum()) if cand != ref else len(pivot)
    q_loss = per.loc[cand, "qlike"] / per.loc[ref, "qlike"] - 1
    if cand != ref and wins >= min_wins and q_loss <= max_qlike_loss:
        chosen, reason = cand, f"IC: mejora en {wins}/{len(pivot)} bloques, QLIKE {q_loss:+.1%}"
    elif cand == ref:
        chosen, reason = ref, "mejor IC y mejor QLIKE a la vez"
    else:
        chosen, reason = ref, (f"QLIKE: la de mejor IC ({cand}) gana en {wins}/{len(pivot)} "
                               f"bloques y su QLIKE es {q_loss:+.1%}")
    return {"config": chosen, "reason": reason, "table": per}


def check_dataset(out_dir: Path, dataset: dict) -> None:
    """Guarda la huella del dataset (dataset.dataset_id) junto a los resultados del ajuste, o
    comprueba que coincide con la guardada. Si no coincide, o hay resultados sin huella (de antes
    de guardarla), no se reanuda: se para con un mensaje."""
    out_dir = Path(out_dir)
    path = out_dir / "dataset.json"
    dataset = json.loads(json.dumps(dataset, default=str))
    if path.exists():
        if json.loads(path.read_text()) != dataset:
            raise SystemExit(f"{out_dir}: los resultados son de otro dataset (dataset.json no "
                             "coincide). No se reanuda: usa otro --name.")
        return
    if (out_dir / "results.parquet").exists():
        raise SystemExit(f"{out_dir}: hay resultados sin huella del dataset; no se puede saber "
                         "si son del actual. No se reanuda: usa otro --name.")
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=1))


def extend_lambda(grid: list[float], chosen: float) -> float | None:
    """Nuevo λ si el elegido está en un extremo de la rejilla (×3 hacia fuera), o None."""
    g = sorted(grid)
    if np.isclose(chosen, g[0]):
        return round(g[0] / 3, 6)
    if np.isclose(chosen, g[-1]):
        return round(g[-1] * 3, 6)
    return None
