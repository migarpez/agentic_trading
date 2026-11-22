"""Horquilla estimada con velas diarias: Abdi y Ranaldo (2017), «A Simple Estimation of Bid-Ask
Spreads from Daily Close, High, and Low Prices», Review of Financial Studies 30(12).

Con c_t el log del cierre y η_t = (log H_t + log L_t) / 2 el punto medio del rango del día:

    S² = 4 · E[(c_t − η_t) · (c_t − η_{t+1})]

La media se toma en cada mes natural, sobre los pares de velas consecutivas válidas; un mes con
media negativa da S = 0 (la versión mensual del artículo). S es la horquilla relativa (fracción
del precio): operar ida y vuelta cuesta ≈ S.

Precios de data/clean/ sin el ajuste por dividendos (P / adj_factor), para que el factor no
cambie entre t y t+1 en las fechas ex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def abdi_ranaldo(df: pd.DataFrame, min_pairs: int = 15) -> pd.Series:
    """Horquilla mensual (fracción) de un activo de data/clean/, indexada por el mes (Period).
    Solo los meses con al menos min_pairs pares de velas consecutivas válidas."""
    f = np.log(df["adj_factor"].to_numpy(float)) if "adj_factor" in df else 0.0
    c = np.log(df["close"].to_numpy(float)) - f
    eta = (np.log(df["high"].to_numpy(float)) + np.log(df["low"].to_numpy(float))) / 2 - f
    valid = df["valid"].to_numpy(bool) if "valid" in df else np.ones(len(df), bool)
    ok = valid[:-1] & valid[1:]
    prod = ((c[:-1] - eta[:-1]) * (c[:-1] - eta[1:]))[ok]
    month = pd.DatetimeIndex(df.index[:-1][ok]).to_period("M")
    g = pd.Series(prod, index=month).groupby(level=0)
    m = g.mean()[g.size() >= min_pairs]
    return np.sqrt(np.maximum(4 * m, 0.0))
