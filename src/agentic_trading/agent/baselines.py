"""Líneas base del agente (handoff §11): reglas fijas con los mismos μ y σ del modelo, decididas
antes de ver los resultados. μ y σ en retornos (μ_5·σ_ventana y σ_5·σ_ventana); solo son
elegibles los activos con predicción ese día.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_weights(mu: pd.Series, sigma: pd.Series, min_weight: float | None = 0.01) -> pd.Series:
    """A. Tipo Kelly sin apalancamiento: pesos ∝ max(μ, 0)/σ², normalizados para que sumen 1.

    Con la regla del 1 % del juego (posiciones menores, a 0), se quedan las de mayor puntuación
    que, normalizadas entre ellas, pesan al menos min_weight, y suman 1; las demás, 0. Sin esto,
    con cientos de acciones casi todos los pesos quedarían por debajo del 1 % y la cartera, en
    caja. Si ninguna tiene μ > 0, todo a caja."""
    score = (np.maximum(mu, 0.0) / sigma ** 2).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = pd.Series(0.0, index=mu.index)
    pos = score[score > 0].sort_values(ascending=False, kind="stable")
    if pos.empty:
        return w
    k = len(pos)
    if min_weight:
        share = pos / pos.cumsum()          # peso de la k-ésima si se quedan las k primeras
        k = int((share >= min_weight * (1 - 1e-9)).sum())  # share decrece con k
    keep = pos.iloc[:k]
    w[keep.index] = keep / keep.sum()
    return w


def top_n_weights(mu: pd.Series, held, n: int = 20, exit_rank: int = 60) -> pd.Series:
    """B. Las n acciones de μ más alto, a partes iguales (1/n cada una). Una que ya está en
    cartera solo sale cuando cae fuera de las exit_rank primeras. Las posiciones sin predicción
    ese día (no están en mu) se mantienen y ocupan su hueco."""
    rank = mu.rank(ascending=False, method="first")
    held = list(held)
    kept = [s for s in held if s not in rank.index]
    slots = max(n - len(kept), 0)
    stay = sorted((s for s in held if s in rank.index and rank[s] <= exit_rank),
                  key=rank.get)[:slots]
    new = [s for s in rank.sort_values().index if s not in stay][:slots - len(stay)]
    w = pd.Series(0.0, index=mu.index)
    w[stay + new] = 1.0 / n
    return w


def equal_weights(mu: pd.Series) -> pd.Series:
    """Referencia: todos los activos con predicción, a partes iguales."""
    return pd.Series(1.0 / len(mu), index=mu.index) if len(mu) else pd.Series(dtype=float)


def weekly_dates(dates: pd.Series, min_share: float = 0.8) -> pd.DatetimeIndex:
    """Una fecha de decisión por semana: el último día laborable de cada semana con predicciones
    de al menos min_share de los activos del día con más de esa semana (evita decidir un día en
    que solo cotiza una parte del universo)."""
    cnt = pd.Series(dates).value_counts().sort_index()
    cnt = cnt[cnt.index.dayofweek < 5]
    week = cnt.index.to_period("W")
    full = cnt >= min_share * cnt.groupby(week).transform("max")
    ok = cnt[full]
    return pd.DatetimeIndex(ok.groupby(ok.index.to_period("W")).apply(lambda s: s.index.max()))
