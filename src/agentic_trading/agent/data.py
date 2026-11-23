"""Datos del agente (handoff §10): observaciones del modelo y precios, en un panel por día de
decisión.

- Predicciones: el ensemble de una ejecución de validación cruzada (por defecto final20, los 20
  miembros: data/predictions/<run>/cv*.parquet), fuera de muestra en 2000–2019. El test (2020–)
  no se toca: WindowData sin allow_test, y nada pasa del 31-12-2019.
- Días de decisión: los laborables con predicción para al menos MIN_ASSETS activos (los fines de
  semana solo tienen cripto, y unos pocos festivos casi nada). En un festivo de un mercado, sus
  posiciones se mantienen (sin predicción ese día).
- Variables por activo y día (las de §2, más el estado de la posición, que añade el entorno):
  μ_h (h = 5, 10, 20; unidades de σ_ventana, como sale del modelo), log σ_h en retornos
  (σ_h · σ_ventana: el stop d está en precio, así que la red necesita la volatilidad absoluta;
  en unidades de σ_ventana, σ_h ≈ √h en todos los activos) y s_h (unidades de σ_ventana). Se
  estandarizan con la media y la desviación típica de todo el entrenamiento (2000–2015), las
  mismas para todos los días: así la red ve el régimen (p. ej., si sube la volatilidad de todo
  el mercado). z_mu5: μ_5 estandarizado entre activos cada día, para iniciar la puntuación (§10).
- ew_logret[k]: log-retorno de los pesos iguales entre los cierres de los días de decisión k y
  k + 1, sobre los activos con predicción en k (sin costes; cierres empalmados del simulador).
  Sirve para la recompensa relativa (ppo.PPOConfig.reward): no depende de las acciones del
  agente, así que restarlo no cambia la política óptima, solo quita la varianza del mercado.
- freq = "weekly" (plan B de §10, 21-11-2026): un día de decisión por semana, el de
  baselines.weekly_dates (el último laborable de la semana con predicción para al menos el 80 %
  de los activos del día con más de esa semana), el mismo de las líneas base. El simulador sigue
  avanzando día a día entre decisiones (stops diarios).
- signal (prueba de capacidad, 21-11-2026): sustituye μ_h por una señal sintética con un IC
  conocido, hecha con el futuro: s_h = ρ·z(y_h − media del día) + √(1 − ρ²)·ε, con z la
  estandarización entre activos de cada día y ε ~ N(0, 1) (semilla fija). Sirve para separar
  «el RL no aprende» de «no hay señal que aprender»; nunca para operar. Con prior = "none",
  z_mu5 = 0: la puntuación inicial no usa la señal y la red tiene que aprender a usarla.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.agent.simulator import Market
from agentic_trading.config import DATA_DIR
from agentic_trading.data.dataset import WindowData, dataset_id

H = (5, 10, 20)
BASE_FEATURES = [f"mu_{h}" for h in H] + [f"logsig_{h}" for h in H] + [f"s_{h}" for h in H]
MIN_ASSETS = 20
END = pd.Timestamp("2019-12-31")
TRAIN_END = pd.Timestamp("2015-12-31")
VALID_START = pd.Timestamp("2016-01-01")


@dataclass
class Panel:
    market: Market
    dec_t: np.ndarray           # índice en market.dates de cada día de decisión
    dec_dates: pd.DatetimeIndex
    elig: list[np.ndarray]      # por día de decisión: activos con predicción (índices de market)
    feats: list[np.ndarray]     # por día: [n_elig, len(BASE_FEATURES)] estandarizadas
    z_mu5: list[np.ndarray]     # por día: μ_5 estandarizado entre activos
    close_ffill: np.ndarray     # último cierre válido hasta cada fecha (T × N)
    norm: dict                  # media y desviación típica de las variables (entrenamiento)
    fingerprint: dict
    ew_logret: np.ndarray | None = None     # [n_dec]; el último, NaN

    def k_range(self, start, end) -> tuple[int, int]:
        """Índices [k0, k1) de los días de decisión entre start y end (incluidos)."""
        d = self.dec_dates
        return int(d.searchsorted(pd.Timestamp(start))), int(d.searchsorted(pd.Timestamp(end),
                                                                          "right"))


def file_hash(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(map(Path, paths)):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load_predictions(data: WindowData, run: str) -> tuple[pd.DataFrame, dict]:
    pdir = DATA_DIR / "predictions" / run
    paths = sorted(pdir.glob("cv[0-9].parquet"))
    if not paths:
        raise SystemExit(f"No hay predicciones de {run}")
    pr = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    smp = data.samples.iloc[pr["sample"].to_numpy()]
    pr["date"] = smp["date"].to_numpy()
    pr["asset"] = smp["asset"].to_numpy()
    pr["sigma_v"] = np.asarray(data.arr["sigma"])[smp["row"].to_numpy()].astype(float)
    pr = pr[pr["date"] <= END]
    cfg = json.loads((pdir / "config.json").read_text())
    fp = {"run": run, "predictions_sha256": file_hash(paths), "members": cfg.get("members"),
          "dataset": dataset_id(data.meta)}
    return pr, fp


def synthetic_signal(y: np.ndarray, dates: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """ρ·z(y − media del día) + √(1 − ρ²)·ε, por día (ver el docstring del módulo)."""
    y = pd.Series(y)
    g = y.groupby(dates)
    zy = ((y - g.transform("mean")) / g.transform("std")).fillna(0.0).to_numpy()
    eps = np.random.default_rng(seed).normal(size=len(zy))
    return rho * zy + np.sqrt(1 - rho ** 2) * eps


def build_panel(run: str = "final20", data: WindowData | None = None,
                symbols_limit: int | None = None, freq: str = "daily",
                signal: float | None = None, signal_seed: int = 0,
                prior: str = "mu5", rank: bool = False) -> Panel:
    """Panel de 2000–2019 (ver el docstring del módulo). symbols_limit: solo para pruebas.
    freq: "daily" o "weekly"; signal: ρ de la señal sintética (None: las predicciones);
    prior: "mu5" o "none" (z_mu5 = 0). rank (23-11-2026, clonado de la regla fija): añade al
    final de las variables el puesto de μ_5 entre los activos del día (0 = el mayor) / 60, con
    tope 5, sin estandarizar (la regla fija vende al salir de los 60 primeros)."""
    assert freq in ("daily", "weekly") and prior in ("mu5", "none")
    data = data or WindowData(DATA_DIR / "dataset")
    pr, fp = load_predictions(data, run)
    fp = {**fp, "freq": freq, "signal": signal, "signal_seed": signal_seed, "prior": prior}
    if rank:
        fp["rank"] = True
    if signal is not None:
        rows = data.samples["row"].to_numpy()[pr["sample"].to_numpy()]
        for i, h in enumerate(H):
            y = np.asarray(data.arr[f"y_{h}"])[rows].astype(float)
            pr[f"mu_{h}"] = synthetic_signal(y, pr["date"].to_numpy(), signal,
                                             signal_seed * 10 + i)
    symbols = list(data.meta["symbols"])
    assets = np.sort(pr["asset"].unique())
    if symbols_limit:
        assets = assets[:symbols_limit]
        pr = pr[pr["asset"].isin(assets)]
    names = [symbols[a] for a in assets]
    clean = DATA_DIR / "clean"
    start = pr["date"].min() - pd.Timedelta(days=10)
    market = Market.from_frames({s: pd.read_parquet(clean / f"{s}.parquet") for s in names},
                                start=start, end=END)
    col = {a: j for j, a in enumerate(assets)}
    pr = pr.assign(j=pr["asset"].map(col))
    for h in H:
        pr[f"logsig_{h}"] = np.log(pr[f"sigma_{h}"] * pr["sigma_v"])

    cnt = pr.groupby("date").size()
    days = cnt[(cnt.index.dayofweek < 5) & (cnt >= MIN_ASSETS)].index
    if freq == "weekly":
        days = days.intersection(weekly_dates(pr["date"]))
    pr = pr[pr["date"].isin(days)]
    train = pr["date"] <= TRAIN_END
    mean = pr.loc[train, BASE_FEATURES].mean()
    sd = pr.loc[train, BASE_FEATURES].std()
    X = ((pr[BASE_FEATURES] - mean) / sd).to_numpy(np.float32)
    mu5 = pr["mu_5"].to_numpy()
    g = pr.groupby("date")["mu_5"]
    z = ((mu5 - g.transform("mean").to_numpy()) / g.transform("std").to_numpy()).astype(np.float32)
    z = np.nan_to_num(z)
    if prior == "none":
        z[:] = 0.0

    pos = market.dates.get_indexer(days)
    assert (pos >= 0).all(), "días de decisión sin fecha en el mercado"
    order = np.argsort(pr["date"].to_numpy(), kind="stable")
    dsorted = pr["date"].to_numpy()[order]
    bounds = np.searchsorted(dsorted, days.to_numpy())
    bounds = np.r_[bounds, len(order)]
    jj = pr["j"].to_numpy()
    elig, feats, zs = [], [], []
    for k in range(len(days)):
        rows = order[bounds[k]:bounds[k + 1]]
        rows = rows[np.argsort(jj[rows], kind="stable")]
        elig.append(jj[rows].astype(np.int64))
        f = X[rows]
        if rank:
            r = np.empty(len(rows))
            r[np.argsort(-mu5[rows], kind="stable")] = np.arange(len(rows))
            f = np.c_[f, np.minimum(r / 60.0, 5.0)].astype(np.float32)
        feats.append(f)
        zs.append(z[rows])
    close_ffill = pd.DataFrame(market.close).ffill().to_numpy()
    norm = {"mean": mean.to_dict(), "sd": sd.to_dict()}
    dec_t = pos.astype(np.int64)
    return Panel(market, dec_t, pd.DatetimeIndex(days), elig, feats, zs, close_ffill, norm, fp,
                 equal_weight_logret(close_ffill, dec_t, elig))


def equal_weight_logret(close_ffill: np.ndarray, dec_t: np.ndarray, elig: list) -> np.ndarray:
    """Log-retorno de los pesos iguales (activos con predicción en k) de la decisión k a k + 1."""
    out = np.full(len(dec_t), np.nan)
    for k in range(len(dec_t) - 1):
        a, b = close_ffill[dec_t[k], elig[k]], close_ffill[dec_t[k + 1], elig[k]]
        ok = np.isfinite(a) & np.isfinite(b)
        out[k] = np.log(np.mean(b[ok] / a[ok])) if ok.any() else 0.0
    return out
