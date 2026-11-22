"""Métricas de evaluación (handoff §9).

μ:
- IC de selección: Spearman entre activos, cada día; se resume con la media y su t de
  Newey-West (retardos = horizonte, porque los targets se solapan).
- Frente a las líneas base: diferencia diaria de IC con la mezcla fija de las dos (cada día, la
  media a partes iguales de los rangos entre activos de la reversión de 1 mes y del momentum de
  12 meses), con su t de Newey-West.
- Dirección: Spearman en el tiempo entre la media diaria de μ y el retorno medio del universo.
- Bucket test: retorno medio por quintil de μ (cada día) y diferencia Q5 − Q1 con t de NW.
- Acierto de signo por quintiles de s_h: diagnóstico, fuera del criterio de éxito (§9).
σ:
- QLIKE fuera de muestra y calibración (varianza predicha frente a realizada por deciles).
Consistencia entre horizontes y líneas base: momentum de 12 meses y reversión de 1 mes para μ;
HAR (log v_h sobre volatilidad diaria, semanal y mensual) para σ.

Todo en unidades de σ_ventana, como los targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_trading.data.dataset import yang_zhang

HORIZONS = (5, 10, 20)


# ----------------------------------------------------------------------------- estadística

def nw_tstat(x, lags: int) -> tuple[float, float]:
    """Media y t de Newey-West (kernel de Bartlett) de una serie."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    m = x.mean()
    e = x - m
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return m, m / np.sqrt(var / n) if var > 0 else np.nan


def daily_rank(dates: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rango percentil de x dentro de cada día."""
    return pd.Series(x).groupby(dates).rank(pct=True).to_numpy()


def daily_ic(dates, pred, target, min_n: int = 20) -> pd.Series:
    """IC de selección: correlación de Spearman entre activos, por día (Pearson de los rangos)."""
    p, y = daily_rank(dates, pred), daily_rank(dates, target)
    df = pd.DataFrame({"d": dates, "p": p, "y": y, "pp": p * p, "yy": y * y, "py": p * y})
    g = df.groupby("d").agg(["sum", "count"])
    n = g[("p", "count")]
    mp, my = g[("p", "sum")] / n, g[("y", "sum")] / n
    cov = g[("py", "sum")] / n - mp * my
    var_p, var_y = g[("pp", "sum")] / n - mp ** 2, g[("yy", "sum")] / n - my ** 2
    ic = cov / np.sqrt(var_p * var_y)
    return ic[(n >= min_n) & (var_p > 0) & (var_y > 0)].dropna()


def rank_residual(dates, x, z) -> np.ndarray:
    """Residuo, día a día (MCO), de los rangos de x sobre los de z: la parte de la ordenación de x
    que no explica z."""
    rx, rz = daily_rank(dates, x), daily_rank(dates, z)
    df = pd.DataFrame({"d": dates, "x": rx, "z": rz})
    cx = df["x"] - df.groupby("d")["x"].transform("mean")
    cz = df["z"] - df.groupby("d")["z"].transform("mean")
    num = (cx * cz).groupby(df["d"]).transform("sum")
    den = (cz * cz).groupby(df["d"]).transform("sum")
    b = (num / den.where(den > 0)).fillna(0.0)
    return (cx - b * cz).to_numpy()


def baseline_mix(dates, reversal, momentum) -> np.ndarray:
    """Mezcla fija de las dos líneas base de μ (handoff §9): cada día, la media a partes iguales
    de los rangos entre activos de la reversión de 1 mes y del momentum de 12 meses."""
    return 0.5 * daily_rank(dates, reversal) + 0.5 * daily_rank(dates, momentum)


def vs_mix(ic_model: pd.Series, ic_mix: pd.Series, lags: int) -> dict:
    """Diferencia diaria de IC entre el modelo y la mezcla (mismos días), con su t de
    Newey-West (lags: el horizonte, porque las muestras se solapan)."""
    diff = (ic_model - ic_mix).dropna().sort_index()
    mean, t = nw_tstat(diff.to_numpy(), lags)
    return {"dif. IC con la mezcla": mean, "t dif. IC mezcla": t}


def direction(dates, pred, target, lags: int) -> dict:
    """Spearman en el tiempo entre la media diaria de μ y el retorno medio del universo."""
    d = pd.DataFrame({"d": dates, "p": pred, "y": target}).groupby("d")[["p", "y"]].mean()
    rp, ry = d.p.rank(), d.y.rank()
    zp, zy = (rp - rp.mean()) / rp.std(), (ry - ry.mean()) / ry.std()
    rho, t = nw_tstat(zp * zy, lags)
    return {"rho": rho, "t": t, "days": len(d)}


def bucket_test(dates, pred, target, lags: int, n: int = 5) -> dict:
    """Retorno medio por quintil diario de μ y diferencia Q_n − Q_1."""
    q = np.minimum((daily_rank(dates, pred) * n).astype(int), n - 1)
    df = pd.DataFrame({"d": dates, "q": q, "y": target})
    m = df.groupby(["d", "q"]).y.mean().unstack()
    spread = (m[n - 1] - m[0]).dropna()
    mean, t = nw_tstat(spread, lags)
    return {"buckets": m.mean().round(4).tolist(), "spread": mean, "t": t}


def sign_by_confidence(mu, s, y, n: int = 5) -> list[float]:
    """Acierto de signo por quintiles de s (de más a menos confianza)."""
    q = pd.qcut(pd.Series(s).rank(method="first"), n, labels=False)
    hit = pd.Series(np.sign(mu) == np.sign(y))
    return hit.groupby(q.to_numpy()).mean().round(4).tolist()


def qlike(v, vhat):
    r = (v / vhat) ** 2
    return r - np.log(r) - 1


def sigma_calibration(vhat, v, n: int = 10) -> dict:
    """Varianza predicha frente a realizada por deciles de la predicción."""
    q = pd.qcut(pd.Series(vhat).rank(method="first"), n, labels=False).to_numpy()
    df = pd.DataFrame({"q": q, "p": vhat ** 2, "r": v ** 2})
    g = df.groupby("q").mean()
    return {"pred_var": g.p.round(3).tolist(), "real_var": g.r.round(3).tolist(),
            "ratio": float(df.r.mean() / df.p.mean())}


# ----------------------------------------------------------------------------- líneas base

def baseline_features(data, rows: np.ndarray) -> dict:
    """Predictores de las líneas base para las filas (velas t) dadas."""
    a = {k: np.asarray(data.arr[k]) for k in ("log_open", "log_high", "log_low", "log_close",
                                              "sigma")}
    lc, sig = a["log_close"], a["sigma"][rows].astype(float)
    out = {
        "momentum": lc[rows - 21] - lc[rows - 251],     # 12 meses saltándose el último
        "reversal": -(lc[rows] - lc[rows - 21]),        # 1 mes, con signo invertido
    }
    # HAR: volatilidad diaria (Parkinson), semanal y mensual (Yang-Zhang), en unidades de σ.
    pk = (a["log_high"] - a["log_low"])[rows] / (2 * np.sqrt(np.log(2)))
    eps = 0.01 * sig
    yz5 = yang_zhang(a["log_open"], a["log_high"], a["log_low"], lc, 5)
    yz22 = yang_zhang(a["log_open"], a["log_high"], a["log_low"], lc, 22)
    out["har_x"] = np.column_stack([
        np.log((pk + eps) / sig),
        np.log((yz5[rows - 5] + eps) / sig),            # velas t−4 … t
        np.log((yz22[rows - 22] + eps) / sig),          # velas t−21 … t
    ])
    return out


def fit_har(x: np.ndarray, v: np.ndarray, mask: np.ndarray) -> dict:
    """MCO de log v² sobre [1, x] por horizonte; predice E[v²] con corrección lognormal."""
    coefs = []
    X = np.column_stack([np.ones(len(x)), x])
    for j in range(v.shape[1]):
        m = mask[:, j] & np.isfinite(X).all(1)
        b, *_ = np.linalg.lstsq(X[m], np.log(v[m, j] ** 2), rcond=None)
        s2 = np.var(np.log(v[m, j] ** 2) - X[m] @ b)
        coefs.append((b, s2))
    return {"coefs": coefs}


def predict_har(model: dict, x: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(x)), x])
    return np.column_stack([np.sqrt(np.exp(X @ b + s2 / 2)) for b, s2 in model["coefs"]])


# ----------------------------------------------------------------------------- informe

def evaluate(data, pred: pd.DataFrame, har: np.ndarray | None = None) -> dict:
    """Métricas de un conjunto de predicciones del ensemble (columnas sample, mu_h, sigma_h,
    s_h). har: predicción de v_h de la línea base HAR para las mismas muestras."""
    idx = pred["sample"].to_numpy()
    smp = data.samples.iloc[idx]
    rows, dates = smp["row"].to_numpy(), smp["date"].to_numpy()
    base = baseline_features(data, rows)
    mix = baseline_mix(dates, base["reversal"], base["momentum"])
    report = {}
    for j, h in enumerate(HORIZONS):
        y = np.asarray(data.arr[f"y_{h}"])[rows].astype(float)
        v = np.asarray(data.arr[f"v_{h}"])[rows].astype(float)
        m = smp[f"mask_sigma_{h}"].to_numpy()
        mu, s = pred[f"mu_{h}"].to_numpy(), pred[f"s_{h}"].to_numpy()
        vhat = pred[f"sigma_{h}"].to_numpy() / np.sqrt(h)
        r, ics = {}, {}
        for name, p in (("modelo", mu), ("momentum", base["momentum"]),
                        ("reversión", base["reversal"]), ("mezcla", mix)):
            ics[name] = daily_ic(dates, p, y)
            mean, t = nw_tstat(ics[name], h)
            r[f"IC {name}"] = mean
            r[f"t IC {name}"] = t
        r.update(vs_mix(ics["modelo"], ics["mezcla"], h))
        r["dirección"] = direction(dates, mu, y, h)
        r["buckets"] = bucket_test(dates, mu, y, h)
        r["signo por confianza"] = sign_by_confidence(mu, s, y)     # diagnóstico
        r["QLIKE modelo"] = float(qlike(v[m], vhat[m]).mean())
        r["QLIKE σ_ventana"] = float(qlike(v[m], np.ones(m.sum())).mean())
        if har is not None:
            r["QLIKE HAR"] = float(qlike(v[m], har[m, j]).mean())
        r["calibración σ"] = sigma_calibration(vhat[m], v[m])
        report[h] = r
    ics = {h: daily_ic(dates, pred[f"mu_{h}"].to_numpy(), pred["mu_5"].to_numpy())
           for h in (10, 20)}
    s5, s10, s20 = (pred[f"sigma_{h}"].to_numpy() for h in HORIZONS)
    report["consistencia"] = {
        "Spearman μ_5–μ_10": float(ics[10].mean()), "Spearman μ_5–μ_20": float(ics[20].mean()),
        "σ_5 ≤ σ_10 ≤ σ_20": float(((s5 <= s10) & (s10 <= s20)).mean()),
    }
    return report


MU_HORIZONS = (5, 10)
T_MIN = 2.0


def success_checks(report: dict) -> dict:
    """Criterio de éxito del handoff §9 (fijado el 9-11-2026; se aplica al test, en desarrollo es
    orientativo). Devuelve {"μ": {...}, "σ": {...}}, cada uno con "ok" y el detalle por horizonte.

    - μ: a h = 5 o 10, la diferencia diaria de IC con la mezcla fija de las dos líneas base es
      positiva con t de Newey-West > 2 (retardos = h, ya aplicados en evaluate).
    - σ: QLIKE menor que la de HAR en los tres horizontes; si falta la de HAR, falla.
    La confianza (s_h) no entra en el criterio."""
    mu = {}
    for h in MU_HORIZONS:
        r = report[h]
        d, t = r.get("dif. IC con la mezcla", np.nan), r.get("t dif. IC mezcla", np.nan)
        mu[h] = {"IC μ": r.get("IC modelo", np.nan), "IC mezcla": r.get("IC mezcla", np.nan),
                 "dif.": d, "t NW": t, "ok": bool(d > 0 and t > T_MIN)}
    sig = {}
    for h in HORIZONS:
        q, qh = report[h]["QLIKE modelo"], report[h].get("QLIKE HAR", np.nan)
        sig[h] = {"QLIKE": q, "QLIKE HAR": qh, "ok": bool(np.isfinite(qh) and q < qh)}
    return {"μ": {"criterio": f"dif. diaria de IC con la mezcla > 0, t NW > {T_MIN:g}, "
                              "a 5 o 10 días",
                  "ok": any(v["ok"] for v in mu.values()), "por horizonte": mu},
            "σ": {"criterio": "QLIKE mejor que HAR en los tres horizontes",
                  "ok": all(v["ok"] for v in sig.values()), "por horizonte": sig}}
