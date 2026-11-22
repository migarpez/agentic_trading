"""Limpieza de las velas a partir de las incidencias de la auditoría.

Reglas (en este orden), todas trazadas en una tabla de acciones:

1. Cortar el inicio del histórico
   - en el último salto de volumen ≥ ×100 (acciones): datos de otra entidad antes de cotizar;
   - tras un tramo denso de errores de precio (≥ 5 en cadena con < 365 días entre ellos) que
     termina en la primera mitad del histórico. Si termina después, el tramo se marca no válido.
2. Reparar saltos con ratio conocido (split registrado sin ajustar, split sin registrar,
   peniques/libras): se reescalan los precios anteriores al salto (y el volumen, en los splits).
3. Quitar velas: spikes (error), spikes de unidad, sesiones sin negociación, precios congelados
   (salvo la primera vela del tramo), H < L, precio ≤ 0, OHLC ausente, fin de semana y días que
   casi ningún par de su bolsa cotiza.
4. OHLC: mechas y aperturas aisladas absurdas (≥ 0,5 en log, ×1,65, frente al cuerpo o al cierre
   anterior; p. ej. un mínimo ÷100 en Londres) se reparan confiando en el cierre: la apertura pasa
   al cierre anterior y la mecha se recorta al cuerpo (ohlc_repaired=True). Si el cierre está dentro de [L, H] y la apertura fuera, la apertura se lleva al rango;
   después H y L se amplían para contener O y C. Si el desajuste era > 1 %, la vela se marca
   ohlc_repaired=True (sigue siendo válida); con ohlc_large="mask" pasa además a no válida.
5. Marcar no válidas (valid=False) las incidencias graves sin reparar: saltos con ratio de split
   sin corroborar, los saltos extremos (≥ ×2 con z ≥ 50) sin ratio de split, la vela siguiente a un precio ≤ 0 quitado, la vela que sigue a un hueco de
   más de 10 días naturales, los años con precios en escalones ≥ 2 % (coarse_prices) y los años
   con O/H/L corruptos (rango diario mediano ≥ 8× el |retorno| mediano, corrupt_ohlc).
6. Ajuste por dividendos con un factor propio, calculado con la columna de dividendos de Yahoo
   (no con su adj_close, que en Londres aplica los dividendos ÷100 y tiene tramos corruptos):
   f_t = Π_{e > t} (1 − D_e / C_{e−1}) sobre las fechas ex e, ignorando los dividendos que la
   auditoría marca como bad_dividend. OHLC × f; el volumen no se ajusta.
7. Tras la última auditoría (clean_universe): si queda un salto grande a ≤ 2 velas de un split
   registrado, de un dividendo ≥ 5 % del precio (aplicado o no) o de una reparación, esa vela se
   marca no válida (operación corporativa mal registrada).

Salida por activo: open, high, low, close (ajustados), volume, adj_factor, valid, ohlc_repaired.
Una muestra (ventana + target) es utilizable solo si todas sus velas son válidas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]
ACTION_COLUMNS = ["symbol", "date", "action", "reason", "detail"]

REPAIR_CHECKS = ["split_not_adjusted", "unrecorded_split", "unit_shift"]
SPLIT_REPAIRS = ["split_not_adjusted", "unrecorded_split"]  # cambian el nº de acciones; unit_shift no
DROP_CHECKS = ["spike", "unit_spike", "no_trade", "stale_price", "high_lt_low",
               "nonpositive_price", "missing_ohlc", "weekend_bar", "extra_day"]
MASK_CHECKS = ["split_like_move", "coarse_prices", "extreme_move", "corrupt_ohlc"]
YEAR_MASKS = ["coarse_prices", "corrupt_ohlc"]
OUTLIER_CHECKS = ["wick_outlier", "open_outlier"]
DENSE_CHECKS = ["spike", "unit_spike", "high_lt_low", "nonpositive_price", "missing_ohlc"]


@dataclass(frozen=True)
class CleanParams:
    ohlc_clip_max: float = 0.01      # desajuste OHLC por encima del cual es grave
    ohlc_large: str = "repair"       # "repair": se repara y marca; "mask": además no válida
    outlier_repair: float = 0.5      # mecha o apertura aislada a ≥ esto en log (×1,65): se repara
    volume_jump_truncate: float = 100.0
    dense_errors: int = 5
    dense_gap_days: int = 365
    dense_truncate_frac: float = 0.5
    max_gap_days: int = 10           # la vela tras un hueco mayor (días naturales) no es válida
    corp_window: int = 2             # velas alrededor de una operación corporativa
    corp_min_yield: float = 0.05     # dividendo (fracción del precio) que cuenta como operación
    window: int = 252
    horizon: int = 20


DEFAULT_CLEAN = CleanParams()


def _log(actions, symbol, date, action, reason, detail=""):
    actions.append((symbol, pd.Timestamp(date), action, reason, detail))


def _dense_regions(dates: pd.Series, min_n: int, gap_days: int) -> list[tuple]:
    """Tramos de fechas encadenadas (separación < gap_days) con al menos min_n eventos."""
    d = pd.Series(sorted(dates.unique()))
    if len(d) < min_n:
        return []
    cut = (d.diff().dt.days >= gap_days).cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in d.groupby(cut) if len(g) >= min_n]


def dividend_factor(df: pd.DataFrame, ignore=()) -> pd.Series:
    """Factor de ajuste por dividendos: f_t = Π_{e > t} (1 − D_e / C_{e−1}) sobre las fechas ex e
    (salvo las de `ignore`). Vale 1 en la última vela y baja hacia el pasado."""
    y = (df["dividends"].fillna(0.0) / df["close"].shift()).fillna(0.0)
    use = (y > 0) & (y < 1) & ~df.index.isin(pd.DatetimeIndex(list(ignore)))
    g = np.where(use, np.log1p(-y.where(use, 0.0)), 0.0)
    cum = np.cumsum(g)
    return pd.Series(np.exp(cum[-1] - cum), index=df.index) if len(cum) else y * 0 + 1.0


def corporate_action_dates(raw: pd.DataFrame, min_yield: float) -> pd.DatetimeIndex:
    """Fechas con operaciones que pueden mover mucho el precio: splits registrados y dividendos
    de al menos min_yield del precio."""
    splits = raw["stock_splits"].fillna(0)
    y = (raw["dividends"].fillna(0) / raw["close"].shift()).fillna(0)
    return raw.index[(~splits.isin([0, 1])) | (y >= min_yield)]


def _mask_dates(index: pd.DatetimeIndex, r) -> pd.DatetimeIndex:
    """Velas que marca una incidencia de MASK_CHECKS: su fecha, o su año para YEAR_MASKS."""
    if r.check in YEAR_MASKS:
        return index[index.year == r.date.year]
    return index[index == r.date]


def _outlier_dates(flags: pd.DataFrame, p: CleanParams) -> pd.DatetimeIndex:
    f = flags[flags["check"].isin(OUTLIER_CHECKS) & (flags["value"].abs() >= p.outlier_repair)]
    return pd.DatetimeIndex(f["date"].unique())


def _repair_outliers(df: pd.DataFrame, flags: pd.DataFrame, dates: pd.DatetimeIndex,
                     p: CleanParams, act: list, symbol: str) -> pd.Series:
    """Mechas y aperturas aisladas absurdas (p. ej. un mínimo ÷100 en Londres), confiando en el
    cierre: la apertura pasa al cierre anterior y la mecha que queda a ≥ outlier_repair del
    cuerpo se recorta al cuerpo. Modifica df; devuelve las velas reparadas."""
    dates = dates[dates.isin(df.index)]
    rep = pd.Series(False, index=df.index)
    opens = flags[(flags["check"] == "open_outlier")
                  & (flags["value"].abs() >= p.outlier_repair)]["date"]
    prev = df["close"].shift()
    for d in opens[opens.isin(df.index)]:
        if np.isfinite(prev[d]):
            _log(act, symbol, d, "fix_open", "open_outlier",
                 f"{df.at[d, 'open']:.4g} → {prev[d]:.4g}")
            df.at[d, "open"] = prev[d]
            rep[d] = True
    o, h, low, c = (df.loc[dates, k] for k in ("open", "high", "low", "close"))
    top, bot = np.maximum(o, c), np.minimum(o, c)
    hi = np.log(h / top) >= p.outlier_repair
    lo = np.log(bot / low) >= p.outlier_repair
    for d in dates[hi.to_numpy()]:
        _log(act, symbol, d, "fix_wick", "wick_outlier", f"H {h[d]:.4g} → {top[d]:.4g}")
    for d in dates[lo.to_numpy()]:
        _log(act, symbol, d, "fix_wick", "wick_outlier", f"L {low[d]:.4g} → {bot[d]:.4g}")
    df.loc[dates[hi.to_numpy()], "high"] = top[hi]
    df.loc[dates[lo.to_numpy()], "low"] = bot[lo]
    rep[dates[(hi | lo).to_numpy()]] = True
    return rep


def clean_asset(raw: pd.DataFrame, flags: pd.DataFrame, symbol: str, asset_class: str = "equity",
                params: CleanParams = DEFAULT_CLEAN) -> tuple[pd.DataFrame, list]:
    """Aplica las reglas a un activo. flags: incidencias de ese activo."""
    p, act = params, []
    df = raw[~raw.index.duplicated(keep="last")].sort_index().copy()
    err = flags[flags["severity"] == "error"]
    valid = pd.Series(True, index=df.index)

    # 0) factor de dividendos, sobre el histórico completo (cuentan también los dividendos de
    #    velas que luego se quitan)
    bad_div = flags[flags["check"] == "bad_dividend"]
    factor = dividend_factor(df, bad_div["date"])
    for r in bad_div.itertuples():
        _log(act, symbol, r.date, "ignore_dividend", "bad_dividend", r.detail)

    # 1) corte del inicio
    start = df.index[0]
    if asset_class == "equity":
        vs = flags[(flags["check"] == "volume_shift")
                   & (flags["value"] >= np.log(p.volume_jump_truncate))]
        if len(vs):
            start = max(start, vs["date"].max())
            _log(act, symbol, start, "truncate", "volume_shift", f"×{np.exp(vs.value.max()):.3g}")
    dense = err[err["check"].isin(DENSE_CHECKS)]
    span = (df.index[-1] - df.index[0])
    for a, b in _dense_regions(dense["date"], p.dense_errors, p.dense_gap_days):
        n = int(((dense["date"] >= a) & (dense["date"] <= b)).sum())
        if (b - df.index[0]) <= p.dense_truncate_frac * span:
            nxt = df.index[df.index > b]
            if len(nxt) and nxt[0] > start:
                start = nxt[0]
                _log(act, symbol, start, "truncate", "dense_errors",
                     f"{n} errores {a.date()}…{b.date()}")
        else:
            valid[(valid.index >= a) & (valid.index <= b)] = False
            _log(act, symbol, a, "mask", "dense_errors", f"{n} errores hasta {b.date()}")
    df, valid = df[df.index >= start], valid[valid.index >= start]
    flags = flags[flags["date"] >= start]
    err = err[err["date"] >= start]

    # 2) reparaciones por ratio conocido
    for r in err[err["check"].isin(REPAIR_CHECKS) & err["fix"].notna()].itertuples():
        before = df.index < r.date
        k = np.exp(r.fix)
        df.loc[before, PRICE_COLS] *= k
        if r.check in SPLIT_REPAIRS:
            df.loc[before, "volume"] /= k
        _log(act, symbol, r.date, "rescale_history", r.check, f"precios ×{k:.4g} antes de la fecha")

    # 3) quitar velas
    drop: dict[pd.Timestamp, str] = {}
    pos = {d: i for i, d in enumerate(df.index)}
    for r in flags[flags["check"].isin(DROP_CHECKS)].itertuples():
        if r.check == "spike" and r.severity != "error":
            continue
        if r.date not in pos:
            continue
        i = pos[r.date]
        rng = range(i + 1, i + r.span) if r.check == "stale_price" else range(i, i + r.span)
        for j in rng:
            if j < len(df):
                drop.setdefault(df.index[j], r.check)
    for d, why in drop.items():
        _log(act, symbol, d, "drop", why)
    # la vela siguiente a un precio <= 0 quitado no tiene un retorno con sentido
    keep_idx = df.index[~df.index.isin(list(drop))]
    for d, why in drop.items():
        if why == "nonpositive_price":
            nxt = keep_idx[keep_idx > d]
            if len(nxt):
                valid[nxt[0]] = False
                _log(act, symbol, nxt[0], "mask", "after_nonpositive")
    df, valid = df.loc[keep_idx], valid.loc[keep_idx]

    # 4) coherencia OHLC
    repaired = pd.Series(False, index=df.index)
    out_dates = _outlier_dates(flags, p)
    if len(out_dates):
        repaired |= _repair_outliers(df, flags, out_dates, p, act, symbol)
    o, h, low, c = (df[k] for k in ("open", "high", "low", "close"))
    viol = np.maximum(np.log(np.maximum(o, c) / h), np.log(low / np.minimum(o, c)))
    bad = viol > 1e-4
    open_out = bad & (c <= h) & (c >= low) & ((o > h) | (o < low))
    df.loc[open_out, "open"] = o[open_out].clip(low[open_out], h[open_out])
    o = df["open"]
    # siempre, también por debajo de la tolerancia: deja H ≥ máx(O, C) y L ≤ mín(O, C) exactos
    df["high"] = np.maximum.reduce([h, o, c])
    df["low"] = np.minimum.reduce([low, o, c])
    if bad.any():
        large = viol > p.ohlc_clip_max
        for d in df.index[bad]:
            what = "open" if open_out[d] else "high_low"
            _log(act, symbol, d, f"fix_{what}", "ohlc_inconsistent", f"{viol[d]:.4f}")
        repaired[large] = True
        if p.ohlc_large == "mask":
            valid[large] = False
            for d in df.index[large]:
                _log(act, symbol, d, "mask", "ohlc_large", f"{viol[d]:.4f}")

    # 5) incidencias graves sin reparar
    for r in flags[flags["check"].isin(MASK_CHECKS) & (flags["severity"] != "info")].itertuples():
        ds = _mask_dates(valid.index, r)
        if len(ds):
            valid[ds] = False
            _log(act, symbol, r.date, "mask", r.check, r.detail)
    gap = valid.index.to_series().diff().dt.days
    for d in valid.index[(gap > p.max_gap_days).to_numpy()]:
        valid[d] = False
        _log(act, symbol, d, "mask", "gap", f"{int(gap[d])} días naturales sin velas")

    # 6) ajuste por dividendos
    f = factor.reindex(df.index)
    out = pd.DataFrame({k: df[k] * f for k in ("open", "high", "low", "close")})
    out["volume"] = df["volume"]
    out["adj_factor"] = f
    out["valid"] = valid.astype(bool)
    out["ohlc_repaired"] = repaired.astype(bool)
    return out, act


def usable_samples(valid: pd.Series, window: int = 252, horizon: int = 20) -> int:
    """Número de fechas t con ventana [t-window+1, t] y target [t+1, t+horizon] todo válido."""
    v = valid.to_numpy(dtype=bool)
    n, span = len(v), window + horizon
    if n < span:
        return 0
    bad = np.r_[0, np.cumsum(~v)]
    return int(((bad[span:] - bad[:-span]) == 0).sum())


def _actionable(flags: pd.DataFrame, equities: set[str],
                params: CleanParams = DEFAULT_CLEAN) -> pd.DataFrame:
    """Incidencias que cambiarían los datos si se limpiara otra vez (reparaciones, velas a
    quitar, OHLC incoherente o un corte por volumen). Si queda alguna, hace falta otra pasada."""
    chk, sev = flags["check"], flags["severity"]
    repair = chk.isin(REPAIR_CHECKS) & (sev == "error")
    drop = chk.isin(DROP_CHECKS) & ((chk != "spike") | (sev == "error"))
    ohlc = chk == "ohlc_inconsistent"
    outlier = chk.isin(OUTLIER_CHECKS) & (flags["value"].abs() >= params.outlier_repair)
    trunc = ((chk == "volume_shift") & (flags["value"] >= np.log(params.volume_jump_truncate))
             & flags["symbol"].isin(equities))
    return flags[repair | drop | ohlc | outlier | trunc]


def clean_universe(universe: pd.DataFrame, flags: pd.DataFrame, load,
                   params: CleanParams = DEFAULT_CLEAN, max_passes: int = 4):
    """Limpia, re-audita y repite sobre los activos que aún tengan algo que corregir; después
    aplica las máscaras de la auditoría final y las de operaciones corporativas.

    Devuelve ({símbolo: datos limpios}, acciones, resumen, incidencias finales)."""
    from agentic_trading.data.audit import JUMP_CHECKS, audit_universe

    cls = dict(zip(universe["symbol"], universe["asset_class"]))
    equities = {s for s, c in cls.items() if c == "equity"}
    frames, raws, actions = {}, {}, []
    todo, current = list(universe["symbol"]), flags
    for n in range(1, max_passes + 1):
        by_sym = dict(tuple(current.groupby("symbol")))
        empty = current.iloc[:0]
        for s in todo:
            if n == 1:
                try:
                    raws[s] = load(s)
                except FileNotFoundError:
                    continue
                src = raws[s]
            else:
                src = as_audit_input(frames[s])
            out, act = clean_asset(src, by_sym.get(s, empty), s, cls[s], params)
            if n > 1:  # conserva el factor de dividendos y las máscaras de pasadas anteriores
                prev = frames[s]
                out["adj_factor"] = prev["adj_factor"].reindex(out.index)
                out["valid"] &= prev["valid"].reindex(out.index, fill_value=True)
                out["ohlc_repaired"] |= prev["ohlc_repaired"].reindex(out.index, fill_value=False)
            frames[s] = out
            actions += [a + (n,) for a in act]
            if not len(out):  # cortado entero (universo ampliado, 22-11-2026): se pierde
                del frames[s]
                actions.append((s, pd.NaT, "drop", "empty", "sin velas tras limpiar", n))
        current, _ = audit_universe(universe[universe["symbol"].isin(frames)],
                                    lambda s: as_audit_input(frames[s]))
        todo = sorted(set(_actionable(current, equities, params)["symbol"]))
        if not todo:
            break

    # Máscaras que aparecen en la auditoría final (no cambian datos, no requieren otra pasada).
    for r in current[current["check"].isin(MASK_CHECKS)
                     & (current["severity"] != "info")].itertuples():
        f = frames.get(r.symbol)
        if f is None:
            continue
        ds = _mask_dates(f.index, r)
        ds = ds[f.loc[ds, "valid"].to_numpy(bool)]
        if len(ds):
            f.loc[ds, "valid"] = False
            actions.append((r.symbol, r.date, "mask", r.check, r.detail, n))

    # Saltos que quedan junto a una operación corporativa: mal registrada, no se puede reparar.
    repaired_at: dict[str, list] = {}
    for a in actions:
        if a[2] == "rescale_history":
            repaired_at.setdefault(a[0], []).append(a[1])
    for s, g in current[current["check"].isin(JUMP_CHECKS)].groupby("symbol"):
        f = frames[s]
        corp = corporate_action_dates(raws[s], params.corp_min_yield)
        corp = corp.union(pd.DatetimeIndex(repaired_at.get(s, [])))
        if not len(corp):
            continue
        cpos = f.index.searchsorted(corp)
        for r in g.itertuples():
            if r.date not in f.index or not f.at[r.date, "valid"]:
                continue
            if np.abs(cpos - f.index.get_loc(r.date)).min() <= params.corp_window:
                f.at[r.date, "valid"] = False
                actions.append((s, r.date, "mask", "corporate_action_jump", r.detail, n))

    rows = []
    for s, out in frames.items():
        raw_ok = pd.Series(True, index=raws[s].index)
        rows.append((s, len(raws[s]), len(out), int((~out["valid"]).sum()),
                     usable_samples(raw_ok, params.window, params.horizon),
                     usable_samples(out["valid"], params.window, params.horizon),
                     out.index[0] if len(out) else pd.NaT))
    actions = (pd.DataFrame(actions, columns=ACTION_COLUMNS + ["pass"])
               .drop_duplicates(["symbol", "date", "action", "reason"], keep="first")
               .reset_index(drop=True))
    summary = pd.DataFrame(rows, columns=["symbol", "raw_rows", "clean_rows", "invalid_rows",
                                          "raw_samples", "usable_samples", "first_date"])
    return frames, actions, summary, current


def as_audit_input(clean: pd.DataFrame) -> pd.DataFrame:
    """Datos limpios en el formato que espera la auditoría (para re-auditar)."""
    df = clean[["open", "high", "low", "close", "volume"]].copy()
    df["adj_close"] = df["close"]
    df["dividends"] = 0.0
    df["stock_splits"] = 0.0
    return df
