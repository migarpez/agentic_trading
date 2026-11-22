"""Auditoría de las velas diarias descargadas.

No modifica los datos: devuelve una tabla de incidencias (una fila por símbolo, fecha y check)
que después usa el paso de limpieza. Cada check tiene una severidad por defecto:

- error: el dato está mal con seguridad o casi (hay que corregirlo o descartarlo).
- warning: sospechoso; revisar o tratar con una regla.
- info: anomalía probablemente legítima (movimiento grande real, volumen 0, huecos cortos...).

Saltos de precio. Se calculan sobre el retorno logarítmico de cierre menos la mediana del
mercado de su país ese día (así un crash general no se marca como error), normalizado por una
escala robusta móvil. Cada salto grande se clasifica en:

- split_not_adjusted: hay un split registrado y el precio salta en su ratio (Yahoo no ajustó).
- unit_spike / unit_shift (solo Londres): salto ×100 o ÷100 por mezcla de peniques y libras,
  que revierte (spike) o se mantiene (shift).
- spike: el precio vuelve a su nivel en ≤ 2 sesiones (tick erróneo). Error si el salto es ≥ 25 %
  y revierte en 1 sesión; si no, warning.
- unrecorded_split: ratio de split típico (2, 3, 1/2...) sin split registrado, con la vela entera
  ya en el nivel nuevo (la apertura salta lo mismo que el cierre y el rango del día es normal) y
  el volumen cambiando de nivel en el ratio inverso.
- split_like_move: como unrecorded_split pero sin corroboración del volumen.
- large_move: salto grande que no encaja en lo anterior (probablemente real). Incluye los saltos
  con ratio de split que ocurren dentro de la sesión (un split no puede producirse a mitad de día).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- configuración

CHECKS = {
    # check: (severidad, descripción)
    "duplicate_date": ("error", "fecha repetida o índice no ordenado"),
    "missing_ohlc": ("error", "falta O, H o L con C presente"),
    "nonpositive_price": ("error", "precio ≤ 0"),
    "high_lt_low": ("error", "H < L"),
    "ohlc_inconsistent": ("warning", "H < máx(O, C) o L > mín(O, C); error si > 1 %"),
    "negative_volume": ("error", "volumen < 0"),
    "missing_volume": ("warning", "volumen ausente"),
    "zero_volume": ("info", "volumen 0 (acciones)"),
    "flat_bar": ("info", "O = H = L = C con volumen > 0"),
    "no_trade": ("warning", "O = H = L = C con volumen 0: sesión sin negociación"),
    "stale_price": ("warning", "cierre idéntico ≥ 5 sesiones seguidas"),
    "weekend_bar": ("warning", "vela en sábado o domingo (no cripto)"),
    "gap": ("warning", "más de 10 días naturales sin velas"),
    "missing_days": ("info", "sesiones del calendario del país sin vela; warning si ≥ 5"),
    "extra_day": ("info", "vela en un día que casi ningún par de su país cotiza"),
    "split_not_adjusted": ("error", "split registrado sin ajustar en precios"),
    "unit_spike": ("error", "salto ×100/÷100 que revierte (peniques/libras)"),
    "unit_shift": ("error", "cambio de nivel ×100/÷100 (peniques/libras)"),
    "spike": ("error", "salto ≥ 25 % que revierte en 1 sesión; si no, warning"),
    "unrecorded_split": ("error", "ratio de split + volumen corroborando, sin split registrado"),
    "split_like_move": ("warning", "salto con ratio de split sin corroborar"),
    "large_move": ("info", "salto grande frente al mercado, no clasificado como error"),
    "extreme_move": ("warning", "salto ≥ ×2 frente al mercado con z ≥ 50 sin ratio de split"),
    "wick_outlier": ("warning", "H o L muy lejos de O/C frente al rango habitual"),
    "open_outlier": ("warning", "apertura aislada lejos del cierre anterior y del cierre"),
    "volume_shift": ("warning", "cambio de nivel del volumen sin split ni salto de precio"),
    # El adj_close de Yahoo no se usa (en Londres aplica los dividendos ÷100 y tiene tramos
    # corruptos): el ajuste se calcula con la columna de dividendos. Estos checks son informativos.
    "adj_factor_jump": ("info", "adj_close/close de Yahoo cambia sin dividendo ni split (no se usa)"),
    "adj_factor_gt1": ("info", "adj_close de Yahoo > close (no se usa)"),
    "adj_factor_invalid": ("info", "adj_close de Yahoo ≤ 0 o ausente (no se usa)"),
    "large_dividend": ("info", "dividendo > 20 % del precio"),
    "bad_dividend": ("warning", "dividendo ≥ precio o > 5 % sin su caída de precio: no se aplica"),
    "coarse_prices": ("warning", "año con precios en escalones ≥ 2 %: el año se marca no válido"),
    "corrupt_ohlc": ("warning", ("año con rango H/L mediano ≥ 8× el |retorno| mediano: O/H/L "
                     "corruptos, el año se marca no válido")),
    "short_history": ("info", "menos de 272 velas: no da ni una ventana con target"),
}

# Ratios de split que se buscan. 5:4 y 3:2 se excluyen: un ±20-40 % es indistinguible de un
# movimiento por resultados, y el volumen no basta para separarlos.
SPLIT_RATIOS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50, 100])


@dataclass(frozen=True)
class AuditParams:
    z_jump: float = 8.0          # umbral en escalas robustas del residuo frente a mercado
    min_jump: float = 0.10       # y como mínimo este |log retorno|
    extreme_jump: float = 0.7    # salto extremo: |log| ≥ esto (×2)
    extreme_z: float = 50.0      # y |z| ≥ esto (una caída real como AIG 2008 da z ≈ 20)
    scale_window: int = 63
    ratio_tol: float = 0.03      # tolerancia en log para casar un ratio de split
    split_open_tol: float = 0.10  # split: |salto en apertura − salto en cierre| ≤ 10 % del salto
    split_range_mult: float = 4.0  # split: rango H/L del día ≤ 4× el habitual
    spike_horizon: int = 2       # sesiones máximas para revertir
    spike_revert: float = 0.25   # vuelve a ≤ 25 % del salto
    spike_error: float = 0.25    # spike error: |log salto| ≥ esto y revierte en 1 sesión
    cluster_min: int = 3         # activos del mismo país saltando el mismo día = evento de mercado
    cluster_frac: float = 0.01   # o esta fracción del grupo, lo que sea mayor
    vol_window: int = 40         # sesiones antes/después para el nivel de volumen
    vol_skip: int = 5            # se ignoran las sesiones junto al salto (volumen de noticia)
    vol_rel_tol: float = 0.35    # |Δlog vol − log k| < 35 % de log k
    ohlc_tol: float = 1e-4
    stale_run: int = 5
    gap_days: int = 10
    wick_mult: float = 8.0
    min_wick: float = 0.10
    adj_tol: float = 0.005
    div_check: float = 0.05      # dividendos mayores (fracción del precio) se contrastan con la caída
    div_tol: float = 0.08        # |Δlog cierre en la fecha ex − log(1 − D/C)| máximo para aceptarlo
    coarse_grid: float = 0.02    # escalón mediano entre niveles de precio distintos / precio
    coarse_min_bars: int = 50    # velas con volumen mínimas en el año para medirlo
    corrupt_range: float = 8.0   # rango H/L mediano / |log retorno de cierre| mediano en el año
    min_rows: int = 272


DEFAULT_PARAMS = AuditParams()

# span: velas afectadas desde `date`. fix: factor log nominal que repara el salto (o NaN).
FLAG_COLUMNS = ["symbol", "date", "check", "severity", "value", "detail", "span", "fix"]


def _flag(rows, symbol, date, check, value=np.nan, detail="", severity=None, span=1,
          fix=np.nan):
    rows.append((symbol, pd.Timestamp(date), check, severity or CHECKS[check][0],
                 float(value), detail, int(span), float(fix)))


def _runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Tramos [inicio, fin] (posiciones) donde mask es True."""
    m = mask.to_numpy(dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


# ----------------------------------------------------------------------------- checks

def _structural(df, sym, rows, p: AuditParams, asset_class: str):
    idx = df.index
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        _flag(rows, sym, idx[0], "duplicate_date", idx.duplicated().sum())

    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    for d in df.index[o.isna() | h.isna() | l.isna()]:
        _flag(rows, sym, d, "missing_ohlc")
    for d in df.index[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]:
        _flag(rows, sym, d, "nonpositive_price")
    for d, x in (l / h)[h < l].items():
        _flag(rows, sym, d, "high_lt_low", x)

    up = np.log(np.maximum(o, c) / h)
    dn = np.log(l / np.minimum(o, c))
    viol = np.maximum(up, dn)
    c_out = (c > h) | (c < l)
    for d, x in viol[viol > p.ohlc_tol].items():
        _flag(rows, sym, d, "ohlc_inconsistent", x, "close fuera de [L, H]" if c_out[d]
              else "open fuera de [L, H]", severity="error" if x > 0.01 else None)

    for d in df.index[v < 0]:
        _flag(rows, sym, d, "negative_volume")
    for d in df.index[v.isna()]:
        _flag(rows, sym, d, "missing_volume")
    if asset_class == "equity":
        for a, b in _runs(v == 0):
            _flag(rows, sym, idx[a], "zero_volume", b - a + 1, f"hasta {idx[b].date()}",
                  span=b - a + 1)
    flat = (o == h) & (h == l) & (l == c)
    for a, b in _runs(flat & (v > 0)):
        _flag(rows, sym, idx[a], "flat_bar", b - a + 1, f"hasta {idx[b].date()}", span=b - a + 1)
    for a, b in _runs(flat & (v == 0)):
        _flag(rows, sym, idx[a], "no_trade", b - a + 1, f"hasta {idx[b].date()}", span=b - a + 1)

    same = c.eq(c.shift())
    for a, b in _runs(same):
        n = b - a + 2  # sesiones con el mismo cierre
        if n >= p.stale_run:
            _flag(rows, sym, idx[a - 1], "stale_price", n, f"hasta {idx[b].date()}", span=n)

    if len(df) < p.min_rows:
        _flag(rows, sym, idx[0], "short_history", len(df))


def _calendar(df, sym, rows, p: AuditParams, asset_class, calendar: pd.DatetimeIndex | None,
              rare_days: pd.DatetimeIndex | None):
    idx = df.index
    if asset_class != "crypto":
        for d in idx[idx.dayofweek >= 5]:
            _flag(rows, sym, d, "weekend_bar")
    step = idx.to_series().diff().dt.days
    for d, n in step[step > p.gap_days].items():
        _flag(rows, sym, d, "gap", n, "días naturales desde la vela anterior")

    if calendar is not None:
        cal = calendar[(calendar >= idx[0]) & (calendar <= idx[-1])]
        missing = ~cal.isin(idx)
        for a, b in _runs(pd.Series(missing)):
            n = b - a + 1
            _flag(rows, sym, cal[a], "missing_days", n, f"hasta {cal[b].date()}",
                  severity="warning" if n >= 5 else None)
    if rare_days is not None:
        for d in idx[idx.isin(rare_days)]:
            _flag(rows, sym, d, "extra_day")


def _robust_scale(x: pd.Series, window: int) -> pd.Series:
    s = x.abs().rolling(window, min_periods=20).median().shift(1) * 1.4826
    fallback = x.abs().median() * 1.4826
    return s.fillna(fallback).clip(lower=max(fallback * 0.25, 1e-4))


def _nanmedian(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def _match_ratio(logk: float, tol: float) -> float | None:
    """Ratio de split (>1) que casa con |log k|, o None."""
    err = np.abs(np.abs(logk) - np.log(SPLIT_RATIOS))
    i = err.argmin()
    return float(SPLIT_RATIOS[i]) if err[i] < tol else None


def _jumps(df, sym, rows, p: AuditParams, market: pd.Series | None, london: bool):
    c, v = df["close"], df["volume"]
    lc = np.log(c.where(c > 0))  # precios ≤ 0 ya se marcan en _structural
    r = lc.diff()
    m = market.reindex(df.index).fillna(0.0) if market is not None else 0.0
    e = r - m
    z = e / _robust_scale(e, p.scale_window)
    cand = np.flatnonzero(((z.abs() > p.z_jump) & (e.abs() > p.min_jump)).to_numpy())

    splits = df["stock_splits"].fillna(0)
    split_pos = {i: s for i, s in enumerate(splits.to_numpy()) if s not in (0, 1)}
    lcv, lv = lc.to_numpy(), np.log(v.where(v > 0)).to_numpy(dtype=float)
    lov = np.log(df["open"].where(df["open"] > 0)).to_numpy(dtype=float)
    rng = np.log(df["high"] / df["low"])
    rng_med = rng.rolling(p.scale_window, min_periods=20).median().shift(1).fillna(rng.median())
    rngv, rng_medv = rng.to_numpy(dtype=float), rng_med.to_numpy(dtype=float)
    n, skip = len(df), set()
    idx = df.index

    for i in cand:
        if i in skip or i == 0:
            continue
        logk, d = r.iloc[i], idx[i]
        detail = f"ratio {np.exp(logk):.4g}, z {z.iloc[i]:.1f}"

        # 1) split registrado cerca y el precio salta en su ratio inverso
        near = [s for j, s in split_pos.items() if abs(j - i) <= 2]
        if any(abs(logk + np.log(s)) < p.ratio_tol for s in near):
            s = min(near, key=lambda s: abs(logk + np.log(s)))
            _flag(rows, sym, d, "split_not_adjusted", logk, detail + f", split {s:g}",
                  fix=-np.log(s))
            continue

        # 2) ¿revierte en pocas sesiones?
        rev = None
        for j in range(1, p.spike_horizon + 1):
            if i + j < n and abs(lcv[i + j] - lcv[i - 1]) < p.spike_revert * abs(logk):
                rev = j
                break
        unit = london and abs(abs(logk) - np.log(100)) < 0.1
        if rev is not None:
            sev = None if unit or (rev == 1 and abs(logk) >= p.spike_error) else "warning"
            _flag(rows, sym, d, "unit_spike" if unit else "spike", logk,
                  detail + f", revierte en {rev}", severity=sev, span=rev)
            skip.update(range(i + 1, i + rev + 1))
            continue
        if unit:
            _flag(rows, sym, d, "unit_shift", logk, detail, fix=np.sign(logk) * np.log(100))
            continue

        # 3) ratio de split sin split registrado: ¿corrobora el volumen?
        ratio = _match_ratio(logk, p.ratio_tol)
        if ratio is not None:
            # un split deja la vela entera en el nivel nuevo: la apertura ya salta y el rango
            # del día es normal; si el salto ocurre dentro de la sesión, es un movimiento real
            gap = lov[i] - lcv[i - 1]
            at_new_level = (abs(gap - logk) <= p.split_open_tol * abs(logk)
                            and rngv[i] <= p.split_range_mult * rng_medv[i])
            if not at_new_level:
                _flag(rows, sym, d, "large_move", logk,
                      detail + f", ratio de split dentro de la sesión (apertura {gap:+.3f}, "
                      f"rango {rngv[i] / rng_medv[i]:.1f}× habitual)")
                continue
            w, k = p.vol_window, p.vol_skip
            dv = _nanmedian(lv[i + k:i + k + w]) - _nanmedian(lv[max(0, i - k - w):max(0, i - k)])
            expected = -logk  # precio ÷k  ->  volumen ×k
            corroborated = np.isfinite(dv) and abs(dv - expected) < p.vol_rel_tol * abs(expected)
            _flag(rows, sym, d, "unrecorded_split" if corroborated else "split_like_move", logk,
                  detail + f", split {ratio:g}, Δlog vol {dv:.2f}",
                  fix=np.sign(logk) * np.log(ratio))
            continue

        extreme = abs(logk) >= p.extreme_jump and abs(z.iloc[i]) >= p.extreme_z
        _flag(rows, sym, d, "extreme_move" if extreme else "large_move", logk, detail)


def _intraday(df, sym, rows, p: AuditParams):
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    rng = np.log(h / l)
    med = rng.rolling(p.scale_window, min_periods=20).median().shift(1)
    med = med.fillna(rng.median())
    wick = np.maximum(np.log(h / np.maximum(o, c)), np.log(np.minimum(o, c) / l))
    bad = (wick > p.wick_mult * med) & (wick > p.min_wick)
    for d, x in wick[bad].items():
        _flag(rows, sym, d, "wick_outlier", x, f"rango habitual {med[d]:.3f}")

    g = np.log(o / c.shift())
    b = np.log(c / o)
    sg = _robust_scale(g.dropna(), p.scale_window).reindex(g.index)
    iso = (g.abs() > p.z_jump * sg) & (g.abs() > p.min_jump) \
        & (np.sign(g) != np.sign(b)) & (b.abs() > 0.8 * g.abs())
    for d, x in g[iso].items():
        _flag(rows, sym, d, "open_outlier", x, f"cierre/apertura {np.exp(b[d]):.3g}")


def _volume_levels(df, sym, rows, p: AuditParams, flagged_dates: set):
    v = df["volume"]
    lv = np.log(v.where(v > 0))
    w = p.vol_window * 3
    before = lv.rolling(w, min_periods=w // 2).median()
    after = lv[::-1].rolling(w, min_periods=w // 2).median()[::-1].shift(-1)
    step = (after - before)
    big = step.abs() > np.log(20)
    splits = df["stock_splits"].fillna(0)
    near_split = splits.ne(0).rolling(2 * w, center=True, min_periods=1).max().astype(bool)
    lvv = lv.to_numpy(dtype=float)
    k = 10  # ventana corta para localizar el punto exacto dentro del tramo
    for a, b in _runs(big & ~near_split):
        cand = range(max(k, a - w // 2), min(len(lvv) - k, b + w // 2) + 1)
        score = [abs(_nanmedian(lvv[i:i + k]) - _nanmedian(lvv[i - k:i])) for i in cand]
        if not score or not np.isfinite(score).any():
            continue
        i = cand[int(np.nanargmax(score))]
        d = lv.index[i]
        if any(abs((d - f).days) <= 5 for f in flagged_dates):
            continue
        seg = step.iloc[a:b + 1]
        size = seg[seg.abs().idxmax()]  # tamaño del cambio, de las ventanas largas
        _flag(rows, sym, d, "volume_shift", size, f"×{np.exp(size):.3g}")


def _adjustment(df, sym, rows, p: AuditParams, market: pd.Series | None = None):
    c, ac = df["close"], df["adj_close"]
    inv = ac.isna() | (ac <= 0)
    for a, b in _runs(inv):
        _flag(rows, sym, df.index[a], "adj_factor_invalid", b - a + 1,
              f"hasta {df.index[b].date()}", span=b - a + 1)
    f = np.log(ac.where(~inv) / c)
    for d, x in f[f > 1e-3].items():
        _flag(rows, sym, d, "adj_factor_gt1", x)
    df_ = f.diff()
    events = (df["dividends"].fillna(0) != 0) | (df["stock_splits"].fillna(0) != 0)
    near = events.shift(-1, fill_value=False) | events | events.shift(1, fill_value=False)
    for d, x in df_[(df_.abs() > p.adj_tol) & ~near].items():
        _flag(rows, sym, d, "adj_factor_jump", x)
    div = df["dividends"].fillna(0)
    c_prev = c.shift()
    y = div / c_prev
    for d, x in y[y > 0.2].items():
        _flag(rows, sym, d, "large_dividend", x, f"dividendo {div[d]:.4g}")
    # Dividendos no fiables: imposibles (≥ precio) o grandes sin la caída de precio que tocaría
    # (medida frente al mercado de ese día, para no confundirla con una subida general).
    m = market.reindex(df.index).fillna(0.0) if market is not None else 0.0
    drop = np.log(c / c_prev) - m
    expected = np.log1p(-y.clip(upper=0.999))
    impossible = y >= 1
    mismatch = (y > p.div_check) & ~impossible & ((drop - expected).abs() >= p.div_tol)
    for d in df.index[impossible | mismatch]:
        why = ("≥ precio" if impossible[d]
               else f"Δlog cierre frente al mercado {drop[d]:+.3f}, esperado {expected[d]:+.3f}")
        _flag(rows, sym, d, "bad_dividend", y[d], f"D/C {y[d]:.4g}, {why}")


def _coarse_prices(df, sym, rows, p: AuditParams):
    """Años con precios en una rejilla gruesa: escalón mediano entre cierres distintos ≥ 2 %."""
    traded = df.loc[df["volume"] > 0, "close"]
    for year, g in traded.groupby(traded.index.year):
        levels = np.unique(g.to_numpy())
        if len(g) < p.coarse_min_bars or len(levels) < 2:
            continue
        grid = float(np.median(np.diff(levels)) / g.median())
        if grid >= p.coarse_grid:
            n = int((df.index.year == year).sum())
            _flag(rows, sym, g.index[0], "coarse_prices", grid,
                  f"año {year}, escalón mediano {grid:.1%}", span=n)


def _corrupt_ohlc(df, sym, rows, p: AuditParams):
    """Años en que O/H/L no cuadran con los cierres (p. ej. VOD.L 2003–2006, que abre cada día un
    13 % por encima del cierre anterior): rango diario mediano ≥ corrupt_range × |retorno| mediano."""
    r = np.log(df["close"]).diff().abs()
    rng = np.log(df["high"] / df["low"])
    for year, g in pd.DataFrame({"r": r, "rng": rng}).groupby(df.index.year):
        if len(g) < p.coarse_min_bars:
            continue
        mr, mrng = g["r"].median(), g["rng"].median()
        if mr > 0 and mrng >= p.corrupt_range * mr:
            _flag(rows, sym, g.index[0], "corrupt_ohlc", mrng / mr,
                  f"año {year}, rango mediano {mrng:.3f} frente a |retorno| {mr:.4f}", span=len(g))


# ----------------------------------------------------------------------------- API

def audit_asset(df: pd.DataFrame, symbol: str, asset_class: str = "equity",
                market: pd.Series | None = None, calendar: pd.DatetimeIndex | None = None,
                rare_days: pd.DatetimeIndex | None = None,
                params: AuditParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Audita un activo. df: columnas open, high, low, close, adj_close, volume, dividends,
    stock_splits, indexado por fecha."""
    rows: list = []
    with np.errstate(divide="ignore", invalid="ignore"):  # precios ≤ 0 se marcan aparte
        _structural(df, symbol, rows, params, asset_class)
        _calendar(df, symbol, rows, params, asset_class, calendar, rare_days)
        _jumps(df, symbol, rows, params, market, london=symbol.endswith(".L"))
        _intraday(df, symbol, rows, params)
        _corrupt_ohlc(df, symbol, rows, params)
        flagged = {r[1] for r in rows if r[2] in JUMP_CHECKS}
        _volume_levels(df, symbol, rows, params, flagged)
        _adjustment(df, symbol, rows, params, market)
        _coarse_prices(df, symbol, rows, params)
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


def peer_context(frames: dict[str, pd.DataFrame], groups: dict[str, str],
                 min_peers: int = 5) -> dict[str, dict]:
    """Por grupo (país): retorno de mercado (mediana), calendario de sesiones y días raros."""
    out = {}
    for g in sorted(set(groups.values())):
        members = [s for s, gg in groups.items() if gg == g and s in frames]
        if len(members) < min_peers:
            continue
        close = pd.DataFrame({s: frames[s]["close"] for s in members}).sort_index()
        alive = close.ffill().notna() & close.bfill().notna()
        frac = close.notna().sum(axis=1) / alive.sum(axis=1)
        ret = np.log(close).diff()
        out[g] = {
            "market": ret.median(axis=1, skipna=True),
            "calendar": close.index[frac >= 0.5],
            "rare_days": close.index[frac < 0.1],
        }
    return out


def audit_universe(universe: pd.DataFrame, load, params: AuditParams = DEFAULT_PARAMS):
    """Audita todo el universo. load(symbol) -> DataFrame. Devuelve (flags, resumen)."""
    frames = {}
    for s in universe["symbol"]:
        try:
            frames[s] = load(s)
        except FileNotFoundError:
            pass
    cls = dict(zip(universe["symbol"], universe["asset_class"]))
    # Grupo de calendario/mercado: país para acciones; commodities con EE. UU.; cripto sin grupo.
    grp = {s: (c if isinstance(c, str) else "US")
           for s, c, a in zip(universe["symbol"], universe["country"], universe["asset_class"])
           if a != "crypto"}
    eq_grp = {s: g for s, g in grp.items() if cls[s] == "equity"}
    ctx = peer_context(frames, eq_grp)

    flags = []
    for s, df in frames.items():
        c = ctx.get(grp.get(s), {})
        market = c.get("market") if cls[s] == "equity" else None
        flags.append(audit_asset(df, s, cls[s], market, c.get("calendar"),
                                 c.get("rare_days") if cls[s] == "equity" else None, params))
    flags = [f for f in flags if len(f)]
    flags = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame(columns=FLAG_COLUMNS)
    flags = mark_market_events(flags, grp, params)

    missing = sorted(set(universe["symbol"]) - set(frames))
    summary = summarize(flags, universe, frames)
    summary.loc[summary["symbol"].isin(missing), "status"] = "missing"
    return flags, summary


JUMP_CHECKS = ["split_not_adjusted", "unit_spike", "unit_shift", "spike", "unrecorded_split",
               "split_like_move", "large_move", "extreme_move"]
DOWNGRADE_ON_CLUSTER = ["spike", "unrecorded_split", "split_like_move", "extreme_move"]


def mark_market_events(flags: pd.DataFrame, groups: dict[str, str],
                       params: AuditParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Si muchos activos de un mismo grupo saltan el mismo día, es un evento de mercado: los
    spikes y saltos con ratio de split de ese día pasan a info."""
    if flags.empty:
        return flags
    size = pd.Series(groups).value_counts()
    j = flags[flags["check"].isin(JUMP_CHECKS)].assign(group=lambda d: d["symbol"].map(groups))
    n = j.dropna(subset=["group"]).groupby(["group", "date"])["symbol"].nunique()
    thr = n.index.get_level_values("group").map(
        lambda g: max(params.cluster_min, params.cluster_frac * size[g]))
    events = set(n[n.to_numpy() >= np.asarray(thr)].index)
    hit = [(groups.get(s), d) in events and c in DOWNGRADE_ON_CLUSTER
           for s, d, c in zip(flags["symbol"], flags["date"], flags["check"])]
    hit = np.asarray(hit, dtype=bool)
    flags = flags.copy()
    counts = [n.get((groups.get(s), d), 0) for s, d in zip(flags["symbol"][hit], flags["date"][hit])]
    flags.loc[hit, "severity"] = "info"
    flags.loc[hit, "detail"] = [f"{x} · evento de mercado ({k} activos)"
                                for x, k in zip(flags.loc[hit, "detail"], counts)]
    return flags


def summarize(flags: pd.DataFrame, universe: pd.DataFrame, frames: dict) -> pd.DataFrame:
    base = universe[["symbol", "asset_class", "country"]].copy()
    base["rows"] = base["symbol"].map(lambda s: len(frames[s]) if s in frames else 0)
    counts = flags.pivot_table(index="symbol", columns="severity", values="check",
                               aggfunc="count", fill_value=0)
    base = base.merge(counts, left_on="symbol", right_index=True, how="left")
    for sev in ("error", "warning", "info"):
        base[sev] = base.get(sev, 0)
    base[["error", "warning", "info"]] = base[["error", "warning", "info"]].fillna(0).astype(int)
    base["status"] = np.select([base["error"] > 0, base["warning"] > 0], ["error", "warning"],
                               "ok")
    return base
