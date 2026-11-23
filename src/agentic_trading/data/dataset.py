"""Dataset de ventanas y targets (handoff §4, §5 y §9).

Construcción (build_dataset): a partir de las series limpias se precalculan, por vela, todo lo
que no depende de la ventana concreta (σ_ventana, mediana del volumen, targets) y se guarda en
arrays planos con un índice de muestras. Las ventanas no se materializan: se cortan al vuelo
(WindowData.batch), lo que evita los ~30 GB de guardarlas.

Convenciones (t = posición de la última vela de la ventana, "hoy"):
- Ventana: velas t−251 … t. Horizontes h en velas (no en días naturales); cripto incluye fines
  de semana y se asume la diferencia.
- σ_ventana: desviación típica (ddof=1) de los 251 log-retornos de cierre dentro de la ventana.
- Precios: x = log(P_d / C_t) / σ_ventana para O, H, L, C.
- Volumen: v = log(V_d / mediana de V>0 en la ventana); V_d = 0 se sustituye por la mediana (v=0).
- Escalar: log σ_ventana.
- μ target: y_h = log(C_{t+h} / C_t) / σ_ventana.
- σ target: v_h = σ_YZ(t+1 … t+h) / σ_ventana (Yang-Zhang, escala diaria).
- Muestra válida: todas las velas t−251 … t+20 con valid=True y targets finitos.
- mask_sigma_h: False si el target a h contiene una vela con ohlc_repaired o si σ_YZ = 0
  (no cuenta en L_σ).
- Velas agregadas (opcionales, batch(coarse=...); prueba de entrada en dos escalas): "month", 12
  velas de 21 sesiones (t−251 … t); "week", 50 velas de 5 sesiones (t−249 … t). Apertura de la
  primera sesión, máximo de los máximos, mínimo de los mínimos, cierre de la última y volumen
  total. Precios como los diarios; volumen v = log(V / (n · mediana de V>0 en la ventana)), con
  n las sesiones de la vela.
- Vela del mercado (opcional, batch(market=True); prueba de mu_tree, data.market): 4 canales más
  (O, H, L, C del mercado de la región del activo), normalizados igual que el activo:
  (log P_mercado − log C_mercado(t)) / σ_ventana del activo. También en las velas agregadas.
- Para L_μ (pruebas de mu_tree): date_ranks (rango gaussianizado del target entre las muestras
  de entrenamiento de su fecha) y tail_weights (peso de las muestras del decil alto de su fecha).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20)
WINDOW = 252

SPLITS = {
    "dev": ("2000-01-01", "2019-12-31"),    # validación cruzada por bloques (model/folds.py)
    "test": ("2020-01-01", "2026-12-31"),   # ciego, walk-forward
}

COARSE = {"month": (12, 21), "week": (50, 5)}   # velas agregadas: número y sesiones por vela

BAR_ARRAYS = ["log_open", "log_high", "log_low", "log_close", "volume", "sigma", "vol_median"]
TARGET_ARRAYS = [f"y_{h}" for h in HORIZONS] + [f"v_{h}" for h in HORIZONS]


@dataclass(frozen=True)
class DatasetParams:
    window: int = WINDOW
    horizons: tuple = HORIZONS


DEFAULT_DATASET = DatasetParams()


# ----------------------------------------------------------------------------- cálculo por activo

def _rolling_sums(x: np.ndarray, w: int):
    """Sumas móviles (n, Σx, Σx²) de x[i−w+1 … i], con NaN tratado como ausente."""
    ok = np.isfinite(x)
    z = np.where(ok, x, 0.0)
    cs = [np.r_[0.0, np.cumsum(q)] for q in (ok.astype(float), z, z * z)]
    return [(c[w:] - c[:-w]) for c in cs]


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Desviación típica (ddof=1) de x[i−w+1 … i]; NaN si falta algún valor de la ventana."""
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    n, s1, s2 = _rolling_sums(x, w)
    var = (s2 - s1 * s1 / w) / (w - 1)
    out[w - 1:] = np.where(n == w, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return out


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    n, s1, _ = _rolling_sums(x, w)
    out[w - 1:] = np.where(n == w, s1 / w, np.nan)
    return out


def yang_zhang(log_o, log_h, log_l, log_c, n: int) -> np.ndarray:
    """σ_YZ de las n velas t+1 … t+n, indexado por t (NaN si no hay n velas futuras)."""
    o = log_o - np.r_[np.nan, log_c[:-1]]   # salto nocturno
    c = log_c - log_o                       # sesión
    u, d = log_h - log_o, log_l - log_o
    rs = u * (u - c) + d * (d - c)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_o = _rolling_std(o, n) ** 2
    var_c = _rolling_std(c, n) ** 2
    m = len(rs)
    rs_mean = _rolling_mean(rs, n)
    var = var_o + k * var_c + (1 - k) * rs_mean    # ventana que termina en i = t+n
    out = np.full(m, np.nan)
    out[:m - n] = np.sqrt(var[n:])                  # reindexar a t
    return out


def asset_arrays(df: pd.DataFrame, params: DatasetParams = DEFAULT_DATASET) -> dict:
    """Arrays por vela de un activo limpio (columnas open, high, low, close, volume, valid,
    ohlc_repaired)."""
    w, hs = params.window, params.horizons
    lo, lh, ll, lc = (np.log(df[k].to_numpy(float)) for k in ("open", "high", "low", "close"))
    vol = df["volume"].to_numpy(float)
    n = len(df)

    r = np.r_[np.nan, np.diff(lc)]
    sigma = _rolling_std(r[1:], w - 1)              # 251 retornos dentro de la ventana
    sigma = np.r_[np.nan, sigma]

    vpos = pd.Series(np.where(vol > 0, vol, np.nan))
    vmed = vpos.rolling(w, min_periods=1).median().to_numpy()
    vmed[: w - 1] = np.nan

    out = {"log_open": lo, "log_high": lh, "log_low": ll, "log_close": lc, "volume": vol,
           "sigma": sigma, "vol_median": vmed}
    for h in hs:
        fut = np.r_[lc[h:], np.full(h, np.nan)]
        out[f"y_{h}"] = (fut - lc) / sigma
        out[f"v_{h}"] = yang_zhang(lo, lh, ll, lc, h) / sigma

    # validez de la muestra: ventana + horizonte máximo, todo válido
    hmax = max(hs)
    valid = df["valid"].to_numpy(bool)
    bad = np.r_[0, np.cumsum(~valid)]
    ok = np.zeros(n, bool)
    t = np.arange(w - 1, n - hmax)
    ok[t] = (bad[t + hmax + 1] - bad[t - w + 1]) == 0
    finite = np.isfinite(sigma) & (sigma > 0)
    for k in TARGET_ARRAYS:
        finite &= np.isfinite(out[k])
    out["sample_ok"] = ok & finite

    rep = np.r_[0, np.cumsum(df["ohlc_repaired"].to_numpy(bool))]
    for h in hs:
        m = np.zeros(n, bool)
        th = np.arange(0, n - h)
        m[th] = (rep[th + h + 1] - rep[th + 1]) == 0  # velas t+1 … t+h
        out[f"mask_sigma_{h}"] = m & (out[f"v_{h}"] > 0)  # σ_YZ = 0 haría infinita la QLIKE
    return out


# ----------------------------------------------------------------------------- construcción

def fingerprint(directory: Path) -> str:
    """Huella sha256 de los Parquet de un directorio (nombres y contenido)."""
    hsh = hashlib.sha256()
    for p in sorted(directory.glob("*.parquet")):
        hsh.update(p.name.encode())
        hsh.update(p.read_bytes())
    return hsh.hexdigest()


def build_dataset(universe: pd.DataFrame, load, out_dir: Path,
                  params: DatasetParams = DEFAULT_DATASET, clean_dir: Path | None = None) -> dict:
    """Construye y guarda el dataset. load(symbol) -> serie limpia. Devuelve los metadatos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = universe["symbol"].tolist()
    chunks = {k: [] for k in BAR_ARRAYS + TARGET_ARRAYS}
    samples, offsets, offset, zeros = [], {}, 0, {"sigma": 0, "v": 0}
    for aid, s in enumerate(symbols):
        df = load(s)
        a = asset_arrays(df, params)
        for k, parts in chunks.items():
            parts.append(a[k].astype(np.float64 if k.startswith("log_") else np.float32))
        pos = np.flatnonzero(a["sample_ok"])
        hmax = max(params.horizons)
        dates = df.index.to_numpy()
        zeros["sigma"] += int((a["sigma"] == 0).sum())
        zeros["v"] += int(sum((a[f"v_{h}"] == 0).sum() for h in params.horizons))
        smp = pd.DataFrame({
            "asset": np.int16(aid), "row": (offset + pos).astype(np.int64),
            "date": dates[pos], "target_end": dates[pos + hmax],
        })
        for h in params.horizons:
            smp[f"mask_sigma_{h}"] = a[f"mask_sigma_{h}"][pos]
        samples.append(smp)
        offsets[s] = (offset, len(df))
        offset += len(df)

    for k, parts in chunks.items():
        np.save(out_dir / f"{k}.npy", np.concatenate(parts))
    samples = pd.concat(samples, ignore_index=True)
    samples["month"] = (samples["date"].dt.year * 12 + samples["date"].dt.month - 1).astype(np.int32)
    samples.to_parquet(out_dir / "samples.parquet", index=False)

    meta = {
        "params": asdict(params), "symbols": symbols, "offsets": offsets,
        "n_rows": offset, "n_samples": len(samples), "splits": SPLITS,
        "zero_counts": zeros,
        "clean_fingerprint": fingerprint(clean_dir) if clean_dir else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ----------------------------------------------------------------------------- lectura

def split_mask(samples: pd.DataFrame, name: str, splits: dict = SPLITS) -> np.ndarray:
    """Muestras del bloque: fecha dentro y target completo antes del fin (hueco de 20 velas)."""
    start, end = (pd.Timestamp(x) for x in splits[name])
    return ((samples["date"] >= start) & (samples["date"] <= end)
            & (samples["target_end"] <= end)).to_numpy()


def dataset_id(meta: dict) -> dict:
    """Huella de un dataset (para no mezclar resultados de datasets distintos): la de los datos
    limpios, los parámetros y el número de muestras."""
    ident = {"clean_fingerprint": meta.get("clean_fingerprint"), "params": meta["params"],
             "n_samples": meta["n_samples"]}
    return json.loads(json.dumps(ident, default=str))


def asset_info(symbols: list[str], universe: pd.DataFrame) -> pd.DataFrame:
    """Clase, país, grupo (país, o la clase si no tiene) y región (US, Europa, crypto) de cada
    activo del dataset, en el orden de sus ids (meta.json["symbols"]). Se asigna por símbolo, no
    por la posición en universe.parquet, que cambia si el universo se rehace. Un símbolo que no
    está en el universo queda con grupo y región "?"."""
    u = universe.drop_duplicates("symbol").set_index("symbol")
    info = pd.DataFrame({"symbol": list(symbols)})
    info["asset_class"] = info["symbol"].map(u["asset_class"])
    info["country"] = info["symbol"].map(u["country"])
    group = info["country"].fillna(info["asset_class"])
    info["group"] = group.fillna("?")
    info["region"] = np.where(group.isna(), "?",
                              np.where(group.isin(["US", "crypto"]), group, "Europa"))
    return info


class WindowData:
    """Acceso a las muestras: corta y normaliza las ventanas al vuelo, por lotes."""

    def __init__(self, directory: Path, allow_test: bool = False):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.window = self.meta["params"]["window"]
        self.horizons = tuple(self.meta["params"]["horizons"])
        self.arr = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r")
                    for k in BAR_ARRAYS + TARGET_ARRAYS}
        self.samples = pd.read_parquet(self.dir / "samples.parquet")
        self.allow_test = allow_test
        self._regions = None
        self._market = None
        self._size = None

    @property
    def regions(self) -> np.ndarray:
        """Código de región de cada activo (data.market.REGION_CODES: 0 EE. UU. y cripto, 1
        Europa), del universo en <dataset>/../universe.parquet."""
        if self._regions is None:
            from agentic_trading.data.market import asset_regions
            uni = pd.read_parquet(self.dir.parent / "universe.parquet")
            self._regions = asset_regions(self.meta["symbols"], uni)
        return self._regions

    @property
    def market(self) -> dict:
        """Arrays por fila de la vela del mercado (data.market.build_market), en <dataset>/market."""
        if self._market is None:
            from agentic_trading.data.market import MARKET_ARRAYS
            d = self.dir / "market"
            if not (d / "meta.json").exists():
                raise FileNotFoundError(f"falta la vela del mercado en {d} (data.market."
                                        "build_market)")
            meta = json.loads((d / "meta.json").read_text())
            fp = self.meta.get("clean_fingerprint")
            if meta["n_rows"] != self.meta["n_rows"] or fp not in (None,
                                                                   meta["clean_fingerprint"]):
                raise ValueError(f"{d}: la vela del mercado es de otro dataset")
            self._market = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in MARKET_ARRAYS}
        return self._market

    @property
    def size(self) -> np.ndarray:
        """Percentil de tamaño de cada muestra (data.size.build_size), en <dataset>/size."""
        if self._size is None:
            d = self.dir / "size"
            if not (d / "meta.json").exists():
                raise FileNotFoundError(f"falta el tamaño en {d} (data.size.build_size)")
            meta = json.loads((d / "meta.json").read_text())
            fp = self.meta.get("clean_fingerprint")
            if meta["n_samples"] != len(self.samples) or fp not in (None,
                                                                    meta["clean_fingerprint"]):
                raise ValueError(f"{d}: el tamaño es de otro dataset")
            self._size = np.load(d / "pct.npy")
        return self._size

    def split(self, name: str) -> np.ndarray:
        """Índices de muestra de un bloque. El test exige allow_test=True (se abre una vez)."""
        if name == "test" and not self.allow_test:
            raise PermissionError("El test es ciego: abre WindowData(..., allow_test=True)")
        return np.flatnonzero(split_mask(self.samples, name, self.meta["splits"]))

    def epoch_indices(self, idx: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
        """Submuestra aleatoria (sin reemplazo) de una fracción de las muestras, barajada."""
        n = max(1, round(len(idx) * frac))
        return rng.choice(idx, size=n, replace=False)

    def date_means(self, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Para el target relativo: por muestra, la media de y_h de las muestras de idx de su
        fecha (n_muestras × horizontes) y cuántas son.

        Solo cuentan las muestras de idx: en el entrenamiento, las del bloque (train_idx), así que
        una muestra que no está en idx no cambia ninguna media. Los arrays tienen una fila por
        muestra del dataset, con NaN y 0 fuera de idx. Por defecto, idx = todas las muestras."""
        n_all = len(self.samples)
        idx = np.arange(n_all) if idx is None else np.unique(np.asarray(idx))
        rows = self.samples["row"].to_numpy()[idx]
        codes, _ = pd.factorize(self.samples["date"].to_numpy()[idx])
        n = np.bincount(codes)
        means = np.full((n_all, len(self.horizons)), np.nan, np.float32)
        for j, h in enumerate(self.horizons):
            sums = np.bincount(codes, weights=np.asarray(self.arr[f"y_{h}"])[rows]
                               .astype(np.float64))
            means[idx, j] = (sums / n)[codes]
        counts = np.zeros(n_all, np.int64)
        counts[idx] = n[codes]
        return means, counts

    def _date_ranks_pct(self, idx: np.ndarray | None):
        """Rango (medio en empates) de y_h entre las muestras de idx de su fecha, n y los códigos."""
        n_all = len(self.samples)
        idx = np.arange(n_all) if idx is None else np.unique(np.asarray(idx))
        rows = self.samples["row"].to_numpy()[idx]
        codes, _ = pd.factorize(self.samples["date"].to_numpy()[idx])
        n = np.bincount(codes)[codes]
        ranks = np.column_stack([
            pd.Series(np.asarray(self.arr[f"y_{h}"])[rows].astype(np.float64))
            .groupby(codes).rank(method="average").to_numpy() for h in self.horizons])
        return idx, ranks, n, codes

    def date_ranks(self, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Target de rango (TrainConfig.target_rank): por muestra, Φ⁻¹((rango − 0,5) / n), con el
        rango de y_h entre las n muestras de idx de su fecha (n_muestras × horizontes), y n. Como
        date_means: NaN y 0 fuera de idx; solo cuentan las muestras de idx."""
        from scipy.special import ndtri
        n_all = len(self.samples)
        idx, ranks, n, _ = self._date_ranks_pct(idx)
        out = np.full((n_all, len(self.horizons)), np.nan, np.float32)
        out[idx] = ndtri((ranks - 0.5) / n[:, None])
        counts = np.zeros(n_all, np.int64)
        counts[idx] = n
        return out, counts

    def tail_weights(self, idx: np.ndarray | None = None, weight: float = 3.0) -> np.ndarray:
        """Pesos de L_μ (TrainConfig.tail_weight): `weight` para las muestras cuyo y_h está en el
        decil alto de su fecha entre las de idx (rango / n > 0,9), 1 para las demás, divididos
        por la media de su fecha (peso medio 1 en cada fecha). n_muestras × horizontes; NaN
        fuera de idx."""
        n_all = len(self.samples)
        idx, ranks, n, codes = self._date_ranks_pct(idx)
        w = np.where(ranks / n[:, None] > 0.9, float(weight), 1.0)
        for j in range(w.shape[1]):
            w[:, j] /= (np.bincount(codes, weights=w[:, j]) / np.bincount(codes))[codes]
        out = np.full((n_all, len(self.horizons)), np.nan, np.float32)
        out[idx] = w
        return out

    def batch(self, idx: np.ndarray, window: int | None = None,
              coarse: str | None = None, market: bool = False, size: bool = False) -> dict:
        """Lote de muestras: x (B, window, 5), scalar (B, 1), y (B, 3), v (B, 3), mask_sigma (B, 3).

        window: solo las últimas `window` velas de la ventana (por defecto, la ventana entera).
        σ_ventana, la mediana del volumen y los targets siguen siendo los de la ventana entera.
        coarse: "month" o "week" añade xc (B, n_velas, 5), las velas agregadas (coarse_bars).
        market: x (y xc) llevan 4 canales más, la vela del mercado de la región (self.market),
        normalizada con σ_ventana del activo: (B, window, 9).
        size: scalar lleva una columna más, el percentil de tamaño (self.size): (B, 2)."""
        w = self.window if window is None else int(window)
        if not 1 <= w <= self.window:
            raise ValueError(f"window debe estar entre 1 y {self.window}")
        s = self.samples.iloc[idx]
        rows = s["row"].to_numpy()
        win = rows[:, None] + np.arange(-w + 1, 1)[None, :]
        a = self.arr
        sigma = np.asarray(a["sigma"])[rows].astype(np.float64)
        lc_t = np.asarray(a["log_close"])[rows]
        x = np.empty((len(rows), w, 5), np.float32)
        for j, k in enumerate(("log_open", "log_high", "log_low", "log_close")):
            x[:, :, j] = (np.asarray(a[k])[win] - lc_t[:, None]) / sigma[:, None]
        vol = np.asarray(a["volume"])[win]
        med = np.asarray(a["vol_median"])[rows].astype(np.float64)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(vol > 0, np.log(vol / med), 0.0)
        x[:, :, 4] = np.where(np.isfinite(v), v, 0.0)
        if market:
            m = self.market
            mc_t = np.asarray(m["m_log_close"])[rows]
            xm = np.stack([(np.asarray(m[k])[win] - mc_t[:, None]) / sigma[:, None]
                           for k in ("m_log_open", "m_log_high", "m_log_low", "m_log_close")], 2)
            x = np.concatenate([x, xm.astype(np.float32)], 2)
        out = {
            "x": x,
            "scalar": np.log(sigma)[:, None].astype(np.float32),
            "y": np.stack([np.asarray(a[f"y_{h}"])[rows] for h in self.horizons], 1),
            "v": np.stack([np.asarray(a[f"v_{h}"])[rows] for h in self.horizons], 1),
            "mask_sigma": s[[f"mask_sigma_{h}" for h in self.horizons]].to_numpy(bool),
            "asset": s["asset"].to_numpy(), "date": s["date"].to_numpy(),
        }
        if coarse is not None:
            out["xc"] = self.coarse_bars(rows, coarse, market=market)
        if size:
            out["scalar"] = np.concatenate([out["scalar"], self.size[idx][:, None]], 1)
        return out

    def coarse_bars(self, rows: np.ndarray, kind: str, market: bool = False) -> np.ndarray:
        """Velas agregadas de las ventanas que terminan en las filas rows: (B, n_velas, 5).

        Cada vela junta n sesiones consecutivas (COARSE[kind]) y la última termina hoy (t):
        apertura de la primera, máximo de los máximos, mínimo de los mínimos, cierre de la última
        y volumen total. Precios: log(P / C_t) / σ_ventana. Volumen: log(V / (n · mediana)), con
        la mediana diaria de V > 0 de la ventana. Como en las velas diarias, un volumen 0 cuenta
        como la mediana (si lo son todos los de la vela, v = 0). market: 4 canales más, las velas
        agregadas del mercado (self.market) con la misma regla: (B, n_velas, 9)."""
        if kind not in COARSE:
            raise ValueError(f"coarse debe ser uno de {sorted(COARSE)}, no {kind!r}")
        n_bars, n_ses = COARSE[kind]
        span = n_bars * n_ses
        if span > self.window:
            raise ValueError(f"{kind}: {span} velas no caben en la ventana de {self.window}")
        rows = np.asarray(rows)
        pos = rows[:, None] + np.arange(-span + 1, 1)[None, :]
        shape = (len(rows), n_bars, n_ses)
        a = self.arr
        sigma = np.asarray(a["sigma"])[rows].astype(np.float64)[:, None]
        lc_t = np.asarray(a["log_close"])[rows][:, None]
        bars = (np.asarray(a["log_open"])[pos[:, ::n_ses]],
                np.asarray(a["log_high"])[pos].reshape(shape).max(2),
                np.asarray(a["log_low"])[pos].reshape(shape).min(2),
                np.asarray(a["log_close"])[pos[:, n_ses - 1::n_ses]])
        xc = np.empty((len(rows), n_bars, 5), np.float32)
        for j, p in enumerate(bars):
            xc[:, :, j] = (p - lc_t) / sigma
        med = np.asarray(a["vol_median"])[rows].astype(np.float64)[:, None]
        vol = np.asarray(a["volume"])[pos].astype(np.float64)
        vol = np.where(vol > 0, vol, med).reshape(shape).sum(2)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log(vol / (n_ses * med))
        xc[:, :, 4] = np.where(np.isfinite(v), v, 0.0)
        if market:
            m = self.market
            mc_t = np.asarray(m["m_log_close"])[rows][:, None]
            mb = (np.asarray(m["m_log_open"])[pos[:, ::n_ses]],
                  np.asarray(m["m_log_high"])[pos].reshape(shape).max(2),
                  np.asarray(m["m_log_low"])[pos].reshape(shape).min(2),
                  np.asarray(m["m_log_close"])[pos[:, n_ses - 1::n_ses]])
            xm = np.stack([(p - mc_t) / sigma for p in mb], 2).astype(np.float32)
            xc = np.concatenate([xc, xm], 2)
        return xc
