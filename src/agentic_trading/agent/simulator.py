"""Simulador de cartera con las reglas del juego del agente (handoff §10).

Reglas del juego (Rules):
- Sin apalancamiento ni cortos: pesos ≥ 0 y caja ≥ 0. Si la caja no llega para las compras del
  día, se reducen todas en proporción.
- Regla del 1 % (min_weight): los pesos objetivo por debajo se ponen a 0. min_weight_to dice
  adónde va ese peso: "cash" (a la liquidez; lo que hacía siempre el simulador) o "spread" (se
  reparte entre las demás posiciones operadas, en proporción a su peso). Provisional en el
  handoff ("por decidir"); por defecto, "cash". Con close_below_min (reglas del agente), también
  se cierran las posiciones mantenidas cuyo peso ha caído por debajo, si el activo tiene
  predicción ese día; esos cierres no cuentan para el tamaño mínimo de operación.
- Tamaño mínimo de operación (min_trade, 0,5 % con las reglas del agente): no se ejecuta una
  orden si |peso objetivo − peso actual| < min_trade; la posición se queda como está, sin
  operar. Excepción: los cierres (objetivo 0), que solo pueden ser menores que min_trade si los
  exige la regla del 1 % (una compra nueva ya pesa ≥ min_weight).
- Stop por posición, con pérdida máxima stop_loss (3 %) del capital. Dos modos:
  - "fixed" (el de las líneas base, 8-11-2026): precio_stop = P_medio × (1 − stop_loss / w),
    con w el peso objetivo de la última operación en ese activo (al comprar o vender, el stop se
    recalcula con el precio medio y el nuevo peso); con w ≤ stop_loss no hay stop.
  - "agent" (21-11-2026): la política elige en cada decisión la distancia d desde el último
    cierre: precio_stop = C_último × (1 − d), con d ≤ d_max = mín(stop_loss / w, stop_cap) (con
    w ≤ 3 %, el tope del 50 %). w es el peso objetivo si se opera, o el peso actual si se
    mantiene. El simulador recorta d a d_max (w·d ≤ stop_loss al decidir; con la ejecución en
    la apertura siguiente, el peso real puede variar algo). Una posición sin d en una decisión
    (p. ej., sin predicción ese día) conserva su stop; una compra sin d, d_max.
  En los dos, salta si el mínimo del día lo toca: se vende al precio del stop, o a la apertura si
  abre por debajo (con huecos, la pérdida puede pasar del 3 %). Después se puede volver a entrar.
- Costes: cost × importe operado, en cada compra y en cada venta.

Simulador:
- Decide con los cierres del día y opera en la apertura siguiente de cada activo: su próxima vela
  válida (cada mercado tiene su calendario). Una orden pendiente que no se ha ejecutado cuando
  llega la decisión siguiente se sustituye.
- Todo en la moneda de cada activo, como si el cambio estuviera cubierto: cada posición evoluciona
  con su retorno local. Con precios ajustados por dividendos, es rentabilidad total.
- Las velas no válidas no existen para el simulador, y el salto de precio a través de ellas se
  empalma con retorno 0 (Market.from_frames).
- Caja sin interés.

Mantener u operar: en cada decisión se dan los pesos objetivo de los activos que se operan. Una
posición abierta que no está entre ellos se mantiene sin operar y conserva el peso que le dejan
los precios (igual que una posición en un activo sin predicción ese día). Si los pesos objetivo no
caben con las mantenidas, se reducen en proporción.

Dos formas de usarlo: simulate() con una política que recibe un State y devuelve pesos objetivo
(pd.Series por símbolo; las líneas base), o Portfolio, paso a paso (el entorno del agente).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rules:
    cost: float = 0.001                 # c × importe operado
    min_weight: float | None = 0.01     # posiciones objetivo menores, a 0 (None: sin regla)
    stop_loss: float | None = 0.03      # pérdida máxima por posición (None: sin stops)
    stop_mode: str = "fixed"            # "fixed" (líneas base) o "agent" (d elegido)
    stop_cap: float = 0.5               # tope de d en el modo "agent"
    min_trade: float | None = None      # órdenes con |Δw| menor no se ejecutan
    min_weight_to: str = "cash"         # adónde va el peso que la regla del 1 % pone a 0
    close_below_min: bool = False       # cerrar posiciones mantenidas por debajo de min_weight


DEFAULT_RULES = Rules()


def agent_rules(cost: float = 0.001, **kw) -> Rules:
    """Reglas del juego del agente (§10, 21-11-2026)."""
    return Rules(cost=cost, stop_mode="agent", min_trade=0.005, close_below_min=True, **kw)


def reference_rules(cost: float) -> Rules:
    """Carteras de referencia (comprar y mantener, pesos iguales): solo costes."""
    return Rules(cost=cost, min_weight=None, stop_loss=None)


def max_stop_distance(w, rules: Rules) -> np.ndarray:
    """d_max = mín(stop_loss / w, stop_cap): la distancia de stop más lejana permitida."""
    w = np.asarray(w, float)
    with np.errstate(divide="ignore"):
        return np.minimum(rules.stop_loss / np.maximum(w, 0.0), rules.stop_cap)


@dataclass
class Market:
    """Panel de precios (fechas × activos), NaN donde el activo no tiene vela válida."""
    dates: pd.DatetimeIndex
    symbols: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    breaks: int = 0             # cortes empalmados (ver from_frames)

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame], start=None, end=None) -> Market:
        """frames: símbolo → DataFrame con índice de fechas y columnas open, high, low, close
        (y, si la tiene, valid). Solo cuentan las velas válidas con OHLC finito y positivo.

        Cortes: entre dos velas válidas con alguna no válida en medio (lo grave sin reparar de
        la limpieza: splits sin corroborar, huecos de más de 10 días, otra entidad...), el salto
        de precio no es fiable (hay saltos de ×57). Se empalma: los precios desde el corte se
        reescalan para que su apertura sea el último cierre válido. Una posición que cruza el
        corte gana o pierde 0 en él; después sigue con sus retornos reales."""
        cols = ["open", "high", "low", "close"]
        parts: dict[str, dict] = {k: {} for k in cols}
        n_breaks = 0
        for s, df in frames.items():
            df = df.loc[start:end]
            ok = df[cols].notna().all(axis=1) & (df[cols] > 0).all(axis=1)
            if "valid" in df:
                ok &= df["valid"].astype(bool)
            px = df.loc[ok, cols].astype(float)
            pos = np.flatnonzero(ok.to_numpy())
            cut = np.flatnonzero(np.diff(pos) > 1) + 1          # primera válida tras un corte
            if len(cut):
                f = np.ones(len(px))
                f[cut] = px["close"].to_numpy()[cut - 1] / px["open"].to_numpy()[cut]
                px = px.mul(np.cumprod(f), axis=0)
                n_breaks += len(cut)
            for k in cols:
                parts[k][s] = px[k]
        dates = pd.DatetimeIndex(sorted(set().union(*(p.index for p in parts["close"].values()))))
        syms = list(frames)
        arr = {k: pd.DataFrame(parts[k]).reindex(index=dates, columns=syms).to_numpy(float)
               for k in cols}
        return cls(dates, syms, arr["open"], arr["high"], arr["low"], arr["close"], n_breaks)


@dataclass
class State:
    """Lo que ve la política al decidir (cierre del día)."""
    date: pd.Timestamp
    nav: float
    cash: float
    weights: pd.Series          # posiciones abiertas: valor / NAV
    stop_distance: pd.Series    # (precio − stop) / precio, de las posiciones con stop


@dataclass
class Result:
    nav: pd.Series              # valor de la cartera al cierre
    traded: pd.Series           # importe operado (compras + ventas) en el día
    costs: pd.Series
    cash: pd.Series
    positions: pd.Series        # nº de posiciones abiertas al cierre
    stops: pd.DataFrame         # fecha, símbolo, precio y si saltó en la apertura o en el día


def _stop_price(avg: np.ndarray, w: np.ndarray, loss: float | None) -> np.ndarray:
    if loss is None:
        return np.full(len(avg), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = avg * (1 - loss / w)
    return np.where(np.isfinite(avg) & np.isfinite(w) & (w > loss), s, np.nan)


class Portfolio:
    """Estado de la cartera, día a día: run_day(t) procesa la fecha t (órdenes pendientes en la
    apertura, stops y valoración al cierre); order() fija las órdenes con los cierres de hoy."""

    def __init__(self, market: Market, rules: Rules = DEFAULT_RULES, nav0: float = 1.0,
                 record_stops: bool = True):
        N = len(market.symbols)
        self.m, self.rules = market, rules
        self.n = np.zeros(N)                    # unidades de cada activo
        self.avg = np.full(N, np.nan)           # precio medio de compra
        self.stop = np.full(N, np.nan)
        self.last = np.full(N, np.nan)          # último cierre válido
        self.pend_n = np.full(N, np.nan)        # órdenes pendientes: unidades objetivo ...
        self.pend_w = np.full(N, np.nan)        # ... peso objetivo (stop "fixed") ...
        self.pend_stop = np.full(N, np.nan)     # ... y precio de stop (stop "agent")
        self.cash = float(nav0)
        self.nav = float(nav0)
        self.record_stops = record_stops
        self.stops: list[tuple] = []
        self.n_stops = 0

    # ------------------------------------------------------------------ estado
    def values(self) -> np.ndarray:
        return self.n * np.nan_to_num(self.last)

    def weights(self) -> np.ndarray:
        return self.values() / self.nav if self.nav > 0 else np.zeros(len(self.n))

    def stop_distance(self) -> np.ndarray:
        """(último cierre − stop) / último cierre; NaN sin posición o sin stop."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.n > 0, (self.last - self.stop) / self.last, np.nan)

    # ------------------------------------------------------------------ un día
    def run_day(self, t: int) -> tuple[float, float]:
        """Procesa la fecha t. Devuelve (NAV al cierre, importe operado)."""
        m, r = self.m, self.rules
        o, lo, cl = m.open[t], m.low[t], m.close[t]
        n, avg, c = self.n, self.avg, r.cost
        bar = np.isfinite(cl)
        traded = 0.0

        # 1) órdenes pendientes, en la apertura de hoy de cada activo con vela
        j = np.flatnonzero(bar & np.isfinite(self.pend_n))
        if len(j):
            dn, px = self.pend_n[j] - n[j], o[j]
            s = dn < 0
            if s.any():
                amt = float((-dn[s] * px[s]).sum())
                self.cash += amt * (1 - c)
                traded += amt
                n[j[s]] += dn[s]
            b = dn > 0
            if b.any():
                need = float((dn[b] * px[b]).sum()) * (1 + c)
                q = dn[b] * min(1.0, max(self.cash, 0.0) / need)      # sin apalancamiento
                jb, old = j[b], n[j[b]]
                avg[jb] = np.where(old > 0, (old * np.nan_to_num(avg[jb]) + q * px[b]) /
                                   np.maximum(old + q, 1e-300), px[b])
                n[jb] = old + q
                amt = float((q * px[b]).sum())
                self.cash -= amt * (1 + c)
                traded += amt
            gone = n[j] <= 1e-12
            n[j[gone]], avg[j[gone]] = 0.0, np.nan
            if r.stop_mode == "agent":
                self.stop[j] = np.where(n[j] > 0, self.pend_stop[j], np.nan)
            else:
                self.stop[j] = _stop_price(avg[j], self.pend_w[j], r.stop_loss)
            self.pend_n[j], self.pend_w[j], self.pend_stop[j] = np.nan, np.nan, np.nan

        # 2) stops, con la vela de hoy
        k = np.flatnonzero(bar & (n > 0) & np.isfinite(self.stop))
        if len(k):
            gap = o[k] <= self.stop[k]
            hit = gap | (lo[k] <= self.stop[k])
            if hit.any():
                kk = k[hit]
                px = np.where(gap[hit], o[kk], self.stop[kk])
                amt = float((n[kk] * px).sum())
                self.cash += amt * (1 - c)
                traded += amt
                self.n_stops += len(kk)
                if self.record_stops:
                    d = m.dates[t]
                    self.stops += [(d, m.symbols[i], p, "apertura" if g else "stop")
                                   for i, p, g in zip(kk, px, gap[hit])]
                n[kk], avg[kk], self.stop[kk] = 0.0, np.nan, np.nan
                self.pend_n[kk], self.pend_w[kk], self.pend_stop[kk] = np.nan, np.nan, np.nan

        # 3) valoración al cierre
        self.last = np.where(bar, cl, self.last)
        self.nav = self.cash + float(self.values().sum())
        return self.nav, traded

    # ------------------------------------------------------------------ decisión
    def order(self, je: np.ndarray, w: np.ndarray, stop_d: np.ndarray | None = None,
              eligible: np.ndarray | None = None) -> None:
        """Órdenes con los cierres de hoy, para la apertura siguiente.

        je, w: activos que se operan y su peso objetivo (fracción del NAV). Las posiciones
        abiertas fuera de je se mantienen sin operar.
        stop_d (modo "agent"): distancia de stop d por activo (longitud N; NaN: sin cambio),
        para las posiciones que se mantienen y para las que se operan.
        eligible (con close_below_min): activos con decisión hoy; sus posiciones mantenidas por
        debajo de min_weight se cierran."""
        r, nav, last = self.rules, self.nav, self.last
        if nav <= 0:
            return
        je = np.asarray(je, dtype=int)
        w = np.clip(np.nan_to_num(np.asarray(w, float)), 0.0, None)
        w[~np.isfinite(last[je])] = 0.0                  # sin precio todavía: no se compra
        held = np.flatnonzero(self.n > 0)
        cur = self.values() / nav
        forced = np.zeros(len(je), bool)
        if r.close_below_min and r.min_weight and eligible is not None:
            el = np.zeros(len(self.n), bool)
            el[np.asarray(eligible, dtype=int)] = True
            small = np.setdiff1d(held[el[held] & (cur[held] < r.min_weight)], je)
            je = np.r_[je, small]
            w = np.r_[w, np.zeros(len(small))]
            forced = np.r_[forced, np.ones(len(small), bool)]
        kept = np.setdiff1d(held, je)                    # en cartera y no operadas: se quedan
        room = max(0.0, 1.0 - float(cur[kept].sum()))
        if w.sum() > room:
            w *= room / w.sum()
        if r.min_weight:
            total = w.sum()
            w[w < r.min_weight] = 0.0
            if r.min_weight_to == "spread" and w.sum() > 0:
                w *= total / w.sum()
        if r.min_trade:
            skip = (np.abs(w - cur[je]) < r.min_trade) & (w > 0) & ~forced
            if skip.any():                               # no se ejecutan: se mantienen
                kept = np.union1d(kept, je[skip & (self.n[je] > 0)])
                keep = ~skip
                je, w = je[keep], w[keep]
        self.pend_n[kept], self.pend_w[kept], self.pend_stop[kept] = np.nan, np.nan, np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            self.pend_n[je] = np.where(w > 0, w * nav / last[je], 0.0)
        self.pend_w[je] = w
        if r.stop_mode == "agent" and r.stop_loss is not None:
            d = np.full(len(self.n), np.nan) if stop_d is None else np.asarray(stop_d, float)
            # operadas: d recortado al d_max de su peso objetivo (sin d, d_max)
            dj = np.where(np.isfinite(d[je]), d[je], np.inf)
            dj = np.clip(dj, 0.0, max_stop_distance(w, r))
            self.pend_stop[je] = np.where(w > 0, last[je] * (1 - dj), np.nan)
            # mantenidas: nuevo stop desde el último cierre, si lo hay
            dk = d[kept]
            ok = np.isfinite(dk)
            if ok.any():
                kk = kept[ok]
                dk = np.clip(dk[ok], 0.0, max_stop_distance(cur[kk], r))
                self.stop[kk] = last[kk] * (1 - dk)


def simulate(market: Market, policy: Callable[[State], pd.Series], decision_dates,
             rules: Rules = DEFAULT_RULES, nav0: float = 1.0) -> Result:
    """Simula la cartera día a día con una política de pesos objetivo (ver el docstring del
    módulo). Los activos de la serie que devuelve la política se operan; el resto de posiciones
    se mantienen. En el modo de stop "agent", sin d, cada compra lleva el stop en d_max."""
    T = len(market.dates)
    col = {s: j for j, s in enumerate(market.symbols)}
    syms = np.asarray(market.symbols, dtype=object)
    pf = Portfolio(market, rules, nav0)
    decide = set(pd.DatetimeIndex(decision_dates))
    out = {k: np.zeros(T) for k in ("nav", "traded", "costs", "cash", "positions")}
    for t, d in enumerate(market.dates):
        nav, traded = pf.run_day(t)
        out["nav"][t], out["traded"][t], out["costs"][t] = nav, traded, traded * rules.cost
        out["cash"][t], out["positions"][t] = pf.cash, int((pf.n > 0).sum())
        if d in decide and nav > 0:
            held = np.flatnonzero(pf.n > 0)
            val = pf.values()
            dist = pf.stop_distance()[held]
            state = State(d, nav, pf.cash, pd.Series(val[held] / nav, index=syms[held]),
                          pd.Series(dist, index=syms[held]).dropna())
            tgt = policy(state)
            tgt = tgt[tgt.index.isin(list(col))]
            je = np.array([col[s] for s in tgt.index], dtype=int)
            pf.order(je, tgt.to_numpy(float), eligible=je)

    idx = market.dates
    return Result(*(pd.Series(out[k], index=idx) for k in ("nav", "traded", "costs", "cash",
                                                          "positions")),
                  stops=pd.DataFrame(pf.stops, columns=["date", "symbol", "price", "how"]))


def performance(res: Result, start=None, end=None) -> dict:
    """Rentabilidad anual (compuesta), volatilidad y Sharpe anuales (sin tipo de interés: la caja
    no rinde), caída máxima y rotación anual ((compras + ventas) / 2 / NAV, sumada y por año)."""
    nav = res.nav.loc[start:end]
    traded = res.traded.loc[start:end]
    if len(nav) < 2:
        return {}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    r = nav.pct_change().dropna()
    per_year = len(r) / years
    sd = r.std()
    return {"rentabilidad anual": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
            "volatilidad": sd * np.sqrt(per_year),
            "Sharpe": r.mean() / sd * np.sqrt(per_year) if sd > 0 else np.nan,
            "caída máxima": float((nav / nav.cummax() - 1).min()),
            "rotación anual": float((traded / nav.shift().fillna(nav)).sum() / 2 / years),
            "años": years}
