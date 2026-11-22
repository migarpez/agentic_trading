"""Entorno del agente (handoff §10): un paso es un día de decisión.

En cada paso, la política ve los activos con predicción ese día (Obs) y da las órdenes; el
simulador (Portfolio, con agent_rules) las ejecuta en la apertura siguiente de cada mercado y
avanza, día a día (con los stops), hasta el cierre del día de decisión siguiente. Recompensa:
log(NAV_siguiente / NAV_hoy), neta de costes.

Episodios: n_steps días de decisión desde un inicio dado (o aleatorio dentro de [k_lo, k_hi)),
empezando 100 % en liquidez. El episodio se corta por tiempo (no es terminal): el entrenamiento
arranca con el valor del estado final.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentic_trading.agent.data import Panel
from agentic_trading.agent.simulator import Portfolio, Rules, agent_rules

N_PORT = 4          # variables de la cartera: liquidez, caída desde el máximo, n.º de posiciones,
                    # capital en riesgo Σ w·d
POS_SCALE = 10.0    # peso y distancia al stop, ×10 en la entrada de la red


@dataclass
class Obs:
    elig: np.ndarray        # activos con predicción (índices de market), ordenados
    x: np.ndarray           # [n, F]: variables del modelo + peso actual + distancia al stop
    z_mu5: np.ndarray       # [n]
    held: np.ndarray        # [n] bool: posición abierta
    weight: np.ndarray      # [n] peso actual
    port: np.ndarray        # [N_PORT]
    room_kept: float        # peso de las posiciones abiertas sin predicción hoy (se mantienen)


class TradingEnv:
    def __init__(self, panel: Panel, rules: Rules | None = None, n_steps: int = 252):
        self.p = panel
        self.rules = rules or agent_rules()
        self.n_steps = n_steps
        self.record = False         # guardar NAV e importe operado de cada día (evaluación)
        self.pf: Portfolio | None = None

    # ------------------------------------------------------------------ episodio
    def reset(self, k0: int) -> Obs:
        p = self.p
        self.k0, self.k = k0, k0
        self.k_end = min(k0 + self.n_steps, len(p.dec_t) - 1)
        self.pf = Portfolio(p.market, self.rules, record_stops=False)
        t0 = p.dec_t[k0]
        self.pf.last = p.close_ffill[t0].copy()     # precios conocidos al empezar
        self.pf.nav = 1.0
        self.peak = 1.0
        return self.obs()

    def obs(self) -> Obs:
        p, pf, k = self.p, self.pf, self.k
        elig = p.elig[k]
        w_all = pf.weights()
        dist_all = np.nan_to_num(pf.stop_distance())
        w, dist = w_all[elig], dist_all[elig]
        held = pf.n[elig] > 0
        x = np.concatenate([p.feats[k], (POS_SCALE * w)[:, None],
                            (POS_SCALE * dist)[:, None]], 1).astype(np.float32)
        held_all = pf.n > 0
        risk = float((w_all * dist_all)[held_all].sum())
        npos = int(held_all.sum())
        cash = pf.cash / pf.nav if pf.nav > 0 else 1.0
        dd = pf.nav / self.peak - 1
        port = np.array([cash, dd * POS_SCALE, npos / 20.0, risk * POS_SCALE], np.float32)
        out_elig = np.ones(len(pf.n), bool)
        out_elig[elig] = False
        kept = float(w_all[held_all & out_elig].sum())
        return Obs(elig, x, p.z_mu5[k], held, w, port, kept)

    def step_order(self, order: dict):
        """step con la orden de policy.act (d de los candidatos que acaban con posición)."""
        sd = np.full(len(self.pf.n), np.nan)
        sd[order["cand"]] = order["d"]
        return self.step(order["je"], order["w"], sd)

    def step(self, je: np.ndarray, w: np.ndarray, stop_d: np.ndarray | None = None
             ) -> tuple[Obs, float, bool, dict]:
        """je, w: activos operados (índices de market) y pesos objetivo; stop_d: d por activo
        (longitud N, NaN: sin cambio). Devuelve (obs, recompensa, fin, info)."""
        p, pf = self.p, self.pf
        nav0 = pf.nav
        if stop_d is None:
            stop_d = np.full(len(pf.n), np.nan)
        pf.order(je, w, stop_d, eligible=p.elig[self.k])
        traded, stops0 = 0.0, pf.n_stops
        days = []
        for t in range(p.dec_t[self.k] + 1, p.dec_t[self.k + 1] + 1):
            nav, tr = pf.run_day(t)
            traded += tr
            if self.record:
                days.append((t, nav, tr))
        self.k += 1
        self.peak = max(self.peak, pf.nav)
        r = float(np.log(max(pf.nav, 1e-12) / nav0))
        done = self.k >= self.k_end or pf.nav <= 1e-6
        info = {"traded": traded / nav0, "nav": pf.nav, "stops": pf.n_stops - stops0,
                "days": days}
        return self.obs(), r, done, info
