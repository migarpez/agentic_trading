"""Evaluación del agente (handoff §10, revisada el 21-11-2026 tras las pruebas de humo).

Las pruebas de humo enseñaron dos problemas: la misma política estocástica da en 2016–19 entre
+10 % y +29 % al año según la semilla del muestreo, y la determinista depende del umbral p > 0,5
de mantener/operar (con p cerca de 0,5, una red casi igual opera 4 o 25 activos al día). Así que:

- Principal: la política estocástica (la que optimiza PPO, con la exploración ε de entrenamiento)
  con n semillas de muestreo; media y error típico entre semillas.
- Referencia: la política inicial congelada (la red con la misma semilla antes de entrenar),
  evaluada con las mismas semillas. Lo aprendido es la diferencia; con las mismas semillas, la
  diferencia se toma por pares (no es un emparejamiento exacto: las trayectorias divergen).
- Secundaria: la determinista (una ejecución).
- Regla fija con las mismas reglas del juego y el mismo entorno (rule_order): las 20 de mayor
  μ_5 (la variable que ve la red, o la señal sintética), a partes iguales; una sale al caer fuera
  de las 60 primeras; stop en d_max. Dice cuánto de la señal se aprovecha sin aprender nada.
- Pesos iguales: log-retorno bruto entre decisiones (Panel.ew_logret, sin costes), solo como
  referencia del mercado del periodo.

Cada episodio empieza el primer día de decisión del periodo, 100 % en liquidez.
"""

from __future__ import annotations

import numpy as np

from agentic_trading.agent.data import Panel
from agentic_trading.agent.env import Obs, TradingEnv
from agentic_trading.agent.policy import AgentNet
from agentic_trading.agent.ppo import run_policy
from agentic_trading.agent.simulator import Rules, max_stop_distance, performance

METRICS = ("rentabilidad anual", "Sharpe", "caída máxima", "rotación anual")


def rule_order(obs: Obs, rules: Rules, n: int = 20, exit_rank: int = 60,
               col: int = 0) -> dict:
    """Orden de la regla fija (ver el docstring del módulo) para TradingEnv.step_order."""
    score = obs.x[:, col]
    rank = np.empty(len(score), int)
    rank[np.argsort(-score, kind="stable")] = np.arange(len(score))
    held = obs.held
    stay = np.flatnonzero(held & (rank < exit_rank))
    slots = max(n - len(stay) - round(obs.room_kept * n), 0)
    new = [i for i in np.argsort(-score, kind="stable") if not held[i]][:slots]
    keep = np.r_[stay, new].astype(int)
    sell = np.flatnonzero(held & (rank >= exit_rank))
    je_idx = np.r_[keep, sell].astype(int)
    w = np.r_[np.full(len(keep), 1.0 / n), np.zeros(len(sell))]
    d = max_stop_distance(w[:len(keep)], rules)
    return {"je": obs.elig[je_idx], "w": w, "cand": obs.elig[keep], "d": d,
            "cash": 1 - w.sum(), "n_trade": len(je_idx)}


def run_rule(panel: Panel, start, end, rules: Rules, **kw) -> dict:
    res = run_orders(panel, start, end, rules, lambda obs, env: rule_order(obs, rules, **kw))
    return {k: float(v) for k, v in performance(res).items()}


def run_orders(panel: Panel, start, end, rules: Rules, order_fn):
    """Simula en el entorno del agente una política fija order_fn(obs, env) → orden (la de
    TradingEnv.step_order) entre start y end, y devuelve el Result diario (NAV e importe
    operado)."""
    k0, k1 = panel.k_range(start, end)
    env = TradingEnv(panel, rules, n_steps=k1 - 1 - k0)
    env.record = True
    obs = env.reset(k0)
    days, done = [(panel.dec_t[k0], 1.0, 0.0)], False
    while not done:
        obs, _, done, info = env.step_order(order_fn(obs, env))
        days += info["days"]
    return _result(panel, days)


def kelly_order(obs: Obs, env: TradingEnv, rules: Rules) -> dict:
    """Orden de la regla tipo Kelly (baselines.kelly_weights, §11.4) en el entorno del agente:
    pesos ∝ max(μ_5, 0)/σ_5² en retornos (Panel.raw), con la regla del 1 %, sobre el peso que no
    ocupan las posiciones sin predicción; se venden las que quedan fuera; stop en d_max."""
    import pandas as pd

    from agentic_trading.agent.baselines import kelly_weights
    raw = env.p.raw[env.k]
    w = kelly_weights(pd.Series(raw[:, 0]), pd.Series(raw[:, 1])).to_numpy()
    w = w * (1 - obs.room_kept)
    keep = np.flatnonzero(w > 0)
    sell = np.flatnonzero(obs.held & (w <= 0))
    idx = np.r_[keep, sell].astype(int)
    ww = np.r_[w[keep], np.zeros(len(sell))]
    d = max_stop_distance(ww[:len(keep)], rules)
    return {"je": obs.elig[idx], "w": ww, "cand": obs.elig[keep], "d": d,
            "cash": 1 - ww.sum(), "n_trade": len(idx)}


def _result(panel: Panel, days):
    import pandas as pd

    from agentic_trading.agent.simulator import Result
    t, nav, traded = map(np.array, zip(*days))
    idx = panel.market.dates[t]
    z = pd.Series(0.0, index=idx)
    return Result(pd.Series(nav, index=idx), pd.Series(traded, index=idx), z, z, z,
                  pd.DataFrame(columns=["date", "symbol", "price", "how"]))


def ew_annual(panel: Panel, start, end) -> float:
    k0, k1 = panel.k_range(start, end)
    lr = panel.ew_logret[k0:k1 - 1]
    years = (panel.market.dates[panel.dec_t[k1 - 1]] - panel.market.dates[panel.dec_t[k0]]
             ).days / 365.25
    return float(np.exp(np.nansum(lr) / years) - 1)


def policy_runs(panel: Panel, net: AgentNet, start, end, rules: Rules, K: int,
                seeds) -> dict:
    """Métricas por semilla de la política estocástica y de la determinista."""
    out = {"stochastic": [], "deterministic": None}
    for s in seeds:
        res, diag = run_policy(panel, net, start, end, rules, K, explore=True, seed=int(s))
        m = {k: float(v) for k, v in performance(res).items()}
        m["operados por decisión"] = float(np.mean(diag["n_trade"]))
        m["posiciones"] = float(np.mean(diag["npos"]))
        m["liquidez"] = float(np.mean(diag["cash"]))
        out["stochastic"].append(m)
    res, diag = run_policy(panel, net, start, end, rules, K, explore=False)
    out["deterministic"] = {k: float(v) for k, v in performance(res).items()}
    out["deterministic"]["operados por decisión"] = float(np.mean(diag["n_trade"]))
    return out


def summarize(rows: list[dict], keys=METRICS + ("operados por decisión", "posiciones",
                                                "liquidez")) -> dict:
    """Media y error típico entre semillas."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    return out


def paired(a: list[dict], b: list[dict], key: str) -> tuple[float, float, float]:
    """Diferencia media a − b por semilla, su error típico y su t."""
    d = np.array([x[key] - y[key] for x, y in zip(a, b)])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(se), float(d.mean() / se) if se > 0 else float("nan")


def _perf(panel: Panel, days) -> dict:
    return {k: float(v) for k, v in performance(_result(panel, days)).items()}
