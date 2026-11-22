"""PPO para el agente (handoff §10), en PyTorch y CPU.

- n_envs entornos en el mismo proceso (la red decide para cada uno; el simulador es numpy).
  Episodios de `episode` días de decisión con inicio aleatorio dentro del entrenamiento
  (2000–2015), empezando 100 % en liquidez. El corte del episodio es por tiempo, no terminal: la
  última recompensa lleva γ·V(estado final).
- Recompensa: log-retorno diario neto de costes, ×reward_scale (100) para el crítico; no cambia
  el óptimo. Con reward = "relative", menos el log-retorno de los pesos iguales ese paso
  (Panel.ew_logret): no depende de las acciones, así que tampoco cambia el óptimo (la suma es
  log(W_final/W_0) − log(EW_final/EW_0)), y quita la varianza del mercado. Con "absolute", la
  de §10 tal cual.
- GAE (λ = 0,95), ventajas normalizadas por lote, ratio recortado (clip), pérdida del crítico
  MSE sobre los retornos, bonificación de entropía pequeña, recorte del gradiente.
- Ratio (ratio): "joint", el de la acción conjunta del día (suma de ~60 log-probabilidades:
  operar, puntuación y stop de cada candidato, y la liquidez); con tantas dimensiones toca el
  recorte enseguida aunque cada componente cambie poco (prueba de humo del 21-11-2026). O
  "component": un ratio recortado por componente, con la misma ventaja, y la media de los
  objetivos de los componentes de cada paso (como si fueran independientes; aproximación
  habitual con acciones de muchas dimensiones). Las log-probabilidades de partida se calculan
  con la red al empezar la actualización, que es la que recogió los datos. En este modo, la
  entropía también se promedia por componente (si no, pesa ~60 veces más que el objetivo y la
  red solo aprende a subir el ruido: prueba smoke_relcomp_g095_s0).
- trade_bias: sesgo inicial de operar/mantener (p(operar) = σ(trade_bias); 0 → 0,5).
- Determinista con semilla: torch con un hilo y generadores de numpy por entorno.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from agentic_trading.agent.data import TRAIN_END, Panel
from agentic_trading.agent.env import TradingEnv
from agentic_trading.agent.policy import (
    AgentNet,
    act,
    component_logprobs,
    evaluate_actions,
    obs_tensors,
    record_tensors,
)
from agentic_trading.agent.simulator import Result, Rules, agent_rules


@dataclass
class PPOConfig:
    gamma: float = 0.95
    gae_lambda: float = 0.95
    n_envs: int = 8
    rollout: int = 64               # pasos por entorno y actualización
    epochs: int = 4
    minibatch: int = 128
    lr: float = 3e-4
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    max_grad: float = 0.5
    reward_scale: float = 100.0
    K: int = 20
    eps: float = 0.05
    episode: int = 252
    cost: float = 0.001
    reward: str = "absolute"        # "absolute" (§10) o "relative" (menos los pesos iguales)
    ratio: str = "joint"            # "joint" o "component"
    trade_bias: float = 0.0         # p(operar) inicial = σ(trade_bias)
    hidden: int = 64
    emb: int = 32
    seed: int = 0


class Trainer:
    def __init__(self, panel: Panel, cfg: PPOConfig, rules: Rules | None = None):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.p, self.cfg = panel, cfg
        self.rules = rules or agent_rules(cfg.cost)
        self.net = AgentNet(hidden=cfg.hidden, emb=cfg.emb, trade_bias=cfg.trade_bias)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        k_lo, k_hi = panel.k_range(panel.dec_dates[0], TRAIN_END)
        self.k_lo, self.k_hi = k_lo, k_hi - 1 - cfg.episode   # el último paso acaba ≤ 2015
        assert self.k_hi > self.k_lo, "entrenamiento demasiado corto para un episodio"
        self.rngs = [np.random.default_rng([cfg.seed, i]) for i in range(cfg.n_envs)]
        self.envs = [TradingEnv(panel, self.rules, cfg.episode) for _ in range(cfg.n_envs)]
        self.obs = [e.reset(int(r.integers(self.k_lo, self.k_hi + 1)))
                    for e, r in zip(self.envs, self.rngs)]
        self.ep_ret = np.zeros(cfg.n_envs)
        self.updates, self.steps = 0, 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ recogida
    def collect(self) -> tuple[list, dict]:
        cfg, net = self.cfg, self.net
        E, T = cfg.n_envs, cfg.rollout
        buf = [[None] * T for _ in range(E)]
        stats = {k: [] for k in ("reward", "reward_rel", "turnover", "npos", "cash", "wd",
                                 "stops", "ep_return", "n_trade")}
        for t in range(T):
            for i, env in enumerate(self.envs):
                o, k = self.obs[i], env.k
                order, rec, lp, v = act(net, o, self.rules, self.rngs[i], True, cfg.K, cfg.eps)
                o2, r, done, info = env.step_order(order)
                r_rel = r - float(self.p.ew_logret[k])
                rs = (r_rel if cfg.reward == "relative" else r) * cfg.reward_scale
                if done:
                    with torch.no_grad():
                        h = net.heads(**obs_tensors([o2]))
                    rs += cfg.gamma * float(h["value"][0])
                buf[i][t] = (o, rec, lp, v, rs, done)
                self.ep_ret[i] += r
                pf = env.pf
                held = pf.n > 0
                stats["reward"].append(r)
                stats["reward_rel"].append(r_rel)
                stats["turnover"].append(info["traded"] / 2)
                stats["npos"].append(int(held.sum()))
                stats["cash"].append(pf.cash / pf.nav)
                stats["stops"].append(info["stops"])
                stats["n_trade"].append(order["n_trade"])
                wd = (pf.weights() * np.nan_to_num(pf.stop_distance()))[held]
                stats["wd"] += wd.tolist()
                if done:
                    stats["ep_return"].append(self.ep_ret[i])
                    self.ep_ret[i] = 0.0
                    o2 = env.reset(int(self.rngs[i].integers(self.k_lo, self.k_hi + 1)))
                self.obs[i] = o2
        with torch.no_grad():
            last_v = net.heads(**obs_tensors(self.obs))["value"].numpy()
        self.steps += E * T
        return self._gae(buf, last_v), stats

    def _gae(self, buf, last_v) -> list:
        cfg, out = self.cfg, []
        for i, seq in enumerate(buf):
            adv, nxt = 0.0, float(last_v[i])
            rows = []
            for o, rec, lp, v, r, done in reversed(seq):
                if done:
                    nxt, adv = 0.0, 0.0
                delta = r + cfg.gamma * nxt - v
                adv = delta + cfg.gamma * cfg.gae_lambda * adv
                rows.append((o, rec, lp, adv, adv + v))
                nxt = v
            out += rows[::-1]
        return out

    # ------------------------------------------------------------------ actualización
    def update(self, data: list) -> dict:
        cfg, net = self.cfg, self.net
        n = len(data)
        adv = np.array([d[3] for d in data], np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rng = np.random.default_rng([cfg.seed, 10_000 + self.updates])
        logs = {k: [] for k in ("loss_pi", "loss_v", "entropy", "kl", "clipfrac")}
        old = self._old_components(data) if cfg.ratio == "component" else None
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                ob = obs_tensors([data[j][0] for j in idx])
                rc = record_tensors([data[j][1] for j in idx])
                lp_old = torch.tensor([data[j][2] for j in idx])
                a = torch.from_numpy(adv[idx])
                ret = torch.tensor([data[j][4] for j in idx], dtype=torch.float32)
                if old is None:
                    lp, ent, v = evaluate_actions(net, ob, rc)
                    ratio = (lp - lp_old).exp()
                    l_pi = -torch.min(ratio * a,
                                      ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                    cmask = torch.ones_like(ratio)
                    lp_c, lp_old_c = lp, lp_old
                else:
                    lp_c, cmask, ent, v = component_logprobs(net, ob, rc)
                    lp_old_c = self._pad_old([old[j] for j in idx], rc["cand"].shape[1])
                    ratio = (lp_c - lp_old_c).exp()
                    a2 = a[:, None]
                    surr = torch.min(ratio * a2, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a2)
                    l_pi = -((surr * cmask).sum(1) / cmask.sum(1)).mean()
                    ent = ent / cmask.sum(1)        # misma escala que el objetivo (por componente)
                l_v = ((v - ret) ** 2).mean()
                loss = l_pi + cfg.vf_coef * l_v - cfg.ent_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad)
                self.opt.step()
                with torch.no_grad():
                    logs["loss_pi"].append(float(l_pi))
                    logs["loss_v"].append(float(l_v))
                    logs["entropy"].append(float(ent.mean()))
                    nm = cmask.sum()
                    logs["kl"].append(float(((lp_old_c - lp_c) * cmask).sum() / nm))
                    logs["clipfrac"].append(float((((ratio - 1).abs() > cfg.clip).float()
                                                   * cmask).sum() / nm))
        self.updates += 1
        return {k: float(np.mean(v)) for k, v in logs.items()}

    def _old_components(self, data: list) -> list:
        """Log-probabilidades de partida por componente de cada paso, sin relleno: (operar,
        puntuación, stop) de sus c candidatos y la liquidez."""
        out = []
        with torch.no_grad():
            for s in range(0, len(data), self.cfg.minibatch):
                chunk = data[s:s + self.cfg.minibatch]
                ob = obs_tensors([d[0] for d in chunk])
                rc = record_tensors([d[1] for d in chunk])
                lp, _, _, _ = component_logprobs(self.net, ob, rc)
                C = rc["cand"].shape[1]
                for i, d in enumerate(chunk):
                    c = len(d[1].cand)
                    out.append((lp[i, :c].clone(), lp[i, C:C + c].clone(),
                                lp[i, 2 * C:2 * C + c].clone(), lp[i, 3 * C].clone()))
        return out

    @staticmethod
    def _pad_old(rows: list, C: int) -> torch.Tensor:
        t = torch.zeros(len(rows), 3 * C + 1)
        for i, (tr, sc, fr, lq) in enumerate(rows):
            c = len(tr)
            t[i, :c], t[i, C:C + c], t[i, 2 * C:2 * C + c], t[i, 3 * C] = tr, sc, fr, lq
        return t

    def train_step(self) -> dict:
        t0 = time.time()
        data, st = self.collect()
        t1 = time.time()
        lg = self.update(data)
        t2 = time.time()
        wd = np.array(st["wd"]) if st["wd"] else np.array([np.nan])
        row = {"update": self.updates, "steps": self.steps,
               "reward_mean": float(np.mean(st["reward"])),
               "reward_rel_mean": float(np.mean(st["reward_rel"])),
               "ep_return": float(np.mean(st["ep_return"])) if st["ep_return"] else None,
               "turnover_day": float(np.mean(st["turnover"])), "npos": float(np.mean(st["npos"])),
               "cash": float(np.mean(st["cash"])), "stops_day": float(np.mean(st["stops"])),
               "n_trade": float(np.mean(st["n_trade"])),
               "wd_p10": float(np.nanpercentile(wd, 10)), "wd_p50": float(np.nanmedian(wd)),
               "wd_p90": float(np.nanpercentile(wd, 90)),
               "std": self.net.log_std.exp().item(), "a": self.net.a.item(),
               "t_collect": t1 - t0, "t_update": t2 - t1, **lg}
        self.history.append(row)
        return row

    # ------------------------------------------------------------------ guardar
    def state_dict(self) -> dict:
        return {"cfg": asdict(self.cfg), "net": self.net.state_dict(),
                "opt": self.opt.state_dict(), "updates": self.updates, "steps": self.steps,
                "history": self.history,
                "rngs": [r.bit_generator.state for r in self.rngs],
                "torch_rng": torch.get_rng_state(),
                "envs": [(e.k0, e.k) for e in self.envs]}

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path) -> None:
        """Reanuda: red, optimizador, contadores y generadores. Los entornos empiezan episodio
        nuevo (el estado de la cartera no se guarda)."""
        ck = torch.load(path, weights_only=False)
        self.net.load_state_dict(ck["net"])
        self.opt.load_state_dict(ck["opt"])
        self.updates, self.steps, self.history = ck["updates"], ck["steps"], ck["history"]
        for r, s in zip(self.rngs, ck["rngs"]):
            r.bit_generator.state = s
        torch.set_rng_state(ck["torch_rng"])
        self.obs = [e.reset(int(r.integers(self.k_lo, self.k_hi + 1)))
                    for e, r in zip(self.envs, self.rngs)]


def run_policy(panel: Panel, net: AgentNet, start, end, rules: Rules, K: int = 20,
               explore: bool = False, seed: int = 0) -> tuple[Result, dict]:
    """Un episodio de start a end (100 % en liquidez al empezar), con la salida determinista
    (explore=False). Devuelve el Result diario (para simulator.performance) y diagnósticos."""
    k0, k1 = panel.k_range(start, end)
    env = TradingEnv(panel, rules, n_steps=k1 - 1 - k0)
    env.record = True
    obs = env.reset(k0)
    rng = np.random.default_rng(seed)
    t0 = panel.dec_t[k0]
    days = [(t0, 1.0, 0.0)]
    diag = {k: [] for k in ("npos", "cash", "wd", "n_trade")}
    done = False
    while not done:
        order, *_ = act(net, obs, rules, rng, explore, K)
        obs, _, done, info = env.step_order(order)
        days += info["days"]
        pf = env.pf
        held = pf.n > 0
        diag["npos"].append(int(held.sum()))
        diag["cash"].append(pf.cash / pf.nav)
        diag["n_trade"].append(order["n_trade"])
        diag["wd"] += (pf.weights() * np.nan_to_num(pf.stop_distance()))[held].tolist()
    t, nav, traded = map(np.array, zip(*days))
    idx = panel.market.dates[t]
    z = pd.Series(0.0, index=idx)
    res = Result(pd.Series(nav, index=idx), pd.Series(traded, index=idx),
                 pd.Series(traded * rules.cost, index=idx), z, z,
                 pd.DataFrame(columns=["date", "symbol", "price", "how"]))
    res.n_stops = env.pf.n_stops
    return res, diag


def write_jsonl(path, row: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
