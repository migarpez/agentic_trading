"""Clonado de la regla fija (behavior cloning) y arranque del crítico (23-11-2026; logs/agent/bc.md).

La regla fija del entorno (agent.evaluation.rule_order: las 20 de mayor μ_5 a partes iguales,
mantener mientras sigan entre las 60 primeras, vender al salir, stop en d_max) se escribe como
acciones de la política (policy.act), con sus propias distribuciones:

- Candidatos: las K de mayor μ_5 del día ∪ las posiciones en cartera (∪ las compras de la regla,
  que ya están entre las K primeras).
- Mantener/operar: la regla rebalancea a 1/20 las posiciones que se quedan y vende las que salen,
  así que todas las posiciones en cartera «operan»; un candidato sin posición opera si la regla lo
  compra.
- Puntuaciones (logits del softmax de los que se operan): κ·z(μ_5) para las que se quedan o se
  compran (κ pequeño: el orden de μ se conserva para elegir los candidatos, y los pesos salen casi
  iguales: con κ = 0,05, ±5 % del peso entre la primera y la vigésima); κ·z − GAP para las que se
  venden (peso < 1 %: la regla del 1 % las cierra). Liquidez: la media de las que se quedan +
  log(liquidez de la regla / (1/20)), con un suelo del 0,1 % de liquidez.
- Stop: d / d_max = FRAC (0,98; la regla usa d_max, y la beta no tiene máximo en 1).

train_bc maximiza la log-verosimilitud de esas acciones (la media de cada tipo de componente:
operar, puntuación, stop y liquidez, a partes iguales), con a = κ al empezar (la puntuación inicial
de la política es a·z(μ_5) + ψ), y con la desviación típica del ruido de las puntuaciones fija (bc_std): si se
aprendiera, se iría a 0 (los objetivos se pueden ajustar casi sin error) y PPO no exploraría.

warm_critic: antes de la primera actualización de PPO, ajusta solo el crítico con los retornos de
la política clonada (recogidas de PPO en 2000–15 con su recompensa y su γ; objetivos de GAE que se
recalculan con el crítico de cada iteración).
"""

from __future__ import annotations

import numpy as np
import torch

from agentic_trading.agent.env import Obs, TradingEnv
from agentic_trading.agent.policy import (
    FRAC_EPS,
    AgentNet,
    Record,
    act,
    component_logprobs,
    obs_tensors,
    record_tensors,
)
from agentic_trading.agent.simulator import Rules

KAPPA, GAP, FRAC, CASH_FLOOR = 0.05, 10.0, 0.98, 1e-3


def rule_record(obs: Obs, K: int = 20, n: int = 20, exit_rank: int = 60, col: int = 0,
                kappa: float = KAPPA) -> Record:
    """Acción de la política equivalente a la orden de la regla fija en obs (ver el docstring)."""
    score = obs.x[:, col]
    order = np.argsort(-score, kind="stable")
    rank = np.empty(len(score), int)
    rank[order] = np.arange(len(score))
    held = obs.held
    stay = np.flatnonzero(held & (rank < exit_rank))
    slots = max(n - len(stay) - round(obs.room_kept * n), 0)
    new = np.array([i for i in order if not held[i]][:slots], int)
    sell = np.flatnonzero(held & (rank >= exit_rank))
    zz = obs.z_mu5
    top = np.argsort(-zz, kind="stable")[:K]
    cand = np.union1d(np.union1d(top, np.flatnonzero(held)), new).astype(int)
    keep = np.union1d(stay, new)
    trade = np.isin(cand, keep) | np.isin(cand, sell)
    s = (kappa * zz[cand]).astype(np.float32)
    s[np.isin(cand, sell)] -= GAP
    kept = np.isin(cand, keep)
    if kept.any():
        cash = max(1.0 - obs.room_kept - len(keep) / n, 0.0)
        liq = float(s[kept].mean() + np.log(max(cash, CASH_FLOOR) * n))
    else:
        liq = float(kappa * zz.max() + GAP)
    rel = trade | held[cand]
    frac = np.full(len(cand), FRAC, np.float32)
    return Record(cand, trade, s, liq, frac, rel)


def collect_rule(panel, rules: Rules, k_lo: int, k_hi: int, episode: int = 52,
                 stride: int = 4) -> list[tuple[Obs, Record]]:
    """Estados y acciones de la regla: episodios de `episode` decisiones desde liquidez, con un
    inicio cada `stride` decisiones en [k_lo, k_hi] (el último acaba antes de k_hi + episode),
    más un episodio largo de k_lo al final del entrenamiento."""
    from agentic_trading.agent.evaluation import rule_order
    out = []
    starts = [(k, episode) for k in range(k_lo, k_hi + 1, stride)]
    starts.append((k_lo, k_hi + episode - k_lo))
    for k0, steps in starts:
        env = TradingEnv(panel, rules, n_steps=steps)
        obs, done = env.reset(k0), False
        while not done:
            out.append((obs, rule_record(obs)))
            obs, _, done, _ = env.step_order(rule_order(obs, rules))
    return out


def collect_clone(panel, net: AgentNet, rules: Rules, k_lo: int, k_hi: int, episode: int = 52,
                  stride: int = 8, seed: int = 0) -> list[tuple[Obs, Record]]:
    """DAgger: estados que visita la política (estocástica), etiquetados con la regla."""
    rng = np.random.default_rng(seed)
    out = []
    for k0 in range(k_lo, k_hi + 1, stride):
        env = TradingEnv(panel, rules, n_steps=episode)
        obs, done = env.reset(k0), False
        while not done:
            out.append((obs, rule_record(obs)))
            order, *_ = act(net, obs, rules, rng, True)
            obs, _, done, _ = env.step_order(order)
    return out


def bc_loss(net: AgentNet, obs_list: list[Obs], recs: list[Record]) -> tuple[torch.Tensor, dict]:
    """− log-verosimilitud media por componente, y aciertos de operar/mantener."""
    ob, rc = obs_tensors(obs_list), record_tensors(recs)
    lp, mask, _, _ = component_logprobs(net, ob, rc)
    C = rc["cand"].shape[1]
    parts = [(lp[:, :C], mask[:, :C]), (lp[:, C:2 * C], mask[:, C:2 * C]),
             (lp[:, 2 * C:3 * C], mask[:, 2 * C:3 * C]), (lp[:, 3 * C:], mask[:, 3 * C:])]
    # cada tipo de componente (operar, puntuación, stop, liquidez) pesa lo mismo: si no, las
    # puntuaciones (con desviación típica 0,1) tapan a operar/mantener
    loss = -sum(x.sum() / m.sum().clamp_min(1.0) for x, m in parts) / len(parts)
    with torch.no_grad():
        h = net.heads(**ob)
        tl = torch.gather(h["trade_logit"], 1, rc["cand"])
        cm = rc["cmask"]
        acc = ((tl > 0).float() == rc["trade"])[cm].float().mean()
    return loss, {"trade_acc": float(acc)}


def train_bc(net: AgentNet, data: list, epochs: int = 30, lr: float = 1e-3, batch: int = 64,
             std: float = 0.1, seed: int = 0, log=print) -> list[dict]:
    """Preentrena la política (no el crítico) por máxima verosimilitud de las acciones de data."""
    import math
    torch.manual_seed(seed)
    with torch.no_grad():
        net.log_std.fill_(math.log(std))
        if not getattr(net, "_bc_init", False):    # puntuación inicial κ·z (ver el docstring)
            net.a.fill_(KAPPA)
            net._bc_init = True
    net.log_std.requires_grad_(False)
    params = [p for n, p in net.named_parameters()
              if p.requires_grad and not n.startswith("critic")]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(data))
        losses, accs = [], []
        for s in range(0, len(data), batch):
            idx = perm[s:s + batch]
            loss, info = bc_loss(net, [data[j][0] for j in idx], [data[j][1] for j in idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            accs.append(info["trade_acc"])
        row = {"epoch": ep + 1, "nll": float(np.mean(losses)), "trade_acc": float(np.mean(accs)),
               "a": float(net.a)}
        hist.append(row)
        log(f"BC época {ep + 1}: nll {row['nll']:.3f} · acierto operar {row['trade_acc']:.4f} · "
            f"a {row['a']:.3f}")
    net.log_std.requires_grad_(True)
    return hist


def warm_critic(trainer, iters: int = 30, epochs: int = 4, lr: float = 1e-3, log=print
                ) -> list[float]:
    """Ajusta solo el crítico de trainer.net con recogidas de la política (ver el docstring).
    La política no cambia; los entornos y generadores de trainer avanzan."""
    net, cfg = trainer.net, trainer.cfg
    params = list(net.critic.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng([cfg.seed, 777])
    out = []
    for it in range(iters):
        data, _ = trainer.collect()
        trainer.steps -= cfg.n_envs * cfg.rollout        # no cuentan como pasos de PPO
        losses = []
        for _ in range(epochs):
            perm = rng.permutation(len(data))
            for s in range(0, len(data), cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                ob = obs_tensors([data[j][0] for j in idx])
                ret = torch.tensor([data[j][4] for j in idx], dtype=torch.float32)
                v = net.heads(**ob)["value"]
                loss = ((v - ret) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad)
                opt.step()
                losses.append(float(loss))
        out.append(float(np.mean(losses)))
        log(f"crítico {it + 1}/{iters}: L_v {out[-1]:.3f}")
    return out


__all__ = ["FRAC_EPS", "bc_loss", "collect_clone", "collect_rule", "rule_record", "train_bc",
           "warm_critic"]
