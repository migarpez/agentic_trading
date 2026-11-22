"""Política del agente (handoff §10).

    1. φ: cada activo con predicción → vector (red compartida, sin identidad).
    2. Resumen del universo: media de φ sobre todos los activos del día.
    3. ψ([φ_i, resumen]) → (puntuación, stop, mantener/operar). Puntuación = a·z(μ_5) + ψ_0, con
       a aprendible (empieza en 1) y la última capa de ψ casi a 0: el primer top K es el de μ.
    4. Candidatos: top K por puntuación media ∪ posiciones en cartera; en entrenamiento, con
       probabilidad ε entran 1 o 2 activos al azar de fuera del top K. La selección es parte del
       entorno: el ratio de PPO se calcula sobre el conjunto guardado.
    5. Liquidez: red pequeña sobre [resumen, estado de la cartera] → puntuación.
    6. Reparto: las posiciones que se mantienen conservan su peso; softmax(candidatos que se
       operan + liquidez) reparte el resto.
    Crítico (solo para entrenar): [resumen, media de φ ponderada por peso de las posiciones,
    estado de la cartera] → valor.

Distribuciones de muestreo (en real, la salida directa):
- Pesos: logística-normal, como ruido gaussiano sobre las puntuaciones de los candidatos que se
  operan y la de la liquidez (desviación típica aprendida, común), y softmax después. La acción
  que se guarda son las puntuaciones con ruido, así que su log-probabilidad es una suma de
  normales, sin jacobiano, y vale para un número de candidatos que cambia cada día. Frente a
  Dirichlet: no necesita una concentración aparte, no tiene problemas con pesos casi 0 (log 0) y
  en modo determinista es exactamente el softmax de la salida.
- Stop: beta para d / d_max (d ∈ (0, d_max]), con α, β ≥ 1 (unimodal), solo para los candidatos
  que acaban con posición (se operan o se mantienen en cartera).
- Mantener/operar: Bernoulli por candidato (operar = 1). En un candidato sin posición, operar es
  comprar; mantener, no hacer nada.
En modo determinista: operar si p > 0,5, puntuaciones sin ruido y d / d_max = α / (α + β).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Beta, Normal
from torch.nn import functional as F

from agentic_trading.agent.env import N_PORT, Obs
from agentic_trading.agent.simulator import Rules, max_stop_distance

N_FEAT = 11


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class AgentNet(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, hidden: int = 64, emb: int = 32,
                 log_std: float = math.log(0.5), trade_bias: float = 0.0):
        super().__init__()
        self.phi = mlp(n_feat, hidden, emb)
        self.psi = mlp(2 * emb, hidden, 4)
        self.liq = mlp(emb + N_PORT, hidden, 1)
        self.critic = mlp(2 * emb + N_PORT, hidden, 1)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.log_std = nn.Parameter(torch.tensor(float(log_std)))
        for head in (self.psi, self.liq):                 # salidas ≈ 0 al empezar
            nn.init.normal_(head[-1].weight, std=1e-3)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():                             # p(operar) inicial = σ(trade_bias)
            self.psi[-1].bias[3] = trade_bias

    def heads(self, x, valid, z, weight, port) -> dict:
        """x [B, N, F], valid [B, N] bool, z [B, N], weight [B, N], port [B, N_PORT]."""
        phi = self.phi(x)
        v = valid.float().unsqueeze(-1)
        summ = (phi * v).sum(1) / v.sum(1).clamp_min(1.0)
        out = self.psi(torch.cat([phi, summ[:, None].expand_as(phi)], -1))
        w = (weight * valid).unsqueeze(-1)
        hold = (phi * w).sum(1) / w.sum(1).clamp_min(1e-6)
        return {"score": self.a * z + out[..., 0],
                "alpha": 1 + F.softplus(out[..., 1]), "beta": 1 + F.softplus(out[..., 2]),
                "trade_logit": out[..., 3],
                "liq": self.liq(torch.cat([summ, port], -1))[..., 0],
                "value": self.critic(torch.cat([summ, hold, port], -1))[..., 0]}


@dataclass
class Record:
    """Lo que se guarda de un paso para recalcular la log-probabilidad (índices en obs)."""
    cand: np.ndarray        # [c] candidatos (posiciones en obs.elig)
    trade: np.ndarray       # [c] bool
    s: np.ndarray           # [c] puntuaciones con ruido (cuentan las de los que se operan)
    liq: float              # puntuación con ruido de la liquidez
    frac: np.ndarray        # [c] d / d_max
    rel: np.ndarray         # [c] bool: acaban con posición (cuenta su stop)


FRAC_EPS = 1e-4


def obs_tensors(obs_list: list[Obs]) -> dict:
    """Lote con relleno: [B, N_max, ...]."""
    B, n = len(obs_list), max(len(o.elig) for o in obs_list)
    x = np.zeros((B, n, N_FEAT), np.float32)
    z, wt = np.zeros((B, n), np.float32), np.zeros((B, n), np.float32)
    valid = np.zeros((B, n), bool)
    for i, o in enumerate(obs_list):
        m = len(o.elig)
        x[i, :m], z[i, :m], wt[i, :m], valid[i, :m] = o.x, o.z_mu5, o.weight, True
    port = np.stack([o.port for o in obs_list])
    return {"x": torch.from_numpy(x), "valid": torch.from_numpy(valid), "z": torch.from_numpy(z),
            "weight": torch.from_numpy(wt), "port": torch.from_numpy(port)}


def record_tensors(recs: list[Record]) -> dict:
    B, c = len(recs), max(len(r.cand) for r in recs)
    out = {k: np.zeros((B, c), dt) for k, dt in (("cand", np.int64), ("trade", np.float32),
                                                  ("s", np.float32), ("frac", np.float32),
                                                  ("rel", bool), ("cmask", bool))}
    for i, r in enumerate(recs):
        m = len(r.cand)
        out["cand"][i, :m], out["trade"][i, :m], out["s"][i, :m] = r.cand, r.trade, r.s
        out["frac"][i, :m], out["rel"][i, :m], out["cmask"][i, :m] = r.frac, r.rel, True
        out["frac"][i, m:] = 0.5
    t = {k: torch.from_numpy(v) for k, v in out.items()}
    t["liq"] = torch.tensor([r.liq for r in recs], dtype=torch.float32)
    return t


def component_logprobs(net: AgentNet, ob: dict, rc: dict) -> tuple[torch.Tensor, ...]:
    """Log-probabilidad de cada componente de la acción, por paso: [B, 3·C + 1] con el orden
    (operar de cada candidato, puntuación de cada candidato, stop de cada candidato, liquidez),
    su máscara (los componentes que existen), la entropía y el valor."""
    h = net.heads(**ob)
    g = lambda k: torch.gather(h[k], 1, rc["cand"])
    cm = rc["cmask"].float()
    op = cm * rc["trade"]
    rel = cm * rc["rel"].float()
    std = net.log_std.exp()
    bern = Bernoulli(logits=g("trade_logit"))
    beta = Beta(g("alpha"), g("beta"))
    lp = torch.cat([bern.log_prob(rc["trade"]), Normal(g("score"), std).log_prob(rc["s"]),
                    beta.log_prob(rc["frac"].clamp(FRAC_EPS, 1 - FRAC_EPS)),
                    Normal(h["liq"], std).log_prob(rc["liq"])[:, None]], 1)
    mask = torch.cat([cm, op, rel, torch.ones_like(cm[:, :1])], 1)
    n_ent = 0.5 + 0.5 * math.log(2 * math.pi) + net.log_std
    ent = ((bern.entropy() * cm).sum(1) + n_ent * (op.sum(1) + 1)
           + (beta.entropy() * rel).sum(1))
    return lp * mask, mask, ent, h["value"]


def evaluate_actions(net: AgentNet, ob: dict, rc: dict) -> tuple[torch.Tensor, ...]:
    """Log-probabilidad conjunta de las acciones guardadas, entropía y valor, por paso."""
    lp, _, ent, v = component_logprobs(net, ob, rc)
    return lp.sum(1), ent, v


def act(net: AgentNet, obs: Obs, rules: Rules, rng: np.random.Generator, explore: bool = True,
        K: int = 20, eps: float = 0.05) -> tuple[dict, Record, float, float]:
    """Decisión de un día. Devuelve (orden para TradingEnv.step, Record, log-prob, valor)."""
    with torch.no_grad():
        ob = obs_tensors([obs])
        h = {k: v[0].numpy() for k, v in net.heads(**ob).items()}
        liq, value = float(h["liq"]), float(h["value"])
        std = float(net.log_std.exp())
    n = len(obs.elig)
    score = h["score"]
    top = np.argsort(-score, kind="stable")[:K]
    cand = np.union1d(top, np.flatnonzero(obs.held))
    if explore and rng.random() < eps:
        rest = np.setdiff1d(np.arange(n), cand)
        if len(rest):
            extra = rng.choice(rest, size=min(len(rest), int(rng.integers(1, 3))), replace=False)
            cand = np.union1d(cand, extra)
    al, be, tl = h["alpha"][cand], h["beta"][cand], h["trade_logit"][cand]
    p_trade = 1 / (1 + np.exp(-tl))
    if explore:
        trade = rng.random(len(cand)) < p_trade
        s = score[cand] + std * rng.normal(size=len(cand))
        liq_s = liq + std * rng.normal()
        frac = np.clip(rng.beta(al, be), FRAC_EPS, 1 - FRAC_EPS)
    else:
        trade = p_trade > 0.5
        s, liq_s, frac = score[cand].copy(), liq, al / (al + be)
    held_c = obs.held[cand]
    w_c = obs.weight[cand]
    room = max(0.0, 1.0 - float(w_c[held_c & ~trade].sum()) - obs.room_kept)
    logits = np.r_[s[trade], liq_s]
    e = np.exp(logits - logits.max())
    sm = e / e.sum() * room
    w_new = np.where(trade, 0.0, w_c)
    w_new[trade] = sm[:-1]
    rel = trade | held_c
    d = frac * max_stop_distance(w_new, rules)
    rec = Record(cand, trade, s.astype(np.float32), float(liq_s), frac.astype(np.float32), rel)
    with torch.no_grad():
        lp, _, _ = evaluate_actions(net, ob, record_tensors([rec]))
    order = {"je": obs.elig[cand[trade]], "w": sm[:-1], "cand": obs.elig[cand[rel]],
             "d": d[rel], "cash": float(sm[-1]), "n_trade": int(trade.sum())}
    return order, rec, float(lp[0]), value
