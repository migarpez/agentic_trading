"""Entrenamiento, predicción y ensemble (handoff §8).

Cada miembro del ensemble tiene su semilla y un remuestreo por bloques de meses (meses con
reemplazo). En cada época se usa una submuestra aleatoria de fechas distinta (por el solapamiento
entre muestras consecutivas).

Opciones de prueba (TrainConfig; los valores por defecto reproducen lo de siempre):
- target="rel": el target de μ es y_h − ȳ_h(fecha), con ȳ_h(fecha) la media de y_h de las
  muestras de entrenamiento del bloque (train_idx) de esa fecha: las que no entrenan (hueco, test)
  no cuentan. Las fechas con menos de REL_MIN_SAMPLES muestras de entrenamiento no cuentan en L_μ
  (sí en L_σ). Al predecir, μ_h = μ_rel + deriva_h, la media de y_h en las muestras de
  entrenamiento del bloque (se guarda en el checkpoint).
- huber=δ: L_μ con Huber de umbral δ·√h en vez de MSE (ver joint_loss).
- window: la red recibe solo las últimas `window` velas, con las capas justas para cubrirlas.
- coarse="month" | "week": además, el año entero en velas agregadas (data.dataset.COARSE), con
  una segunda rama en la red (model.tcn). Con None (por defecto), todo queda como siempre.

Opciones de mu_tree (22-11-2026), desactivadas por defecto (sin ellas, bit a bit lo de antes):
- market: la vela del mercado de la región como entrada (data.market; model.tcn).
- target_rank: en L_μ, el target es su rango gaussianizado entre las muestras de entrenamiento de
  la fecha, Φ⁻¹((rango − 0,5)/n) (WindowData.date_ranks), con la misma regla de fechas (menos de
  REL_MIN_SAMPLES no cuentan) y pesos iguales por horizonte (con el rango, la varianza ya no crece
  con h). μ sale en unidades de rango (≈ N(0, 1) entre activos de un día) y sin deriva: no es
  un retorno esperado, solo sirve para ordenar los activos de cada día.
- xsec: resumen aprendido del universo (model.tcn). Se entrena por fechas: cada lote son todas
  las muestras de un día (de una misma extracción del remuestreo por meses, para no repetir
  muestras en un lote), con una submuestra del epoch_frac de esos días en cada época. Al
  predecir, igual: lotes de días enteros, y la media es la de las muestras predichas ese día.
- tail_weight = w: en L_μ, peso w para las muestras cuyo target está en el decil alto de su fecha
  (entre las de entrenamiento), normalizado para que el peso medio de cada fecha sea 1
  (WindowData.tail_weights).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from agentic_trading.data.dataset import COARSE, WINDOW, WindowData
from agentic_trading.model.tcn import HORIZONS, TCN, dilations_for, joint_loss

REL_MIN_SAMPLES = 20   # target relativo: fechas con menos muestras no cuentan en L_μ


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    epoch_frac: float = 0.2      # fracción de las muestras de entrenamiento por época
    batch: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lam: float = 1.0             # peso de L_σ frente a L_μ (pendiente de ajustar)
    bootstrap: bool = True       # remuestreo por meses
    threads: int = 8             # con 12 (núcleos P+E) va más lento
    target: str = "abs"          # "abs": y_h; "rel": y_h − media de y_h de la fecha
    huber: float | None = None   # None: MSE; δ: Huber con umbral δ·√h
    window: int = WINDOW         # velas que ve la red (las últimas de la ventana de 252)
    coarse: str | None = None    # None, "month" o "week": el año entero en velas agregadas
    market: bool = False         # vela del mercado de la región como entrada
    target_rank: bool = False    # L_μ sobre el rango gaussianizado del target en la fecha
    xsec: bool = False           # resumen aprendido del universo (lotes por fecha)
    tail_weight: float | None = None   # peso del decil alto de la fecha en L_μ


def month_bootstrap(samples: pd.DataFrame, idx: np.ndarray, rng: np.random.Generator,
                    return_draw: bool = False):
    """Remuestreo por bloques de meses: se sortean tantos meses como hay, con reemplazo, y se
    toman todas las muestras de cada mes sorteado. return_draw: además, el número de extracción
    de cada muestra (un mes sorteado dos veces da dos extracciones)."""
    months = samples["month"].to_numpy()[idx]
    order = np.argsort(months, kind="stable")
    uniq, start, count = np.unique(months[order], return_index=True, return_counts=True)
    pick = rng.integers(0, len(uniq), len(uniq))
    out = np.concatenate([idx[order[start[i]:start[i] + count[i]]] for i in pick])
    if not return_draw:
        return out
    return out, np.repeat(np.arange(len(pick)), count[pick])


def date_groups(samples: pd.DataFrame, idx: np.ndarray, draw: np.ndarray | None = None):
    """Grupos (xsec) de idx: las muestras de una misma fecha (y extracción). Devuelve el orden y
    el inicio y tamaño de cada grupo dentro de idx[orden]."""
    codes, _ = pd.factorize(samples["date"].to_numpy()[idx], sort=True)
    key = codes.astype(np.int64)
    if draw is not None:
        key = np.asarray(draw, np.int64) * (int(codes.max()) + 1) + key
    order = np.argsort(key, kind="stable")
    _, start, count = np.unique(key[order], return_index=True, return_counts=True)
    return order, start, count


def batch_groups(data, b_idx: np.ndarray) -> np.ndarray:
    """Grupo de cada muestra del lote para la media de xsec: fecha × región."""
    s = data.samples.iloc[b_idx]
    codes, _ = pd.factorize(s["date"].to_numpy())
    return codes.astype(np.int64) * 2 + data.regions[s["asset"].to_numpy()]


def _batch(data, b_idx, cfg_or_model):
    """data.batch con las opciones de entrada de una configuración o de un modelo."""
    g = cfg_or_model
    kw = {"window": getattr(g, "window", None), "coarse": getattr(g, "coarse", None)}
    if getattr(g, "market", False):
        kw["market"] = True
    return data.batch(b_idx, **kw)


def _tensors(b: dict):
    xc = torch.from_numpy(b["xc"]) if "xc" in b else None     # velas agregadas (coarse)
    return (torch.from_numpy(b["x"]), torch.from_numpy(b["scalar"]),
            torch.from_numpy(b["y"]), torch.from_numpy(b["v"]), torch.from_numpy(b["mask_sigma"]),
            xc)


def build_model(cfg: TrainConfig | dict) -> TCN:
    """TCN con las capas justas para la ventana de la configuración (252: la de siempre) y, con
    coarse, la rama de las velas agregadas (los checkpoints de antes no lo tienen: None)."""
    if not isinstance(cfg, dict):
        cfg = asdict(cfg)
    window, coarse = cfg.get("window", WINDOW), cfg.get("coarse")
    extra = {k: True for k in ("market", "xsec") if cfg.get(k)}
    model = TCN(dilations=dilations_for(window), coarse=coarse, **extra)
    model.window = window
    return model


def train_model(data: WindowData, train_idx: np.ndarray, cfg: TrainConfig, seed: int,
                log=print) -> tuple[TCN, list[dict]]:
    if cfg.target not in ("abs", "rel"):
        raise ValueError(f"target desconocido: {cfg.target}")
    if cfg.coarse is not None and cfg.coarse not in COARSE:
        raise ValueError(f"coarse desconocido: {cfg.coarse}")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    draw = None
    if cfg.bootstrap and cfg.xsec:
        idx, draw = month_bootstrap(data.samples, train_idx, rng, return_draw=True)
    else:
        idx = month_bootstrap(data.samples, train_idx, rng) if cfg.bootstrap else train_idx
    model = build_model(cfg)
    rel = cfg.target == "rel" and not cfg.target_rank
    if cfg.target_rank:
        y_rank, n_date = data.date_ranks(train_idx)
        horizon_w = np.ones(len(HORIZONS))
    else:
        horizon_w = None
    tail = data.tail_weights(train_idx, cfg.tail_weight) if cfg.tail_weight else None
    if rel:
        y_mean, n_date = data.date_means(train_idx)   # solo muestras de entrenamiento
        rows = data.samples["row"].to_numpy()[train_idx]
        model.drift = np.array([np.asarray(data.arr[f"y_{h}"])[rows].astype(np.float64).mean()
                                for h in HORIZONS])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.xsec:
        g_order, g_start, g_count = date_groups(data.samples, idx, draw)
        n_ep = max(1, round(len(g_start) * cfg.epoch_frac))    # días por época
        steps = n_ep
    else:
        n_ep = max(1, round(len(idx) * cfg.epoch_frac))
        steps = int(np.ceil(n_ep / cfg.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps)
    history = []
    for ep in range(cfg.epochs):
        t0, model_sums, n = time.perf_counter(), np.zeros(3), 0
        model.train()
        if cfg.xsec:
            gsel = rng.choice(len(g_start), size=n_ep, replace=False)
            batches = (idx[g_order[g_start[g]:g_start[g] + g_count[g]]] for g in gsel)
        else:
            ep_idx = rng.choice(idx, size=n_ep, replace=False)
            batches = (ep_idx[i:i + cfg.batch] for i in range(0, n_ep, cfg.batch))
        for b_idx in batches:
            x, s, y, v, m, xc = _tensors(_batch(data, b_idx, cfg))
            mask_mu = None
            if cfg.target_rank:
                y = torch.from_numpy(y_rank[b_idx])
                mask_mu = torch.from_numpy(n_date[b_idx] >= REL_MIN_SAMPLES)
            elif rel:
                y = y - torch.from_numpy(y_mean[b_idx])
                mask_mu = torch.from_numpy(n_date[b_idx] >= REL_MIN_SAMPLES)
            group = torch.from_numpy(batch_groups(data, b_idx)) if cfg.xsec else None
            mu, logv = (model(x, s, xc, group=group) if cfg.xsec else model(x, s, xc))
            kw = {}
            if tail is not None:
                kw["weight_mu"] = torch.from_numpy(tail[b_idx])
            if horizon_w is not None:
                kw["horizon_w"] = horizon_w
            loss, l_mu, l_sig = joint_loss(mu, logv, y, v, m, cfg.lam, mask_mu=mask_mu,
                                           huber=cfg.huber, **kw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model_sums += np.array([loss.item(), l_mu.item(), l_sig.item()]) * len(y)
            n += len(y)
        rec = dict(zip(["loss", "l_mu", "l_sigma"], model_sums / n), epoch=ep,
                   seconds=time.perf_counter() - t0)
        history.append(rec)
        log(f"  época {ep + 1}/{cfg.epochs} · L {rec['loss']:.4f} (μ {rec['l_mu']:.4f}, "
            f"σ {rec['l_sigma']:.4f}) · {rec['seconds']:.0f} s")
    return model, history


@torch.no_grad()
def predict(model: TCN, data: WindowData, idx: np.ndarray, batch: int = 8192) -> pd.DataFrame:
    """μ_h y log v_h de un modelo para las muestras idx (con target relativo, μ_h incluye la
    deriva del entrenamiento; con target_rank, unidades de rango y sin deriva). Con xsec, los
    lotes son de días enteros y la media de cada grupo es la de las muestras de idx."""
    model.eval()
    idx = np.asarray(idx)
    xsec = getattr(model, "xsec", False)
    if xsec:
        order, start, count = date_groups(data.samples, idx)
        chunks, cur, size = [], [], 0
        for a, c in zip(start, count):
            cur.append(order[a:a + c])
            size += c
            if size >= batch:
                chunks.append(np.concatenate(cur))
                cur, size = [], 0
        if cur:
            chunks.append(np.concatenate(cur))
    else:
        chunks = [np.arange(i, min(i + batch, len(idx))) for i in range(0, len(idx), batch)]
    mus, logvs = [], []
    for pos in chunks:
        b_idx = idx[pos]
        x, s, *_, xc = _tensors(_batch(data, b_idx, model))
        if xsec:
            mu, logv = model(x, s, xc, group=torch.from_numpy(batch_groups(data, b_idx)))
        else:
            mu, logv = model(x, s, xc)
        mus.append(mu.numpy())
        logvs.append(logv.numpy())
    mu, logv = np.concatenate(mus), np.concatenate(logvs)
    if xsec:
        inv = np.empty(len(idx), np.int64)
        inv[np.concatenate(chunks)] = np.arange(len(idx))
        mu, logv = mu[inv], logv[inv]
    drift = getattr(model, "drift", None)
    if drift is not None:
        mu = mu + np.asarray(drift, dtype=mu.dtype)[None, :]
    out = {"sample": idx}
    for j, h in enumerate(HORIZONS):
        out[f"mu_{h}"] = mu[:, j]
        out[f"logv_{h}"] = logv[:, j]
    return pd.DataFrame(out)


def ensemble(preds: list[pd.DataFrame]) -> pd.DataFrame:
    """Salidas para el agente (handoff §2), en unidades de σ_ventana:
    μ_h = media de los μ; σ_h = √h · √(media de v²); s_h = desviación típica de los μ."""
    out = pd.DataFrame({"sample": preds[0]["sample"].to_numpy()})
    for h in HORIZONS:
        mu = np.stack([p[f"mu_{h}"].to_numpy() for p in preds])
        v2 = np.stack([np.exp(2 * p[f"logv_{h}"].to_numpy()) for p in preds])
        out[f"mu_{h}"] = mu.mean(0)
        out[f"sigma_{h}"] = np.sqrt(h) * np.sqrt(v2.mean(0))
        out[f"s_{h}"] = mu.std(0, ddof=1) if len(preds) > 1 else 0.0
    return out


def save_model(model: TCN, path, cfg: TrainConfig, seed: int, history: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {"state": model.state_dict(), "config": asdict(cfg), "seed": seed, "history": history}
    if getattr(model, "drift", None) is not None:
        ck["drift"] = [float(d) for d in model.drift]
    torch.save(ck, path)


def load_model(path) -> tuple[TCN, dict]:
    """Modelo de un checkpoint de save_model, listo para predict (ventana y deriva incluidas)."""
    ck = torch.load(path, weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["state"])
    if ck.get("drift") is not None:
        model.drift = np.array(ck["drift"])
    return model, ck
