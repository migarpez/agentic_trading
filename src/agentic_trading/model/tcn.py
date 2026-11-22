"""TCN causal pequeña y pérdida conjunta (handoff §6 y §7).

Entrada: ventana (B, T, 5) con o, h, l, c normalizados por σ_ventana y el volumen relativo (las
últimas T velas de la ventana de 252), más el escalar log σ_ventana. Salida: μ_5, μ_10, μ_20 y
log v_5, log v_10, log v_20.

Opcional, coarse (prueba de entrada en dos escalas): una segunda rama ve el año entero en velas
agregadas (data.dataset.COARSE). Con "month", un MLP sobre las 12 velas de 21 sesiones (60
números); con "week", una TCN como la diaria, con las capas justas para las 50 velas de 5
sesiones. Su salida (32 valores) se junta con el vector de la última vela diaria y log σ_ventana
antes del MLP de salida. Sin coarse, la red es la de siempre: mismos parámetros y, con la misma
semilla, la misma inicialización.

Opciones de mu_tree (22-11-2026; sin ellas, la red es la de siempre, bit a bit):
- market: la entrada diaria y la agregada llevan 4 canales más, la vela del mercado de la región
  (data.market): n_in = 9 en la rama diaria y 12 × 9 en la mensual.
- xsec: resumen aprendido del universo. El vector de cada activo antes del MLP de salida (el de
  la última vela, el de la rama agregada y log σ_ventana) se junta con la media de esos vectores
  entre los activos del mismo grupo del lote (misma fecha y región; forward(group=...)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from agentic_trading.data.dataset import COARSE

HORIZONS = (5, 10, 20)

# Escala fija de los canales de precio: x = log(P/C_t)/σ crece como √(distancia a hoy), del orden
# de ±16 a un año (√252). Dividir por 16 deja la entrada en O(1). Es una constante, equivalente a
# reescalar los pesos de la primera capa; solo ayuda a la optimización.
PRICE_SCALE = 1 / 16


def dilations_for(window: int, kernel: int = 3) -> tuple[int, ...]:
    """Dilataciones 1, 2, 4… justas para que el campo receptivo cubra la ventana
    (252 → 1–64, 7 capas, campo 255; 63 → 1–16, 5 capas, campo 63)."""
    d = [1]
    while 1 + (kernel - 1) * sum(d) < window:
        d.append(d[-1] * 2)
    return tuple(d)


class CausalConv(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalStack(nn.Module):
    """Convoluciones causales con conexiones residuales y dilataciones que se doblan; da el vector
    de la última posición."""

    def __init__(self, n_in: int = 5, channels: int = 32, kernel: int = 3,
                 dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = CausalConv(n_in, channels, kernel, dilations[0])
        self.skip = nn.Conv1d(n_in, channels, 1)
        self.layers = nn.ModuleList(CausalConv(channels, channels, kernel, d) for d in dilations[1:])
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def features_full(self, x):
        """Salida de la última capa en todas las posiciones (referencia, lenta)."""
        h = F.gelu(self.inp(x)) + self.skip(x)
        for conv in self.layers:
            h = h + F.gelu(conv(h))
        return h[:, :, -1]

    def features_last(self, x):
        """Lo mismo, calculando solo las posiciones de las que depende la última vela.

        Con dilataciones que se doblan, la capa de dilatación d solo necesita las posiciones
        T, T−d, T−2d... de la capa anterior: se submuestrea con paso 2 (alineado con el final) y
        la capa siguiente pasa a ser una convolución de dilatación d/paso acumulado. El resultado
        es idéntico y el coste ~3,5× menor."""
        h = F.gelu(self.inp(x)) + self.skip(x)
        stride = self.inp.conv.dilation[0]
        for conv in self.layers:
            d = conv.conv.dilation[0]
            step = d // stride
            if step > 1:
                h = h[:, :, (h.shape[-1] - 1) % step::step]
                stride = d
            pad = (conv.conv.kernel_size[0] - 1) * (d // stride)
            hp = F.pad(h, (pad, 0))
            h = h + F.gelu(F.conv1d(hp, conv.conv.weight, conv.conv.bias, dilation=d // stride))
        return h[:, :, -1]


class TCN(CausalStack):
    def __init__(self, n_in: int = 5, channels: int = 32, kernel: int = 3,
                 dilations=(1, 2, 4, 8, 16, 32, 64), hidden: int = 64, n_out: int = 6,
                 coarse: str | None = None, market: bool = False, xsec: bool = False):
        n_base = n_in
        if market:
            n_in = n_base + 4
        super().__init__(n_in, channels, kernel, dilations)
        self.coarse = coarse
        self.market, self.xsec = market, xsec
        n_coarse = 0
        if coarse is not None:
            if coarse not in COARSE:
                raise ValueError(f"coarse debe ser None o uno de {sorted(COARSE)}, no {coarse!r}")
            n_bars = COARSE[coarse][0]
            if coarse == "month":       # MLP sobre las 12 velas × 5 canales
                self.coarse_net = nn.Sequential(nn.Linear(n_bars * n_in, hidden), nn.GELU(),
                                                nn.Linear(hidden, channels))
            else:                       # TCN con las capas justas para las 50 velas
                self.coarse_net = CausalStack(n_in, channels, kernel,
                                              dilations_for(n_bars, kernel))
            n_coarse = channels
        n_feat = channels + n_coarse + 1
        self.head = nn.Sequential(nn.Linear(n_feat * (2 if xsec else 1), hidden), nn.GELU(),
                                  nn.Linear(hidden, n_out))
        scale = torch.ones(n_in)
        scale[:4] = PRICE_SCALE
        if market:
            scale[n_base:n_base + 4] = PRICE_SCALE      # precios del mercado
        self.register_buffer("scale", scale)

    def coarse_features(self, xc):
        """Salida de la rama de las velas agregadas: (B, n_velas, 5) → (B, 32)."""
        if xc is None:
            raise ValueError("esta red necesita las velas agregadas (xc)")
        xc = xc * self.scale
        if self.coarse == "month":
            return self.coarse_net(xc.flatten(1))
        return self.coarse_net.features_last(xc.transpose(1, 2))

    def forward(self, x, scalar, xc=None, group=None):
        """x: (B, T, 5) (9 con market); scalar: (B, 1); xc: (B, n_velas, 5 o 9), solo con coarse;
        group: (B,) enteros, solo con xsec (la media se toma dentro de cada grupo). Devuelve
        (μ (B, 3), log v (B, 3))."""
        if self.coarse is None and xc is not None:
            raise ValueError("esta red no usa velas agregadas (coarse = None)")
        x = (x * self.scale).transpose(1, 2)
        feats = [self.features_last(x)]
        if self.coarse is not None:
            feats.append(self.coarse_features(xc))
        h = torch.cat([*feats, scalar], dim=1)
        if getattr(self, "xsec", False):
            if group is None:
                raise ValueError("esta red necesita los grupos (xsec)")
            h = torch.cat([h, group_mean(h, group)], dim=1)
        out = self.head(h)
        return out[:, :3], out[:, 3:]


def group_mean(h, group):
    """Media de las filas de h (B, D) dentro de cada grupo, repetida en cada fila: (B, D)."""
    uniq, inv = torch.unique(group, return_inverse=True)
    sums = torch.zeros(len(uniq), h.shape[1], dtype=h.dtype).index_add(0, inv, h)
    cnt = torch.bincount(inv, minlength=len(uniq)).to(h.dtype)[:, None]
    return (sums / cnt)[inv]


def joint_loss(mu, logv, y, v, mask, lam: float = 1.0, horizons=HORIZONS, mask_mu=None,
               huber: float | None = None, weight_mu=None, horizon_w=None):
    """L = L_μ + λ·L_σ.

    L_μ = Σ_h MSE_h / h.  L_σ = media sobre h de la QLIKE sobre la varianza,
    v²/v̂² − log(v²/v̂²) − 1, solo en las muestras con mask (σ válida para ese horizonte).

    Opcionales (por defecto, lo de arriba):
    - mask_mu (B,): solo esas muestras cuentan en L_μ (media sobre ellas).
    - huber = δ: cada término de L_μ pasa a Huber con umbral d = δ·√h (unidades de σ_ventana):
      e² si |e| ≤ d (igual que el MSE) y 2·d·|e| − d² a partir de ahí.
    - weight_mu (B, n_h): peso de cada muestra y horizonte en L_μ (tail_weight); la suma se divide
      igual por el número de muestras (con pesos de media 1, la escala no cambia).
    - horizon_w: pesos de los horizontes en vez de 1/h (target_rank: iguales)."""
    if horizon_w is None:
        w = 1.0 / torch.tensor(horizons, dtype=mu.dtype, device=mu.device)
    else:
        w = torch.as_tensor(horizon_w, dtype=mu.dtype, device=mu.device)
    e = mu - y
    if huber is None:
        term = e ** 2
    else:
        d = huber * torch.sqrt(torch.tensor(horizons, dtype=mu.dtype, device=mu.device))
        a = e.abs()
        term = torch.where(a <= d, e ** 2, 2 * d * a - d ** 2)
    if weight_mu is not None:
        term = term * weight_mu
    if mask_mu is None:
        mse = term.mean(0)
    else:
        mm = mask_mu.to(term.dtype)[:, None]
        mse = (term * mm).sum(0) / mm.sum().clamp(min=1)
    l_mu = (mse * w).sum()
    log_ratio = 2 * (torch.log(torch.where(mask, v, torch.ones_like(v))) - logv)
    q = torch.where(mask, torch.exp(log_ratio) - log_ratio - 1, torch.zeros_like(v))
    l_sig = (q.sum(0) / mask.sum(0).clamp(min=1)).mean()
    return l_mu + lam * l_sig, l_mu.detach(), l_sig.detach()
