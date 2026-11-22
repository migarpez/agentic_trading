"""Gráficos para revisar incidencias de la auditoría."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_flag(df: pd.DataFrame, symbol: str, date, check: str, detail: str = "",
              window: int = 60, path: Path | None = None):
    """Cierre (log), rango H-L y volumen alrededor de `date`."""
    date = pd.Timestamp(date)
    i = df.index.searchsorted(date)
    w = df.iloc[max(0, i - window): i + window + 1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.vlines(w.index, w["low"], w["high"], color="0.6", lw=1)
    ax1.plot(w.index, w["close"], color="C0", lw=1.2, label="close")
    ax1.plot(w.index, w["adj_close"], color="C1", lw=0.8, ls="--", label="adj_close")
    ax1.set_yscale("log")
    ax1.axvline(date, color="C3", lw=0.8)
    for d in w.index[w["stock_splits"].fillna(0) != 0]:
        ax1.axvline(d, color="C2", lw=1, ls=":")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_title(f"{symbol} · {date.date()} · {check}\n{detail}", fontsize=10)
    ax2.bar(w.index, w["volume"], color="0.5", width=1)
    ax2.axvline(date, color="C3", lw=0.8)
    ax2.set_yscale("log")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=90)
        plt.close(fig)
    return fig
