"""Informe de una ejecución de train_agent.py (prueba de humo, handoff §10): curvas por
actualización (recompensa, pérdida del crítico, entropía, rotación, posiciones, liquidez, w·d) en
logs/agent/<name>_curves.png, tabla por tramos y comportamiento de la política determinista
(logs/agent/<name>_eval.json, si existe), en logs/agent/<name>.md.

Uso: python scripts/agent_smoke_report.py --name smoke_g095_s0
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentic_trading.config import ROOT

COLS = [("reward_mean", "recompensa media por paso (log-ret.)"), ("loss_v", "pérdida del crítico"),
        ("entropy", "entropía (por paso)"), ("turnover_day", "rotación por día"),
        ("npos", "posiciones"), ("cash", "liquidez"), ("wd_p50", "w·d mediana"),
        ("n_trade", "operados por día")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    d = ROOT / "logs" / "agent"
    h = pd.read_json(d / f"{args.name}.jsonl", lines=True)
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    for ax, (c, label) in zip(axes.flat, COLS):
        ax.plot(h["steps"], h[c], lw=0.6, alpha=0.4)
        ax.plot(h["steps"], h[c].rolling(10, min_periods=1).mean(), lw=1.5)
        ax.set_title(label, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("pasos de entorno")
    fig.tight_layout()
    fig.savefig(d / f"{args.name}_curves.png", dpi=110)

    sps = (h["steps"].diff().fillna(h["steps"].iloc[0]) / (h["t_collect"] + h["t_update"]))
    n = len(h)
    parts = np.array_split(np.arange(n), 5)
    L = [f"# Agente: `{args.name}`", "",
         (f"{n} actualizaciones, {int(h['steps'].iloc[-1]):,} pasos de entorno; "
         f"{sps.mean():.0f} pasos/s de media (recogida {h['t_collect'].sum() / 60:.1f} min, "
         f"actualización {h['t_update'].sum() / 60:.1f} min)."), "",
         ("| tramo (actualizaciones) | recompensa/paso | rotación/día | posiciones | liquidez | "
         "w·d p10 / p50 / p90 | stops/día | Lv | entropía | std | a |"),
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in parts:
        s = h.iloc[p]
        L.append(f"| {p[0] + 1}–{p[-1] + 1} | {s.reward_mean.mean():+.5f} | "
                 f"{s.turnover_day.mean():.3f} | {s.npos.mean():.1f} | {s.cash.mean():.2f} | "
                 f"{s.wd_p10.mean():.4f} / {s.wd_p50.mean():.4f} / {s.wd_p90.mean():.4f} | "
                 f"{s.stops_day.mean():.2f} | {s.loss_v.mean():.1f} | {s.entropy.mean():.1f} | "
                 f"{s['std'].mean():.3f} | {s.a.mean():.3f} |")
    ev = d / f"{args.name}_eval.json"
    if ev.exists():
        L += ["", "## Política determinista", "", "```", ev.read_text(), "```"]
    L += ["", f"Curvas: logs/agent/{args.name}_curves.png"]
    (d / f"{args.name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps({"filas": n}))


if __name__ == "__main__":
    main()
