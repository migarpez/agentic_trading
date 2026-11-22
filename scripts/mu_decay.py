"""Diagnóstico sin entrenar (22-11-2026): ¿la caída de la señal después de 2012 es del mercado o
de nuestro modelo?

Con la medida de logs/mu_tree/deciles.md (exceso semanal de apertura a apertura sobre la media de
las acciones, en pb, en las fechas semanales del agente; scripts/mu_deciles.py), por bloque:

1. Dispersión entre acciones: desviación típica entre acciones del log-retorno semanal (de
   apertura a apertura), media de las semanas del bloque, en pb.
2. Piernas larga (D9–D10) y corta (D1–D2) de la mezcla de success_checks (cada fecha, media de
   los rangos de la reversión de 1 mes y del momentum de 12 meses entre las acciones de la fecha),
   y por separado de la reversión y del momentum.
3. Las mismas piernas de final20 (20 miembros, μ_5), y la diferencia entre piernas (larga −
   corta) dividida por la dispersión del bloque (también para las líneas base).
4. IC de selección de final20 y de la mezcla por bloque (media de h = 5 y 10, todas las muestras;
   como en mu_compare.py).

Lectura fijada antes de verlo: r = (diferencia entre piernas en 2012–19) / (en 2000–11), con la
media de las semanas de cada periodo.
- Si r de la mezcla está dentro de un factor 1,5 de r de final20 (r_mezcla / r_final20 entre
  1/1,5 y 1,5), la caída es del mercado (dispersión, régimen o sesgo), no del modelo.
- Si r de la mezcla es más de 1,5 veces el de final20, es el modelo, que no se adapta al periodo
  reciente.
- (Caso no previsto: r de la mezcla menor que r de final20 / 1,5; se informa tal cual.)

Solo lee datos y predicciones de 2000–2019. Escribe logs/mu_tree/decay.md y decay.json.

Uso: PYTHONPATH=. python scripts/mu_decay.py
"""

import json

import numpy as np
import pandas as pd

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.model.metrics import baseline_features, baseline_mix
from agentic_trading.model.mu_metrics import (
    BLOCK_NAMES,
    OpenPrices,
    StdContext,
    block_of,
    std_metrics,
    weekly_frame,
)
from scripts.mu_compare import BLOCK_LABEL, fmt, load_run
from scripts.mu_deciles import decile_series

OUT = ROOT / "logs" / "mu_tree"
EARLY, LATE = ("cv0", "cv1", "cv2"), ("cv3", "cv4")
FACTOR = 1.5


def main() -> None:
    data = WindowData(DATA_DIR / "dataset")
    pred = load_run("final20")
    samples = pred["sample"].to_numpy()
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    equity = (info["asset_class"] == "equity").to_numpy()
    smp = data.samples.iloc[samples]
    dates = smp["date"]
    wk = weekly_dates(dates[dates.dt.dayofweek < 5])
    n_eq = smp[equity[smp["asset"].to_numpy()]].groupby("date").size()
    wk = wk[wk.isin(n_eq[n_eq >= 20].index)]
    prices = OpenPrices.from_clean(DATA_DIR / "clean",
                                   [data.meta["symbols"][a] for a in np.flatnonzero(equity)])
    wf = weekly_frame(data, samples, equity, wk, prices)
    rows = data.samples["row"].to_numpy()[wf["sample"].to_numpy()]
    base = baseline_features(data, rows)
    d = wf["date"].to_numpy()
    signals = {
        "final20": pred["mu_5"].to_numpy()[np.searchsorted(samples, wf["sample"].to_numpy())],
        "mezcla": baseline_mix(d, base["reversal"], base["momentum"]),
        "reversión 1 m": base["reversal"],
        "momentum 12 m": base["momentum"],
    }
    disp = 1e4 * wf.groupby("date")["ret_oo"].std()
    disp_blk = block_of(disp.index)
    disp_b = {b: float(disp[disp_blk == b].mean()) for b in BLOCK_NAMES}

    res = {"dispersion": disp_b, "signals": {}}
    for name, x in signals.items():
        s = decile_series(wf, x)
        s["dif"] = s["larga"] - s["corta"]
        blk = block_of(s.index)
        per = {}
        for b in BLOCK_NAMES:
            q = s[blk == b]
            per[b] = {"larga": float(q["larga"].mean()), "corta": float(q["corta"].mean()),
                      "dif": float(q["dif"].mean()), "dif/disp": float(q["dif"].mean() / disp_b[b])}
        early = float(s.loc[np.isin(blk, EARLY), "dif"].mean())
        late = float(s.loc[np.isin(blk, LATE), "dif"].mean())
        res["signals"][name] = {"blocks": per, "dif_2000_11": early, "dif_2012_19": late,
                                "r": late / early}

    ctx = StdContext(data, samples)
    std = std_metrics(ctx, pred)
    res["IC"] = {b: {"final20": std[b]["IC"],
                     "mezcla": float(np.mean([ctx.ic_mix[b][h].mean() for h in (5, 10)]))}
                 for b in BLOCK_NAMES}
    r_mix, r_f = res["signals"]["mezcla"]["r"], res["signals"]["final20"]["r"]
    ratio = r_mix / r_f
    if 1 / FACTOR <= ratio <= FACTOR:
        verdict = "del mercado (dispersión, régimen o sesgo), no del modelo"
    elif ratio > FACTOR:
        verdict = "del modelo, que no se adapta al periodo reciente"
    else:
        verdict = ("caso no previsto: la mezcla cae más que final20 (r_mezcla < r_final20 / 1,5); "
                   "no apunta al modelo")
    res["verdict"] = {"r_mezcla": r_mix, "r_final20": r_f, "ratio": ratio, "lectura": verdict}
    (OUT / "decay.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    L = ["# Caída de la señal después de 2012: ¿mercado o modelo? (22-11-2026)", "",
         ("Ver el docstring de scripts/mu_decay.py. Exceso semanal de apertura a apertura sobre la "
          "media de las acciones, pb por semana; dif. = larga (D9–D10) − corta (D1–D2)."), "",
         (f"**r = dif. 2012–19 / dif. 2000–11: mezcla {r_mix:.2f}, final20 {r_f:.2f}; cociente "
          f"{ratio:.2f} → la caída es {verdict}.**"), "",
         "| señal | dif. 2000–11 | dif. 2012–19 | r |", "|---|---|---|---|"]
    for name, v in res["signals"].items():
        L.append(f"| {name} | {fmt(v['dif_2000_11'])} | {fmt(v['dif_2012_19'])} | "
                 f"{v['r']:.2f} |")
    L += ["", "## Por bloque", "",
          ("| bloque | dispersión (pb) | IC final20 / mezcla | "
           + " | ".join(f"{n}: larga / corta / dif. (dif./disp.)" for n in signals) + " |"),
          "|---|---|---|" + "---|" * len(signals)]
    for b in BLOCK_NAMES:
        cells = []
        for n in signals:
            p = res["signals"][n]["blocks"][b]
            cells.append(f"{fmt(p['larga'])} / {fmt(p['corta'])} / {fmt(p['dif'])} "
                         f"({p['dif/disp']:.3f})")
        ic = res["IC"][b]
        L.append(f"| {BLOCK_LABEL[b]} | {disp_b[b]:.0f} | {ic['final20']:.4f} / "
                 f"{ic['mezcla']:.4f} | " + " | ".join(cells) + " |")
    (OUT / "decay.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
