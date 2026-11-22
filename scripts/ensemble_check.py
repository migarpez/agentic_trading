"""Prueba de decisión para μ con un ensemble de 10 miembros en un bloque (por defecto cv4, 2016–19).

Lee data/predictions/<run>/<bloque>_members.parquet (de train_cv.py) y calcula:
- IC (media de h = 5 y 10, como en el ajuste) de cada miembro y del ensemble de los k primeros;
- comprobación: el ensemble de los miembros 0 y 1 (mismas semillas que el ajuste) debe reproducir
  el IC del ajuste;
- a h = 5 y 10, la diferencia diaria de IC entre el ensemble completo y la reversión de 1 mes, con su
  t de Newey-West (retardos = h);
- para k = 1, 2 y 10: dispersión de μ_5 entre activos frente a IC × la de y_5, y MSE de μ frente a
  μ = 0 (total, entre activos y media diaria).

Regla (fijada antes de ver los resultados): μ funciona si, a h = 5 o 10, la diferencia de IC con la
reversión de 1 mes es positiva con t > 2.

Uso: python scripts/ensemble_check.py [--run ens10] [--fold 4] [--expected-ic 0.0319]
"""

import argparse

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR
from agentic_trading.data.dataset import WindowData
from agentic_trading.model.metrics import baseline_features, daily_ic, nw_tstat
from agentic_trading.model.train import ensemble
from agentic_trading.model.tuning import fold_scores

H = (5, 10, 20)


def mse_parts(dates, mu, y) -> dict:
    """MSE de μ relativo a μ = 0 (en %): total, entre activos (quitando la media diaria) y de la
    media diaria."""
    g = pd.Series(mu).groupby(dates)
    mu_d = g.transform("mean").to_numpy()
    y_d = pd.Series(y).groupby(dates).transform("mean").to_numpy()
    rel = lambda e, e0: 100 * (np.mean(e ** 2) / np.mean(e0 ** 2) - 1)
    return {"total": rel(y - mu, y), "entre activos": rel((y - y_d) - (mu - mu_d), y - y_d),
            "media diaria": rel(y_d - mu_d, y_d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ens10")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--expected-ic", type=float, default=0.0319,
                    help="IC del ajuste para los miembros 0 y 1 (λ = 3, lr = 0,001, cv4)")
    ap.add_argument("--tol", type=float, default=0.0005)
    args = ap.parse_args()

    data = WindowData(DATA_DIR / "dataset")
    mem = pd.read_parquet(DATA_DIR / "predictions" / args.run / f"cv{args.fold}_members.parquet")
    preds = [g.drop(columns="member").reset_index(drop=True)
             for _, g in mem.groupby("member", sort=True)]
    n = len(preds)
    idx = preds[0]["sample"].to_numpy()
    assert all((p["sample"].to_numpy() == idx).all() for p in preds), "miembros desalineados"
    smp = data.samples.iloc[idx]
    rows, dates = smp["row"].to_numpy(), smp["date"].to_numpy()
    y = {h: np.asarray(data.arr[f"y_{h}"])[rows].astype(float) for h in H}

    # 1) IC por miembro y por ensemble de los k primeros
    member_ic = [fold_scores(data, ensemble([p]))["ic"] for p in preds]
    ens = {k: ensemble(preds[:k]) for k in (1, 2, 5, 10) if k <= n}
    ens_ic = {k: fold_scores(data, e)["ic"] for k, e in ens.items()}
    print(f"IC por miembro (media h=5,10): {np.round(member_ic, 4).tolist()}")
    print(f"  media {np.mean(member_ic):.4f} · sd {np.std(member_ic, ddof=1):.4f}")
    print("IC del ensemble de los k primeros:", {k: round(v, 4) for k, v in ens_ic.items()})

    # 2) comprobación con el ajuste
    ok = abs(ens_ic[2] - args.expected_ic) <= args.tol
    print(f"\nComprobación: ensemble 0+1 IC {ens_ic[2]:.4f} frente a {args.expected_ic:.4f} del "
          f"ajuste → {'OK' if ok else 'NO CUADRA'}")
    if not ok:
        raise SystemExit("El ensemble de los miembros 0 y 1 no reproduce el IC del ajuste: "
                         "revisar antes de seguir.")

    # 3) diferencia diaria de IC frente a la reversión de 1 mes
    full = ens[max(ens)]
    rev = baseline_features(data, rows)["reversal"]
    print(f"\nEnsemble de {max(ens)} frente a reversión de 1 mes:")
    verdict = False
    for h in (5, 10):
        ic_m = daily_ic(dates, full[f"mu_{h}"].to_numpy(), y[h])
        ic_r = daily_ic(dates, rev, y[h])
        diff = (ic_m - ic_r).dropna()
        mean, t = nw_tstat(diff.to_numpy(), h)
        verdict |= bool(mean > 0 and t > 2)
        print(f"  h={h}: IC modelo {ic_m.mean():+.4f} · reversión {ic_r.mean():+.4f} · "
              f"diferencia {mean:+.4f} (t NW {t:+.2f}, {len(diff)} días)")

    # 4) dispersión de μ_5 y MSE frente a μ = 0
    sd_y = pd.Series(y[5]).groupby(dates).std().mean()
    print("\nDispersión de μ_5 entre activos y MSE de μ frente a μ = 0 (%):")
    for k in (1, 2, 10):
        if k not in ens:
            continue
        e = ens[k]
        mu5 = e["mu_5"].to_numpy()
        ic5 = daily_ic(dates, mu5, y[5]).mean()
        sd_mu = pd.Series(mu5).groupby(dates).std().mean()
        print(f"  k={k}: sd μ_5 {sd_mu:.3f} · IC_5 × sd y_5 {ic5 * sd_y:.3f} "
              f"(ratio {sd_mu / (ic5 * sd_y):.1f}×)")
        for h in H:
            m = mse_parts(dates, e[f"mu_{h}"].to_numpy(), y[h])
            print(f"     h={h}: " + " · ".join(f"{a} {b:+.2f}" for a, b in m.items()))

    print(f"\nVeredicto (diferencia de IC con la reversión > 0 y t > 2 a h = 5 o 10): "
          f"{'μ FUNCIONA' if verdict else 'μ NO FUNCIONA'}")


if __name__ == "__main__":
    main()
