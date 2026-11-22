"""Construye el dataset de ventanas y targets en data/dataset/ a partir de data/clean/.

Salidas: arrays .npy por vela (precios en log, volumen, σ_ventana, mediana de volumen, targets),
samples.parquet (índice de muestras válidas) y meta.json (parámetros, símbolos, huella de los
datos limpios).
"""

import time

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR
from agentic_trading.data.dataset import (
    DEFAULT_DATASET,
    HORIZONS,
    WindowData,
    asset_info,
    build_dataset,
    split_mask,
)


def main() -> None:
    clean = DATA_DIR / "clean"
    out = DATA_DIR / "dataset"
    universe = pd.read_parquet(DATA_DIR / "universe.parquet")
    # Sin datos limpios (no descargados o cortados enteros; en el universo actual no hay):
    have = universe["symbol"].map(lambda s: (clean / f"{s}.parquet").exists())
    if not have.all():
        print(f"Sin datos limpios, fuera del dataset: {sorted(universe.loc[~have, 'symbol'])}")
        universe = universe[have].reset_index(drop=True)
    # Menos velas que el horizonte máximo (Yahoo da 1 vela de algunos del universo ampliado):
    # asset_arrays no las admite y no darían muestras. En el universo actual no hay.
    hmax = max(DEFAULT_DATASET.horizons)
    short = [s for s in universe["symbol"]
             if len(pd.read_parquet(clean / f"{s}.parquet", columns=["close"])) < hmax]
    if short:
        print(f"Menos de {hmax} velas, fuera del dataset: {short}")
        universe = universe[~universe["symbol"].isin(short)].reset_index(drop=True)
    meta = build_dataset(universe, lambda s: pd.read_parquet(clean / f"{s}.parquet"), out,
                         clean_dir=clean)

    smp = pd.read_parquet(out / "samples.parquet")
    group = asset_info(meta["symbols"], universe)["group"].to_numpy()  # por símbolo (meta.json)
    smp["group"] = group[smp["asset"].to_numpy()]

    size = sum(p.stat().st_size for p in out.glob("*")) / 1e6
    print(f"Velas: {meta['n_rows']:,} · muestras válidas: {meta['n_samples']:,} · {size:,.0f} MB")
    print(f"Valores exactamente 0: σ_ventana {meta['zero_counts']['sigma']}, "
          f"σ_YZ {meta['zero_counts']['v']}\n")

    rows = []
    for name in ("dev", "test"):
        m = split_mask(smp, name)
        s = smp[m]
        rows.append({"bloque": name, "muestras": m.sum(), "activos": s["asset"].nunique(),
                     "días": s["date"].nunique(), "desde": s["date"].min().date(),
                     "hasta": s["date"].max().date(),
                     **{f"σ excl. h={h}": f"{1 - s[f'mask_sigma_{h}'].mean():.2%}"
                        for h in HORIZONS}})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nMuestras por grupo y bloque:")
    smp["bloque"] = np.select([split_mask(smp, n) for n in ("dev", "test")], ["dev", "test"],
                              "hueco")
    print(smp.pivot_table(index="group", columns="bloque", values="row", aggfunc="size",
                          fill_value=0).to_string())

    # targets y velocidad de lectura (solo desarrollo, el test no se mira)
    data = WindowData(out)
    tr = data.split("dev")
    rng = np.random.default_rng(0)
    b = data.batch(rng.choice(tr, 50_000, replace=False))
    print("\nTargets en desarrollo (muestra de 50.000):")
    q = [0.01, 0.5, 0.99]
    for k in ("y", "v"):
        for j, h in enumerate(HORIZONS):
            print(f"  {k}_{h}: media {b[k][:, j].mean():+.3f} · cuantiles 1/50/99 % "
                  + " / ".join(f"{x:+.3f}" for x in np.quantile(b[k][:, j], q)))
    print(f"  |x| máx en ventanas: {np.abs(b['x']).max():.1f}")
    t0 = time.perf_counter()
    for _ in range(20):
        data.batch(rng.choice(tr, 1024, replace=False))
    print(f"\nLectura: {20 * 1024 / (time.perf_counter() - t0):,.0f} muestras/s (lotes de 1024)")


if __name__ == "__main__":
    main()
