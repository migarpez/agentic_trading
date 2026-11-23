"""Dataset combinado del paso 2 de la ampliación (22-11-2026): universo actual + ampliado en
data/comb/ (data/size.py), con el pipeline de siempre (scripts/build_dataset.py con
AGENTIC_TRADING_DATA=data/comb) y el escalar de tamaño (data/comb/dataset/size/). No toca
data/universe.parquet, data/dataset ni data/ext.

Uso: python scripts/build_comb.py   (o `make comb`, después de `make ext`)
"""

import os
import subprocess
import sys

from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.size import build_combined, build_size, fx_history

sys.path.insert(0, str(ROOT))
from scripts.universe_ext import group_of

BASE, EXT, COMB = DATA_DIR, DATA_DIR / "ext", DATA_DIR / "comb"


def main() -> None:
    import pandas as pd
    uni = build_combined(BASE, EXT, COMB, group_of(pd.read_parquet(EXT / "universe.parquet")))
    print(uni.groupby("size_group").size().to_string())
    fx = fx_history()
    fx.to_parquet(COMB / "fx.parquet")
    print("FX desde:", fx.apply(lambda s: s.first_valid_index()).to_dict())
    env = {**os.environ, "AGENTIC_TRADING_DATA": str(COMB)}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset.py")], cwd=ROOT,
                   env=env, check=True)
    pct = build_size(COMB, fx)
    print(f"Tamaño: {len(pct):,} muestras, media {pct.mean():.3f}")


if __name__ == "__main__":
    main()
