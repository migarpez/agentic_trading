"""Rutas del proyecto."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AGENTIC_TRADING_DATA", ROOT / "data"))
