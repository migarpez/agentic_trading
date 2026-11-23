"""Modelo de costes por activo para el simulador (23-11-2026; logs/universe_ext3.md, §11.6).

Para cada activo, dos parámetros de Rules (agent/simulator.py):
- cost_asset: c en cada compra y en cada venta = la mitad de la horquilla estimada de su grupo
  (size_group del universo combinado) en 2012–19, con un mínimo del 0,1 %. Horquillas de
  Abdi–Ranaldo de logs/universe_ext.json (la mediana entre las acciones de cada mes, media de los
  meses); para la cripto, la misma cuenta con BTC y ETH (scripts/universe_ext.spread_by_month).
- buy_tax: impuesto de su país, solo en las compras (BUY_TAX). Por el país de la bolsa (columna
  country del universo), salvo Irlanda: las acciones de Londres con sede en Irlanda según Yahoo
  (hq_country, en caché en <data>/hq_country.json) pagan el 1 % irlandés, no el 0,5 % británico.

Tipos vigentes a 23-11-2026 (si hay duda, el más alto, como pidió el usuario):
- Reino Unido, 0,5 % (Stamp Duty Reserve Tax; gov.uk/tax-buy-shares). Se aplica a todas las de
  Londres, aunque no paga SDRT una empresa extranjera sin registro en el Reino Unido, ni una
  recién cotizada (exención de 3 años de 2025): el tipo más alto.
- Francia, 0,5 %. El tipo confirmado es el 0,4 % desde el 1-4-2025 (loi de finances pour 2025,
  n.º 2025-127, art. 235 ter ZD del CGI; BOFiP). En el PLF 2026 hubo enmiendas al 0,5–0,8 %
  (la del 0,8 %, rechazada en el Senado) y no pude confirmar el texto final: ante la duda, el
  0,5 %. Solo grava a las francesas de más de 1.000 M€ de capitalización; se aplica a todas.
- España, 0,2 % (Impuesto sobre las Transacciones Financieras, Ley 5/2020; sede de la Agencia
  Tributaria). Solo las de más de 1.000 M€; se aplica a todas.
- Italia, 0,2 % (imposta sulle transazioni finanziarie; la Ley de presupuestos de 2026, n.º
  199/2025, la duplicó desde el 0,1 % en mercados regulados, en vigor desde el 1-1-2026; resúmenes
  de EY, A&O Shearman, DWF y Bird & Bird). Exentas las de capitalización pequeña (lista de
  Consob); se aplica a todas.
- Suiza, 0,15 % (Umsatzabgabe sobre valores suizos; ESTV). Solo con un intermediario suizo, y
  se reparte entre las partes (0,075 % cada una) y se cobra también al vender; aquí, el 0,15 %
  entero solo en las compras, como pide la prueba.
- Irlanda, 1 % (stamp duty sobre acciones irlandesas). Desde el 1-1-2026 están exentas las de
  menos de 1.000 M€ (Finance Act 2025; resúmenes de Walkers, Maples y Clearstream); se aplica a
  todas. Las irlandesas que cotizan en EE. UU. (a través de DTC) no pagan: no se aplica en EE. UU.
- Alemania, Países Bajos y EE. UU.: sin impuesto a la compra (la tasa de la SEC sobre las ventas,
  ~0,003 %, se ignora). Cripto: sin impuesto.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_COST = 0.001
BUY_TAX = {"GB": 0.005, "FR": 0.005, "ES": 0.002, "IT": 0.002, "CH": 0.0015, "IE": 0.01}
IRELAND = "IE"


def hq_country(universe: pd.DataFrame, cache: Path, countries=("GB",)) -> pd.Series:
    """País de la sede según Yahoo (Ticker.info["country"]) de las acciones que cotizan en
    `countries`, en caché en `cache` (JSON símbolo → país o null)."""
    known = json.loads(cache.read_text()) if cache.exists() else {}
    syms = universe.loc[universe["country"].isin(countries) &
                        (universe["asset_class"] == "equity"), "symbol"]
    todo = [s for s in syms if s not in known]
    if todo:
        import yfinance as yf
        for s in todo:
            try:
                known[s] = yf.Ticker(s).info.get("country")
            except Exception:  # noqa: BLE001  (yfinance lanza de todo; sin datos: sin sede)
                known[s] = None
            cache.write_text(json.dumps(known, indent=1, ensure_ascii=False))
    return pd.Series({s: known.get(s) for s in syms}, dtype=object)


def tax_country(universe: pd.DataFrame, hq: pd.Series) -> pd.Series:
    """País del impuesto de cada símbolo: el de la bolsa, IE para las de Londres con sede en
    Irlanda, "cripto" para la cripto."""
    u = universe.set_index("symbol")
    c = u["country"].astype(object).where(u["asset_class"] == "equity", "cripto")
    irish = hq.reindex(c.index).eq("Ireland") & c.eq("GB")
    c[irish] = IRELAND
    return c


def asset_costs(symbols, groups: pd.Series, countries: pd.Series, half_spread: dict,
                scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(cost_asset, buy_tax) alineados con symbols. half_spread: grupo → c (ya con el mínimo).
    scale multiplica los dos (prueba de robustez)."""
    g = groups.reindex(list(symbols))
    if g.isna().any():
        raise ValueError(f"sin grupo: {list(g.index[g.isna()])[:5]}")
    cost = g.map(half_spread).to_numpy(float)
    tax = countries.reindex(list(symbols)).map(BUY_TAX).fillna(0.0).to_numpy(float)
    if np.isnan(cost).any():
        raise ValueError("grupo sin horquilla")
    return cost * scale, tax * scale
