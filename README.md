# agentic_trading

Modelo de ML sobre velas diarias OHLCV (TCN causal + ensemble) que alimenta a agentes de trading autónomos.

## Instalación

Requiere Python 3.10–3.12. Entrenamiento en CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m ipykernel install --user --name agentic_trading
```

## Pipeline de datos

```bash
make data       # universe + download + audit + clean
make universe   # componentes de los índices -> data/universe.parquet
make download   # OHLCV diario de Yahoo -> data/raw/yahoo/ (salta lo ya descargado)
make audit      # incidencias -> data/audit/{flags,summary}.parquet y data/audit/plots/
make clean      # limpieza + re-auditoría -> data/clean/ (una serie por activo + acciones)
make dataset    # ventanas y targets -> data/dataset/ (arrays + índice de muestras)
make test
```

## Estructura

```
src/agentic_trading/
  data/    universo, descarga y auditoría de datos
  model/   TCN, pérdidas y ensemble
  agent/   simulador, entorno, política y PPO del agente
scripts/   puntos de entrada
notebooks/ exploración
tests/     pruebas
data/      datos en Parquet (fuera de git)
```
