PY := .venv/bin/python

.PHONY: data universe download audit clean dataset test lint

data: download audit clean  ## pipeline de datos completo, con el universo congelado

universe:  ## componentes de los índices (Wikipedia) -> data/universe.parquet; solo a propósito
	$(PY) scripts/build_universe.py

download:  ## OHLCV diario -> data/raw/yahoo/
	$(PY) scripts/download_data.py

audit:  ## auditoría -> data/audit/
	$(PY) scripts/audit_data.py

clean:  ## limpieza + re-auditoría -> data/clean/
	$(PY) scripts/clean_data.py

dataset:  ## ventanas y targets -> data/dataset/
	$(PY) scripts/build_dataset.py

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src scripts tests
