.PHONY: install dev test lint format type clean reproduce smoke release-check

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev,notebook]"
	pre-commit install

test:
	$(PYTHON) -m pytest --cov=dsa_scout --cov-report=term --cov-fail-under=80

lint:
	$(PYTHON) -m ruff check dsa_scout/ tests/ scripts/
	$(PYTHON) -m ruff format --check dsa_scout/ tests/ scripts/

format:
	$(PYTHON) -m ruff format dsa_scout/ tests/ scripts/

type:
	$(PYTHON) -m mypy --strict dsa_scout/ scripts/

smoke:
	$(PYTHON) -m dsa_scout.cli smoke

reproduce:
	$(PYTHON) -m dsa_scout.cli reproduce

release-check:
	$(PYTHON) scripts/verify_release.py

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist']]; [p.unlink() for pattern in ['plots/*.png', 'plots/*.svg', 'results/*.json', 'results/*.pt'] for p in Path('.').glob(pattern)]"
