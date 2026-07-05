.PHONY: setup lint format typecheck test check clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .

format:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/black .

typecheck:
	$(VENV)/bin/mypy

test:
	$(VENV)/bin/pytest

check: lint typecheck test

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
