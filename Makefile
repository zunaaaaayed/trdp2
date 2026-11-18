# Override with: make check PYTHON=/path/to/tooling-venv/bin/python
PYTHON ?= python

.PHONY: check format format-diff
check:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-diff:
	$(PYTHON) -m ruff format --diff .
