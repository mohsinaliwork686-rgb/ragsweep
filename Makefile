VENV ?= .venv
PY   := $(VENV)/bin/python

.PHONY: help install test lint check demo clean

help:
	@echo "install  create the virtualenv and install ragsweep with dev extras"
	@echo "test     run the test suite"
	@echo "lint     run ruff"
	@echo "check    lint + test, what CI runs"
	@echo "demo     sweep the example handbook and write results.json"
	@echo "clean    remove build artefacts and caches"

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

check: lint test

demo:
	cd examples && ../$(PY) -m ragsweep run

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
