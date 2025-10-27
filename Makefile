# -----------------------------------------------------------------------------
# Project Makefile
# -----------------------------------------------------------------------------
# Usage examples:
#   make setup          # create venv + install deps
#   make lint           # ruff lint
#   make format         # black format
#   make test           # run pytest
#   make eval           # run scripts/eval_all.sh
#   make clean          # remove caches
#   make clean-all      # remove caches + outputs + venv
# -----------------------------------------------------------------------------

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# ---- Config -----------------------------------------------------------------
PYTHON ?= python3
VENV ?= .venv
VENVPY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
ACTIVATE := source $(VENV)/bin/activate

REQUIREMENTS := requirements.txt
CI_REQUIREMENTS := requirements-ci.txt

# Export project root to Python so src/ is importable in tests/ and scripts/
export PYTHONPATH := $(CURDIR)

# ---- Phony targets ----------------------------------------------------------
.PHONY: help setup install format lint check test eval precommit clean clean-outputs clean-venv clean-all

# ---- Help -------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  setup         Create venv (if missing) and install all dependencies"
	@echo "  install       Install requirements into existing venv"
	@echo "  format        Run black"
	@echo "  lint          Run ruff"
	@echo "  check         Run ruff (lint) + black --check"
	@echo "  test          Run pytest"
	@echo "  eval          Run scripts/eval_all.sh (uses .venv python)"
	@echo "  precommit     Run pre-commit on all files"
	@echo "  clean         Remove caches/artifacts"
	@echo "  clean-outputs Remove outputs/results/logs"
	@echo "  clean-venv    Remove virtual environment"
	@echo "  clean-all     Clean everything (outputs + venv)"

# ---- Setup / Install --------------------------------------------------------
$(VENV):
	$(PYTHON) -m venv $(VENV)
	@echo "Created virtualenv at $(VENV)"

setup: $(VENV)
	$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS)" ]; then \
	  $(PIP) install -r $(REQUIREMENTS); \
	fi
	@if [ -f "$(CI_REQUIREMENTS)" ]; then \
	  $(PIP) install -r $(CI_REQUIREMENTS); \
	fi
	@echo "Environment ready."

install:
	@test -x "$(VENVPY)" || (echo "Missing venv. Run 'make setup' first." && exit 1)
	$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS)" ]; then \
	  $(PIP) install -r $(REQUIREMENTS); \
	fi
	@if [ -f "$(CI_REQUIREMENTS)" ]; then \
	  $(PIP) install -r $(CI_REQUIREMENTS); \
	fi

# ---- Quality ----------------------------------------------------------------
format:
	@test -x "$(VENVPY)" || (echo "Missing venv. Run 'make setup' first." && exit 1)
	$(VENV)/bin/black .

lint:
	@test -x "$(VENVPY)" || (echo "Missing venv. Run 'make setup' first." && exit 1)
	$(VENV)/bin/ruff check src scripts tests

check:
	@test -x "$(VENVPY)" || (echo "Missing venv. Run 'make setup' first." && exit 1)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .

# ---- Tests ------------------------------------------------------------------
test:
	@test -x "$(VENVPY)" || (echo "Missing venv. Run 'make setup' first." && exit 1)
	$(VENV)/bin/pytest -q

# ---- Evaluation -------------------------------------------------------------
eval:
	@test -x "$(VENVPY)" || (echo "Missing venv. Run 'make setup' first." && exit 1)
	chmod +x scripts/eval_all.sh
	$(ACTIVATE) && bash scripts/eval_all.sh

# ---- Pre-commit -------------------------------------------------------------
precommit:
	@test -x "$(VENVPY)" || (echo "Missing venv. Run 'make setup' first." && exit 1)
	@if ! command -v $(VENV)/bin/pre-commit >/dev/null 2>&1; then \
	  echo "Installing pre-commit..."; \
	  $(PIP) install pre-commit; \
	fi
	$(VENV)/bin/pre-commit run --all-files

# ---- Clean ------------------------------------------------------------------
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache .coverage .mypy_cache .benchmarks

clean-outputs:
	rm -rf outputs results logs checkpoints

clean-venv:
	rm -rf $(VENV)

clean-all: clean clean-outputs clean-venv
	@echo "Cleaned everything."
