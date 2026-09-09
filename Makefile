# -----------------------------------------------------------------------------
# Project Makefile
# -----------------------------------------------------------------------------
# Common workflows for local development, quality assurance, testing,
# evaluation, and repository maintenance.
#
# Usage:
#   make help           # show available targets
#   make setup          # create venv + install dependencies + pre-commit
#   make install        # install/update dependencies in existing venv
#   make format         # format Python code with Black
#   make lint           # lint Python code with Ruff
#   make check          # Ruff + Black formatting check
#   make test           # run pytest
#   make precommit      # run all pre-commit hooks
#   make verify         # pre-commit + tests
#   make eval           # run the evaluation workflow
#   make clean          # remove caches and temporary artifacts
#   make clean-outputs  # remove generated outputs while preserving results/
#   make clean-venv     # remove the virtual environment
#   make clean-all      # clean caches, generated outputs, and venv
# -----------------------------------------------------------------------------

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# ---- Configuration ----------------------------------------------------------

PYTHON ?= python3
VENV ?= .venv

VENVPY := $(VENV)/bin/python
PIP := $(VENVPY) -m pip

REQUIREMENTS := requirements.txt
CI_REQUIREMENTS := requirements-ci.txt

# Export the repository root so project modules are importable from tests
# and scripts without requiring an editable package installation.
export PYTHONPATH := $(CURDIR)

# ---- Phony targets ----------------------------------------------------------

.PHONY: \
	help \
	setup \
	install \
	format \
	lint \
	check \
	test \
	precommit \
	verify \
	eval \
	clean \
	clean-outputs \
	clean-venv \
	clean-all

# ---- Help -------------------------------------------------------------------

help:
	@echo "Available targets:"
	@echo ""
	@echo "  setup          Create .venv and install project/development dependencies"
	@echo "  install        Install or update dependencies in the existing .venv"
	@echo "  format         Format Python files with Black"
	@echo "  lint           Run Ruff lint checks"
	@echo "  check          Run Ruff and Black in non-destructive check mode"
	@echo "  test           Run the pytest test suite"
	@echo "  precommit      Run all configured pre-commit hooks"
	@echo "  verify         Run pre-commit checks followed by the test suite"
	@echo "  eval           Run scripts/eval_all.sh using the project environment"
	@echo "  clean          Remove Python/tool caches and temporary artifacts"
	@echo "  clean-outputs  Remove generated outputs/logs/checkpoints; preserve results/"
	@echo "  clean-venv     Remove the local virtual environment"
	@echo "  clean-all      Remove caches, generated outputs, and the virtual environment"

# ---- Environment ------------------------------------------------------------

$(VENV):
	$(PYTHON) -m venv $(VENV)
	@echo "Created virtual environment at $(VENV)."

setup: $(VENV)
	$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS)" ]; then \
		$(PIP) install -r "$(REQUIREMENTS)"; \
	fi
	@if [ -f "$(CI_REQUIREMENTS)" ]; then \
		$(PIP) install -r "$(CI_REQUIREMENTS)"; \
	fi
	@if [ ! -x "$(VENV)/bin/pre-commit" ]; then \
		$(PIP) install pre-commit; \
	fi
	$(VENV)/bin/pre-commit install
	@echo "Development environment ready."

install:
	@test -x "$(VENVPY)" || \
		(echo "Missing virtual environment. Run 'make setup' first." && exit 1)
	$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS)" ]; then \
		$(PIP) install -r "$(REQUIREMENTS)"; \
	fi
	@if [ -f "$(CI_REQUIREMENTS)" ]; then \
		$(PIP) install -r "$(CI_REQUIREMENTS)"; \
	fi

# ---- Code quality -----------------------------------------------------------

format:
	@test -x "$(VENVPY)" || \
		(echo "Missing virtual environment. Run 'make setup' first." && exit 1)
	$(VENV)/bin/black .

lint:
	@test -x "$(VENVPY)" || \
		(echo "Missing virtual environment. Run 'make setup' first." && exit 1)
	$(VENV)/bin/ruff check .

check:
	@test -x "$(VENVPY)" || \
		(echo "Missing virtual environment. Run 'make setup' first." && exit 1)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .

# ---- Tests ------------------------------------------------------------------

test:
	@test -x "$(VENVPY)" || \
		(echo "Missing virtual environment. Run 'make setup' first." && exit 1)
	$(VENV)/bin/pytest -q

# ---- Pre-commit / full verification -----------------------------------------

precommit:
	@test -x "$(VENVPY)" || \
		(echo "Missing virtual environment. Run 'make setup' first." && exit 1)
	@if [ ! -x "$(VENV)/bin/pre-commit" ]; then \
		echo "Installing pre-commit..."; \
		$(PIP) install pre-commit; \
	fi
	$(VENV)/bin/pre-commit run --all-files

verify: precommit test
	@echo "All repository checks and tests passed."

# ---- Evaluation -------------------------------------------------------------

eval:
	@test -x "$(VENVPY)" || \
		(echo "Missing virtual environment. Run 'make setup' first." && exit 1)
	@test -f "scripts/eval_all.sh" || \
		(echo "Missing scripts/eval_all.sh." && exit 1)
	PATH="$(CURDIR)/$(VENV)/bin:$$PATH" bash scripts/eval_all.sh

# ---- Cleanup ----------------------------------------------------------------

clean:
	rm -rf \
		.pytest_cache \
		.ruff_cache \
		.mypy_cache \
		.benchmarks \
		.coverage \
		htmlcov
	find . \
		-type d \
		-name "__pycache__" \
		-not -path "./$(VENV)/*" \
		-prune \
		-exec rm -rf {} +

clean-outputs:
	rm -rf outputs logs checkpoints

clean-venv:
	rm -rf "$(VENV)"

clean-all: clean clean-outputs clean-venv
	@echo "Cleaned caches, generated outputs, and the virtual environment."
