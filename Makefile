.PHONY: setup format lint test eval clean

VENV = .venv
ACTIVATE = . $(VENV)/bin/activate

setup:
	python -m venv $(VENV)
	$(ACTIVATE) && pip install -r requirements.txt -r requirements-ci.txt

format:
	$(ACTIVATE) && black .

lint:
	$(ACTIVATE) && ruff check src scripts tests

test:
	$(ACTIVATE) && pytest -q

eval:
	$(ACTIVATE) && bash scripts/eval_all.sh

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache .coverage .mypy_cache
