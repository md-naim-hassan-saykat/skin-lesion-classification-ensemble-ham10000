.PHONY: setup format lint test eval

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-ci.txt

format:
	black .

lint:
	ruff check .

test:
	pytest -q

eval:
	bash scripts/eval_all.sh
