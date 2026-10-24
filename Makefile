.PHONY: install dev test lint format demo eval serve docker clean

PY ?= python3

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

demo:
	actiongraph demo --db demo.db

eval:
	actiongraph eval --provider offline

serve:
	actiongraph serve --db demo.db

docker:
	docker build -t actiongraph .

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info demo.db actiongraph.db
