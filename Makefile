.PHONY: install lint format typecheck test check eval compare redteam real-redteam ab reports demo serve docker clean-local

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

eval:
	sdr-crew eval --prompt-version v1 --trace traces/eval_v1.jsonl
	sdr-crew eval --prompt-version v2 --trace traces/eval_v2.jsonl

compare: eval
	sdr-crew compare reports/eval_v1.json reports/eval_v2.json

redteam:
	sdr-crew redteam

# Real public attack datasets + real B2B pages (network on first run; cached in data/real/).
real-redteam:
	$(PY) -m pip install -q -e ".[dev,real]"
	$(PY) scripts/eval_real_injection.py

ab:
	sdr-crew ab simulate

reports: compare redteam ab

demo:
	sdr-crew demo

serve:
	sdr-crew serve

docker:
	docker build -t ai-sdr-crew .

clean-local:
	rm -rf data traces exports
