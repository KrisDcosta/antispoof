.PHONY: setup test smoke eda train summarize project

PYTHON ?= ./venv/bin/python
CONFIG ?= configs/asvspoof2019_gmm.json
SMOKE_CONFIG ?= configs/asvspoof2019_smoke.json
NEURAL_CONFIG ?= configs/neural_lcnn.json
NEURAL_SMOKE_CONFIG ?= configs/neural_lcnn_smoke.json

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests

smoke:
	$(PYTHON) scripts/run_project.py --config $(SMOKE_CONFIG) --limit 100

neural-smoke:
	$(PYTHON) scripts/train_neural.py --config $(NEURAL_SMOKE_CONFIG)

neural-train:
	$(PYTHON) scripts/train_neural.py --config $(NEURAL_CONFIG)

eda:
	$(PYTHON) scripts/eda.py --data data/LA --output results/eda

train:
	$(PYTHON) scripts/run_project.py --config $(CONFIG) --skip-eda --skip-summary

summarize:
	$(PYTHON) scripts/summarize_results.py --results results/baseline --output results/baseline/summary

project:
	$(PYTHON) scripts/run_project.py --config $(CONFIG)
