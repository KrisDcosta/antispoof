.PHONY: setup test smoke eda train summarize project neural-smoke neural-train aasist-smoke aasist-train ssl-ready ssl-cache-smoke ssl-smoke ssl-cache ssl-train phase4-fusion

PYTHON ?= ./venv/bin/python
CONFIG ?= configs/asvspoof2019_gmm.json
SMOKE_CONFIG ?= configs/asvspoof2019_smoke.json
NEURAL_CONFIG ?= configs/neural_lcnn.json
NEURAL_SMOKE_CONFIG ?= configs/neural_lcnn_smoke.json
AASIST_CONFIG ?= configs/neural_aasist_lite.json
AASIST_SMOKE_CONFIG ?= configs/neural_aasist_lite_smoke.json
SSL_CONFIG ?= configs/neural_ssl_wavlm_frozen.json
SSL_SMOKE_CONFIG ?= configs/neural_ssl_wavlm_frozen_smoke.json

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

aasist-smoke:
	$(PYTHON) scripts/train_neural.py --config $(AASIST_SMOKE_CONFIG)

aasist-train:
	$(PYTHON) scripts/train_neural.py --config $(AASIST_CONFIG)

ssl-ready:
	$(PYTHON) scripts/check_phase3_ssl_ready.py --config $(SSL_CONFIG)

ssl-cache-smoke:
	$(PYTHON) scripts/cache_ssl_embeddings.py --config $(SSL_SMOKE_CONFIG)

ssl-smoke:
	$(PYTHON) scripts/train_ssl_head.py --config $(SSL_SMOKE_CONFIG)

ssl-cache:
	$(PYTHON) scripts/cache_ssl_embeddings.py --config $(SSL_CONFIG)

ssl-train:
	$(PYTHON) scripts/train_ssl_head.py --config $(SSL_CONFIG)

phase4-fusion:
	$(PYTHON) scripts/run_score_fusion.py

eda:
	$(PYTHON) scripts/eda.py --data data/LA --output results/eda

train:
	$(PYTHON) scripts/run_project.py --config $(CONFIG) --skip-eda --skip-summary

summarize:
	$(PYTHON) scripts/summarize_results.py --results results/baseline --output results/baseline/summary

project:
	$(PYTHON) scripts/run_project.py --config $(CONFIG)
