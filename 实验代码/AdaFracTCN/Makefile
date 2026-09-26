PYTHON ?= python

.PHONY: help install data dry-run smoke semantic-audit robustness-check reproduce figures clean

help:
	@echo "AdaFracTCN common commands"
	@echo "  make install           Install full dependencies"
	@echo "  make data              Rebuild offline S&P 500 preprocessing arrays"
	@echo "  make dry-run           Print the full reproduction pipeline"
	@echo "  make smoke             Run lightweight unit/smoke tests"
	@echo "  make semantic-audit    Run implementation/protocol audit"
	@echo "  make robustness-check  Prepare/check robustness inputs without training"
	@echo "  make reproduce         Run the full standard pipeline (expensive)"
	@echo "  make figures           Generate figures from completed outputs"
	@echo "  make clean             Remove generated caches/results/figures"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	cd experiments && $(PYTHON) download_data.py --mode standard --offline

dry-run:
	cd experiments && $(PYTHON) run_all.py --dry-run

smoke:
	$(PYTHON) -m unittest discover -s tests -v

semantic-audit:
	cd experiments && $(PYTHON) audit_semantics.py

robustness-check:
	cd experiments && $(PYTHON) robustness_experiment.py --mode standard --check-only --prepare-gk

reproduce:
	cd experiments && $(PYTHON) run_all.py

figures:
	cd experiments && $(PYTHON) plot_paper_figures.py

clean:
	rm -rf experiments/data/h* experiments/data/robustness_* experiments/figures
	rm -rf experiments/results/checkpoints experiments/results/raw_predictions experiments/results/ablation_raw_predictions
	rm -rf experiments/results/rolling_arima experiments/results/tuned_raw
	find experiments/results -maxdepth 1 -type f ! -name README.md -delete
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
