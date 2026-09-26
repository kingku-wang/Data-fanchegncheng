# AdaFracTCN

Code and reproducibility materials for **Adaptive Fractional-Order Temporal Convolutional Network with Learnable Memory for Multi-Horizon Volatility Forecasting**.

AdaFracTCN combines a learnable short convolution with truncated Grünwald–Letnikov memory and an input-conditional dual-order mechanism for multi-horizon volatility forecasting. The repository is organized around the manuscript protocol: S&P 500 daily data, input length `L=256`, horizons `1/5/10/20`, matched-capacity neural baselines, multi-seed evaluation, robustness checks, and manuscript-facing diagnostics.

中文说明见 [README_zh-CN.md](README_zh-CN.md).

## Repository layout

```text
AdaFracTCN/
├── experiments/                  # models, data pipeline, experiment runners, audits
│   ├── adafractcn.py             # AdaFracTCN and ablation variants
│   ├── baselines.py              # statistical and neural baselines
│   ├── param_budget.py           # shared architecture/protocol constants
│   ├── download_data.py          # data acquisition and preprocessing
│   ├── main_experiment.py        # main multi-horizon benchmark
│   ├── ablation_experiment.py    # matched ablation study
│   ├── hyperparam_sensitivity.py # M / depth / K / fixed-alpha sensitivity
│   ├── regime_analysis.py        # volatility-regime diagnostics
│   ├── residual_analysis.py      # residual-memory diagnostics
│   ├── robustness_experiment.py  # cross-asset / expanding-window / GK checks
│   ├── additional_analyses.py    # rolling ARIMA, tuning, kernel diagnostics, HAC, bootstrap
│   ├── audit_semantics.py        # implementation/protocol invariants
│   ├── audit_results.py          # completed-run result consistency audit
│   ├── plot_paper_figures.py     # manuscript-facing figures from completed outputs
│   ├── run_all.py                # fixed-order reproduction orchestrator
│   ├── data/                     # cached S&P 500 input + generated preprocessing artifacts
│   └── results/                  # generated outputs + compact reference artifacts
├── docs/                         # reproducibility and manuscript-to-code mapping
├── tests/                        # lightweight structural smoke tests
├── .github/workflows/            # GitHub Actions smoke CI
├── requirements.txt              # full experiment dependencies
├── requirements-ci.txt           # lightweight CI dependencies
├── Makefile                      # common commands
├── CITATION.cff
└── LICENSE
```

## Quick start

Recommended: Python **3.11** on Linux/macOS or a recent 64-bit Windows environment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository includes a cached S&P 500 daily-price file for deterministic offline preprocessing. Rebuild the manuscript windows with:

```bash
cd experiments
python download_data.py --mode standard --offline
```

Inspect the complete reproduction pipeline without starting training:

```bash
python run_all.py --dry-run
```

Run lightweight validation:

```bash
cd ..
python -m unittest discover -s tests -v
cd experiments
python audit_semantics.py
python robustness_experiment.py --mode standard --check-only --prepare-gk
```

## Full reproduction

The standard pipeline is computationally expensive because the paper uses multiple horizons, multiple model families, and ten random seeds for the main trainable-model comparison.

```bash
cd experiments
python run_all.py
```

Stages can be selected explicitly:

```bash
python run_all.py --steps data main ablation hyperparam
python run_all.py --steps regime residual robustness additional
python run_all.py --steps semantic-audit results-audit figures
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the protocol, expected artifacts, and validation levels.

## Manuscript-to-code map

A section-by-section mapping is provided in [docs/MANUSCRIPT_CODE_MAP.md](docs/MANUSCRIPT_CODE_MAP.md). In short:

- model definition: `adafractcn.py`
- parameter budget and receptive-field bookkeeping: `param_budget.py`
- baselines: `baselines.py`
- data and split construction: `download_data.py`
- main benchmark and DM inference: `main_experiment.py`
- ablations: `ablation_experiment.py`
- hyperparameter sensitivity: `hyperparam_sensitivity.py`
- regime and residual diagnostics: `regime_analysis.py`, `residual_analysis.py`
- robustness: `robustness_experiment.py`
- additional diagnostics: `additional_analyses.py`

## Data

The main study uses daily S&P 500 data. The cached `experiments/data/sp500.csv` is included to make offline preprocessing deterministic. The robustness runner can additionally use `^IXIC`, `^DJI`, `AAPL`, `JPM`, and `XOM`; those extra asset caches are intentionally not bundled and can be prepared separately.

The data source is Yahoo Finance via `yfinance`. Users are responsible for complying with the data provider's terms when downloading or redistributing market data.

## Reference results

`experiments/results/reference/` contains compact tables retained from the manuscript workflow, such as seed-ensemble summaries, rolling-ARIMA checks, fitted-kernel diagnostics, and selected tuning configurations. Generated run outputs are intentionally ignored by Git so that running experiments does not dirty the repository.

See [experiments/results/README.md](experiments/results/README.md).

## Reproducibility design

The repository intentionally separates three levels of validation:

1. **Structural smoke tests** — imports, forward shapes, receptive fields, and parameter-budget contracts.
2. **Semantic audit** — protocol invariants, chronology, transformations, DM/HAC conventions, and model semantics.
3. **Completed-run audit** — recomputes reported summaries from saved forecasts and completed standard-run artifacts.

This separation lets contributors verify code structure without accidentally launching long training jobs.

## Citation

If you use this repository, please cite the accompanying manuscript. Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff). Update the DOI/URL fields after publication or archival release.

## License

Code is released under the [MIT License](LICENSE). The cached financial data remain subject to the original data provider's terms.
