# Experiment entry points

All commands in this directory are designed to be run from `experiments/`.

## Standard pipeline

```bash
python run_all.py
```

The default order is:

1. `download_data.py` — offline S&P 500 preprocessing
2. `main_experiment.py` — main multi-horizon benchmark
3. `ablation_experiment.py` — matched ablation study
4. `hyperparam_sensitivity.py` — `M`, depth, `K`, and fixed-`alpha` sensitivity
5. `regime_analysis.py` — regime-stratified diagnostics from saved forecasts
6. `residual_analysis.py` — residual-memory diagnostics from saved forecasts
7. `robustness_experiment.py` — unified robustness suite
8. `additional_analyses.py` — rolling ARIMA, tuning, fitted kernels, HAC, bootstrap
9. `audit_semantics.py` — implementation/protocol invariants
10. `audit_results.py` — completed-run consistency audit
11. `plot_paper_figures.py` — manuscript-facing figures from completed outputs

Inspect or select stages:

```bash
python run_all.py --dry-run
python run_all.py --steps data main ablation
python run_all.py --steps regime residual robustness additional
```

## Additional analyses

```bash
python additional_analyses.py all
```

Available subcommands include:

```bash
python additional_analyses.py ensemble-summary
python additional_analyses.py rolling-arima
python additional_analyses.py baseline-tuning
python additional_analyses.py tuned-test
python additional_analyses.py effective-kernels --sample-per-regime 64
python additional_analyses.py ablation-timeseries
python additional_analyses.py residual-uncertainty --bootstrap-reps 1000 --block-len 5
```

## Robustness suite

```bash
python robustness_experiment.py --mode standard --prepare-gk
```

The unified robustness runner uses the manuscript settings `h=10`, `L=256`, and seed `42` for trainable models. It covers the six-asset descriptive check, eight S&P 500 expanding-window blocks, and the isolated S&P 500 Garman–Klass target check.

Validate the configuration without training:

```bash
python robustness_experiment.py --mode standard --check-only --prepare-gk
```

## Audits

```bash
python audit_semantics.py
python audit_results.py
```

`audit_semantics.py` can run after preprocessing. `audit_results.py` requires completed standard-run forecasts/checkpoints.

## Output policy

- Generated preprocessing arrays: `data/h*/`
- Generated figures: `figures/`
- Generated training/result artifacts: `results/`
- Compact manuscript-side reference tables: `results/reference/`

The first three groups are ignored by Git. Reference artifacts are committed for transparency and comparison.
