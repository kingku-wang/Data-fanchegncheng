# Result artifacts

The repository distinguishes **reference artifacts** from **generated run artifacts**.

## Bundled reference artifacts

Stored under `experiments/results/reference/`:

- `reference_results.csv` — compact benchmark anchors retained from the manuscript workflow
- `seed_ensemble_summary.csv` — seed dispersion and ensemble summaries
- `rolling_arima.csv` — rolling-ARIMA robustness comparison
- `baseline_tuning_selected.json` — selected equal-budget validation configurations
- `tuned_test_summary.csv` — tuned test summaries
- `ablation_hac.csv` — time-series-aware ablation inference summary
- `residual_hurst_uncertainty.csv` — residual-memory uncertainty summary
- `effective_kernels.csv` — fitted effective-kernel profiles
- `effective_kernel_cancellation.csv` — seed/block/branch/stage cancellation diagnostics
- `effective_kernel_cancellation_summary.csv` — aggregate cancellation diagnostics
- `effective_kernel_tail_summary.csv` — fitted tail-mass summaries

These files are provided for transparency and comparison. They are not read by the training loop as labels or hidden configuration.

## Generated outputs

Long-running experiment outputs are written directly under `experiments/results/` and are ignored by Git. Depending on the selected stages, these include:

- main result tables
- raw seed-level predictions
- model checkpoints
- ablation per-seed predictions and confidence intervals
- regime summaries
- residual-memory summaries
- robustness outputs
- completed-run audit reports

Generated figures are written under `experiments/figures/` and are also ignored by Git.
