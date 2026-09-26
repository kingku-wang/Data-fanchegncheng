# Manuscript-to-code map

This document maps the current manuscript concepts to their implementation entry points.

| Manuscript component | Primary implementation |
|---|---|
| Data acquisition and preprocessing | `experiments/download_data.py` |
| Shared training protocol / parameter budgets | `experiments/param_budget.py` |
| Fractional causal convolution | `experiments/adafractcn.py` |
| Adaptive dual-order AdaFracTCN block | `experiments/adafractcn.py` |
| Statistical and neural baselines | `experiments/baselines.py` |
| Metric inverse-scaling and positivity projection | `experiments/exp_common.py` |
| Main multi-horizon benchmark | `experiments/main_experiment.py` |
| Diebold–Mariano tests and multiplicity inputs | `experiments/main_experiment.py` |
| Matched ablation study | `experiments/ablation_experiment.py` |
| `M`, depth, `K`, fixed-`alpha` sensitivity | `experiments/hyperparam_sensitivity.py` |
| Regime-stratified performance and adaptive orders | `experiments/regime_analysis.py` |
| Residual-memory diagnostics | `experiments/residual_analysis.py` |
| Cross-asset / expanding-window / Garman–Klass checks | `experiments/robustness_experiment.py` |
| Rolling ARIMA, equal-budget tuning, fitted kernels, HAC, bootstrap | `experiments/additional_analyses.py` |
| Receptive-field bookkeeping | `experiments/param_budget.py` |
| Protocol/implementation invariants | `experiments/audit_semantics.py` |
| Completed-run numerical consistency | `experiments/audit_results.py` |
| Manuscript-facing data figures | `experiments/plot_paper_figures.py` |
| Pipeline orchestration | `experiments/run_all.py` |

## Important conventions

### Receptive field

The repository uses the manuscript bookkeeping convention centralized in `param_budget.py`. Under `L=256`:

- TCN: structural RF `61`, usable RF `61`
- RF-matched TCN: structural RF `509`, usable RF `256`
- AdaFracTCN: structural RF `1006`, usable RF `256`

### Adaptive block order

The AdaFracTCN implementation follows the current manuscript computational order for the adaptive block, including the placement of normalization, activation, dropout, and the residual connection.

### Across-seed dispersion

Reported across-seed standard deviations use sample SD (`ddof=1`).

### Robustness

The unified robustness runner fixes `h=10` and keeps the three robustness families in one public entry point so that their configuration cannot silently diverge.
