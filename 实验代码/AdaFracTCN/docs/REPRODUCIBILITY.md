# Reproducibility guide

This repository separates quick structural validation from expensive end-to-end retraining. The goal is to let users verify the implementation before committing substantial compute.

## Protocol snapshot

The standard manuscript configuration uses:

- input length: `L = 256`
- forecast horizons: `1, 5, 10, 20` trading days
- main trainable-model seeds: `42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144`
- main dataset: S&P 500 daily returns / volatility targets
- chronological train/validation/test splits
- sample standard deviations (`ddof=1`) for across-seed dispersion
- positive-output projection using a floor derived from the training split only

Shared architecture and training constants are centralized in `experiments/param_budget.py` and imported by the model/baseline code to reduce configuration drift.

## Level 1 — structural smoke validation

No training is launched.

```bash
python -m unittest discover -s tests -v
cd experiments
python run_all.py --dry-run
```

The smoke suite checks core forward passes, input length 256 support, and the manuscript receptive-field bookkeeping.

## Level 2 — deterministic preprocessing and semantic audit

Rebuild the S&P 500 arrays from the bundled cache:

```bash
cd experiments
python download_data.py --mode standard --offline
python audit_semantics.py
```

For the manuscript configuration, preprocessing should create these window counts:

| Horizon | Train | Validation | Test |
|---:|---:|---:|---:|
| 1 | 4252 | 756 | 1004 |
| 5 | 4248 | 752 | 1000 |
| 10 | 4243 | 747 | 995 |
| 20 | 4233 | 737 | 985 |

The semantic audit checks chronology, scaling conventions, parameter counts, model semantics, receptive-field bookkeeping, DM/HAC behavior, and related invariants.

## Level 3 — completed-run audit

After the long-running standard experiments have produced raw forecasts and checkpoints:

```bash
cd experiments
python audit_results.py
```

This recomputes summary statistics from completed-run artifacts and writes:

- `results/results_audit.md`
- `results/results_summary.json`

It is designed to detect discrepancies between saved forecast artifacts and reported aggregate tables.

## Full standard pipeline

```bash
cd experiments
python run_all.py
```

The fixed order is:

1. data preprocessing
2. main multi-horizon benchmark
3. ablation study
4. hyperparameter sensitivity
5. regime analysis
6. residual-memory analysis
7. robustness suite
8. additional diagnostics
9. semantic audit
10. completed-result audit
11. figure generation

Use `--steps` to run a subset.

## Robustness suite

The unified runner implements the three robustness families used by the manuscript:

```bash
python robustness_experiment.py --mode standard --prepare-gk
```

It fixes `h=10`, `L=256`, and seed `42` for trainable models in the robustness protocol, and covers:

- cross-asset descriptive checks
- eight-block expanding-window S&P 500 evaluation
- an isolated Garman–Klass target evaluation

To validate data/configuration without training:

```bash
python robustness_experiment.py --mode standard --check-only --prepare-gk
```

The additional cross-asset caches are not bundled. Missing assets are reported explicitly.

## Reference artifacts versus generated artifacts

`experiments/results/reference/` contains compact manuscript-side reference tables retained for transparency. These are not used as hidden inputs to model training.

Generated training outputs are intentionally Git-ignored, including raw seed-level forecasts, checkpoints, preprocessing arrays, and figures.

Because model implementation details were aligned to the manuscript during repository preparation, a final end-to-end standard rerun is recommended before declaring an archival release numerically identical to the manuscript tables.

## Determinism

The code sets explicit experiment seeds, but exact floating-point equality can still depend on operating system, BLAS backend, PyTorch version, and hardware. For the closest reproduction, use one environment consistently and record package versions with:

```bash
python -m pip freeze > environment-freeze.txt
```

Do not commit local freeze files unless you intentionally want to archive that exact environment.
