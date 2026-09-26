# Data directory

The repository commits only the cached S&P 500 input required for deterministic offline preprocessing:

| File | Purpose |
|---|---|
| `sp500.csv` | Cached S&P 500 daily OHLC data used by the manuscript pipeline |

All preprocessing parameters, descriptive-statistics files, and horizon-specific NumPy arrays are generated locally and ignored by Git.

## Rebuild the manuscript windows

From `experiments/`:

```bash
python download_data.py --mode standard --offline
```

For `L=256` and horizons `1/5/10/20`, the expected window counts are:

| Horizon | Train | Validation | Test |
|---:|---:|---:|---:|
| 1 | 4252 | 756 | 1004 |
| 5 | 4248 | 752 | 1000 |
| 10 | 4243 | 747 | 995 |
| 20 | 4233 | 737 | 985 |

The robustness suite may use additional assets (`^IXIC`, `^DJI`, `AAPL`, `JPM`, `XOM`). Those caches are intentionally not bundled in the public repository.
