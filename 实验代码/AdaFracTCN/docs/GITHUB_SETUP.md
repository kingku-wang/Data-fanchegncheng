# GitHub setup

The repository directory is ready for a normal GitHub push.

## Suggested repository metadata

**Name:** `AdaFracTCN`

**Description:**

> Adaptive fractional-order temporal convolutional network for multi-horizon volatility forecasting — research code and reproducibility materials.

**Suggested topics:**

`time-series`, `volatility-forecasting`, `fractional-calculus`, `tcn`, `deep-learning`, `financial-time-series`, `pytorch`, `reproducible-research`

## First push

```bash
git init
git add .
git commit -m "Initial public repository"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/AdaFracTCN.git
git push -u origin main
```

After the repository exists, add its URL to `CITATION.cff` and to the manuscript Code/Data Availability statement.

## Recommended release sequence

1. Push the repository privately or as a draft public repository.
2. Complete the end-to-end standard rerun after the implementation is frozen.
3. Run `audit_results.py` and compare against `results/reference/`.
4. Update any reference artifacts that are intentionally superseded by the final run.
5. Tag the validated archival release as `v1.0.0`.
6. Optionally connect the GitHub repository to Zenodo and add the resulting DOI to `CITATION.cff`.
