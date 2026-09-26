# Contributing

This is a research repository tied to a specific manuscript protocol. Contributions are welcome, but changes that alter model definitions, data splits, metrics, or evaluation semantics should be explicit and testable.

## Before opening a pull request

```bash
python -m unittest discover -s tests -v
cd experiments
python run_all.py --dry-run
python download_data.py --mode standard --offline
python audit_semantics.py
```

For changes to training or reported statistics, also run the affected experiment stages and document whether reference results change.

## Style of changes

- Keep architecture/protocol constants centralized in `param_budget.py` where possible.
- Do not hard-code manuscript result values into training or analysis code.
- Do not use test-period information for preprocessing, tuning, early stopping, or positivity-floor construction.
- Keep generated data, checkpoints, raw predictions, and figures out of Git unless they are intentionally promoted to a compact reference artifact.
- Update `docs/MANUSCRIPT_CODE_MAP.md` if an entry point moves.
