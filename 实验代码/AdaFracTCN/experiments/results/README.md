# Results directory

This directory has two roles:

1. `reference/` contains compact manuscript-side reference artifacts committed to Git.
2. The directory root is used for generated outputs from local experiment runs and is ignored by Git.

## Bundled reference artifacts

See [`../../docs/RESULT_ARTIFACTS.md`](../../docs/RESULT_ARTIFACTS.md) for descriptions of the files in `reference/`.

## Generated outputs

A full run can create raw seed-level predictions, checkpoints, aggregate tables, robustness outputs, audit reports, and temporary analysis directories. These are intentionally not committed because they are larger and are reproducible from the experiment pipeline.
