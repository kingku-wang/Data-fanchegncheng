# Public release checklist

Use this checklist before turning the repository into an archival release.

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python experiments/run_all.py --dry-run` and confirm every referenced script exists.
- [ ] Rebuild offline S&P 500 preprocessing with `download_data.py --mode standard --offline`.
- [ ] Run `audit_semantics.py` successfully.
- [ ] Prepare the additional cross-asset data required by the robustness suite, or document which assets are intentionally omitted.
- [ ] Run the full standard pipeline after the final implementation is frozen.
- [ ] Run `audit_results.py` on the completed standard run.
- [ ] Compare generated outputs with `experiments/results/reference/` and investigate any material mismatch.
- [ ] Regenerate manuscript-facing figures from completed outputs.
- [ ] Update the manuscript Data Availability / Code Availability statement with the final repository URL.
- [ ] Add the final GitHub repository URL to `CITATION.cff` after the repository is created.
- [ ] Add DOI metadata to `CITATION.cff` if the code is archived on Zenodo or another repository.
- [ ] Confirm that MIT is the intended software license.
- [ ] Confirm that redistribution of any cached market-data files is acceptable for the intended public release.
- [ ] Tag the validated archival release as `v1.0.0`.
