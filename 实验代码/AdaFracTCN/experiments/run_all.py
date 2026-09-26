# -*- coding: utf-8 -*-
"""Run the AdaFracTCN reproduction pipeline in a fixed order.

The orchestrator contains no manuscript result constants.  Long-running stages
can be selected individually with ``--steps``.  The default order reproduces
the paper's data products, additional analyses, audits, and manuscript figures.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STEPS = {
    "data": [PY, "download_data.py", "--mode", "standard", "--offline"],
    "main": [PY, "main_experiment.py", "--mode", "standard"],
    "ablation": [PY, "ablation_experiment.py", "--mode", "standard"],
    "hyperparam": [PY, "hyperparam_sensitivity.py", "--mode", "standard"],
    "regime": [PY, "regime_analysis.py", "--mode", "standard", "--source", "saved"],
    "residual": [PY, "residual_analysis.py", "--mode", "standard", "--source", "saved"],
    "robustness": [PY, "robustness_experiment.py", "--mode", "standard", "--prepare-gk"],
    "additional": [PY, "additional_analyses.py", "all"],
    "semantic-audit": [PY, "audit_semantics.py"],
    "results-audit": [PY, "audit_results.py"],
    "figures": [PY, "plot_paper_figures.py"],
}

DEFAULT_ORDER = [
    "data", "main", "ablation", "hyperparam", "regime", "residual",
    "robustness", "additional", "semantic-audit",
    "results-audit", "figures",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the AdaFracTCN paper reproduction pipeline")
    ap.add_argument(
        "--steps", nargs="+", choices=list(STEPS), default=DEFAULT_ORDER,
        help="subset of stages to run in the given order",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for name in args.steps:
        cmd = STEPS[name]
        print(f"\n=== {name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
