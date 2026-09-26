# -*- coding: utf-8 -*-
"""robustness_experiment.py — Section 4.7 robustness checks.

This is the single runner for the three h=10 robustness analyses reported in
Section 4.7 of the manuscript:

1. Cross-asset descriptive check on ^GSPC, ^IXIC, ^DJI, AAPL, JPM and XOM.
   Ratio = AdaFracTCN MSE / minimum MSE among Persistence, Uncond mean,
   HAR-RV and TCN.
2. Eight chronological expanding-window blocks for S&P 500.
3. Alternative Garman--Klass target on S&P 500, comparing Persistence,
   HAR-RV, GARCH, TCN, Frac-LSTM and AdaFracTCN.

All three checks use h=10, L=256 and seed 42 for trainable models, exactly as
stated in the manuscript.  This script contains no manuscript result constants.

Examples
--------
Prepare only the Garman--Klass cached windows from the bundled S&P 500 OHLC CSV:
    python robustness_experiment.py --prepare-gk --check-only

Run all three checks (requires the six asset datasets to have been prepared):
    python robustness_experiment.py --prepare-gk

Run selected components:
    python robustness_experiment.py --only cross-asset expanding
    python robustness_experiment.py --only garman-klass --prepare-gk

A lightweight data/configuration check that does not train models:
    python robustness_experiment.py --check-only --prepare-gk
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from adafractcn import get_adafractcn
from baselines import get_model as get_baseline, set_protocol_scaling
from exp_common import (
    PARAMS_FILENAME,
    compute_metrics,
    load_input_scaling,
    load_positivity_floor,
    load_target_scaling,
    sample_std,
    set_seed,
)
import download_data as data_pipeline

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
RESULTS_DIR = HERE / "results"
GK_DATA_DIR = DATA_ROOT / "robustness_gk" / "GSPC"

HORIZON = 10
INPUT_LEN = 256
SEED = 42
DEFAULT_BLOCKS = 8
DEFAULT_ASSET = "^GSPC"
ASSETS = ["^GSPC", "^IXIC", "^DJI", "AAPL", "JPM", "XOM"]

CROSS_ASSET_MODELS = [
    "Persistence", "Uncond mean", "HAR-RV", "TCN", "AdaFracTCN"
]
CROSS_ASSET_COMPARATORS = ["Persistence", "Uncond mean", "HAR-RV", "TCN"]
GK_MODELS = ["Persistence", "HAR-RV", "GARCH", "TCN", "Frac-LSTM", "AdaFracTCN"]

MODE_CONFIG = {
    "standard": {"batch_size": 256},
    "quick": {"batch_size": 128},
}


def asset_slug(asset: str) -> str:
    return asset.lstrip("^")


def standard_asset_dir(asset: str) -> Path:
    return DATA_ROOT if asset == DEFAULT_ASSET else DATA_ROOT / asset_slug(asset)


def _read_params(data_dir: Path) -> dict:
    p = data_dir / PARAMS_FILENAME
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("rb") as fh:
        return pickle.load(fh)


def load_horizon(data_dir: Path, horizon: int = HORIZON):
    """Load one prepared horizon and register its protocol scaling."""
    hdir = data_dir / f"h{horizon}"
    names = [
        "X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
        "X_test.npy", "y_test.npy",
    ]
    if not hdir.is_dir() or not all((hdir / n).exists() for n in names):
        return None
    arrays = tuple(np.load(hdir / n) for n in names)
    scaling = load_target_scaling(str(data_dir), horizon)
    floor = load_positivity_floor(str(data_dir), horizon)
    set_protocol_scaling(
        input_scaling=load_input_scaling(str(data_dir)),
        target_scaling=scaling,
    )
    return arrays, scaling, floor


def make_loader(X, y, batch_size: int, shuffle: bool = False):
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def fit_and_predict(model_name: str, *, mode: str, X_train, y_train,
                    X_val, y_val, X_test, y_test, batch_size: int, device: str):
    """Train one manuscript-specified model with the Section 4.7 seed."""
    set_seed(SEED)
    tr = make_loader(X_train, y_train, batch_size, shuffle=True)
    va = make_loader(X_val, y_val, batch_size, shuffle=False)
    te = make_loader(X_test, y_test, batch_size, shuffle=False)
    if model_name == "AdaFracTCN":
        model = get_adafractcn("AdaFracTCN", input_len=INPUT_LEN,
                               horizon=HORIZON, mode=mode)
    else:
        model = get_baseline(model_name, input_len=INPUT_LEN,
                             horizon=HORIZON, mode=mode)
    model.fit(tr, va, device=device)
    return model.predict(te, device=device)


def evaluate_models(model_names, loaded, *, mode: str):
    arrays, scaling, floor = loaded
    Xtr, ytr, Xva, yva, Xte, yte = arrays
    batch_size = MODE_CONFIG[mode]["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in model_names:
        pred = fit_and_predict(
            name, mode=mode,
            X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
            X_test=Xte, y_test=yte, batch_size=batch_size, device=device,
        )
        metrics = compute_metrics(yte, pred, scaling=scaling, floor=floor)
        rows.append({"Model": name, **metrics})
    return pd.DataFrame(rows)


def prepare_gk_data(mode: str = "standard", overwrite: bool = False) -> Path:
    """Build isolated S&P 500 Garman--Klass windows from the bundled OHLC CSV.

    The main squared-return data under ``data/`` are never overwritten.  This
    function reuses the exact preprocessing functions in ``download_data.py``
    but writes the alternative-target cache to ``data/robustness_gk/GSPC``.
    """
    params = GK_DATA_DIR / PARAMS_FILENAME
    hdir = GK_DATA_DIR / f"h{HORIZON}"
    if params.exists() and hdir.exists() and not overwrite:
        return GK_DATA_DIR

    raw_path = DATA_ROOT / "sp500.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"bundled OHLC cache not found: {raw_path}; run download_data.py first"
        )
    GK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = data_pipeline.read_raw_csv(str(raw_path))
    feat = data_pipeline.compute_features(raw)
    cfg = dict(data_pipeline.MODE_CONFIG[mode])
    # Section 4.7 is h=10 only; restricting the cache avoids unrelated files.
    cfg["horizons"] = [HORIZON]
    masks = data_pipeline.split_masks(feat["Date"], cfg)
    shapes, target_norm, input_norm = data_pipeline.build_and_save_windows(
        feat, masks, cfg, str(GK_DATA_DIR), INPUT_LEN, proxy="sqgk"
    )
    with params.open("wb") as fh:
        pickle.dump({
            "asset": DEFAULT_ASSET,
            "proxy": "sqgk",
            "window_len": INPUT_LEN,
            "horizons": [HORIZON],
            "mode": mode,
            "data_config": cfg,
            "target_normalization": target_norm,
            "input_normalization": {
                "mean": input_norm["mean"], "std": input_norm["std"]
            },
            "shapes": shapes,
        }, fh)
    return GK_DATA_DIR


def run_cross_asset(mode: str, assets=ASSETS) -> tuple[pd.DataFrame, list[str]]:
    """Manuscript cross-asset check: h=10 only, seed 42, stated comparator set."""
    rows, missing = [], []
    for asset in assets:
        d = standard_asset_dir(asset)
        loaded = load_horizon(d, HORIZON)
        if loaded is None:
            missing.append(asset)
            rows.append({"Asset": asset, "Status": "no-data", "Horizon": HORIZON})
            continue
        metrics = evaluate_models(CROSS_ASSET_MODELS, loaded, mode=mode)
        mse = dict(zip(metrics["Model"], metrics["MSE"]))
        best_name = min(CROSS_ASSET_COMPARATORS, key=lambda n: mse[n])
        best = float(mse[best_name])
        ada = float(mse["AdaFracTCN"])
        row = {
            "Asset": asset,
            "Status": "ok",
            "Horizon": HORIZON,
            "Seed": SEED,
            "MSE_AdaFracTCN": ada,
            "BestComparator": best_name,
            "MSE_BestComparator": best,
            "Ratio_Ada_over_Best": ada / best,
        }
        for name in CROSS_ASSET_MODELS:
            row[f"MSE_{name}"] = float(mse[name])
        rows.append(row)
    return pd.DataFrame(rows), missing


def run_expanding_window(mode: str, blocks: int = DEFAULT_BLOCKS) -> pd.DataFrame:
    """Eight chronological S&P 500 h=10 expanding-window evaluation blocks.

    For block b, model fitting may use only observations available before that
    block: original train, original validation, and earlier test blocks.  The
    original validation interval remains the common early-stopping reference;
    the evaluated block itself is never used for fitting or early stopping.
    This preserves the information-set rule stated in the manuscript.
    """
    loaded = load_horizon(standard_asset_dir(DEFAULT_ASSET), HORIZON)
    if loaded is None:
        raise FileNotFoundError("S&P 500 h=10 data are not prepared")
    arrays, scaling, floor = loaded
    Xtr, ytr, Xva, yva, Xte, yte = arrays
    edges = np.linspace(0, len(yte), blocks + 1).astype(int)
    batch_size = MODE_CONFIG[mode]["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for b in range(blocks):
        lo, hi = map(int, (edges[b], edges[b + 1]))
        if hi <= lo:
            continue
        Xfit = np.concatenate([Xtr, Xva, Xte[:lo]], axis=0)
        yfit = np.concatenate([ytr, yva, yte[:lo]], axis=0)
        mse = {}
        for name in CROSS_ASSET_MODELS:
            pred = fit_and_predict(
                name, mode=mode,
                X_train=Xfit, y_train=yfit,
                X_val=Xva, y_val=yva,
                X_test=Xte[lo:hi], y_test=yte[lo:hi],
                batch_size=batch_size, device=device,
            )
            mse[name] = compute_metrics(
                yte[lo:hi], pred, scaling=scaling, floor=floor
            )["MSE"]
        best_name = min(CROSS_ASSET_COMPARATORS, key=lambda n: mse[n])
        best = float(mse[best_name])
        ada = float(mse["AdaFracTCN"])
        row = {
            "Block": b + 1,
            "Start": lo,
            "End": hi,
            "N_block": hi - lo,
            "Horizon": HORIZON,
            "Seed": SEED,
            "MSE_AdaFracTCN": ada,
            "BestComparator": best_name,
            "MSE_BestComparator": best,
            "Ratio_Ada_over_Best": ada / best,
            "AdaFracTCN_below_best": int(ada < best),
        }
        for name in CROSS_ASSET_MODELS:
            row[f"MSE_{name}"] = float(mse[name])
        rows.append(row)
    return pd.DataFrame(rows)


def run_garman_klass(mode: str) -> pd.DataFrame:
    loaded = load_horizon(GK_DATA_DIR, HORIZON)
    if loaded is None:
        raise FileNotFoundError(
            f"Garman--Klass h=10 data not prepared under {GK_DATA_DIR}; "
            "rerun with --prepare-gk"
        )
    df = evaluate_models(GK_MODELS, loaded, mode=mode)
    df.insert(0, "Seed", SEED)
    df.insert(0, "Horizon", HORIZON)
    df.insert(0, "Target", "Garman-Klass")
    return df


def check_inputs(prepare_gk: bool, mode: str) -> dict:
    if prepare_gk:
        prepare_gk_data(mode=mode)
    status = {
        "horizon": HORIZON,
        "input_len": INPUT_LEN,
        "seed": SEED,
        "cross_asset": {},
        "garman_klass": False,
    }
    for asset in ASSETS:
        loaded = load_horizon(standard_asset_dir(asset), HORIZON)
        status["cross_asset"][asset] = loaded is not None
        if loaded is not None:
            Xtr = loaded[0][0]
            if Xtr.ndim != 3 or Xtr.shape[1:] != (INPUT_LEN, 1):
                raise AssertionError(f"{asset}: expected (*,{INPUT_LEN},1), got {Xtr.shape}")
    gk = load_horizon(GK_DATA_DIR, HORIZON)
    status["garman_klass"] = gk is not None
    if gk is not None and gk[0][0].shape[1:] != (INPUT_LEN, 1):
        raise AssertionError("Garman--Klass cache violates L=256 one-channel contract")
    return status


def _print_summary(cross=None, expanding=None, gk=None):
    if cross is not None and not cross.empty:
        ok = cross[cross["Status"] == "ok"]
        if not ok.empty:
            print("\nCross-asset h=10 ratios:")
            for _, r in ok.iterrows():
                print(f"  {r['Asset']:6s}: {r['Ratio_Ada_over_Best']:.3f}")
    if expanding is not None and not expanding.empty:
        ratios = expanding["Ratio_Ada_over_Best"].to_numpy(float)
        print("\nExpanding-window h=10 ratios:")
        print("  " + ", ".join(f"{x:.3f}" for x in ratios))
        print(f"  mean={np.mean(ratios):.3f}, sample SD={sample_std(ratios):.3f}, "
              f"below one={(ratios < 1).sum()}/{len(ratios)}")
    if gk is not None and not gk.empty:
        print("\nGarman--Klass h=10 MSE:")
        for _, r in gk.iterrows():
            print(f"  {r['Model']:12s}: {r['MSE']:.6f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the three manuscript Section 4.7 robustness checks")
    ap.add_argument("--mode", default="standard", choices=sorted(MODE_CONFIG))
    ap.add_argument("--only", nargs="+", choices=["cross-asset", "expanding", "garman-klass"],
                    default=["cross-asset", "expanding", "garman-klass"])
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    ap.add_argument("--prepare-gk", action="store_true",
                    help="build isolated S&P 500 Garman--Klass h=10 windows from cached OHLC")
    ap.add_argument("--overwrite-gk", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="validate required data/configuration without training")
    args = ap.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.prepare_gk:
        prepare_gk_data(args.mode, overwrite=args.overwrite_gk)

    status = check_inputs(False, args.mode)
    if args.check_only:
        print(status)
        missing = [a for a, ok in status["cross_asset"].items() if not ok]
        if missing:
            print("Missing cross-asset caches:", ", ".join(missing))
        if not status["garman_klass"]:
            print("Garman--Klass cache missing; use --prepare-gk")
        return 0

    cross = expanding = gk = None
    if "cross-asset" in args.only:
        cross, missing = run_cross_asset(args.mode)
        cross.to_csv(RESULTS_DIR / "robustness_cross_asset_h10.csv", index=False)
        if missing:
            print("Missing cross-asset data:", ", ".join(missing))
    if "expanding" in args.only:
        expanding = run_expanding_window(args.mode, blocks=args.blocks)
        expanding.to_csv(RESULTS_DIR / "robustness_expanding_h10.csv", index=False)
    if "garman-klass" in args.only:
        gk = run_garman_klass(args.mode)
        gk.to_csv(RESULTS_DIR / "robustness_garman_klass_h10.csv", index=False)

    _print_summary(cross, expanding, gk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
