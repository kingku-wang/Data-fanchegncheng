# -*- coding: utf-8 -*-
"""Semantic/inference audit for the revised AdaFracTCN experiment pipeline.

The audit is intentionally fast. It validates implementation semantics and, when
result files exist, cross-checks inferential outputs against their raw inputs.
It never substitutes for the long forecasting experiment itself.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from adafractcn import FractionalConv1d, get_adafractcn
from baselines import get_model as get_baseline
from param_budget import (baseline_analytic_param_count, manuscript_receptive_field,
                          usable_receptive_field)
from main_experiment import dm_test
from ablation_experiment import paired_difference_interval
from exp_common import adjust_family, load_target_scaling, inverse_target
from download_data import hurst_mfdfa, hurst_rs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
STANDARD_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


def _assert_close(a, b, tol=1e-6, msg=""):
    if not np.allclose(a, b, atol=tol, rtol=tol, equal_nan=True):
        raise AssertionError(msg or f"not close: {a} vs {b}")


def audit_gl_orientation():
    layer = FractionalConv1d(1, 1, kernel_size=1, dilation=1,
                             truncation=4, init_alpha=0.5,
                             learnable_alpha=False, adaptive_alpha=False)
    with torch.no_grad():
        layer.w.fill_(1.0); layer.bias.zero_()
    x0 = torch.zeros(1, 1, 8); x0[0, 0, -1] = 1.0
    x1 = torch.zeros(1, 1, 8); x1[0, 0, -2] = 1.0
    with torch.no_grad():
        y0 = layer(x0)[0, 0, -1].item()
        y1 = layer(x1)[0, 0, -1].item()
    _assert_close(y0, 1.0, msg="lag-0 GL coefficient is not c0=1")
    _assert_close(y1, -0.5, msg="lag-1 GL coefficient is not c1=-alpha")
    return {"lag0": y0, "lag1": y1}


def audit_batch_independence():
    torch.manual_seed(7)
    model = get_adafractcn("AdaFracTCN", input_len=256, horizon=10, mode="standard")
    model.eval()
    x0 = torch.randn(1, 256, 1)
    xa = torch.cat([x0, torch.randn(2, 256, 1)], dim=0)
    xb = torch.cat([x0, torch.randn(2, 256, 1) * 7.0 + 3.0], dim=0)
    with torch.no_grad():
        ya, aa = model.forward_with_alpha(xa)
        yb, ab = model.forward_with_alpha(xb)
    pred_diff = float(torch.max(torch.abs(ya[0] - yb[0])).item())
    alpha_diff = float(torch.max(torch.abs(aa[0] - ab[0])).item())
    if aa.shape != (3, 4, 2):
        raise AssertionError(f"unexpected alpha trace shape: {tuple(aa.shape)}")
    if pred_diff > 1e-6 or alpha_diff > 1e-6:
        raise AssertionError(f"batch dependence detected: pred={pred_diff}, alpha={alpha_diff}")
    return {"prediction_diff": pred_diff, "alpha_diff": alpha_diff,
            "trace_shape": tuple(aa.shape)}


def audit_parameter_counts():
    names = ["LSTM", "GRU", "TCN", "Transformer", "Informer", "Frac-LSTM"]
    rows = {}
    for name in names:
        model = get_baseline(name, horizon=10, mode="standard")
        actual = int(model.count_parameters())
        expected = int(baseline_analytic_param_count(name))
        if actual != expected:
            raise AssertionError(f"{name}: torch={actual} analytic={expected}")
        rows[name] = actual
    ada = get_adafractcn("AdaFracTCN", horizon=10, mode="standard")
    actual = int(ada.count_parameters()); expected = int(baseline_analytic_param_count("AdaFracTCN"))
    if actual != expected:
        raise AssertionError(f"AdaFracTCN: torch={actual} analytic={expected}")
    rows["AdaFracTCN"] = actual
    return rows




def audit_receptive_fields():
    """Check the manuscript RF convention and the RF-matched control."""
    expected = {
        "TCN": (61, 61),
        "TCN-RFmatched": (509, 256),
        "AdaFracTCN": (1006, 256),
    }
    out = {}
    for name, (r_expected, u_expected) in expected.items():
        r, _ = manuscript_receptive_field(name)
        u, _ = usable_receptive_field(name, input_len=256)
        if int(r) != r_expected or int(u) != u_expected:
            raise AssertionError(
                f"{name}: RF/usable={(r,u)}, expected={(r_expected,u_expected)}"
            )
        model = get_adafractcn(name, input_len=256, horizon=10, mode="standard")
        if int(model.receptive_field) != r_expected:
            raise AssertionError(f"{name}: model RF {model.receptive_field} != {r_expected}")
        if int(model.usable_receptive_field) != u_expected:
            raise AssertionError(
                f"{name}: usable RF {model.usable_receptive_field} != {u_expected}"
            )
        out[name] = {"structural": int(r), "usable": int(u)}
    return out

def audit_input_contract():
    """Verify the manuscript's one-channel, 256-day neural input contract."""
    shapes = {}
    for h in (1, 5, 10, 20):
        hd = DATA / f"h{h}"
        for split in ("train", "val", "test"):
            x = np.load(hd / f"X_{split}.npy", mmap_mode="r")
            if x.ndim != 3 or x.shape[1:] != (256, 1):
                raise AssertionError(f"h={h} {split}: expected (*,256,1), got {x.shape}")
            shapes[f"h{h}_{split}"] = tuple(x.shape)
    return shapes

def audit_split_chronology():
    expected = {1: (4252, 756, 1004), 5: (4248, 752, 1000),
                10: (4243, 747, 995), 20: (4233, 737, 985)}
    boundaries = {
        "train": (np.datetime64("2000-02-01"), np.datetime64("2017-12-29")),
        "val":   (np.datetime64("2018-01-02"), np.datetime64("2020-12-31")),
        "test":  (np.datetime64("2021-01-04"), np.datetime64("2024-12-31")),
    }
    out = {}
    for h, counts in expected.items():
        hd = DATA / f"h{h}"
        got = tuple(len(np.load(hd / f"y_{s}.npy")) for s in ("train", "val", "test"))
        if got != counts:
            raise AssertionError(f"h={h}: counts {got}, expected {counts}")
        for split in ("train", "val", "test"):
            origin = np.load(hd / f"origin_dates_{split}.npy").astype("datetime64[D]")
            tend = np.load(hd / f"target_end_dates_{split}.npy").astype("datetime64[D]")
            lo, hi = boundaries[split]
            if tend.min() < lo or tend.max() > hi or np.any(tend <= origin):
                raise AssertionError(f"h={h} split={split}: chronology violation")
        out[h] = got
    return out


def audit_dm_semantics():
    """Check sign convention and overlap-aware HAC lower bound."""
    rng = np.random.default_rng(12345)
    n = 700
    y = rng.normal(size=n)
    # A is intentionally closer to y than B, with serially varying errors.
    a = y + 0.25 * rng.normal(size=n)
    b = y + 0.55 * rng.normal(size=n)
    dm, p, lag = dm_test(y, a, b, horizon=20)
    if lag < 19:
        raise AssertionError(f"h=20 HAC lag {lag} < h-1")
    if dm <= 0:
        raise AssertionError(f"DM sign convention failed: Ada/better model should be positive, got {dm}")
    if not (0 <= p <= 1):
        raise AssertionError(f"invalid DM p-value {p}")
    return {"dm_positive_when_A_better": dm, "p": p, "hac_lag_h20": lag}


def audit_h10_target_hurst():
    y = np.load(DATA / "h10" / "y_test.npy")
    scaling = load_target_scaling(str(DATA), 10)
    target = inverse_target(y, scaling)
    h_dfa = float(hurst_mfdfa(np.abs(target), q=2))
    h_rs = float(hurst_rs(np.abs(target)))
    # These values are deterministic for the bundled data/current preprocessing.
    _assert_close(h_dfa, 1.0680, tol=5e-4, msg="h=10 target MF-DFA anchor drifted")
    _assert_close(h_rs, 0.9072, tol=5e-4, msg="h=10 target R/S anchor drifted")
    return {"MF_DFA": h_dfa, "R_S": h_rs}


def audit_real_outputs_if_present():
    """Cross-check completed outputs after a long run, if they exist."""
    out = {"status": "no raw_predictions bundled in this package"}
    index_path = RESULTS / "raw_predictions" / "index.csv"
    if not index_path.exists():
        return out
    idx = pd.read_csv(index_path)
    if len(idx) != 52:  # 13 models x 4 horizons
        raise AssertionError(f"raw forecast index has {len(idx)} rows, expected 52")
    if not np.all(idx["N_seeds"].to_numpy(int) == 10):
        raise AssertionError("not every main model/horizon has ten saved seeds")
    ada = idx[idx["Model"] == "AdaFracTCN"]
    if len(ada) != 4 or not ada["Alpha_trace"].astype(bool).all():
        raise AssertionError("AdaFracTCN alpha traces missing for one or more horizons")

    dm_path = RESULTS / "dm_test_long.csv"
    if dm_path.exists():
        dm = pd.read_csv(dm_path)
        if len(dm) != 48:
            raise AssertionError(f"real DM family size {len(dm)} != 48")
        if np.any(dm["hac_lag"].to_numpy(float) < dm["Horizon"].to_numpy(float) - 1):
            raise AssertionError("real DM HAC bandwidth violates h-1 lower bound")
        fam = adjust_family(dm["p_unadjusted"].to_numpy(float), alpha=0.05)
        _assert_close(fam["p_holm"], dm["p_holm"], tol=1e-10)
        _assert_close(fam["p_bh"], dm["p_bh"], tol=1e-10)
        out["real_DM_counts"] = (fam["n_significant_unadjusted"], fam["n_significant_holm"], fam["n_significant_bh"])

    ps_path, ci_path = RESULTS / "ablation_perseed.csv", RESULTS / "ablation_paired_ci.csv"
    if ps_path.exists() and ci_path.exists():
        ps, ci = pd.read_csv(ps_path), pd.read_csv(ci_path)
        if "Seed" not in ps.columns:
            raise AssertionError("real ablation_perseed.csv lacks actual Seed ids")
        tcn = ps[ps["Variant"] == "TCN"].sort_values("Seed")
        if set(tcn["Seed"].astype(int)) != set(STANDARD_SEEDS):
            raise AssertionError("TCN ablation seed ids do not match protocol")
        for _, row in ci.iterrows():
            v = ps[ps["Variant"] == row["Variant"]].sort_values("Seed")
            got = paired_difference_interval(tcn["MSE"], v["MSE"])
            _assert_close(got["mean_delta"], row["mean_delta"], tol=1e-10)
            _assert_close(got["exact_low"], row["exact_low"], tol=1e-10)
            _assert_close(got["exact_high"], row["exact_high"], tol=1e-10)
            _assert_close(got["p_exact"], row["p_exact"], tol=1e-10)
        out["real_ablation_pairs"] = len(ci)

    rh_path = RESULTS / "residual_hurst.csv"
    if rh_path.exists():
        rh = pd.read_csv(rh_path)
        if len(rh) < 2 or not np.isfinite(rh[["H_MF_DFA", "H_RS"]].to_numpy(float)).all():
            raise AssertionError("real residual_hurst.csv incomplete/non-finite")
        out["real_residual_rows"] = len(rh)
    out["status"] = "completed outputs present and checked"
    return out


def main():
    lines = []
    def report(label, value):
        line = f"- {label}: {value}"
        print(line); lines.append(line)

    print("AdaFracTCN semantic/inference audit")
    report("GL causal orientation", audit_gl_orientation())
    report("adaptive-alpha batch independence", audit_batch_independence())
    report("parameter counts", audit_parameter_counts())
    report("receptive fields", audit_receptive_fields())
    report("input contract", audit_input_contract())
    report("split chronology", audit_split_chronology())
    report("DM sign/HAC semantics", audit_dm_semantics())
    report("h=10 target Hurst anchor", audit_h10_target_hurst())
    report("completed-output audit", audit_real_outputs_if_present())
    print("AUDIT PASS")
    lines.append("AUDIT PASS")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "semantic_audit.txt").write_text(
        "AdaFracTCN semantic/inference audit\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
