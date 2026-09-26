# -*- coding: utf-8 -*-
"""Audit and summarize completed standard-run outputs for reproducibility.

This script recomputes the manuscript statistics from completed standard-run artifacts.
be run after the long standard experiment and turns the raw forecast artifacts
into a compact audit/report package for manuscript updating.

Usage
-----
    python audit_results.py

Required upstream commands
--------------------------
    python main_experiment.py --mode standard
    python ablation_experiment.py --mode standard
    python regime_analysis.py --mode standard --source saved
    python residual_analysis.py --mode standard --source saved

Outputs
-------
    results/results_audit.md
    results/results_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from exp_common import (adjust_family, apply_positivity_projection,
                        compute_metrics, inverse_target, sample_std)
from main_experiment import dm_test
from ablation_experiment import paired_difference_interval
from download_data import hurst_mfdfa, hurst_rs

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = RESULTS / "raw_predictions"
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
HORIZONS = [1, 5, 10, 20]
MODEL_NAMES = [
    "Persistence", "Uncond mean", "ARIMA", "HAR-RV", "GARCH", "EGARCH",
    "LSTM", "GRU", "TCN", "Transformer", "Informer", "Frac-LSTM", "AdaFracTCN",
]
BASELINES = [m for m in MODEL_NAMES if m != "AdaFracTCN"]


def _slug(name: str) -> str:
    return (name.lower().replace("(1,1)", "11").replace(".", "")
            .replace(" ", "_").replace("-", "_").replace("/", "_"))


def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required completed-run output missing: {path}")
    return path


def _close(a, b, tol=1e-8, label="value"):
    aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape or not np.allclose(aa, bb, rtol=tol, atol=tol, equal_nan=True):
        raise AssertionError(f"{label} mismatch: {aa} vs {bb}")


def load_raw(model: str, h: int) -> dict:
    p = _need(RAW / f"h{h}__{_slug(model)}.npz")
    z = np.load(p, allow_pickle=False)
    req = ["predictions_std", "y_test_std", "origin_dates", "target_mean",
           "target_std", "positivity_floor", "seeds"]
    miss = [k for k in req if k not in z.files]
    if miss:
        raise AssertionError(f"{p.name} missing {miss}")
    out = {k: z[k] for k in z.files}
    pred = np.asarray(out["predictions_std"], float)
    if pred.ndim != 2 or pred.shape[0] != len(SEEDS) or not np.isfinite(pred).all():
        raise AssertionError(f"{p.name}: expected finite predictions shape (10,N), got {pred.shape}")
    if list(np.asarray(out["seeds"], int)) != SEEDS:
        raise AssertionError(f"{p.name}: seed order mismatch")
    if model == "AdaFracTCN":
        if "alpha_trace" not in out:
            raise AssertionError(f"{p.name}: missing operative alpha_trace")
        a = np.asarray(out["alpha_trace"], float)
        if a.ndim != 4 or a.shape[0] != 10 or a.shape[1] != pred.shape[1] or a.shape[-1] != 2:
            raise AssertionError(f"{p.name}: invalid alpha_trace shape {a.shape}")
        if not np.isfinite(a).all() or np.any((a <= 0) | (a >= 1)):
            raise AssertionError(f"{p.name}: non-finite/out-of-range adaptive orders")
    return out


def audit_raw_and_main() -> dict:
    idx = pd.read_csv(_need(RAW / "index.csv"))
    if len(idx) != len(MODEL_NAMES) * len(HORIZONS):
        raise AssertionError(f"raw_predictions/index.csv has {len(idx)} rows, expected 52")

    main = pd.read_csv(_need(RESULTS / "main_results_long.csv"))
    if len(main) != 52:
        raise AssertionError(f"main_results_long.csv has {len(main)} rows, expected 52")

    anchor_by_h = {}
    recomputed = []
    for h in HORIZONS:
        for model in MODEL_NAMES:
            z = load_raw(model, h)
            y = np.asarray(z["y_test_std"], float)
            dates = np.asarray(z["origin_dates"])
            scaling = (float(z["target_mean"]), float(z["target_std"]))
            floor = float(z["positivity_floor"])
            pred = np.asarray(z["predictions_std"], float)
            if h not in anchor_by_h:
                anchor_by_h[h] = (y.copy(), dates.copy(), scaling, floor)
            else:
                y0, d0, sc0, fl0 = anchor_by_h[h]
                _close(y, y0, 1e-12, f"h={h} targets")
                if dates.shape != d0.shape or not np.array_equal(dates, d0):
                    raise AssertionError(f"h={h}: origin dates differ across models")
                _close(scaling, sc0, 1e-12, f"h={h} target scaling")
                _close(floor, fl0, 1e-12, f"h={h} positivity floor")

            m = [compute_metrics(y, p, scaling=scaling, floor=floor) for p in pred]
            row = {
                "model": model, "horizon": h,
                "mse": float(np.mean([r["MSE"] for r in m])),
                "qlike": float(np.mean([r["QLIKE"] for r in m])),
                "mse_std": sample_std([r["MSE"] for r in m]),
                "qlike_std": sample_std([r["QLIKE"] for r in m]),
            }
            recomputed.append(row)
            q = main[(main["model"] == model) & (main["horizon"] == h)]
            if len(q) != 1:
                raise AssertionError(f"main_results_long missing/duplicate {model}, h={h}")
            for key in ["mse", "qlike", "mse_std", "qlike_std"]:
                _close(row[key], float(q.iloc[0][key]), 3e-7, f"main {model} h={h} {key}")
    return {"main_rows": 52, "raw_rows": 52, "recomputed_rows": recomputed}


def audit_dm() -> dict:
    dm = pd.read_csv(_need(RESULTS / "dm_test_long.csv"))
    summ = pd.read_csv(_need(RESULTS / "dm_family_summary.csv"))
    if len(dm) != 48 or len(summ) != 1:
        raise AssertionError("DM outputs must contain 48 comparisons and one family summary")

    expected_rows = []
    for baseline in BASELINES:
        for h in HORIZONS:
            za, zb = load_raw("AdaFracTCN", h), load_raw(baseline, h)
            y_std = np.asarray(za["y_test_std"], float)
            scaling = (float(za["target_mean"]), float(za["target_std"]))
            floor = float(za["positivity_floor"])
            y = inverse_target(y_std, scaling)
            pa = inverse_target(np.asarray(za["predictions_std"], float).mean(axis=0), scaling)
            pb = inverse_target(np.asarray(zb["predictions_std"], float).mean(axis=0), scaling)
            pa = apply_positivity_projection(pa, floor)[0]
            pb = apply_positivity_projection(pb, floor)[0]
            stat, p, lag = dm_test(y, pa, pb, horizon=h)
            expected_rows.append((baseline, h, stat, p, lag))

    # Match by labels, then rederive multiplicity correction from unadjusted p.
    pvals = []
    for baseline, h, stat, p, lag in expected_rows:
        q = dm[(dm["Baseline"] == baseline) & (dm["Horizon"] == h)]
        if len(q) != 1:
            raise AssertionError(f"DM row missing/duplicate {baseline}, h={h}")
        row = q.iloc[0]
        _close(stat, row["DM"], 1e-9, f"DM {baseline} h={h}")
        _close(p, row["p_unadjusted"], 1e-10, f"DM p {baseline} h={h}")
        if int(row["hac_lag"]) != int(lag) or lag < h - 1:
            raise AssertionError(f"DM HAC lag mismatch {baseline} h={h}: {row['hac_lag']} vs {lag}")
        pvals.append(p)
    fam = adjust_family(pvals, alpha=0.05)
    for i, (_, h, _, _, _) in enumerate(expected_rows):
        baseline = expected_rows[i][0]
        row = dm[(dm["Baseline"] == baseline) & (dm["Horizon"] == h)].iloc[0]
        _close(fam["p_holm"][i], row["p_holm"], 1e-10, "Holm p")
        _close(fam["p_bh"][i], row["p_bh"], 1e-10, "BH p")
    s = summ.iloc[0]
    counts = {
        "n_tests": int(fam["n_tests"]),
        "unadjusted": int(fam["n_significant_unadjusted"]),
        "holm": int(fam["n_significant_holm"]),
        "bh": int(fam["n_significant_bh"]),
    }
    _close([counts["n_tests"], counts["unadjusted"], counts["holm"], counts["bh"]],
           [s["n_tests"], s["n_significant_unadjusted"], s["n_significant_holm"], s["n_significant_bh"]],
           0, "DM family summary")
    return counts


def audit_ablation() -> dict:
    ps = pd.read_csv(_need(RESULTS / "ablation_perseed.csv"))
    ci = pd.read_csv(_need(RESULTS / "ablation_paired_ci.csv"))
    variants = [x for x in ps["Variant"].drop_duplicates().tolist() if x != "TCN"]
    tcn = ps[ps["Variant"] == "TCN"].sort_values("Seed")
    if list(tcn["Seed"].astype(int)) != SEEDS:
        raise AssertionError("ablation TCN seed ids do not match standard protocol")
    out = []
    for variant in variants:
        v = ps[ps["Variant"] == variant].sort_values("Seed")
        if list(v["Seed"].astype(int)) != SEEDS:
            raise AssertionError(f"ablation {variant} seed ids do not match")
        calc = paired_difference_interval(tcn["MSE"].to_numpy(float), v["MSE"].to_numpy(float))
        q = ci[ci["Variant"] == variant]
        if len(q) != 1:
            raise AssertionError(f"ablation CI missing/duplicate {variant}")
        r = q.iloc[0]
        mapping = {
            "mean_delta": "mean_delta", "exact_low": "exact_low", "exact_high": "exact_high",
            "bca_low": "bca_low", "bca_high": "bca_high", "p_exact": "p_exact",
        }
        for ck, rk in mapping.items():
            _close(calc[ck], r[rk], 2e-8, f"ablation {variant} {ck}")
        if int(calc["n_pairs"]) != 10:
            raise AssertionError(f"ablation {variant}: expected 10 matched pairs")
        out.append({"variant": variant, **{k: float(calc[k]) for k in mapping}})
    return {"n_variants": len(out), "rows": out}


def audit_regime() -> dict:
    reg = pd.read_csv(_need(RESULTS / "regime_results.csv"))
    alpha = pd.read_csv(_need(RESULTS / "regime_alpha_summary.csv"))
    alpha_seed = pd.read_csv(_need(RESULTS / "regime_alpha_by_seed.csv"))
    if len(alpha) != 1:
        raise AssertionError("regime_alpha_summary must have one row")
    if len(alpha_seed) != 10:
        raise AssertionError(f"regime_alpha_by_seed must have 10 rows, got {len(alpha_seed)}")
    # Verify pooled mean identity using stored calm/turb means and counts when possible.
    # Column names are intentionally discovered rather than assumed for cosmetic fields.
    required = {"Model", "Calm_MSE", "Turbulent_MSE", "Overall_MSE"}
    if not required.issubset(reg.columns):
        raise AssertionError(f"regime_results missing {sorted(required - set(reg.columns))}")
    # The h=10 split contains 459 calm and 536 turbulent test origins; these
    # counts are deterministic functions of the saved windows and training-only
    # median threshold and are also checked by audit_semantics.py.
    n_calm, n_turb = 459, 536
    for _, r in reg.iterrows():
        pooled = (n_calm * float(r["Calm_MSE"]) + n_turb * float(r["Turbulent_MSE"])) / (n_calm + n_turb)
        _close(pooled, float(r["Overall_MSE"]), 5e-6, f"regime pooled {r['Model']}")
    a = alpha.iloc[0].to_dict()
    _close(alpha_seed["long_alpha_mean"].mean(), float(a["long_mean"]), 2e-10, "alpha long seed mean")
    _close(alpha_seed["short_alpha_mean"].mean(), float(a["short_mean"]), 2e-10, "alpha short seed mean")
    _close(alpha_seed["long_alpha_mean"].std(ddof=1), float(a["long_seed_sd"]), 2e-10, "alpha long seed sd")
    _close(alpha_seed["short_alpha_mean"].std(ddof=1), float(a["short_seed_sd"]), 2e-10, "alpha short seed sd")
    for k in ["long_mean", "short_mean", "long_seed_sd", "short_seed_sd",
              "pearson_long", "pearson_short", "spearman_long", "spearman_short",
              "calm_long_mean", "turbulent_long_mean", "calm_short_mean", "turbulent_short_mean",
              "pearson_long_low", "pearson_long_high", "pearson_short_low", "pearson_short_high",
              "spearman_long_low", "spearman_long_high", "spearman_short_low", "spearman_short_high",
              "calm_long_mean_low", "calm_long_mean_high", "turbulent_long_mean_low", "turbulent_long_mean_high",
              "calm_short_mean_low", "calm_short_mean_high", "turbulent_short_mean_low", "turbulent_short_mean_high"]:
        if k not in a or not np.isfinite(float(a[k])):
            raise AssertionError(f"regime alpha summary missing/non-finite {k}")
    return {"n_rows": len(reg), "alpha": {k: float(a[k]) for k in a if isinstance(a[k], (int, float, np.integer, np.floating)) and np.isfinite(a[k])}}


def audit_residual_hurst() -> dict:
    rh = pd.read_csv(_need(RESULTS / "residual_hurst.csv"))
    if len(rh) < 2:
        raise AssertionError("residual_hurst.csv is incomplete")
    ref = rh.iloc[0]
    # Recompute from the stored h=10 target and ensemble forecasts.
    za = load_raw("AdaFracTCN", 10)
    scaling = (float(za["target_mean"]), float(za["target_std"]))
    y = inverse_target(np.asarray(za["y_test_std"], float), scaling)
    abs_y = np.abs(y)
    h0_dfa = float(hurst_mfdfa(abs_y, q=2)); h0_rs = float(hurst_rs(abs_y))
    h0_dfa_no = float(hurst_mfdfa(abs_y[::10], q=2)); h0_rs_no = float(hurst_rs(abs_y[::10]))
    _close(h0_dfa, ref["H_MF_DFA"], 5e-7, "target MF-DFA H")
    _close(h0_rs, ref["H_RS"], 5e-7, "target R/S H")
    if "H_MF_DFA_nonoverlap" not in rh.columns or "H_RS_nonoverlap" not in rh.columns:
        raise AssertionError("residual_hurst.csv missing non-overlapping diagnostics")
    _close(h0_dfa_no, ref["H_MF_DFA_nonoverlap"], 5e-7, "target non-overlap MF-DFA H")
    _close(h0_rs_no, ref["H_RS_nonoverlap"], 5e-7, "target non-overlap R/S H")

    rows = []
    for _, r in rh.iloc[1:].iterrows():
        model = str(r["Model"])
        z = load_raw(model, 10)
        sc = (float(z["target_mean"]), float(z["target_std"]))
        fl = float(z["positivity_floor"])
        pred = inverse_target(np.asarray(z["predictions_std"], float).mean(axis=0), sc)
        pred = apply_positivity_projection(pred, fl)[0]
        yy = inverse_target(np.asarray(z["y_test_std"], float), sc)
        res = np.abs(yy - pred)
        hd = float(hurst_mfdfa(res, q=2)); hr = float(hurst_rs(res))
        hd_no = float(hurst_mfdfa(res[::10], q=2)); hr_no = float(hurst_rs(res[::10]))
        _close(hd, r["H_MF_DFA"], 5e-7, f"residual MF-DFA {model}")
        _close(hr, r["H_RS"], 5e-7, f"residual R/S {model}")
        _close(hd_no, r["H_MF_DFA_nonoverlap"], 5e-7, f"residual non-overlap MF-DFA {model}")
        _close(hr_no, r["H_RS_nonoverlap"], 5e-7, f"residual non-overlap R/S {model}")
        _close(h0_dfa - hd, r["Reduction_MF_DFA"], 5e-7, f"residual reduction {model}")
        _close(h0_rs - hr, r["Reduction_RS"], 5e-7, f"residual reduction R/S {model}")
        rows.append({"model": model, "mfdfa": hd, "rs": hr, "mfdfa_nonoverlap": hd_no, "rs_nonoverlap": hr_no})
    return {"target_mfdfa": h0_dfa, "target_rs": h0_rs, "target_mfdfa_nonoverlap": h0_dfa_no, "target_rs_nonoverlap": h0_rs_no, "rows": rows}


def _fmt(x, digits=4):
    return f"{float(x):.{digits}f}"


def build_outputs(audit: dict) -> None:
    main = pd.read_csv(RESULTS / "main_results_long.csv")
    dm = pd.read_csv(RESULTS / "dm_test_long.csv")
    ci = pd.read_csv(RESULTS / "ablation_paired_ci.csv")
    rh = pd.read_csv(RESULTS / "residual_hurst.csv")
    alpha = pd.read_csv(RESULTS / "regime_alpha_summary.csv").iloc[0]

    def val(model, h, field):
        q = main[(main.model == model) & (main.horizon == h)]
        return float(q.iloc[0][field])

    selected_dm_models = ["HAR-RV", "TCN", "Transformer", "Informer", "Frac-LSTM"]
    selected_dm = {}
    for model in selected_dm_models:
        selected_dm[model] = {}
        for h in HORIZONS:
            r = dm[(dm["Baseline"] == model) & (dm["Horizon"] == h)].iloc[0]
            selected_dm[model][str(h)] = {
                "dm": float(r["DM"]),
                "p_unadjusted": float(r["p_unadjusted"]),
                "p_holm": float(r["p_holm"]),
                "p_bh": float(r["p_bh"]),
            }

    ada_ci = ci[ci["Variant"] == "AdaFracTCN"].iloc[0]
    regime = pd.read_csv(RESULTS / "regime_results.csv")
    summary = {
        "AdaFracTCN": {str(h): {"mse": val("AdaFracTCN", h, "mse"), "qlike": val("AdaFracTCN", h, "qlike")} for h in HORIZONS},
        "Frac-LSTM": {str(h): {"mse": val("Frac-LSTM", h, "mse"), "qlike": val("Frac-LSTM", h, "qlike")} for h in HORIZONS},
        "dm_family": audit["dm"],
        "dm_selected": selected_dm,
        "ablation_ada_vs_tcn": {
            "mean_delta": float(ada_ci["mean_delta"]),
            "exact_low": float(ada_ci["exact_low"]),
            "exact_high": float(ada_ci["exact_high"]),
            "bca_low": float(ada_ci["bca_low"]),
            "bca_high": float(ada_ci["bca_high"]),
            "p_exact": float(ada_ci["p_exact"]),
        },
        "alpha": {k: float(alpha[k]) for k in [
            "long_mean", "short_mean", "long_seed_sd", "short_seed_sd",
            "pearson_long", "pearson_short", "spearman_long", "spearman_short",
            "calm_long_mean", "turbulent_long_mean", "calm_short_mean", "turbulent_short_mean",
            "pearson_long_low", "pearson_long_high", "pearson_short_low", "pearson_short_high",
            "spearman_long_low", "spearman_long_high", "spearman_short_low", "spearman_short_high",
            "calm_long_mean_low", "calm_long_mean_high", "turbulent_long_mean_low", "turbulent_long_mean_high",
            "calm_short_mean_low", "calm_short_mean_high", "turbulent_short_mean_low", "turbulent_short_mean_high"]},
        "regime_h10": regime.to_dict(orient="records"),
        "residual_hurst": audit["residual"],
    }
    (RESULTS / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # A compact human-readable report.
    lines = ["# Completed standard-run audit", "", "All values in this file are recomputed from completed standard-run artifacts.", ""]
    lines += [f"- Main result rows checked: {audit['main']['main_rows']}.",
              f"- Raw forecast artifacts checked: {audit['main']['raw_rows']}.",
              f"- DM family: {audit['dm']['n_tests']} tests; significant unadjusted/Holm/BH = {audit['dm']['unadjusted']}/{audit['dm']['holm']}/{audit['dm']['bh']}.",
              f"- Ablation matched-pair rows checked: {audit['ablation']['n_variants']}.",
              f"- Residual-Hurst rows recomputed: {len(audit['residual']['rows'])} model rows.",
              f"- Regime rows checked: {audit['regime']['n_rows']}.", "", "## AdaFracTCN vs Frac-LSTM MSE", ""]
    lines.append("| h | AdaFracTCN | Frac-LSTM | reduction |")
    lines.append("|---:|---:|---:|---:|")
    for h in HORIZONS:
        a, b = val("AdaFracTCN", h, "mse"), val("Frac-LSTM", h, "mse")
        lines.append(f"| {h} | {a:.6f} | {b:.6f} | {(b-a)/b*100:.2f}% |")
    (RESULTS / "results_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")




def main():
    audit = {
        "main": audit_raw_and_main(),
        "dm": audit_dm(),
        "ablation": audit_ablation(),
        "regime": audit_regime(),
        "residual": audit_residual_hurst(),
    }
    build_outputs(audit)
    print("RESULT AUDIT PASS")
    print(f"Wrote {RESULTS / 'results_audit.md'}")
    print(f"Wrote {RESULTS / 'results_summary.json'}")


if __name__ == "__main__":
    main()
