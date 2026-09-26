# -*- coding: utf-8 -*-
"""Generate manuscript figures from completed experiment outputs only."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig_style

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(fig_style.DATA_DIR)
FIG_DIR = Path(fig_style.FIGURES_DIR)
RESULTS_DIR = SCRIPT_DIR / "results"
HORIZONS = [1, 5, 10, 20]
fig_style.apply(serif=True)

_NAME_MAP = {
    "Uncond. mean": "Uncond mean", "GARCH(1,1)": "GARCH", "EGARCH(1,1)": "EGARCH"
}


def canonical_name(name):
    return _NAME_MAP.get(str(name), str(name))


def _need(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path




def _load_main():
    p = RESULTS_DIR / "main_results_long.csv"
    df = pd.read_csv(_need(p)).rename(columns=str.lower)
    req = {"model", "horizon", "mse", "qlike"}
    if not req.issubset(df.columns):
        raise ValueError(f"{p.name} missing {sorted(req-set(df.columns))}")
    df["model"] = df["model"].map(canonical_name)
    out = {}
    for _, r in df.iterrows():
        name, h = str(r.model), int(r.horizon)
        out.setdefault(name, {"mse": {}, "qlike": {}})
        out[name]["mse"][h] = float(r.mse)
        out[name]["qlike"][h] = float(r.qlike)
    for name, vals in out.items():
        miss = [h for h in HORIZONS if h not in vals["mse"] or h not in vals["qlike"]]
        if miss:
            raise ValueError(f"{p.name}: {name} missing horizons {miss}")
    return out


def _load_alpha():
    r = pd.read_csv(_need(RESULTS_DIR / "regime_alpha_summary.csv")).iloc[0]
    return {
        "long_calm": float(r.calm_long_mean), "short_calm": float(r.calm_short_mean),
        "long_turbulent": float(r.turbulent_long_mean), "short_turbulent": float(r.turbulent_short_mean),
        "long_pearson": float(r.pearson_long), "short_pearson": float(r.pearson_short),
        "long_spearman": float(r.spearman_long), "short_spearman": float(r.spearman_short),
    }


def _load_ablation():
    p = RESULTS_DIR / "ablation_paired_ci.csv"
    df = pd.read_csv(_need(p))
    req = {"Variant", "mean_delta", "exact_low", "exact_high"}
    if not req.issubset(df.columns):
        raise ValueError(f"{p.name} missing {sorted(req-set(df.columns))}")
    return df.copy()


def gl_coefficients(alpha, n):
    c = np.empty(n + 1, dtype=float); c[0] = 1.0
    for k in range(1, n + 1):
        c[k] = c[k - 1] * (k - 1 - alpha) / k
    return c


def normalized_gl_decay(alpha, n):
    c = np.abs(gl_coefficients(alpha, n))
    return c / c[1]


def load_abs_returns():
    path = _need(DATA_DIR / "sp500.csv")
    close = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                close.append(float(row["Close"]))
            except (KeyError, TypeError, ValueError):
                continue
    close = np.asarray(close, dtype=float)
    if close.size < 300:
        raise ValueError("sp500.csv contains too few usable closes")
    return np.abs(np.diff(np.log(close)))


def acf(x, max_lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def empirical_abs_return_acf(max_lag=256, fit_lo=20, fit_hi=150):
    a = load_abs_returns(); rho = acf(a, max_lag); lags = np.arange(max_lag + 1)
    ci95 = 1.96 / np.sqrt(a.size)
    mask = (lags >= fit_lo) & (lags <= fit_hi) & (rho > 0)
    slope, intercept = np.polyfit(np.log(lags[mask]), np.log(rho[mask]), 1)
    return lags, rho, ci95, float(slope), float(intercept)


def save(fig, name, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    return fig_style.save(fig, name, str(outdir))


def fig_main_results(outdir):
    """Focused summary of the main result table without overlapping trajectories.

    The table already contains the exact MSE/QLIKE values.  The figure therefore
    visualizes the percentage MSE reduction of AdaFracTCN relative to every
    comparator at each horizon, which is easier to read than 13 nearly
    coincident multi-horizon lines.
    """
    data = _load_main()
    ada = data["AdaFracTCN"]["mse"]

    preferred_order = [
        "Persistence", "Uncond mean", "HAR-RV", "ARIMA", "GARCH", "EGARCH",
        "LSTM", "GRU", "TCN", "Transformer", "Informer", "Frac-LSTM",
    ]
    names = [n for n in preferred_order if n in data]
    # Keep any unexpected completed-run models rather than silently dropping them.
    names += [n for n in data if n not in names and n != "AdaFracTCN"]

    values = np.array([
        [100.0 * (data[name]["mse"][h] - ada[h]) / data[name]["mse"][h] for h in HORIZONS]
        for name in names
    ], dtype=float)

    labels = {
        "Uncond mean": "Uncond. mean",
        "GARCH": "GARCH(1,1)",
        "EGARCH": "EGARCH(1,1)",
    }
    ylabels = [labels.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(7.15, 4.45))
    vmax = max(10.0, float(np.ceil(values.max() / 5.0) * 5.0))
    im = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_xticklabels([rf"$h={h}$" for h in HORIZONS])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(ylabels, fontsize=8.6)
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_title(r"MSE reduction of AdaFracTCN relative to each comparator", fontsize=11)

    # Group separator between conventional/statistical and neural baselines.
    if "LSTM" in names:
        split = names.index("LSTM") - 0.5
        ax.axhline(split, color="white", linewidth=2.0)
        ax.axhline(split, color=fig_style.COLOR_GREY, linewidth=0.6, alpha=0.65)

    threshold = 0.58 * vmax
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = "white" if v >= threshold else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=7.7, color=color)

    # Subtle emphasis of the Frac-LSTM comparison row without changing the data scale.
    if "Frac-LSTM" in names:
        idx = names.index("Frac-LSTM")
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-0.5, idx - 0.5), len(HORIZONS), 1.0,
                               fill=False, edgecolor=fig_style.COLOR_MAIN,
                               linewidth=1.2))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("MSE reduction (%)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    ax.tick_params(axis="x", labelsize=9.2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    save(fig, "main_results.pdf", outdir)

def fig_alpha_regime(outdir):
    m = _load_alpha()
    calm = [m["long_calm"], m["short_calm"]]; turb = [m["long_turbulent"], m["short_turbulent"]]
    pear = [m["long_pearson"], m["short_pearson"]]; spear = [m["long_spearman"], m["short_spearman"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05)); x = np.arange(2); w = 0.34
    axes[0].bar(x-w/2, calm, w, label="Calm"); axes[0].bar(x+w/2, turb, w, label="Turbulent")
    axes[0].set_xticks(x, ["Long-memory\nchannel", "Short-memory\nchannel"])
    axes[0].set_ylabel(r"Operative order $\alpha(t)$"); axes[0].set_ylim(0, 0.9)
    axes[0].legend(frameon=False); axes[0].set_title("(a) regime means", loc="left")
    axes[1].bar(x-w/2, pear, w, label="Pearson $r$"); axes[1].bar(x+w/2, spear, w, label="Spearman $\\rho$")
    axes[1].axhline(0, linewidth=0.8); axes[1].set_xticks(x, ["Long-memory\nchannel", "Short-memory\nchannel"])
    axes[1].set_ylabel("Correlation with 20-day volatility"); axes[1].set_ylim(-0.65, 0.5)
    axes[1].legend(frameon=False); axes[1].set_title("(b) volatility association", loc="left")
    fig.tight_layout(); save(fig, "alpha_regime.pdf", outdir)


def fig_ablation(outdir):
    df = _load_ablation()
    order = [
        "TCN-RFmatched", "PowerLaw", "FracTCN-fixed", "FracTCN-learnable",
        "FracTCN-dual", "FracTCN-dual-fixed", "FracTCN-dual-indep", "AdaFracTCN",
    ]
    present = [v for v in order if v in set(df["Variant"].astype(str))]
    if not present:
        raise ValueError("ablation_paired_ci.csv contains none of the manuscript variants")
    df = df.set_index("Variant").loc[present].reset_index()
    y = np.arange(len(df))
    x = df["mean_delta"].to_numpy(float)
    lo = df["exact_low"].to_numpy(float)
    hi = df["exact_high"].to_numpy(float)
    xerr = np.vstack([x-lo, hi-x])

    fig, ax = plt.subplots(figsize=(7.15, 4.55))
    colors = []
    for name in df["Variant"]:
        if name == "AdaFracTCN": colors.append("#d62728")
        elif name in {"FracTCN-dual-fixed", "FracTCN-dual-indep"}: colors.append("#e6864a")
        elif name == "TCN-RFmatched": colors.append("#888888")
        else: colors.append(fig_style.COLOR_MAIN)
    for i, (xx, ee, cc) in enumerate(zip(x, xerr.T, colors)):
        ax.errorbar(xx, i, xerr=np.array([[ee[0]], [ee[1]]]), fmt="o",
                    color=cc, ecolor=cc, elinewidth=1.5, capsize=0, markersize=4.5)
        ax.text(hi[i] + max(0.00012, 0.015*(hi.max()-lo.min())), i, f"{xx:.4f}",
                va="center", fontsize=8.6, color=cc)
    ax.axvline(0.0, linestyle="--", linewidth=0.9, color=fig_style.COLOR_GREY)
    ax.set_yticks(y, df["Variant"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired improvement over TCN in MSE, $\Delta=\mathrm{MSE}_{TCN}-\mathrm{MSE}_{variant}$")
    ax.set_title(r"Matched-seed ablation effects at $h=10$")
    ax.grid(axis="x", alpha=0.22, linestyle="--")
    fig.tight_layout()
    save(fig, "ablation_forest.pdf", outdir)


def fig_robustness(outdir):
    cross_p = RESULTS_DIR / "robustness_cross_asset_h10.csv"
    exp_p = RESULTS_DIR / "robustness_expanding_h10.csv"
    gk_p = RESULTS_DIR / "robustness_garman_klass_h10.csv"
    cross = pd.read_csv(_need(cross_p))
    expanding = pd.read_csv(_need(exp_p))
    gk = pd.read_csv(_need(gk_p))

    cross = cross[cross.get("Status", "ok") == "ok"].copy()
    label_map = {"^GSPC":"S&P 500", "^IXIC":"NASDAQ", "^DJI":"DJIA"}
    cross["Label"] = cross["Asset"].map(lambda x: label_map.get(str(x), str(x)))

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 3.25), gridspec_kw={"width_ratios":[1.15,1.0,1.2]})

    ax = axes[0]
    yy = np.arange(len(cross))
    vals = cross["Ratio_Ada_over_Best"].to_numpy(float)
    ax.barh(yy, vals, height=0.62, color=fig_style.COLOR_MAIN)
    ax.set_yticks(yy, cross["Label"].tolist()); ax.invert_yaxis()
    ax.axvline(1.0, linestyle="--", linewidth=0.9, color=fig_style.COLOR_GREY)
    for i,v in enumerate(vals): ax.text(v+0.001, i, f"{v:.3f}", va="center", fontsize=8.3)
    ax.set_xlabel("MSE ratio"); ax.set_title("(a) Cross-asset")
    ax.grid(axis="x", alpha=0.18, linestyle="--")

    ax = axes[1]
    expanding = expanding.sort_values("Block")
    bx = expanding["Block"].to_numpy(int); by = expanding["Ratio_Ada_over_Best"].to_numpy(float)
    ax.plot(bx, by, marker="o", linewidth=1.6, markersize=3.6, color=fig_style.COLOR_MAIN)
    ax.axhline(1.0, linestyle="--", linewidth=0.9, color=fig_style.COLOR_GREY)
    ax.fill_between([bx.min()-0.35,bx.max()+0.35], [0.94,0.94], [1.0,1.0], color=fig_style.COLOR_MAIN, alpha=0.06)
    ax.set_xticks(bx); ax.set_xlabel("Evaluation block"); ax.set_title("(b) Expanding window")
    ymin=min(0.94, float(by.min())-0.006); ymax=max(1.02, float(by.max())+0.006); ax.set_ylim(ymin,ymax)
    ax.grid(alpha=0.18, linestyle="--")

    ax = axes[2]
    order=["Persistence","HAR-RV","GARCH","TCN","Frac-LSTM","AdaFracTCN"]
    gd=gk.set_index("Model")
    names=[n for n in order if n in gd.index]
    gv=np.array([float(gd.loc[n,"MSE"]) for n in names])
    gy=np.arange(len(names))
    cols=[]
    for n in names:
        if n=="AdaFracTCN": cols.append("#d62728")
        elif n=="Frac-LSTM": cols.append("#e6864a")
        elif n=="Persistence": cols.append("#8a8a8a")
        else: cols.append(fig_style.COLOR_MAIN)
    ax.barh(gy, gv, height=0.62, color=cols); ax.set_yticks(gy, ["GARCH(1,1)" if n=="GARCH" else n for n in names]); ax.invert_yaxis()
    for i,v in enumerate(gv): ax.text(v+0.0004, i, f"{v:.3f}", va="center", fontsize=8.2)
    ax.set_xlabel("MSE"); ax.set_title("(c) Garman--Klass target")
    ax.grid(axis="x", alpha=0.18, linestyle="--")

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "robustness_summary.pdf", outdir)

def fig_memory_decay(outdir):
    N, K = 256, 5; k = np.arange(1, N + 1)
    lags, rho, ci95, slope, intercept = empirical_abs_return_acf(N, 20, 150)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    ax = axes[0]
    for alpha, ls in [(0.2, "-"), (0.5, "--"), (0.8, ":")]:
        ax.plot(k, normalized_gl_decay(alpha, N)[1:], linestyle=ls, linewidth=1.8,
                label=rf"reference $\alpha={alpha:.1f}$")
    ax.plot(k, np.exp(-(k-1)/float(K)), linestyle="-.", linewidth=1.5, label=rf"exponential ($\tau=K={K}$)")
    ax.axvline(K, linestyle=(0, (4, 2)), linewidth=1.3, label=rf"TCN cutoff $K={K}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Lag $k$ (trading days)")
    ax.set_ylabel(r"Normalized memory weight $|\tilde{c}_k(\alpha)|$")
    ax.set_xlim(1, N); ax.set_ylim(4e-4, 2.0); ax.grid(alpha=0.22, which="both", linestyle="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False,
              fontsize=7.8, columnspacing=0.9, handlelength=2.5)
    ax.set_title("(a) analytic memory-kernel references", fontsize=10.5, loc="left")

    ax = axes[1]; valid = (lags[1:] > 0) & (rho[1:] > 0)
    ax.plot(lags[1:][valid], rho[1:][valid], linestyle=":", marker="o", markersize=2.7,
            markevery=10, linewidth=1.25, label=r"ACF of $|r_t|$ (S\&P 500)")
    ax.axhline(ci95, linestyle="--", linewidth=1.0)
    fitk = np.linspace(20, 150, 100); fitrho = np.exp(intercept) * fitk ** slope
    ax.plot(fitk, fitrho, linewidth=1.8, label=rf"local fit $\propto k^{{{slope:.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Lag $k$ (trading days)")
    ax.set_ylabel(r"Autocorrelation of $|r_t|$"); ax.set_xlim(1, 200); ax.set_ylim(0.012, 0.62)
    ax.grid(alpha=0.22, which="both", linestyle="--"); ax.legend(loc="upper right", frameon=False, fontsize=8.1)
    ax.text(0.04, 0.05, rf"95% white-noise band: $\pm {ci95:.3f}$" + "\n" +
            rf"local log--log slope: ${slope:.2f}$ (lags 20--150)", transform=ax.transAxes,
            fontsize=8.2, va="bottom")
    ax.set_title(r"(b) empirical long-range dependence of $|r_t|$", fontsize=10.5, loc="left")
    fig.tight_layout(w_pad=1.4); save(fig, "memory_decay.pdf", outdir)


def run_check():
    data = _load_main(); best, ada = [], []
    for h in HORIZONS:
        vals = [d["mse"][h] for n, d in data.items() if n != "AdaFracTCN"]
        best.append(min(vals)); ada.append(data["AdaFracTCN"]["mse"][h])
    margin = 100 * (np.array(best)-np.array(ada))/np.array(best)
    _, _, ci, slope, _ = empirical_abs_return_acf()
    print("MSE margin over lowest-MSE non-Ada evaluated model:", margin)
    print("empirical ACF slope 20-150:", slope, "95% band:", ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()
    if args.check:
        run_check(); return
    fig_main_results(args.outdir)
    fig_alpha_regime(args.outdir)
    fig_ablation(args.outdir)
    fig_memory_decay(args.outdir)
    fig_robustness(args.outdir)
    print("figures written to", args.outdir, "from completed experiment outputs")


if __name__ == "__main__":
    main()
