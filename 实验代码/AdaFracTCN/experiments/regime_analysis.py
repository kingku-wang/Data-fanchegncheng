# -*- coding: utf-8 -*-
"""
regime_analysis.py — 市场 Regime 分析
======================================
分析不同市场状态下各模型的表现，验证 AdaFracTCN 自适应机制的有效性。

输出:
  - results/regime_results.csv              Regime 分析结果表

用法:
    python regime_analysis.py --mode standard
    python regime_analysis.py --mode quick
"""
import argparse
import os
import time
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

from baselines import get_model as get_baseline, set_protocol_scaling
from adafractcn import get_adafractcn
import fig_style
from exp_common import (load_target_scaling, load_input_scaling,
                        compute_metrics,
                        load_positivity_floor, set_seed)

# ============================================================
# 全局配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = fig_style.DATA_DIR
RESULTS_DIR = fig_style.RESULTS_DIR

HORIZON = 10

MODE_CONFIG = {
    "standard": {
        "seeds": [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
        "models": ["Persistence", "HAR-RV", "GARCH", "LSTM", "TCN",
                   "Frac-LSTM", "AdaFracTCN"],
        "batch_size": 256,
    },
    "quick": {
        "seeds": [42],
        "models": ["TCN", "AdaFracTCN"],
        "batch_size": 128,
    },
}


# ============================================================
# 数据加载
# ============================================================

def load_data(horizon):
    """加载指定预测步长的预处理数据。

    返回 (X_train, y_train, X_val, y_val, X_test, y_test, scaling)。scaling
    为目标的 z-score 标定参数，用于回逆变换到百分比波动率尺度。
    """
    h_dir = os.path.join(DATA_DIR, f"h{horizon}")
    if not os.path.exists(h_dir):
        raise FileNotFoundError(f"数据目录不存在: {h_dir}")
    arrs = tuple(np.load(os.path.join(h_dir, f)) for f in
                 ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                  "X_test.npy", "y_test.npy"])
    return arrs + (load_target_scaling(DATA_DIR, horizon),)


def make_loader(X, y, batch_size, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.float32)),
                      batch_size=batch_size, shuffle=shuffle)


# ============================================================
# Regime 划分
# ============================================================

def rolling_vol_from_windows(X, input_scaling, window=20):
    """Information-set 20-day realized-volatility proxy at each forecast origin.

    ``X`` contains standardized daily log returns. We invert only the final
    ``window`` observations, all known at the forecast origin, and compute
    ``100*sqrt(mean(r^2))``. This matches Eq. (rolling_vol) and avoids the
    previous error of deriving regimes from the *future h-day target*.
    """
    X = np.asarray(X, dtype=np.float64)[:, :, 0]
    mu, sd = input_scaling
    raw = X[:, -window:] * sd + mu
    return 100.0 * np.sqrt(np.mean(raw ** 2, axis=1))


def split_regime_from_windows(X_train, X_test, input_scaling, window=20):
    """Classify test origins using a training-only threshold."""
    train_vol = rolling_vol_from_windows(X_train, input_scaling, window)
    test_vol = rolling_vol_from_windows(X_test, input_scaling, window)
    threshold = float(np.median(train_vol))
    labels = np.where(test_vol >= threshold, "turbulent", "calm")
    return labels, test_vol, threshold


# ============================================================
# 评估
# ============================================================

def compute_mse(y_true, y_pred, scaling=None, floor=None):
    """MSE，口径与主实验完全一致。

    旧实现直接在传入的数组上算平方误差，而这些数组是标准化尺度上的
    —— 于是 regime 表的 MSE 与正文表格差一个 std^2，且没有正值投影。
    这里改为转发 exp_common.compute_metrics：回逆变换 -> 正值投影 ->
    MSE。比值（turbulent/calm）本来对尺度不敏感，但绝对水平必须对齐。
    """
    return compute_metrics(y_true, y_pred, scaling=scaling,
                           floor=floor)["MSE"]


# 本地 set_seed 已删除：它只固定随机源、不设 cudnn.deterministic，
# 而协议声明的是确定性 cuDNN。统一用 exp_common.set_seed。



def _raw_slug(name):
    return (name.lower().replace("(1,1)", "11").replace(".", "")
            .replace(" ", "_").replace("-", "_").replace("/", "_"))


def _load_saved_forecast(model_name, horizon):
    path = os.path.join(RESULTS_DIR, "raw_predictions",
                        f"h{horizon}__{_raw_slug(model_name)}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Saved forecast artifact not found: {path}. Run main_experiment.py first.")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _moving_block_indices(n, block_len, rng):
    """Circular moving-block bootstrap indices with deterministic length ``n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    block_len = max(1, min(int(block_len), n))
    out = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        out.extend((start + np.arange(block_len)) % n)
    return np.asarray(out[:n], dtype=int)


def _alpha_block_bootstrap(long_a, short_a, vol, labels,
                           n_boot=2000, block_len=20, seed=20260918):
    """Moving-block bootstrap CIs for alpha/volatility interpretation metrics.

    The resampling unit is the forecast-origin time axis.  Circular blocks keep
    local serial dependence in both the operative alpha traces and the
    post-hoc volatility diagnostic.  The same bootstrap indices are applied to
    alpha, volatility and regime labels.
    """
    rng = np.random.default_rng(seed)
    n = len(vol)
    vals = {
        "pearson_long": [], "pearson_short": [],
        "spearman_long": [], "spearman_short": [],
        "calm_long_mean": [], "turbulent_long_mean": [],
        "calm_short_mean": [], "turbulent_short_mean": [],
    }
    for _ in range(int(n_boot)):
        idx = _moving_block_indices(n, block_len, rng)
        la, sa, vv, ll = long_a[idx], short_a[idx], vol[idx], labels[idx]
        calm, turb = ll == "calm", ll == "turbulent"
        # Both states are numerous in the standard test split; guard the
        # generic routine nevertheless so a short custom sample cannot fail.
        if calm.sum() == 0 or turb.sum() == 0:
            continue
        vals["pearson_long"].append(float(np.corrcoef(la, vv)[0, 1]))
        vals["pearson_short"].append(float(np.corrcoef(sa, vv)[0, 1]))
        vals["spearman_long"].append(float(sp_stats.spearmanr(la, vv).statistic))
        vals["spearman_short"].append(float(sp_stats.spearmanr(sa, vv).statistic))
        vals["calm_long_mean"].append(float(la[calm].mean()))
        vals["turbulent_long_mean"].append(float(la[turb].mean()))
        vals["calm_short_mean"].append(float(sa[calm].mean()))
        vals["turbulent_short_mean"].append(float(sa[turb].mean()))

    ci = {}
    for key, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        if len(a) < max(100, n_boot // 4):
            raise RuntimeError(f"Too few valid bootstrap replicates for {key}: {len(a)}")
        ci[f"{key}_low"] = float(np.quantile(a, 0.025))
        ci[f"{key}_high"] = float(np.quantile(a, 0.975))
    ci["bootstrap_replicates"] = int(n_boot)
    ci["bootstrap_block_len"] = int(block_len)
    return ci


def summarize_saved_alpha(alpha_trace, vol_series, regime_labels):
    """Summarize exact operative alpha traces saved by main_experiment.

    Input has shape (seed, date, block, channel).  Channel identity is fixed by
    the grand test-period mean: the lower-mean channel is called ``long`` and
    the higher-mean channel ``short``.  Point estimates average blocks within a
    seed and then average seeds at each forecast origin.  Across-seed standard
    deviations summarize uncertainty of the time-averaged channel means;
    serial-dependence-aware 95% intervals for correlations and regime means use
    a circular moving-block bootstrap over forecast origins.
    """
    a = np.asarray(alpha_trace, dtype=float)
    if a.ndim != 4 or a.shape[-1] != 2:
        raise RuntimeError(f"Expected alpha_trace (seed,date,block,2), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise RuntimeError("Non-finite values in saved alpha trace")

    # (seed, date, channel), then grand mean decides a stable channel naming.
    seed_date_channel = a.mean(axis=2)
    grand = seed_date_channel.mean(axis=(0, 1))
    long_idx, short_idx = [int(i) for i in np.argsort(grand)]
    long_seed_date = seed_date_channel[:, :, long_idx]
    short_seed_date = seed_date_channel[:, :, short_idx]
    long_seed_mean = long_seed_date.mean(axis=1)
    short_seed_mean = short_seed_date.mean(axis=1)

    long_a = long_seed_date.mean(axis=0)
    short_a = short_seed_date.mean(axis=0)
    n = min(len(long_a), len(vol_series), len(regime_labels))
    long_a, short_a = long_a[:n], short_a[:n]
    vol = np.asarray(vol_series, dtype=float)[:n]
    labels = np.asarray(regime_labels)[:n]
    calm, turb = labels == "calm", labels == "turbulent"

    summary = {
        "long_mean": float(long_seed_mean.mean()),
        "short_mean": float(short_seed_mean.mean()),
        "long_seed_sd": float(long_seed_mean.std(ddof=1)),
        "short_seed_sd": float(short_seed_mean.std(ddof=1)),
        "pearson_long": float(np.corrcoef(long_a, vol)[0, 1]),
        "pearson_short": float(np.corrcoef(short_a, vol)[0, 1]),
        "spearman_long": float(sp_stats.spearmanr(long_a, vol).statistic),
        "spearman_short": float(sp_stats.spearmanr(short_a, vol).statistic),
        "calm_long_mean": float(long_a[calm].mean()),
        "turbulent_long_mean": float(long_a[turb].mean()),
        "calm_short_mean": float(short_a[calm].mean()),
        "turbulent_short_mean": float(short_a[turb].mean()),
        "long_channel_index": long_idx, "short_channel_index": short_idx,
        "n_test": n, "n_seeds": int(a.shape[0]), "n_blocks": int(a.shape[2]),
    }
    summary.update(_alpha_block_bootstrap(long_a, short_a, vol, labels))
    return summary, np.column_stack([long_a, short_a])


def alpha_seed_table(alpha_trace):
    """Return one row per seed with time/block-averaged operative channel means."""
    a = np.asarray(alpha_trace, dtype=float)
    if a.ndim != 4 or a.shape[-1] != 2:
        raise RuntimeError(f"Expected alpha_trace (seed,date,block,2), got {a.shape}")
    seed_date_channel = a.mean(axis=2)
    grand = seed_date_channel.mean(axis=(0, 1))
    long_idx, short_idx = [int(i) for i in np.argsort(grand)]
    return pd.DataFrame({
        "seed_index": np.arange(a.shape[0], dtype=int),
        "long_alpha_mean": seed_date_channel[:, :, long_idx].mean(axis=1),
        "short_alpha_mean": seed_date_channel[:, :, short_idx].mean(axis=1),
    })


def run_regime_from_saved(mode):
    """Re-use exact main-experiment forecasts and operative alpha traces."""
    config = MODE_CONFIG[mode]
    seeds, model_names = config["seeds"], config["models"]
    X_train, y_train, X_val, y_val, X_test, y_test, scaling = load_data(HORIZON)
    floor = load_positivity_floor(DATA_DIR, HORIZON)
    input_scaling = load_input_scaling(DATA_DIR)
    regime_labels, vol_series, median_vol = split_regime_from_windows(
        X_train, X_test, input_scaling, window=20)
    calm_mask, turb_mask = regime_labels == "calm", regime_labels == "turbulent"

    all_results, all_preds = {}, {}
    alpha_summary = alpha_date_channel = None
    alpha_seed_df = None
    for model_name in model_names:
        z = _load_saved_forecast(model_name, HORIZON)
        preds = np.asarray(z["predictions_std"], dtype=float)
        if mode == "standard" and preds.shape[0] != len(seeds):
            raise RuntimeError(f"{model_name}: expected {len(seeds)} seeds, got {preds.shape[0]}")
        if not np.all(np.isfinite(preds)):
            raise RuntimeError(f"{model_name}: non-finite saved forecasts")
        y_saved = np.asarray(z["y_test_std"], dtype=float)
        if len(y_saved) != len(y_test) or not np.allclose(y_saved, y_test, atol=1e-12, rtol=0):
            raise RuntimeError(f"{model_name}: saved target differs from current h={HORIZON} test target")
        all_results[model_name] = {"calm": [], "turbulent": [], "ratio": [], "overall": []}
        for pred in preds:
            mc = compute_mse(y_test[calm_mask], pred[calm_mask], scaling=scaling, floor=floor)
            mt = compute_mse(y_test[turb_mask], pred[turb_mask], scaling=scaling, floor=floor)
            mo = compute_mse(y_test, pred, scaling=scaling, floor=floor)
            all_results[model_name]["calm"].append(mc)
            all_results[model_name]["turbulent"].append(mt)
            all_results[model_name]["ratio"].append(mt / mc)
            all_results[model_name]["overall"].append(mo)
        all_preds[model_name] = preds.mean(axis=0)
        if model_name == "AdaFracTCN":
            if "alpha_trace" not in z:
                raise RuntimeError("AdaFracTCN raw artifact has no alpha_trace")
            alpha_summary, alpha_date_channel = summarize_saved_alpha(
                z["alpha_trace"], vol_series, regime_labels)
            alpha_seed_df = alpha_seed_table(z["alpha_trace"])

    return (all_results, all_preds, y_test, vol_series, regime_labels, median_vol,
            model_names, alpha_summary, alpha_date_channel, alpha_seed_df)

def train_and_predict(model_name, horizon, seed, mode,
                      X_train, y_train, X_val, y_val, X_test, y_test, batch_size, device):
    """训练模型并返回预测值。"""
    set_seed(seed)
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    if model_name == "AdaFracTCN":
        model = get_adafractcn("AdaFracTCN", horizon=horizon, mode=mode)
    else:
        model = get_baseline(model_name, horizon=horizon, mode=mode)

    model.fit(train_loader, val_loader, device=device)
    y_pred = model.predict(test_loader, device=device)
    return y_pred, model


def run_regime_experiment(mode):
    """运行 regime 分析实验。"""
    config = MODE_CONFIG[mode]
    seeds = config["seeds"]
    model_names = config["models"]
    batch_size = config["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"设备: {device}")
    print(f"模式: {mode}, h={HORIZON}")
    print(f"模型: {model_names}, 种子: {seeds}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaling = load_data(HORIZON)
    floor = load_positivity_floor(DATA_DIR, HORIZON)
    # 计量/朴素基线需要输入侧与目标侧的 z-score 常数才能把窗口还原成
    # 收益率 / 已实现方差（见 baselines.set_protocol_scaling）。
    input_scaling = load_input_scaling(DATA_DIR)
    set_protocol_scaling(input_scaling=input_scaling, target_scaling=scaling)
    print(f"数据: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, "
          f"正值投影地板 y_min={floor:.6g}（来源：训练切分）")

    # Regime 必须在预测时可知：只用输入窗口最后 20 日收益构造 RV，
    # 阈值取训练 forecast origins 的中位数，再原样应用到测试期。
    regime_labels, vol_series, median_vol = split_regime_from_windows(
        X_train, X_test, input_scaling, window=20)
    calm_mask = regime_labels == "calm"
    turb_mask = regime_labels == "turbulent"
    print(f"Regime 划分: calm={calm_mask.sum()}, turbulent={turb_mask.sum()}, median_vol={median_vol:.4f}")

    # 存储结果: results[model] = {"calm": [...], "turbulent": [...], "ratio": [...], "overall": [...]}
    all_results = {}
    all_preds = {}  # 保存第一个 seed 的预测用于可视化
    best_ada_model = None

    total = len(model_names) * len(seeds)
    current = 0

    for model_name in model_names:
        all_results[model_name] = {"calm": [], "turbulent": [], "ratio": [], "overall": []}

        for seed in seeds:
            current += 1
            print(f"  [{current}/{total}] {model_name:15s} seed={seed}...", end=" ")

            try:
                y_pred, model = train_and_predict(
                    model_name, HORIZON, seed, mode,
                    X_train, y_train, X_val, y_val, X_test, y_test, batch_size, device
                )

                # 分 regime 计算 MSE。三处共用同一个地板，因此两组的
                # 口径一致，pooled 值也可由两组的加权平均还原。
                mse_calm = compute_mse(y_test[calm_mask], y_pred[calm_mask],
                                       scaling=scaling, floor=floor)
                mse_turb = compute_mse(y_test[turb_mask], y_pred[turb_mask],
                                       scaling=scaling, floor=floor)
                mse_overall = compute_mse(y_test, y_pred,
                                          scaling=scaling, floor=floor)
                ratio = mse_turb / mse_calm if mse_calm > 0 else np.nan

                all_results[model_name]["calm"].append(mse_calm)
                all_results[model_name]["turbulent"].append(mse_turb)
                all_results[model_name]["ratio"].append(ratio)
                all_results[model_name]["overall"].append(mse_overall)

                # 保存第一个 seed 的预测
                if seed == seeds[0]:
                    all_preds[model_name] = y_pred
                    if model_name == "AdaFracTCN":
                        best_ada_model = model

                print(f"calm={mse_calm:.4f} turb={mse_turb:.4f} ratio={ratio:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")
                for k in all_results[model_name]:
                    all_results[model_name][k].append(np.nan)

    return all_results, all_preds, y_test, vol_series, regime_labels, median_vol, model_names, best_ada_model


# ============================================================
# Alpha 轨迹提取
# ============================================================

def extract_alpha_trajectory(model, X_test, device="cpu", return_blocks=False):
    """Extract the operative adaptive orders from the prediction forward pass.

    For the full model ``forward_with_alpha`` returns ``(B,L_net,2)``. By
    default we average over blocks for each forecast origin and return ``(T,2)``;
    ``return_blocks=True`` returns the full ``(T,L_net,2)`` trace. No alpha
    network is re-evaluated out of context, so these values are exactly those
    used to form the GL kernels that produced each prediction.
    """
    if model is None or not getattr(model, "adaptive_alpha", False):
        return np.full((len(X_test), 1), np.nan)
    if not hasattr(model, "forward_with_alpha"):
        raise RuntimeError("model lacks forward_with_alpha; operative alpha cannot be audited")

    model.eval()
    model.to(device)
    traces = []
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_tensor), 128):
            batch = X_tensor[i:i + 128].to(device)
            _, trace = model.forward_with_alpha(batch)
            traces.append(trace.cpu().numpy())
    trace = np.concatenate(traces, axis=0)[:len(X_test)]
    return trace if return_blocks else trace.mean(axis=1)


def compute_alpha_by_regime(ada_model, X_test, regime_labels, device="cpu"):
    """按市场状态（calm/turbulent）统计自适应 alpha 的均值/标准差（供 Table 19）。

    返回 dict: {"calm_mean":..,"calm_std":..,"turbulent_mean":..,"turbulent_std":..}
    双通道模型下每个统计量同时给出两条通道的均值（列表长度为 2），
    单通道模型下为标量；未启用 adaptive_alpha 时返回 NaN。
    """
    empty = {"calm_mean": np.nan, "calm_std": np.nan,
             "turbulent_mean": np.nan, "turbulent_std": np.nan}
    if ada_model is None or not ada_model.adaptive_alpha:
        return empty

    alphas = np.asarray(extract_alpha_trajectory(ada_model, X_test, device))
    labels = np.asarray(regime_labels)
    n = min(len(alphas), len(labels))
    alphas, labels = alphas[:n], labels[:n]

    calm = alphas[labels == "calm"]
    turb = alphas[labels == "turbulent"]

    def _stat(arr, fn):
        if not arr.size:
            return np.nan
        vals = fn(arr, axis=0)
        return float(vals) if np.ndim(vals) == 0 else [float(v) for v in vals]

    result = {
        "calm_mean": _stat(calm, np.mean),
        "calm_std": _stat(calm, np.std),
        "turbulent_mean": _stat(turb, np.mean),
        "turbulent_std": _stat(turb, np.std),
    }
    return result


# ============================================================
# 结果保存
# ============================================================

def save_regime_results(all_results, model_names):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for name in model_names:
        vals = all_results[name]
        calm = np.nanmean(vals["calm"])
        turb = np.nanmean(vals["turbulent"])
        ratio = np.nanmean(vals["ratio"])
        overall = np.nanmean(vals["overall"])
        rows.append({
            "Model": name,
            "Calm_MSE": calm,
            "Turbulent_MSE": turb,
            "Ratio_Tur_Calm": ratio,
            "Overall_MSE": overall,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "regime_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("Table: 市场 Regime 分析结果")
    print("=" * 70)
    print(f"{'Model':<15} {'Calm_MSE':<12} {'Turb_MSE':<12} {'Ratio':<10} {'Overall':<12}")
    print("-" * 61)
    for _, row in df.iterrows():
        print(f"{row['Model']:<15} {row['Calm_MSE']:<12.6f} {row['Turbulent_MSE']:<12.6f} {row['Ratio_Tur_Calm']:<10.4f} {row['Overall_MSE']:<12.6f}")

    print(f"\n结果已保存: {csv_path}")
    return df







# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="市场 Regime 分析")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "quick"], help="运行模式")
    parser.add_argument("--source", type=str, default="saved",
                        choices=["saved", "train"],
                        help="saved: reuse main_experiment raw forecasts (recommended); train: retrain")
    args = parser.parse_args()

    print("=" * 60)
    print("  市场 Regime 分析")
    print("=" * 60)
    start_time = time.time()

    if args.source == "saved":
        (all_results, all_preds, y_test, vol_series, regime_labels, median_vol,
         model_names, alpha_summary, alpha_date_channel, alpha_seed_df) = run_regime_from_saved(args.mode)
        ada_model = None
    else:
        all_results, all_preds, y_test, vol_series, regime_labels, median_vol, model_names, ada_model = \
            run_regime_experiment(args.mode)
        alpha_summary = alpha_date_channel = None
        alpha_seed_df = None

    save_regime_results(all_results, model_names)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.source == "saved":
        if alpha_summary is None:
            raise RuntimeError("No saved AdaFracTCN alpha summary")
        pd.DataFrame([alpha_summary]).to_csv(
            os.path.join(RESULTS_DIR, "regime_alpha_summary.csv"), index=False)
        if alpha_seed_df is None:
            raise RuntimeError("No seed-level operative alpha summary")
        alpha_seed_df.to_csv(os.path.join(RESULTS_DIR, "regime_alpha_by_seed.csv"), index=False)
        pd.DataFrame({
            "regime": ["calm", "turbulent"],
            "long_alpha_mean": [alpha_summary["calm_long_mean"], alpha_summary["turbulent_long_mean"]],
            "short_alpha_mean": [alpha_summary["calm_short_mean"], alpha_summary["turbulent_short_mean"]],
        }).to_csv(os.path.join(RESULTS_DIR, "regime_alpha_by_state.csv"), index=False)
        print("Alpha summary:", alpha_summary)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_train, y_train, X_val, y_val, X_test2, _, _ = load_data(HORIZON)
        alpha_by_regime = compute_alpha_by_regime(ada_model, X_test2, regime_labels, device)
        pd.DataFrame([alpha_by_regime]).to_csv(
            os.path.join(RESULTS_DIR, "regime_alpha_by_state.csv"), index=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("  Regime 分析完成！")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
