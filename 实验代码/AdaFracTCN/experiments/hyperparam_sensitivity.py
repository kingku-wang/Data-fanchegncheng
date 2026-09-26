# -*- coding: utf-8 -*-
"""
hyperparam_sensitivity.py — 超参数敏感性分析
==============================================
分析截断长度 M、网络层数 L_net、卷积核大小 K，以及固定单通道阶 α 对验证集预测精度的影响。
所有敏感性数值只在 validation split 上计算，test split 不参与超参数选择。
生成论文 Figure 5 的数据。

输出:
  - results/sensitivity_M.csv     截断长度敏感性
  - results/sensitivity_L.csv     网络层数敏感性
  - results/sensitivity_K.csv     卷积核大小敏感性
  - results/sensitivity_alpha.csv 固定单通道 α 敏感性
  - figures/hyperparam_sensitivity.pdf  2×2 四面板可视化

用法:
    python hyperparam_sensitivity.py --mode standard
    python hyperparam_sensitivity.py --mode quick
"""
import argparse
import os
import time
import warnings

# 解决 OpenMP 重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# 本地 import
from adafractcn import AdaFracTCN
import fig_style
from exp_common import (load_target_scaling, compute_metrics,
                        load_positivity_floor, set_seed)

# ============================================================
# 全局配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = fig_style.DATA_DIR
RESULTS_DIR = fig_style.RESULTS_DIR
FIGURES_DIR = fig_style.FIGURES_DIR

# 固定预测步长 h=10
HORIZON = 10

MODE_CONFIG = {
    "standard": {
        "seeds": [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
        "M_values": [32, 48, 64, 96, 128, 256],
        "L_values": [2, 3, 4, 6],
        "K_values": [3, 5, 7, 9],
        "alpha_values": [0.1, 0.3, 0.5, 0.7, 0.9],
        "batch_size": 256,
        # 默认值（对齐正文 Table 12 与 AdaFracTCN 标准配置）
        "default_M": 64,
        "default_L": 4,
        "default_K": 5,
        "default_alpha": 0.5,
    },
    "quick": {
        "seeds": [42],
        "M_values": [32, 128],
        "L_values": [3, 4],
        "K_values": [3, 5],
        "alpha_values": [0.2, 0.5, 0.8],
        "batch_size": 128,
        "default_M": 8,
        "default_L": 2,
        "default_K": 5,
        "default_alpha": 0.5,
    },
}

# 协议参考超参（用于感受野 / 误差界推导）：M=64, K=5, L_net=4。
# 感受野一律按论文口径 R_eff = 1 + (M+K-2)(2^L_net - 1) 计算，
# 其中 M+K-2 是带包络卷积层的每层跨度（有效核长度为 M+K-1）。
M_REF = 64
K_REF = 5
L_REF = 4


# ============================================================
# 数据加载
# ============================================================

def load_data(horizon):
    """加载指定预测步长的预处理数据。"""
    h_dir = os.path.join(DATA_DIR, f"h{horizon}")
    if not os.path.exists(h_dir):
        raise FileNotFoundError(f"数据目录不存在: {h_dir}，请先运行 download_data.py")

    X_train = np.load(os.path.join(h_dir, "X_train.npy"))
    y_train = np.load(os.path.join(h_dir, "y_train.npy"))
    X_val = np.load(os.path.join(h_dir, "X_val.npy"))
    y_val = np.load(os.path.join(h_dir, "y_val.npy"))
    X_test = np.load(os.path.join(h_dir, "X_test.npy"))
    y_test = np.load(os.path.join(h_dir, "y_test.npy"))

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            load_target_scaling(DATA_DIR, horizon))


def make_loader(X, y, batch_size, shuffle=False):
    """创建 DataLoader。"""
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# 评估指标
# ============================================================

# 指标口径统一收敛到 exp_common.compute_metrics（回逆变换 + 正值投影
# + QLIKE），本模块不再自带一份实现。
#
# 这里原先有一个两参数的本地 compute_metrics，它遮蔽了上面的 import，
# 而调用处写的是 compute_metrics(y_test, y_pred, scaling=scaling)，
# 于是这段代码运行到那里必然 TypeError。删除本地副本既修掉这个缺陷，
# 也让本脚本的 MSE 与主实验、消融表落在同一口径上。


# ============================================================
# 实验运行
# ============================================================

# 本地 set_seed 已删除：它只固定随机源、不设 cudnn.deterministic，
# 而协议声明的是确定性 cuDNN。统一用 exp_common.set_seed。


def run_single_config(mode, X_train, y_train, X_val, y_val, X_test, y_test,
                      batch_size, device, seed, scaling=(0.0, 1.0),
                      truncation=None, num_layers=None, kernel_size=None,
                      fixed_alpha=None, floor=None):
    """Train one configuration and score it on validation only.

    The test arrays remain in the public signature for compatibility with older
    callers but are not loaded into a DataLoader or used for configuration
    ranking. ``scaling`` maps validation metrics to the reported target scale.
    """
    set_seed(seed)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    # M/L_net/K 扫描使用完整 AdaFracTCN；α 扫描按正文使用固定单通道变体。
    if fixed_alpha is None:
        kwargs = dict(
            horizon=HORIZON, mode=mode,
            use_fractional=True, learnable_alpha=True,
            dual_channel=True, adaptive_alpha=True,
        )
    else:
        kwargs = dict(
            horizon=HORIZON, mode=mode,
            use_fractional=True, learnable_alpha=False,
            dual_channel=False, adaptive_alpha=False,
            fixed_alpha=float(fixed_alpha),
        )
    if truncation is not None:
        kwargs["truncation"] = truncation
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    if kernel_size is not None:
        kwargs["kernel_size"] = kernel_size

    model = AdaFracTCN(**kwargs)

    # 训练（测量时间）
    start_time = time.time()
    model.fit(train_loader, val_loader, device=device)
    train_time = time.time() - start_time

    # Validation-only sensitivity: test arrays are not used.
    y_pred = model.predict(val_loader, device=device)
    metrics = compute_metrics(y_val, y_pred, scaling=scaling, floor=floor)
    metrics["selection_split"] = "validation"
    metrics["train_time"] = train_time
    metrics["params"] = model.count_parameters()

    return metrics


def run_sensitivity_sweep(param_name, param_values, mode,
                          X_train, y_train, X_val, y_val, X_test, y_test,
                          batch_size, device, seeds, scaling=(0.0, 1.0),
                          floor=None):
    """对单个超参数进行扫描。

    scaling: 目标 z-score 标定参数，向下传给 run_single_config 以统一指标口径。
    """
    print(f"\n--- {param_name} 敏感性扫描 ---")
    print(f"  {param_name} 取值: {param_values}")
    print(f"  种子: {seeds}")

    results = []
    total = len(param_values) * len(seeds)
    current = 0

    for val in param_values:
        mse_list = []
        rmse_list = []
        time_list = []
        param_list = []

        for seed in seeds:
            current += 1
            print(f"  [{current}/{total}] {param_name}={val} seed={seed}...", end=" ")

            try:
                # 根据参数名设置 kwargs
                kwargs = {}
                if param_name == "M":
                    kwargs["truncation"] = val
                elif param_name == "L_net":
                    kwargs["num_layers"] = val
                elif param_name == "K":
                    kwargs["kernel_size"] = val
                elif param_name == "alpha":
                    kwargs["fixed_alpha"] = val

                metrics = run_single_config(
                    mode, X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size, device, seed, scaling=scaling,
                    floor=floor, **kwargs
                )
                mse_list.append(metrics["MSE"])
                rmse_list.append(metrics["RMSE"])
                time_list.append(metrics["train_time"])
                param_list.append(metrics["params"])
                print(f"MSE={metrics['MSE']:.6f}")
            except Exception as e:
                print(f"ERROR: {e}")
                mse_list.append(np.nan)
                rmse_list.append(np.nan)
                time_list.append(np.nan)
                param_list.append(np.nan)

        # 计算有效核长度 / 感受野（协议口径）
        # 感受野一律按论文口径：R_eff = 1 + (M+K-2)(2^L_net - 1)。
        # 旧代码在这里用的是 1 + (M-1)(2^L-1)，把每层跨度写成 M-1，
        # 与正文（每层跨度 M+K-2，K=5）不一致，会低估约 30%。
        if param_name == "L_net":
            receptive_field = 1 + (M_REF + K_REF - 2) * (2 ** val - 1)
        elif param_name == "M":
            receptive_field = 1 + (val + K_REF - 2) * (2 ** L_REF - 1)
        elif param_name == "K":
            receptive_field = 1 + (M_REF + val - 2) * (2 ** L_REF - 1)
        else:
            receptive_field = None

        row = {
            param_name: val,
            "selection_split": "validation",
            "MSE_mean": np.nanmean(mse_list),
            "MSE_std": np.nanstd(mse_list, ddof=1),
            "RMSE_mean": np.nanmean(rmse_list),
            "RMSE_std": np.nanstd(rmse_list, ddof=1),
            "time_per_epoch": np.nanmean(time_list) / (80 if mode == "quick" else 300),
            "total_train_time": np.nanmean(time_list),
            "Parameters": param_list[0] if param_list else 0,
        }
        if receptive_field is not None:
            row["ReceptiveField"] = receptive_field

        # 对 M 参数添加理论误差界：bound = (N-1)^(-alpha)/alpha，
        # 其中 N = M + K - 1（有效核长度），alpha=0.5，即 2*(M+K-2)^-0.5。
        # 代入协议参考 K=5：2*(M+3)^-0.5。
        if param_name == "M":
            row["TheoreticalBound"] = 2.0 * (val + K_REF - 2) ** (-0.5)

        results.append(row)

    return results


# ============================================================
# 结果保存
# ============================================================

def save_results(results, param_name, filename):
    """保存结果到 CSV。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(csv_path, index=False)

    print(f"\n  {param_name} 敏感性结果:")
    print(f"  {'Value':<10} {'MSE (mean±std)':<25} {'Time/epoch':<12} {'Params':<12}")
    print("  " + "-" * 59)
    for _, row in df.iterrows():
        val = row[param_name]
        mse_str = f"{row['MSE_mean']:.6f}±{row['MSE_std']:.6f}"
        time_str = f"{row['time_per_epoch']:.2f}s" if not np.isnan(row['time_per_epoch']) else "---"
        param_str = f"{int(row['Parameters']):,}" if not np.isnan(row['Parameters']) else "---"
        print(f"  {val:<10} {mse_str:<25} {time_str:<12} {param_str:<12}")

    print(f"\n  结果已保存: {csv_path}")
    return df


# ============================================================
# 可视化（PDF 矢量图）
# ============================================================

def plot_hyperparam_sensitivity(df_M, df_L, df_K, df_alpha, mode):
    """Plot the four validation-only sensitivity panels reported in the manuscript."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_style.apply(serif=True)

    config = MODE_CONFIG[mode]
    panels = [
        (df_M, "M", config["default_M"], "(a) Truncation length", r"Truncation length $M$"),
        (df_L, "L_net", config["default_L"], "(b) Network depth", r"Network depth $L_{net}$"),
        (df_K, "K", config["default_K"], "(c) Short-kernel width", r"Short-kernel width $K$"),
        (df_alpha, "alpha", config["default_alpha"], "(d) Fixed fractional order", r"Fixed fractional order $\alpha$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for ax, (df, key, default, title, xlabel) in zip(axes.flat, panels):
        x = np.asarray(df[key], dtype=float)
        y = np.asarray(df["MSE_mean"], dtype=float)
        ax.plot(x, y, "o-", linewidth=2.0, markersize=5.5)
        ax.axvline(default, linestyle="--", linewidth=1.0, alpha=0.7)
        valid = np.isfinite(y)
        if valid.any():
            best_local = np.where(valid)[0][np.argmin(y[valid])]
            ax.scatter([x[best_local]], [y[best_local]], s=55, zorder=6)
            if np.isclose(x[best_local], default):
                ax.annotate("Default / best", xy=(x[best_local], y[best_local]),
                            xytext=(6, 8), textcoords="offset points", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation MSE")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.22, linestyle="--")

    fig.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "hyperparam_sensitivity.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n图表已保存: {pdf_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="超参数敏感性分析")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "quick"],
                        help="运行模式")
    args = parser.parse_args()

    print("=" * 60)
    print("  超参数敏感性分析")
    print("=" * 60)

    config = MODE_CONFIG[args.mode]
    seeds = config["seeds"]
    batch_size = config["batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"模式: {args.mode}")
    print(f"预测步长: h={HORIZON}")
    print(f"种子: {seeds}")
    print(f"M 取值: {config['M_values']}")
    print(f"L_net 取值: {config['L_values']}")
    print(f"K 取值: {config['K_values']}")
    print(f"alpha 取值: {config['alpha_values']}")

    start_time = time.time()

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaling = load_data(HORIZON)
    floor = load_positivity_floor(DATA_DIR, HORIZON)
    print(f"数据: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, "
          f"正值投影地板 y_min={floor}")

    # ---- 1. 截断长度 M 敏感性 ----
    print(f"\n{'='*60}")
    print("Part 1: 截断长度 M 敏感性分析")
    print(f"{'='*60}")

    results_M = run_sensitivity_sweep(
        "M", config["M_values"], args.mode,
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size, device, seeds, scaling=scaling, floor=floor
    )
    df_M = save_results(results_M, "M", "sensitivity_M.csv")

    # ---- 2. 网络层数 L_net 敏感性 ----
    print(f"\n{'='*60}")
    print("Part 2: 网络层数 L_net 敏感性分析")
    print(f"{'='*60}")

    results_L = run_sensitivity_sweep(
        "L_net", config["L_values"], args.mode,
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size, device, seeds, scaling=scaling, floor=floor
    )
    df_L = save_results(results_L, "L_net", "sensitivity_L.csv")

    # ---- 3. 卷积核大小 K 敏感性 ----
    print(f"\n{'='*60}")
    print("Part 3: 卷积核大小 K 敏感性分析")
    print(f"{'='*60}")

    results_K = run_sensitivity_sweep(
        "K", config["K_values"], args.mode,
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size, device, seeds, scaling=scaling, floor=floor
    )
    df_K = save_results(results_K, "K", "sensitivity_K.csv")

    # ---- 4. 固定单通道 fractional order alpha 敏感性 ----
    print(f"\n{'='*60}")
    print("Part 4: 固定单通道 alpha 敏感性分析")
    print(f"{'='*60}")

    results_alpha = run_sensitivity_sweep(
        "alpha", config["alpha_values"], args.mode,
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size, device, seeds, scaling=scaling, floor=floor
    )
    df_alpha = save_results(results_alpha, "alpha", "sensitivity_alpha.csv")

    # ---- 5. 可视化 ----
    print(f"\n{'='*60}")
    print("Part 5: 生成可视化图表")
    print(f"{'='*60}")

    plot_hyperparam_sensitivity(df_M, df_L, df_K, df_alpha, args.mode)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("  超参数敏感性分析完成！")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"  图表目录: {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
