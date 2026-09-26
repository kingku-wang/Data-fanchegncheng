# -*- coding: utf-8 -*-
"""
ablation_experiment.py — 消融实验
====================================
验证 AdaFracTCN 各组件的增量贡献：分数阶卷积、可学习 alpha、双通道、自适应 alpha(t)。
生成论文 Table 4 和 Figure 3 的数据。

输出:
  - results/ablation_results.csv
  - results/ablation_perseed.csv
  - results/ablation_paired_ci.csv
  - results/ablation_raw_predictions/  (standard mode)

用法:
    python ablation_experiment.py --mode standard
    python ablation_experiment.py --mode quick
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
from scipy import stats as sp_stats
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# 本地 import
from adafractcn import get_adafractcn, AdaFracTCN, ABLATION_VARIANTS, FractionalConv1d
import fig_style
from exp_common import (load_target_scaling, compute_metrics,
                        load_positivity_floor, sample_std, set_seed)

# ============================================================
# 全局配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = fig_style.DATA_DIR
RESULTS_DIR = fig_style.RESULTS_DIR

MODE_CONFIG = {
    "standard": {
        "horizon": 10,
        "seeds": [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
        # 完整阶梯 + 两个对照（E1 感受野匹配 TCN、E2 同包络非分数阶）。
        # 顺序即 Table 5 的行序：先对照，再阶梯，由浅到深。
        "variants": list(ABLATION_VARIANTS),
        "batch_size": 256,
    },
    "quick": {
        "horizon": 5,
        "seeds": [42],
        "variants": ["TCN", "AdaFracTCN"],
        "batch_size": 128,
    },
}


# ============================================================
# 数据加载（复用 main_experiment 的逻辑）
# ============================================================

def load_data(horizon):
    """加载指定预测步长的预处理数据，并附带目标标定参数。

    返回 (X_train, y_train, X_val, y_val, X_test, y_test, scaling)，
    scaling=(mean, std) 用于把预测/真值回逆变换到百分比波动率尺度。
    """
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
# 注：指标实现统一收敛到 exp_common.compute_metrics（含回逆变换与 QLIKE），
# 本模块不再重复定义，避免各脚本口径漂移。


# ============================================================
# 实验运行
# ============================================================
# 注：set_seed 统一使用 exp_common.set_seed（含 cudnn 确定性设置）。


def run_variant(variant_name, horizon, seed, mode,
                X_train, y_train, X_val, y_val, X_test, y_test, batch_size,
                device, scaling=(0.0, 1.0), floor=None):
    """运行单个消融变体的训练和测试。

    floor: 正值投影地板 y_min（来自训练切分，与模型无关）。消融表里的所有
    变体共用同一个地板，MSE 也在投影后的预测上计算，口径与主实验一致。
    """
    set_seed(seed)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    # 初始化对应变体
    model = get_adafractcn(variant_name, horizon=horizon, mode=mode)

    # 训练
    model.fit(train_loader, val_loader, device=device)

    # 预测
    y_pred = model.predict(test_loader, device=device)

    # 指标：先回逆变换到百分比波动率尺度，再施加与其它变体相同的正值投影
    metrics = compute_metrics(y_test, y_pred, scaling=scaling, floor=floor)
    metrics["params"] = model.count_parameters()
    metrics["receptive_field"] = model.receptive_field

    # Record the actual learned global fractional orders, if the variant has
    # non-adaptive learnable-alpha layers. There can be one alpha per block;
    # reporting the layer mean/min/max avoids pretending they are a single
    # shared scalar. Adaptive alpha(t) is stored by the main experiment as a
    # per-date trace and is intentionally not collapsed here.
    learned = []
    for layer in model.modules():
        if isinstance(layer, FractionalConv1d) and layer.learnable_alpha and not layer.adaptive_alpha:
            learned.append(float(layer.alpha.detach().cpu().item()))
    metrics["alpha_mean"] = float(np.mean(learned)) if learned else np.nan
    metrics["alpha_min"] = float(np.min(learned)) if learned else np.nan
    metrics["alpha_max"] = float(np.max(learned)) if learned else np.nan

    return metrics, y_pred


def run_ablation_experiment(mode):
    """运行消融对比实验。"""
    config = MODE_CONFIG[mode]
    horizon = config["horizon"]
    seeds = config["seeds"]
    variants = config["variants"]
    batch_size = config["batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"模式: {mode}")
    print(f"预测步长: h={horizon}")
    print(f"消融变体: {variants}")
    print(f"种子: {seeds}")

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaling = load_data(horizon)
    floor = load_positivity_floor(DATA_DIR, horizon)
    print(f"数据: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, "
          f"正值投影地板 y_min={floor}")

    # 存储结果: results[variant] = {"MSE": [...], "RMSE": [...], ...}
    all_results = {}
    all_preds = {}
    total = len(variants) * len(seeds)
    current = 0

    for variant in variants:
        all_results[variant] = {
            k: [] for k in ["MSE", "RMSE", "MAE", "R2", "params",
                            "receptive_field", "n_projected", "frac_projected"]}
        all_preds[variant] = []

        for seed in seeds:
            current += 1
            print(f"  [{current}/{total}] {variant:20s} seed={seed}...", end=" ")

            try:
                metrics, y_pred = run_variant(
                    variant, horizon, seed, mode,
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size, device, scaling=scaling, floor=floor
                )
                for k in all_results[variant]:
                    all_results[variant][k].append(metrics[k])
                all_preds[variant].append(y_pred)
                print(f"MSE={metrics['MSE']:.6f}")
            except Exception as e:
                print(f"ERROR: {e}")
                for k in all_results[variant]:
                    all_results[variant][k].append(np.nan)
                all_preds[variant].append(np.full_like(y_test, np.nan))

    return all_results, all_preds, variants, y_test, scaling, floor


def save_ablation_raw_predictions(all_preds, y_test, scaling, floor, seeds, horizon):
    """Persist per-seed ablation forecasts for time-axis inference.

    The saved arrays stay on the standardized target scale, matching the main
    raw-prediction artifacts.  Follow-up scripts invert and apply the common
    training-only positivity floor before constructing date-wise loss
    differentials.
    """
    raw_dir = os.path.join(RESULTS_DIR, "ablation_raw_predictions")
    os.makedirs(raw_dir, exist_ok=True)
    dates = np.load(os.path.join(DATA_DIR, f"h{horizon}", "origin_dates_test.npy"))
    rows = []
    for variant, preds in all_preds.items():
        arr = np.stack(preds, axis=0).astype(np.float32)
        path = os.path.join(raw_dir, f"h{horizon}__{variant.lower().replace('-', '_')}.npz")
        np.savez_compressed(path, predictions_std=arr, y_test_std=np.asarray(y_test, np.float32),
                            origin_dates=dates, target_mean=float(scaling[0]),
                            target_std=float(scaling[1]), positivity_floor=float(floor),
                            seeds=np.asarray(seeds, dtype=np.int64), variant=variant)
        rows.append({"variant": variant, "horizon": int(horizon), "file": os.path.basename(path),
                     "n_seeds": int(arr.shape[0]), "n_test": int(arr.shape[1])})
    pd.DataFrame(rows).to_csv(os.path.join(raw_dir, "index.csv"), index=False)




# ============================================================
# 结果保存
# ============================================================

def _bca_interval(x, n_boot=20000, alpha=0.05, seed=20260918):
    """BCa bootstrap 区间（E3）。x 为逐种子的配对差 Δ_i。

    返回 (lo, hi, theta, z0, a)：偏差校正 z0 与加速度 a 都由 jackknife
    估计，只用到 numpy 与 scipy.stats.norm，不引入额外依赖。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    theta = float(np.mean(x))
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(x, size=n, replace=True))
                      for _ in range(n_boot)])
    prop = float(np.mean(boots < theta))
    z0 = float(sp_stats.norm.ppf(prop)) if 0.0 < prop < 1.0 else 0.0
    jk = np.array([np.mean(np.delete(x, i)) for i in range(n)])
    jk_mean = jk.mean()
    num = float(np.sum((jk_mean - jk) ** 3))
    den = float(6.0 * np.sum((jk_mean - jk) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    def _adj(z):
        zz = z0 + z
        return float(sp_stats.norm.cdf(z0 + zz / (1.0 - a * zz)))

    lo_q = _adj(sp_stats.norm.ppf(alpha / 2.0))
    hi_q = _adj(sp_stats.norm.ppf(1.0 - alpha / 2.0))
    return (float(np.quantile(boots, lo_q)), float(np.quantile(boots, hi_q)),
            theta, z0, a)


def paired_difference_interval(mse_tcn, mse_variant, alpha=0.05):
    """Δ_i = MSE_TCN,i − MSE_variant,i 的两种 95% 区间与配对 t 的 p 值。

    为什么必须算这个：Table 5 那一列区间是用 s_Δ ≤ s_1 + s_2 构造的
    **最宽相容区间**，它只保证把真区间包在里面，因此"含零"只能读成
    inconclusive，不能读成"无差异"。这里给出真正的 seed 级配对统计量
    —— 同一批 seed、逐 i 配对 —— 用来把含零的那一档结案。

    返回 exact_*（Student-t 精确配对区间）与 bca_*（BCa bootstrap 区间）；
    两者都基于同一批 Δ_i，正文按需要引用其一。
    """
    a = np.asarray(mse_tcn, dtype=np.float64)
    b = np.asarray(mse_variant, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return dict(mean_delta=np.nan, exact_low=np.nan, exact_high=np.nan,
                    bca_low=np.nan, bca_high=np.nan, p_exact=np.nan,
                    n_pairs=int(valid.sum()), t_stat=np.nan)
    d = a[valid] - b[valid]
    n = d.size
    mean_d = float(d.mean())
    se = float(d.std(ddof=1)) / np.sqrt(n)
    tcrit = float(sp_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    t_stat, p_exact = sp_stats.ttest_rel(a[valid], b[valid])
    bca_lo, bca_hi, _, _, _ = _bca_interval(d)
    return dict(mean_delta=mean_d,
                exact_low=mean_d - tcrit * se,
                exact_high=mean_d + tcrit * se,
                bca_low=bca_lo, bca_high=bca_hi,
                p_exact=float(p_exact), n_pairs=int(n),
                t_stat=float(t_stat))


def save_ablation_results(all_results, variants, seeds=None):
    """保存消融对比结果到 CSV（含与 TCN 的配对检验 p 值与 seed-noise floor）。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 获取 TCN baseline 的逐种子 MSE（用于配对 t 检验）
    tcn_mse = np.asarray(all_results["TCN"]["MSE"], dtype=np.float64) if "TCN" in all_results else None
    baseline_mse = np.nanmean(tcn_mse) if tcn_mse is not None else np.nan

    rows = []
    for variant in variants:
        vals = all_results[variant]
        mse_mean = np.nanmean(vals["MSE"])
        mse_std = sample_std(vals["MSE"])
        rmse_mean = np.nanmean(vals["RMSE"])
        rmse_std = sample_std(vals["RMSE"])
        mae_mean = np.nanmean(vals["MAE"])
        mae_std = sample_std(vals["MAE"])
        r2_mean = np.nanmean(vals["R2"])
        r2_std = sample_std(vals["R2"])
        params = vals["params"][0] if vals["params"] else 0
        rf = (int(vals["receptive_field"][0])
              if vals.get("receptive_field") else -1)
        n_proj = (int(np.nanmean(vals["n_projected"]))
                  if vals.get("n_projected") else 0)
        frac_proj = (float(np.nanmean(vals["frac_projected"]) * 100)
                     if vals.get("frac_projected") else np.nan)

        # 改进百分比
        if not np.isnan(baseline_mse) and not np.isnan(mse_mean) and baseline_mse > 0:
            improvement = (baseline_mse - mse_mean) / baseline_mse * 100
        else:
            improvement = np.nan

        # 与 TCN 基线的 seed 级配对统计（E3）：Δ_i = MSE_TCN,i - MSE_variant,i。
        # 同一批 seed 逐 i 配对，给出的才是真区间；Table 5 上那列保守界
        # 只是它的外包络。
        paired_p = np.nan
        ci = dict(mean_delta=np.nan, exact_low=np.nan, exact_high=np.nan,
                  bca_low=np.nan, bca_high=np.nan, p_exact=np.nan,
                  n_pairs=0, t_stat=np.nan)
        if variant != "TCN" and tcn_mse is not None:
            ci = paired_difference_interval(tcn_mse, vals["MSE"])
            paired_p = ci["p_exact"]

        rows.append({
            "Variant": variant,
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
            "RMSE_mean": rmse_mean,
            "RMSE_std": rmse_std,
            "MAE_mean": mae_mean,
            "MAE_std": mae_std,
            "R2_mean": r2_mean,
            "R2_std": r2_std,
            "Improvement_pct": improvement,
            "TCN_paired_p": paired_p,
            "Parameters": params,
            "Receptive_field": rf,
            "N_projected": n_proj,
            "Frac_projected_pct": frac_proj,
            "Delta_mean": ci["mean_delta"],
            "Delta_exact_low": ci["exact_low"],
            "Delta_exact_high": ci["exact_high"],
            "Delta_bca_low": ci["bca_low"],
            "Delta_bca_high": ci["bca_high"],
            "Delta_n_pairs": ci["n_pairs"],
            "Alpha_mean": float(np.nanmean(vals.get("alpha_mean", [np.nan]))),
            "Alpha_min": float(np.nanmin(vals.get("alpha_min", [np.nan]))) if np.any(np.isfinite(vals.get("alpha_min", [np.nan]))) else np.nan,
            "Alpha_max": float(np.nanmax(vals.get("alpha_max", [np.nan]))) if np.any(np.isfinite(vals.get("alpha_max", [np.nan]))) else np.nan,
        })

    # ---- seed-noise floor：所有变体 MSE 跨种子 std 的最大值 ----
    if variants:
        seed_noise_floor = float(np.nanmax([sample_std(all_results[v]["MSE"]) for v in variants]))
    else:
        seed_noise_floor = np.nan
    rows.append({
        "Variant": "seed-noise floor",
        "MSE_mean": np.nan,
        "MSE_std": seed_noise_floor,
        "RMSE_mean": np.nan,
        "RMSE_std": np.nan,
        "MAE_mean": np.nan,
        "MAE_std": np.nan,
        "R2_mean": np.nan,
        "R2_std": np.nan,
        "Improvement_pct": np.nan,
        "TCN_paired_p": np.nan,
        "Parameters": np.nan,
    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    df.to_csv(csv_path, index=False)

    # ---- 逐种子 MSE：表里只给 mean±std，而配对检验复核需要原始的 Δ_i，
    #      因此把每个种子的分量单独落盘，不让读者从 std 反推 Δ_i 的分布 ----
    rows_ps = []
    if seeds is None:
        seeds = list(range(max((len(all_results[v]["MSE"]) for v in variants), default=0)))
    for variant in variants:
        n_proj = all_results[variant].get("n_projected")
        for i, m in enumerate(all_results[variant]["MSE"]):
            rows_ps.append({
                "Variant": variant,
                "Seed": seeds[i] if i < len(seeds) else i,
                "MSE": m,
                "n_projected": (n_proj[i] if n_proj else np.nan),
                "alpha_mean": all_results[variant].get("alpha_mean", [np.nan] * len(all_results[variant]["MSE"]))[i],
                "alpha_min": all_results[variant].get("alpha_min", [np.nan] * len(all_results[variant]["MSE"]))[i],
                "alpha_max": all_results[variant].get("alpha_max", [np.nan] * len(all_results[variant]["MSE"]))[i],
            })
    pd.DataFrame(rows_ps).to_csv(
        os.path.join(RESULTS_DIR, "ablation_perseed.csv"), index=False)

    # ---- 配对差 Δ_i = MSE_TCN,i - MSE_variant,i 的两种 95% 区间（E3）----
    rows_ci = []
    if "TCN" in all_results:
        tcn_mse_ps = np.asarray(all_results["TCN"]["MSE"], dtype=np.float64)
        for variant in variants:
            if variant == "TCN":
                continue
            entry = {"Variant": variant}
            entry.update(paired_difference_interval(
                tcn_mse_ps, all_results[variant]["MSE"]))
            rows_ci.append(entry)
    df_ci = pd.DataFrame(rows_ci)
    df_ci.to_csv(os.path.join(RESULTS_DIR, "ablation_paired_ci.csv"), index=False)

    # 打印 Markdown 表格
    print("\n" + "=" * 90)
    print("Table 4: 消融实验结果")
    print("=" * 90)
    # 格式化打印
    print(f"{'Variant':<22} {'MSE (mean±std)':<24} {'Improvement%':<15} {'P(vs TCN)':<12} {'Params':<12}")
    print("-" * 85)
    for _, row in df.iterrows():
        mse_str = f"{row['MSE_mean']:.6f}±{row['MSE_std']:.6f}"
        imp_str = f"{row['Improvement_pct']:.2f}%" if not np.isnan(row['Improvement_pct']) else "---"
        p_str = f"{row['TCN_paired_p']:.4f}" if not np.isnan(row['TCN_paired_p']) else "---"
        param_str = f"{int(row['Parameters']):,}" if not np.isnan(row['Parameters']) else "---"
        print(f"{row['Variant']:<22} {mse_str:<24} {imp_str:<15} {p_str:<12} {param_str:<12}")

    print(f"\nseed-noise floor (所有变体 MSE std 的最大值) = {seed_noise_floor:.6f}")

    if not df_ci.empty:
        print("\n" + "=" * 90)
        print("seed 级配对差 Δ_i = MSE_TCN,i - MSE_variant,i 的 95% 区间 (E3)")
        print("=" * 90)
        print(f"{'Variant':<22} {'Δ mean':>10} {'exact t CI':>22} "
              f"{'BCa CI':>22} {'p':>9}")
        print("-" * 88)
        for _, r in df_ci.iterrows():
            exact = f"[{r['exact_low']:.4f}, {r['exact_high']:.4f}]"
            bca = f"[{r['bca_low']:.4f}, {r['bca_high']:.4f}]"
            print(f"{r['Variant']:<22} {r['mean_delta']:>10.6f} "
                  f"{exact:>22} {bca:>22} {r['p_exact']:>9.4f}")

    print(f"\n结果已保存: {csv_path}")
    return df







# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="消融实验：AdaFracTCN 组件贡献分析")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "quick"],
                        help="运行模式")
    args = parser.parse_args()

    print("=" * 60)
    print("  消融实验：AdaFracTCN 组件贡献分析")
    print("=" * 60)

    start_time = time.time()

    # ---- Part 1: 消融对比实验 ----
    print(f"\n{'='*60}")
    print("Part 1: 消融变体对比实验")
    print(f"{'='*60}")

    ablation_results, ablation_preds, variants, y_test, scaling, floor = run_ablation_experiment(args.mode)
    ablation_df = save_ablation_results(ablation_results, variants, MODE_CONFIG[args.mode]["seeds"])
    if args.mode == "standard":
        save_ablation_raw_predictions(ablation_preds, y_test, scaling, floor,
                                      MODE_CONFIG[args.mode]["seeds"], MODE_CONFIG[args.mode]["horizon"])

    # Hyperparameter/alpha sensitivity is reproduced exclusively by
    # hyperparam_sensitivity.py; keeping a second sweep here would duplicate work.

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("  消融实验完成！")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
