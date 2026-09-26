# -*- coding: utf-8 -*-
"""
residual_analysis.py — 残差长记忆分析
======================================
分析各模型残差的长记忆性，验证 AdaFracTCN 是否能有效捕捉数据的长记忆结构。
残差 Hurst 指数越接近 0.5，说明模型对长记忆的提取越充分。

输出:
  - results/residual_hurst.csv          残差 Hurst 指数结果表

用法:
    python residual_analysis.py --mode standard
    python residual_analysis.py --mode quick
"""
import argparse
import os
import time
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

from baselines import get_model as get_baseline, set_protocol_scaling
from adafractcn import get_adafractcn
from download_data import hurst_mfdfa, hurst_rs
import fig_style
from exp_common import (load_target_scaling, load_input_scaling, inverse_target,
                        load_positivity_floor,
                        apply_positivity_projection, set_seed)

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
        "models": ["ARIMA", "LSTM", "GRU", "TCN", "Transformer",
                   "Informer", "Frac-LSTM", "AdaFracTCN"],
        "batch_size": 256,
    },
    "quick": {
        "seeds": [42],
        "models": ["TCN", "LSTM", "Frac-LSTM", "AdaFracTCN"],
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
        raise FileNotFoundError(f"数据目录不存在: {h_dir}，请先运行 download_data.py")
    arrs = tuple(np.load(os.path.join(h_dir, f)) for f in
                 ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                  "X_test.npy", "y_test.npy"])
    return arrs + (load_target_scaling(DATA_DIR, horizon),)


def make_loader(X, y, batch_size, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.float32)),
                      batch_size=batch_size, shuffle=shuffle)


# ============================================================
# 模型训练与预测
# ============================================================

# 本地 set_seed 已删除：它只固定随机源、不设 cudnn.deterministic，
# 而协议声明的是确定性 cuDNN。统一用 exp_common.set_seed。


def train_and_predict(model_name, horizon, seed, mode,
                      X_train, y_train, X_val, y_val,
                      X_test, y_test, batch_size, device):
    """训练模型并返回测试集预测值。"""
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
    return y_pred


# ============================================================
# 残差 Hurst 指数计算
# ============================================================

def compute_hurst(residuals_abs):
    """使用 MF-DFA 和 R/S 两种方法估计 Hurst 指数。"""
    r = np.asarray(residuals_abs, dtype=float)

    h_dfa = hurst_mfdfa(r, q=2)
    h_rs = hurst_rs(r)

    return h_dfa, h_rs


def compute_acf(x, max_lag=40):
    """计算自相关函数（ACF）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    var = np.var(x, ddof=1)
    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.mean((x[:-lag] - mean) * (x[lag:] - mean)) / var
    return acf


def _raw_slug(name):
    return (name.lower().replace("(1,1)", "11").replace(".", "")
            .replace(" ", "_").replace("-", "_").replace("/", "_"))


def _load_saved_forecast(model_name, horizon):
    """Load per-seed standardized forecasts produced by main_experiment.py."""
    path = os.path.join(RESULTS_DIR, "raw_predictions",
                        f"h{horizon}__{_raw_slug(model_name)}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Saved forecast artifact not found: {path}. Run "
            "`python main_experiment.py --mode standard` first.")
    z = np.load(path, allow_pickle=False)
    required = ["predictions_std", "y_test_std", "target_mean", "target_std",
                "positivity_floor", "seeds"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise RuntimeError(f"{path} missing fields: {missing}")
    return {k: z[k] for k in z.files}


def run_residual_from_saved(mode):
    """Compute h=10 residual-memory diagnostics without retraining.

    The main experiment stores every seed forecast. We average the ten forecasts
    at each test date, inverse-transform and positivity-project exactly as in the
    DM/MSE pipeline, then compute Hurst exponents of |y - yhat_ensemble|.
    """
    config = MODE_CONFIG[mode]
    model_names = config["models"]
    results, all_residuals = {}, {}
    y_test_pct = None
    h_orig_dfa = h_orig_rs = np.nan

    for j, model_name in enumerate(model_names):
        z = _load_saved_forecast(model_name, HORIZON)
        preds = np.asarray(z["predictions_std"], dtype=float)
        expected = len(config["seeds"])
        if mode == "standard" and preds.shape[0] != expected:
            raise RuntimeError(f"{model_name}: expected {expected} seeds, found {preds.shape[0]}")
        if not np.all(np.isfinite(preds)):
            raise RuntimeError(f"{model_name}: non-finite saved forecasts")

        scaling = (float(z["target_mean"]), float(z["target_std"]))
        floor = float(z["positivity_floor"])
        y_std = np.asarray(z["y_test_std"], dtype=float)
        y_pct = inverse_target(y_std, scaling)
        pred_std_ens = preds.mean(axis=0)
        pred_pct = inverse_target(pred_std_ens, scaling)
        pred_pct = apply_positivity_projection(pred_pct, floor)[0]
        residual = y_pct - pred_pct

        if y_test_pct is None:
            y_test_pct = y_pct
            target_abs = np.abs(y_test_pct)
            h_orig_dfa, h_orig_rs = compute_hurst(target_abs)
            h_orig_dfa_no, h_orig_rs_no = compute_hurst(target_abs[::HORIZON])
            print(f"h={HORIZON} test target Hurst: MF-DFA={h_orig_dfa:.4f}, R/S={h_orig_rs:.4f}; "
                  f"non-overlap MF-DFA={h_orig_dfa_no:.4f}, R/S={h_orig_rs_no:.4f}")
        elif not np.allclose(y_test_pct, y_pct, rtol=0, atol=1e-12):
            raise RuntimeError(f"{model_name}: y_test differs from the first saved artifact")

        abs_res = np.abs(residual)
        h_dfa, h_rs = compute_hurst(abs_res)
        # Non-overlapping robustness: take every h-th forecast origin so
        # adjacent targets no longer share future daily returns.
        h_dfa_no, h_rs_no = compute_hurst(abs_res[::HORIZON])
        results[model_name] = {
            "h_dfa": h_dfa, "h_rs": h_rs,
            "h_dfa_nonoverlap": h_dfa_no, "h_rs_nonoverlap": h_rs_no,
            "reduction_dfa": h_orig_dfa - h_dfa,
            "reduction_rs": h_orig_rs - h_rs,
            "n_seeds": int(preds.shape[0]),
        }
        all_residuals[model_name] = residual
        print(f"{model_name:15s}: |ensemble residual| H MF-DFA={h_dfa:.4f}, R/S={h_rs:.4f}")

    return results, all_residuals, y_test_pct, h_orig_dfa, h_orig_rs, model_names


# ============================================================
# 实验主流程
# ============================================================

def run_residual_analysis(mode):
    """运行残差长记忆分析实验。"""
    config = MODE_CONFIG[mode]
    seeds = config["seeds"]
    model_names = config["models"]
    batch_size = config["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"设备: {device}")
    print(f"模式: {mode}, h={HORIZON}")
    print(f"模型: {model_names}")
    print(f"种子: {seeds}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaling = load_data(HORIZON)
    floor = load_positivity_floor(DATA_DIR, HORIZON)
    set_protocol_scaling(input_scaling=load_input_scaling(DATA_DIR),
                         target_scaling=scaling)
    print(f"数据: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, "
          f"正值投影地板 y_min={floor:.6g}（来源：训练切分）")

    # 残差与基准序列一律取在百分比波动率尺度上。Hurst 对正数整体缩放
    # 不变，所以只看比值时尺度无所谓；但正值投影会改变残差的取值，
    # 因此预测侧必须先回逆变换再投影，才与 MSE/QLIKE 的口径一致。
    y_test_pct = inverse_target(y_test, scaling)

    # ---- 原始序列的 Hurst 指数（基准） ----
    print(f"\n计算原始序列 Hurst 指数...")
    orig_abs = np.abs(y_test_pct)
    h_orig_dfa, h_orig_rs = compute_hurst(orig_abs)
    print(f"  h={HORIZON} 测试目标 Hurst: MF-DFA={h_orig_dfa:.4f}, R/S={h_orig_rs:.4f}")

    # ---- 各模型残差 Hurst 指数 ----
    results = {}  # model -> {h_dfa, h_rs, reduction_dfa, residuals_avg, acf}
    all_residuals = {}  # 保存平均残差用于 ACF 可视化

    total = len(model_names) * len(seeds)
    current = 0

    for model_name in model_names:
        print(f"\n--- {model_name} ---")
        # 收集所有种子的残差
        residuals_seeds = []

        for seed in seeds:
            current += 1
            print(f"  [{current}/{total}] seed={seed}...", end=" ")

            try:
                y_pred = train_and_predict(
                    model_name, HORIZON, seed, mode,
                    X_train, y_train, X_val, y_val,
                    X_test, y_test, batch_size, device
                )
                # 残差 e_t = y_t - y_hat_t，两侧口径与评分时完全一致：
                # 回逆变换到百分比尺度，预测再过一次同一个正值投影。
                y_pred_pct = apply_positivity_projection(
                    inverse_target(y_pred, scaling), floor)[0]
                residual = y_test_pct - y_pred_pct
                residuals_seeds.append(residual)
                print(f"残差均值={np.mean(residual):.6f}, std={np.std(residual):.6f}")
            except Exception as e:
                print(f"ERROR: {e}")
                residuals_seeds.append(None)

        # 过滤掉失败的种子
        valid_residuals = [r for r in residuals_seeds if r is not None]
        if not valid_residuals:
            print(f"  [警告] {model_name} 所有种子均失败，跳过")
            results[model_name] = {
                "h_dfa": np.nan, "h_rs": np.nan,
                "reduction_dfa": np.nan, "reduction_rs": np.nan,
            }
            all_residuals[model_name] = np.full_like(y_test, np.nan)
            continue

        # 标准模式：取平均后的残差计算 Hurst
        # 快速模式：只有1个种子，直接使用
        if len(valid_residuals) > 1:
            avg_residual = np.mean(valid_residuals, axis=0)
            print(f"  使用 {len(valid_residuals)} 个种子的平均残差计算 Hurst")
        else:
            avg_residual = valid_residuals[0]

        # 绝对残差
        abs_residual = np.abs(avg_residual)

        # 计算 Hurst 指数
        h_dfa, h_rs = compute_hurst(abs_residual)
        h_dfa_no, h_rs_no = compute_hurst(abs_residual[::HORIZON])
        reduction_dfa = h_orig_dfa - h_dfa
        reduction_rs = h_orig_rs - h_rs

        results[model_name] = {
            "h_dfa": h_dfa,
            "h_rs": h_rs,
            "h_dfa_nonoverlap": h_dfa_no,
            "h_rs_nonoverlap": h_rs_no,
            "reduction_dfa": reduction_dfa,
            "reduction_rs": reduction_rs,
        }
        all_residuals[model_name] = avg_residual

        print(f"  残差 |e| Hurst: MF-DFA={h_dfa:.4f}, R/S={h_rs:.4f}")
        print(f"  Hurst 降低: MF-DFA={reduction_dfa:.4f}, R/S={reduction_rs:.4f}")
        if not np.isnan(h_dfa):
            dist = abs(h_dfa - 0.5)
            if dist < 0.05:
                print(f"  -> 接近 0.5，长记忆结构已被有效提取")
            elif h_dfa < 0.5:
                print(f"  -> 低于 0.5，存在反持久性（均值回复）")
            else:
                print(f"  -> 高于 0.5，仍存在残余长记忆")

    # 返回的是百分比尺度上的目标序列，出图侧的 |y_t| ACF 与它对齐
    return results, all_residuals, y_test_pct, h_orig_dfa, h_orig_rs, model_names


# ============================================================
# 结果保存
# ============================================================

def save_results(results, h_orig_dfa, h_orig_rs, model_names, y_test):
    """保存结果到 CSV 并打印表格。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    h_orig_dfa_no, h_orig_rs_no = compute_hurst(np.abs(np.asarray(y_test, dtype=float))[::HORIZON])
    # 原始序列行
    rows.append({
        "Model": f"Original h={HORIZON} target",
        "H_MF_DFA": h_orig_dfa,
        "H_RS": h_orig_rs,
        "H_MF_DFA_nonoverlap": h_orig_dfa_no,
        "H_RS_nonoverlap": h_orig_rs_no,
        "Reduction_MF_DFA": np.nan,
        "Reduction_RS": np.nan,
    })

    for name in model_names:
        if name in results:
            r = results[name]
            rows.append({
                "Model": name,
                "H_MF_DFA": r["h_dfa"],
                "H_RS": r["h_rs"],
                "H_MF_DFA_nonoverlap": r.get("h_dfa_nonoverlap", np.nan),
                "H_RS_nonoverlap": r.get("h_rs_nonoverlap", np.nan),
                "Reduction_MF_DFA": r["reduction_dfa"],
                "Reduction_RS": r["reduction_rs"],
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "residual_hurst.csv")
    df.to_csv(csv_path, index=False)

    # 打印表格
    print("\n" + "=" * 75)
    print("Table: 残差长记忆分析结果")
    print("=" * 75)
    print(f"{'Model':<18} {'H_MF-DFA':<12} {'H_R/S':<12} {'ΔH_MF-DFA':<12} {'ΔH_R/S':<12}")
    print("-" * 66)
    for _, row in df.iterrows():
        h_dfa = f"{row['H_MF_DFA']:.4f}" if not np.isnan(row['H_MF_DFA']) else "---"
        h_rs = f"{row['H_RS']:.4f}" if not np.isnan(row['H_RS']) else "---"
        red_dfa = f"{row['Reduction_MF_DFA']:.4f}" if not np.isnan(row['Reduction_MF_DFA']) else "---"
        red_rs = f"{row['Reduction_RS']:.4f}" if not np.isnan(row['Reduction_RS']) else "---"
        print(f"{row['Model']:<18} {h_dfa:<12} {h_rs:<12} {red_dfa:<12} {red_rs:<12}")

    print(f"\n结果已保存: {csv_path}")
    return df



# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="残差长记忆分析")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "quick"], help="运行模式")
    parser.add_argument("--source", type=str, default="saved",
                        choices=["saved", "train"],
                        help="saved: reuse raw_predictions from main_experiment (recommended); train: retrain models")
    args = parser.parse_args()

    print("=" * 60)
    print("  残差长记忆分析")
    print("=" * 60)

    start_time = time.time()

    # Prefer the saved per-seed forecasts from the main experiment so the
    # residual diagnostic is exactly tied to the reported forecasts and does
    # not silently create a second, independently trained set of models.
    if args.source == "saved":
        results, all_residuals, y_test, h_orig_dfa, h_orig_rs, model_names = \
            run_residual_from_saved(args.mode)
    else:
        results, all_residuals, y_test, h_orig_dfa, h_orig_rs, model_names = \
            run_residual_analysis(args.mode)

    # 保存结果
    df = save_results(results, h_orig_dfa, h_orig_rs, model_names, y_test)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  残差长记忆分析完成！")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
