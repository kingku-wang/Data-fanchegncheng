# -*- coding: utf-8 -*-
"""
main_experiment.py — 主实验：多步预测精度对比
================================================
对比 AdaFracTCN 与基线模型在 h ∈ {1,5,10,20} 上的预测性能。
生成主实验表格、逐 seed 预测与 DM 检验所需数据。

用法:
    python main_experiment.py --mode standard
    python main_experiment.py --mode quick
"""
import argparse
import os
import warnings

# 解决 OpenMP 重复加载问题（PyTorch + numpy 在 Windows 上的常见冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# 本地 import
from baselines import get_model as get_baseline, set_protocol_scaling
from adafractcn import get_adafractcn
from exp_common import (load_target_scaling as _load_target_scaling,
                        load_input_scaling as _load_input_scaling,
                        inverse_target as _exp_inverse_target,
                        compute_metrics as _compute_metrics,
                        load_positivity_floor as _load_positivity_floor,
                        positivity_floor_detail as _floor_detail,
                        apply_positivity_projection as _apply_projection,
                        adjust_family as _adjust_family,
                        sample_std as _sample_std,
                        set_seed as _exp_set_seed)

# ============================================================
# 全局配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODE_CONFIG = {
    "standard": {
        "horizons": [1, 5, 10, 20],
        "seeds": [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
        "models": ["Persistence", "Uncond mean", "ARIMA", "HAR-RV", "GARCH",
                   "EGARCH", "LSTM", "GRU", "TCN", "Transformer",
                   "Informer", "Frac-LSTM", "AdaFracTCN"],
        "batch_size": 256,
    },
    "quick": {
        "horizons": [1, 10],
        "seeds": [42],
        "models": ["LSTM", "TCN", "AdaFracTCN"],
        "batch_size": 128,
    },
}


# ============================================================
# 数据加载
# ============================================================

def load_data(horizon):
    """加载指定预测步长的预处理数据。

    除 (X, y) 外，同时返回该步长的目标标定参数 (mean, std)，用于把
    标准化尺度上的预测与真值回逆变换到"百分比波动率"尺度后再计算指标。
    标准化尺度上目标存在负值，若不做回逆变换，MAPE 与 QLIKE 在定义上
    不可用（负预测占比可达数十个百分点），指标也会与论文口径不一致。
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
            load_target_scaling(horizon))


def load_target_scaling(horizon):
    """读取 download_data.py 保存的目标 z-score 标定参数。

    直接复用 exp_common 的实现，避免两处口径漂移。
    返回 (mean, std)；若参数文件缺失则返回 (0.0, 1.0)，此时回逆变换为
    恒等映射，并给出明确警告（指标将落在标准化尺度上，论文中不可比）。
    """
    return _load_target_scaling(DATA_DIR, horizon)


def inverse_target(y, scaling):
    """把标准化尺度上的 y 回逆变换到百分比波动率尺度：y * std + mean。"""
    return _exp_inverse_target(y, scaling)


def positivity_floor(horizon):
    """该 horizon 的正值投影地板 y_min，**只由训练切分**确定。

    口径见 exp_common.positivity_floor：
        y_min = max( min{y_i>0, i∈train}, 1e-3 * mean{y_i, i∈train} )
    reproducibility protocol 指出earlier implementation把 y_min 写成不限切分的 data-dependent floor，若最小值
    取自测试集真实 y_i，测试标签就参与了预测后处理。现在地板只由训练切分给出，
    且 compute_metrics 在地板缺失时会直接报错而不是退化为测试集口径。
    """
    return _load_positivity_floor(DATA_DIR, horizon)


def make_loader(X, y, batch_size, shuffle=False):
    """创建 DataLoader。"""
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# 评估指标
# ============================================================

def compute_metrics(y_true, y_pred, scaling=None, floor=None, project=True):
    """统一指标口径，实现见 exp_common.compute_metrics。

    本函数只做转发，不再维护第二套实现。早期版本在这里留了一份独立的
    指标实现，于是与 exp_common 出现口径漂移：这份副本没有正值投影，
    且 QLIKE 的有效掩码按每个模型自己的非正预测各自计算，等于把不同
    模型记在不同日期的子样本上。论文 §4.3 承诺"投影对每个模型相同、
    MSE 也在投影后的预测上计算"，这条承诺现在由 exp_common 一侧强制，
    主实验只保留转发壳，避免又一次漂移。

    参数
    ----
    scaling : (mean, std) 或 None，目标 z-score 标定参数；
    floor   : 正值投影地板 y_min（百分比波动率尺度），None 时由被评分
              目标的最小正值代替；
    project : 是否施加正值投影（默认 True，与论文一致）。
    """
    return _compute_metrics(y_true, y_pred, scaling=scaling,
                            floor=floor, project=project)


def dm_test(y_true, y_pred_a, y_pred_b, horizon=1):
    """Diebold-Mariano 检验（HAC 标准误，Bartlett 窗）。

    H0: 两个模型预测精度相同（基于平方误差损失）。
    返回 (DM 统计量, 双侧 p 值)。正值表示模型 A（AdaFracTCN）更优。

    两个口径要点（对应multiple-testing protocol）
    ------------------------------
    * **输入必须是"十种子平均后的预测"在百分比波动率尺度上的、已施加正值
      投影的数组。** 旧实现取的是 `all_preds[...][0]`，即只用第一个种子的
      预测，与论文"at each test date the ten seed forecasts of a model are
      averaged to a single forecast before the loss differential is formed"
      直接冲突。现在由 save_results 统一做种内平均与投影，本函数只做检验。
    * 损失在**平方误差**上定义，与 Table 3 的 MSE 列同口径（同为投影后的
      预测）。把预测先回逆变换到百分比尺度并不改变 DM 的取值——平方误差
      整体乘 sd^2 是常数因子，会被 HAC 标准差约掉——但正值投影是非线性的，
      在投影**之前**做检验会与表格口径不一致，故这里在投影之后做。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    e_a = (y_true - np.asarray(y_pred_a, dtype=np.float64)) ** 2
    e_b = (y_true - np.asarray(y_pred_b, dtype=np.float64)) ** 2
    d = e_a - e_b  # d_i = loss_A - loss_B, 正值表示 A 更差

    N = len(d)
    d_mean = np.mean(d)

    # HAC 方差估计（Newey-West / Bartlett 窗）
    # Overlapping h-day targets induce mechanical serial correlation through at
    # least h-1 lags. Use that as a lower bound on the automatic Newey-West
    # bandwidth instead of a horizon-agnostic plug-in alone.
    plugin_lag = int(np.floor(4 * (N / 100) ** (2 / 9)))
    max_lag = max(int(horizon) - 1, plugin_lag, 1)
    max_lag = min(max_lag, N - 1)

    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * w * gamma_k

    var_d = gamma_0 + gamma_sum
    if var_d <= 0:
        return 0.0, 1.0, max_lag

    dm_stat = d_mean / np.sqrt(var_d / N)

    # 双侧 p-value。d = loss_A - loss_B 时正值表示 A 的损失更大，
    # 而论文约定 DM 取正值表示 AdaFracTCN 更好，故统计量取负号。
    # p 值只依赖 |统计量|，不受取负影响。
    dm_stat = -dm_stat
    p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value, max_lag


# ============================================================
# 实验运行
# ============================================================

def set_seed(seed):
    """固定随机源与 cudnn 确定性开关，实现见 exp_common.set_seed。

    原先这里只设 numpy / torch 的随机源，漏了 cudnn.deterministic，
    与协议里"deterministic cuDNN settings"的声明不符。
    """
    return _exp_set_seed(seed)


def run_single(model_name, horizon, seed, mode, X_train, y_train,
               X_val, y_val, X_test, y_test, batch_size, device,
               scaling=(0.0, 1.0), floor=None):
    """运行单个模型×步长×种子的实验。

    scaling: (mean, std) 目标 z-score 标定参数。模型在标准化尺度上训练与
    预测，评价前统一回逆变换到百分比波动率尺度并施加正值投影（论文
    eq:inverse_standardization / eq:positivity_projection），因此 MAPE 与
    QLIKE 有定义，且投影对同一 horizon 的所有模型完全相同。
    floor: 正值投影地板 y_min；None 时由 exp_common 退化为评测目标的最小
           正值。主实验传入由训练切分算出、与模型无关的地板。
    """
    set_seed(seed)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    # 初始化模型
    if model_name == "AdaFracTCN":
        model = get_adafractcn("AdaFracTCN", horizon=horizon, mode=mode)
    else:
        model = get_baseline(model_name, horizon=horizon, mode=mode)

    # 训练
    model.fit(train_loader, val_loader, device=device)

    # 预测（标准化尺度）
    y_pred = model.predict(test_loader, device=device)

    # AdaFracTCN 同一已训练模型的 operative alpha trace。这里不重新训练、
    # 也不调用解释阶段的另一套 alpha 网络；forward_with_alpha 与正式 forward
    # 共享完全相同的计算图，只额外返回各 block 真正用于 GL 核的逐样本阶数。
    alpha_trace = None
    if model_name == "AdaFracTCN" and hasattr(model, "forward_with_alpha"):
        model.eval()
        model.to(device)
        traces = []
        with torch.no_grad():
            for xb, _ in test_loader:
                _, a = model.forward_with_alpha(xb.to(device))
                traces.append(a.detach().cpu().numpy())
        alpha_trace = np.concatenate(traces, axis=0)[:len(y_test)]

    # 指标：回逆变换 -> 正值投影 -> MSE/QLIKE 等，全部在 exp_common 里完成，
    # 传入 floor 以保证与其它模型共用同一地板、MSE 也在投影后的预测上算。
    metrics = compute_metrics(y_test, y_pred, scaling=scaling, floor=floor)

    return metrics, y_pred, alpha_trace, model


def run_experiment(mode):
    """运行完整实验。"""
    config = MODE_CONFIG[mode]
    horizons = config["horizons"]
    seeds = config["seeds"]
    model_names = config["models"]
    batch_size = config["batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"模式: {mode}")
    print(f"预测步长: {horizons}")
    print(f"模型: {model_names}")
    print(f"种子: {seeds}")
    print(f"Batch size: {batch_size}")

    # 存储所有结果
    # results[model_name][horizon] = {"MSE": [val_seed1, ...], "RMSE": [...], ...}
    all_results = {}
    # predictions[model_name][horizon] = [pred_array_seed1, ...]
    all_preds = {}
    # y_test 存储（每个 h 一份）
    y_tests = {}
    # AdaFracTCN operative alpha traces[h] = [seed_trace, ...]
    all_alpha_traces = {h: [] for h in horizons}
    # 每个 horizon 的 (目标标定, 正值地板)，供指标与 DM 使用
    scalings, floors = {}, {}

    total = len(model_names) * len(horizons) * len(seeds)
    current = 0

    for h in horizons:
        print(f"\n{'='*60}")
        print(f"预测步长 h={h}")
        print(f"{'='*60}")

        X_train, y_train, X_val, y_val, X_test, y_test, scaling = load_data(h)
        floor = positivity_floor(h)
        scalings[h] = scaling
        floors[h] = floor
        y_tests[h] = y_test
        # 计量/朴素基线（Persistence、HAR-RV、GARCH、EGARCH）需要把标准化
        # 输入窗口还原成收益率 / 已实现方差，并把百分比波动率预测换算回
        # 标准化目标尺度；两组常数在此设置一次，供本轮所有模型共用。
        input_scaling = _load_input_scaling(DATA_DIR)
        set_protocol_scaling(input_scaling=input_scaling, target_scaling=scaling)
        print(f"  数据: train={X_train.shape}, val={X_val.shape}, "
              f"test={X_test.shape}, input scaling=(mean={input_scaling[0]:.6g}, "
              f"std={input_scaling[1]:.6g}), target scaling=(mean={scaling[0]:.6g}, "
              f"std={scaling[1]:.6g}), positivity floor y_min={floor:.6g} "
              f"(来源：训练切分)")

        for model_name in model_names:
            if model_name not in all_results:
                all_results[model_name] = {}
                all_preds[model_name] = {}

            all_results[model_name][h] = {
                k: [] for k in ["MSE", "RMSE", "MAE", "MAPE", "R2", "QLIKE",
                                "n_projected", "frac_projected"]}
            all_preds[model_name][h] = []

            for seed in seeds:
                current += 1
                print(f"  [{current}/{total}] {model_name:15s} h={h:2d} seed={seed}...", end=" ")

                try:
                    metrics, y_pred, alpha_trace, fitted_model = run_single(
                        model_name, h, seed, mode,
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        batch_size, device, scaling=scaling, floor=floor
                    )
                    for k in all_results[model_name][h]:
                        all_results[model_name][h][k].append(metrics[k])
                    all_preds[model_name][h].append(y_pred)
                    if model_name == "AdaFracTCN":
                        all_alpha_traces[h].append(alpha_trace)
                        # Preserve the fitted state needed by the analysis-specific
                        # effective-kernel diagnostic.  This is written only for
                        # the full AdaFracTCN because downstream kernel analysis
                        # needs the learned short filters W together with the
                        # operative alpha trace already stored in raw_predictions.
                        if mode == "standard":
                            ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
                            os.makedirs(ckpt_dir, exist_ok=True)
                            torch.save({
                                "model": "AdaFracTCN",
                                "horizon": int(h),
                                "seed": int(seed),
                                "state_dict": {k: v.detach().cpu() for k, v in fitted_model.state_dict().items()},
                            }, os.path.join(ckpt_dir, f"adafractcn_h{h}_seed{seed}.pt"))
                    print(f"MSE={metrics['MSE']:.6f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    for k in all_results[model_name][h]:
                        all_results[model_name][h][k].append(np.nan)
                    all_preds[model_name][h].append(np.full_like(y_test, np.nan))
                    if model_name == "AdaFracTCN":
                        all_alpha_traces[h].append(None)

    return (all_results, all_preds, y_tests, horizons, model_names,
            scalings, floors, all_alpha_traces, seeds)


# ============================================================
# 结果汇总与保存
# ============================================================

def save_results(all_results, all_preds, y_tests, horizons, model_names,
                 scalings=None, floors=None, seeds=None, all_alpha_traces=None):
    """保存结果到 CSV 并打印 Markdown 表格。

    参数增加 scalings / floors：DM 检验必须在"回逆变换 + 正值投影"之后的
    百分比波动率尺度上进行，才能与 Table 3 的 MSE 同口径；投影所需的地板
    必须显式传入（训练切分口径），不在本函数里临场推断。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    alpha_level = 0.05

    # ---- Table 1: 主结果 (MSE, RMSE, QLIKE, 各 h) ----
    rows = []
    for name in model_names:
        row = {"Model": name}
        for h in horizons:
            vals = all_results[name][h]
            mse_mean = np.nanmean(vals["MSE"])
            mse_std = _sample_std(vals["MSE"])
            rmse_mean = np.nanmean(vals["RMSE"])
            rmse_std = _sample_std(vals["RMSE"])
            qlike_mean = np.nanmean(vals["QLIKE"])
            qlike_std = _sample_std(vals["QLIKE"])
            row[f"h={h}_MSE"] = f"{mse_mean:.6f}±{mse_std:.6f}"
            row[f"h={h}_RMSE"] = f"{rmse_mean:.6f}±{rmse_std:.6f}"
            row[f"h={h}_QLIKE"] = f"{qlike_mean:.6f}±{qlike_std:.6f}"
        rows.append(row)
    df_main = pd.DataFrame(rows)
    df_main.to_csv(os.path.join(RESULTS_DIR, "main_results.csv"), index=False)

    # Machine-readable numeric long form.  This is the canonical file for
    # later result audits and additional analyses; the wide CSV above is formatted
    # for human reading and contains mean±std strings.
    long_main = []
    for name in model_names:
        for h in horizons:
            vals = all_results[name][h]
            long_main.append({
                "model": name, "horizon": int(h),
                "mse": float(np.nanmean(vals["MSE"])),
                "mse_std": float(_sample_std(vals["MSE"])),
                "qlike": float(np.nanmean(vals["QLIKE"])),
                "qlike_std": float(_sample_std(vals["QLIKE"])),
                "n_seeds": int(np.sum(np.isfinite(np.asarray(vals["MSE"], dtype=float)))),
            })
    pd.DataFrame(long_main).to_csv(
        os.path.join(RESULTS_DIR, "main_results_long.csv"), index=False)

    # ---- Table 2: 补充指标 (h=10 的 MAE, MAPE, R2) ----
    h_ref = 10 if 10 in horizons else horizons[-1]
    rows2 = []
    for name in model_names:
        vals = all_results[name][h_ref]
        row = {
            "Model": name,
            "MAE": f"{np.nanmean(vals['MAE']):.6f}±{_sample_std(vals['MAE']):.6f}",
            "MAPE": f"{np.nanmean(vals['MAPE']):.4f}±{_sample_std(vals['MAPE']):.4f}",
            "R2": f"{np.nanmean(vals['R2']):.6f}±{_sample_std(vals['R2']):.6f}",
        }
        rows2.append(row)
    df_full = pd.DataFrame(rows2)
    df_full.to_csv(os.path.join(RESULTS_DIR, "full_metrics.csv"), index=False)

    # ---- Table 4: DM 检验 ----
    #
    # 三个口径修正（multiple-testing protocol）：
    #   * 损失对象是"**十种子平均后**的预测"，而不是某一个种子的预测。
    #     旧实现取 all_preds[...][0]，只用了 seed 42，与论文 §4.3
    #     "at each test date the ten seed forecasts of a model are averaged
    #     to a single forecast before the loss differential is formed" 冲突。
    #     这决定了检验对象是 ensemble-of-seeds 的预测差异，而不是单次运行的
    #     期望表现——论文现在把这一点写进正文。
    #   * 预测先回逆变换到百分比波动率尺度，并施加与 MSE 相同的正值投影，
    #     使 DM 与 Table 3 同口径。
    #   * 48 次检验构成一个检验族，除未校正 p 值外同时给出 Holm（控制 FWER）
    #     与 Benjamini--Hochberg（控制 FDR）校正 p 值，避免只报告未校正的多重比较结论
    #     被读成比实际更强的证据。
    if "AdaFracTCN" in model_names and scalings and floors:
        proj, n_seeds_used = {}, {}
        for name in model_names:
            per_h = {}
            for h in horizons:
                stack = np.vstack([np.asarray(p, dtype=np.float64)
                                   for p in all_preds[name][h]])
                n_seeds_used[name] = int(stack.shape[0])
                with np.errstate(invalid="ignore"):
                    mean_pred = np.nanmean(stack, axis=0)
                pct = _exp_inverse_target(mean_pred, scalings[h])
                pct, _ = _apply_projection(pct, floors[h])
                per_h[h] = pct
            proj[name] = per_h

        y_true_pct = {h: _exp_inverse_target(y_tests[h], scalings[h]) for h in horizons}

        rows3, long_rows, pvals = [], [], []
        for name in model_names:
            if name == "AdaFracTCN":
                continue
            row = {"Baseline": name}
            for h in horizons:
                pa, pb = proj["AdaFracTCN"][h], proj[name][h]
                if (not np.all(np.isfinite(pa))) or (not np.all(np.isfinite(pb))):
                    row[f"h={h}_DM"] = "N/A"
                    row[f"h={h}_pvalue"] = "N/A"
                    pvals.append(np.nan)
                    long_rows.append({"Baseline": name, "Horizon": h, "DM": np.nan,
                                      "p_unadjusted": np.nan, "hac_lag": np.nan,
                                      "n_seeds": n_seeds_used.get(name, 0)})
                    continue
                dm, pval, hac_lag = dm_test(y_true_pct[h], pa, pb, horizon=h)
                row[f"h={h}_DM"] = f"{dm:.4f}"
                row[f"h={h}_pvalue"] = f"{pval:.6f}"
                pvals.append(pval)
                long_rows.append({"Baseline": name, "Horizon": h, "DM": dm,
                                  "p_unadjusted": pval, "hac_lag": hac_lag,
                                  "n_seeds": n_seeds_used.get(name, 0)})
            rows3.append(row)

        # 校正（NaN 不参与族大小；论文的 48 格在本协议下应当全部有效）
        #
        # 宽表 dm_test.csv（rows3，每模型一行）与长表 dm_test_long.csv
        # （long_rows，每 (模型, horizon) 一行）必须共享**同一份**校正结果。
        # 早期实现建了两个迭代器，在 rows3 循环里把它们耗尽后，long_rows
        # 循环再 `next()` 必然 StopIteration；本协议下 48 格全有效，重跑时
        # 这是一定会触发的崩溃点。改为单次有序遍历：idx 只随"有效格"递增，
        # 同时写入宽表单元格与长表行，末尾断言恰好消费 len(valid) 个。
        valid = [v for v in pvals if np.isfinite(v)]
        fam = _adjust_family(valid, alpha=0.05)
        ph_all, pb_all = list(fam["p_holm"]), list(fam["p_bh"])
        nh = len(horizons)

        j = 0
        for mi, row in enumerate(rows3):
            for k, h in enumerate(horizons):
                lr = long_rows[mi * nh + k]
                if row.get(f"h={h}_pvalue") == "N/A":
                    row[f"h={h}_p_holm"] = "N/A"
                    row[f"h={h}_p_bh"] = "N/A"
                    lr["p_holm"] = np.nan
                    lr["p_bh"] = np.nan
                    lr["sig_unadjusted"] = False
                    lr["sig_holm"] = False
                    lr["sig_bh"] = False
                    continue
                ph, pb = ph_all[j], pb_all[j]
                j += 1
                row[f"h={h}_p_holm"] = f"{ph:.6f}"
                row[f"h={h}_p_bh"] = f"{pb:.6f}"
                lr["p_holm"] = ph
                lr["p_bh"] = pb
                lr["sig_unadjusted"] = bool(lr["p_unadjusted"] < alpha_level)
                lr["sig_holm"] = bool(ph < alpha_level)
                lr["sig_bh"] = bool(pb < alpha_level)
        assert j == len(valid), (j, len(valid))

        df_dm = pd.DataFrame(rows3)
        df_dm.to_csv(os.path.join(RESULTS_DIR, "dm_test.csv"), index=False)
        pd.DataFrame(long_rows).to_csv(
            os.path.join(RESULTS_DIR, "dm_test_long.csv"), index=False)
        # 检验族的汇总：族大小与三种口径下的显著计数，供 run_all 的 X9 断言
        pd.DataFrame([{
            "n_tests": fam["n_tests"],
            "n_significant_unadjusted": fam["n_significant_unadjusted"],
            "n_significant_holm": fam["n_significant_holm"],
            "n_significant_bh": fam["n_significant_bh"],
            "alpha": fam["alpha"],
            "n_seeds_averaged": max(n_seeds_used.values()) if n_seeds_used else 0,
        }]).to_csv(os.path.join(RESULTS_DIR, "dm_family_summary.csv"), index=False)
        print(f"\nDM 检验族：m={fam['n_tests']}，未校正显著 {fam['n_significant_unadjusted']}，"
              f"Holm 显著 {fam['n_significant_holm']}，"
              f"BH 显著 {fam['n_significant_bh']}（α={fam['alpha']}）")

    # ---- Table 4: 正值投影命中计数（论文 §4.8 引用的量）----
    # 投影对同一 horizon 的所有模型使用同一地板，因此这里逐模型逐 horizon
    # 记录"有多少天的线性读出会落到非正"，正是正文承诺要记录的数量。
    rows_pos = []
    for name in model_names:
        row = {"Model": name}
        for h in horizons:
            n_list = np.asarray(all_results[name][h]["n_projected"], dtype=np.float64)
            f_list = np.asarray(all_results[name][h]["frac_projected"], dtype=np.float64)
            row[f"h={h}_n_projected"] = int(np.nanmean(n_list)) if n_list.size else 0
            row[f"h={h}_frac_pct"] = (float(np.nanmean(f_list) * 100)
                                      if f_list.size else np.nan)
        rows_pos.append(row)
    df_pos = pd.DataFrame(rows_pos)
    df_pos.to_csv(os.path.join(RESULTS_DIR, "positivity_counts.csv"), index=False)

    # ---- 正值投影地板台账：来源必须可审计（run_all 的 X7）----
    #
    # reproducibility protocol 的核心不是地板的数值，而是它的**来源**。这里把每个 horizon
    # 的地板值、来源切分、系数与训练切分的样本量一并落盘，使"地板只由训练
    # 切分确定"成为一条可核对的记录，而不是一句声称。
    if floors:
        floor_rows = []
        for h in horizons:
            detail = _floor_detail(DATA_DIR, h)
            detail["horizon"] = h
            detail["floor"] = floors[h] if floors[h] is not None else np.nan
            floor_rows.append(detail)
        pd.DataFrame(floor_rows).to_csv(
            os.path.join(RESULTS_DIR, "positivity_floor.csv"), index=False)

    # ---- 打印 Markdown 表格 ----
    print("\n" + "=" * 80)
    print("Table 1: 多步预测主结果 (MSE & RMSE)")
    print("=" * 80)
    print(df_main.to_markdown(index=False))

    print("\n" + "=" * 80)
    print(f"Table 2: 补充指标 (h={h_ref})")
    print("=" * 80)
    print(df_full.to_markdown(index=False))

    if "AdaFracTCN" in model_names:
        print("\n" + "=" * 80)
        print("Table 3: Diebold-Mariano 检验 (AdaFracTCN vs Baselines)")
        print("=" * 80)
        print(df_dm.to_markdown(index=False))

    print("\n" + "=" * 80)
    print("Table 4: 正值投影命中计数 y+ = max(y, y_min)")
    print("=" * 80)
    print(df_pos.to_markdown(index=False))

    if seeds is not None and scalings and floors:
        save_raw_prediction_artifacts(all_preds, y_tests, horizons, model_names,
                                      scalings, floors, seeds, all_alpha_traces)

    return df_main, df_full




def _raw_slug(name):
    """Filesystem-safe model id used by raw forecast artifacts."""
    return (name.lower().replace("(1,1)", "11").replace(".", "")
            .replace(" ", "_").replace("-", "_").replace("/", "_"))


def save_raw_prediction_artifacts(all_preds, y_tests, horizons, model_names,
                                  scalings, floors, seeds, all_alpha_traces=None):
    """Persist per-seed test forecasts so inference never requires retraining.

    Files store standardized forecasts (the model output), standardized targets,
    forecast-origin dates, target scaling, the train-only positivity floor and,
    for AdaFracTCN, the exact operative alpha trace with shape
    ``(seed, date, block, channel)``. Downstream inference scripts invert and
    project from these files using the same protocol as the main table.
    """
    raw_dir = os.path.join(RESULTS_DIR, "raw_predictions")
    os.makedirs(raw_dir, exist_ok=True)
    index_rows = []
    for h in horizons:
        date_path = os.path.join(DATA_DIR, f"h{h}", "origin_dates_test.npy")
        dates = np.load(date_path) if os.path.exists(date_path) else np.arange(len(y_tests[h]))
        for name in model_names:
            stack = np.vstack([np.asarray(p, dtype=np.float64) for p in all_preds[name][h]])
            filename = f"h{h}__{_raw_slug(name)}.npz"
            payload = dict(
                model=np.asarray(name), horizon=np.asarray(h),
                seeds=np.asarray(seeds[:stack.shape[0]], dtype=np.int64),
                predictions_std=stack,
                y_test_std=np.asarray(y_tests[h], dtype=np.float64),
                origin_dates=dates,
                target_mean=np.asarray(scalings[h][0]),
                target_std=np.asarray(scalings[h][1]),
                positivity_floor=np.asarray(floors[h]),
            )
            alpha_ok = False
            if name == "AdaFracTCN" and all_alpha_traces is not None:
                traces = all_alpha_traces.get(h, [])
                if len(traces) == stack.shape[0] and all(t is not None for t in traces):
                    payload["alpha_trace"] = np.stack(traces, axis=0)
                    alpha_ok = True
            np.savez_compressed(os.path.join(raw_dir, filename), **payload)
            index_rows.append({"Model": name, "Horizon": h, "File": filename,
                               "N_seeds": stack.shape[0], "N_test": stack.shape[1],
                               "Alpha_trace": alpha_ok})
    pd.DataFrame(index_rows).to_csv(os.path.join(raw_dir, "index.csv"), index=False)
    print(f"Raw per-seed forecasts saved to: {raw_dir}")


def assert_complete_run(all_results, all_preds, horizons, model_names, expected_seeds):
    """Fail closed if a standard run has any missing seed/model/horizon output."""
    problems = []
    for name in model_names:
        for h in horizons:
            preds = all_preds.get(name, {}).get(h, [])
            if len(preds) != expected_seeds:
                problems.append(f"{name} h={h}: {len(preds)}/{expected_seeds} prediction arrays")
                continue
            for i, pred in enumerate(preds):
                if not np.all(np.isfinite(np.asarray(pred))):
                    problems.append(f"{name} h={h}: non-finite prediction at seed index {i}")
            for key, vals in all_results[name][h].items():
                arr = np.asarray(vals, dtype=float)
                if len(arr) != expected_seeds or not np.all(np.isfinite(arr)):
                    problems.append(f"{name} h={h} metric {key}: incomplete/non-finite")
    if problems:
        raise RuntimeError("Standard run incomplete; inferential outputs are blocked:\n  - " +
                           "\n  - ".join(problems[:50]))
    return True



# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="主实验：多步预测精度对比")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "quick"],
                        help="运行模式")
    args = parser.parse_args()

    print("=" * 60)
    print("  主实验：多步预测精度对比")
    print("=" * 60)

    # 运行实验
    (all_results, all_preds, y_tests, horizons, model_names, scalings, floors,
     all_alpha_traces, seeds) = run_experiment(args.mode)

    # 标准运行必须完整：任何 seed 失败都阻止生成显著性结论。
    if args.mode == "standard":
        assert_complete_run(all_results, all_preds, horizons, model_names, len(seeds))

    # 保存结果 + 原始逐 seed 预测/operative alpha。
    df_main, df_full = save_results(
        all_results, all_preds, y_tests, horizons, model_names,
        scalings=scalings, floors=floors, seeds=seeds,
        all_alpha_traces=all_alpha_traces)

    print("\n" + "=" * 60)
    print("  实验完成！")
    print(f"  结果目录: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()