# -*- coding: utf-8 -*-
"""exp_common.py — 实验脚本共享工具
=====================================
集中放置所有实验脚本都需要的五件事，避免各脚本各自复制一份实现而产生
口径漂移：

1. `load_target_scaling(data_dir, horizon)` / `inverse_target(y, scaling)`
   目标 z-score 标定参数的回逆变换。download_data.py 在构建窗口时对目标
   （平方收益 / GKing 方差）做了 z-score 标准化，并把 (mean, std) 写进
   `<data>/{asset}/preprocess_params.pkl` 的 `target_normalization` 字段。
   **所有报告给论文的指标必须在回逆变换后的"百分比波动率"尺度上计算**，
   否则：
     * MSE/RMSE/MAE 的量纲与表格不符；
     * 目标含负值，MAPE 分母无意义；
     * QLIKE 按定义要求 y_true, y_pred > 0，在标准化尺度上会成片失效。
   This inverse transformation is required for reproducible metrics on the reported scale。

2. 正值投影（论文式 eq:inverse_standardization / eq:positivity_projection）

       yhat_i  = sd_tgt * zhat_i + mean_tgt          (eq:inverse_standardization)
       yhat+_i = max(yhat_i, y_min)                  (eq:positivity_projection)

   **地板 y_min 只由训练切分的目标决定，绝不接触被评分的数据。** protocol
   第 1 条意见指出的正是这条：旧实现把 y_min 写成 `min{ y_i : y_i > 0 }`
   而不限定切分，若最小值取自测试集真实 y_i，测试标签就参与了预测后处理，
   构成 test leakage，并会污染 MSE、QLIKE 以及其后的 DM 检验。本模块现在
   用两层手段把这条钉死：

     * 口径层：`y_min := max( min{ y_i : y_i > 0, i ∈ train },
                              1e-3 * mean{ y_i : i ∈ train } )`
       第一项是protocol specification的主力方案（只依赖训练切分）；第二项对应protocol
       并列给出的"预先固定一个与测试集无关的 eps"那一支。加第二项是必要
       的，不是保险：S&P 500 训练集在 h=1 上存在陈旧价格日，其
       `100|r_t|` 只有 1.4e-9，若直接用它作地板，投影在数值上等同于把负
       预测夹到 0，QLIKE 的方差比值会溢出。两项都只依赖训练切分，因此都
       不构成泄漏。由当前训练集地板口径约束。

     * 执行层：`compute_metrics` 在地板缺失时**报错**，不再静默退化为
       "取被评分目标自身的最小正值"——那正是泄漏本身。诊断用途（例如
       对照"有无投影"两条口径）必须显式传 `allow_evaluation_floor=True`，
       并把返回的 `floor_source` 一起落盘，使口径来源可审计。

   由此得到的性质：
     * 投影对同一 horizon 的所有模型完全相同（同一地板、同一日期集合），
       MSE 等全部指标都在投影后的预测上计算；
     * 投影后恒有 yhat+ > 0，QLIKE 的有效掩码退化为 `y_true > 0`；
     * 命中次数与比例为 `n_projected` / `frac_projected`，落盘供论文
       §4.8 引用（"the number and fraction of dates on which it binds"）。

3. `compute_metrics(y_true, y_pred, scaling=None, floor=None, project=True)`
   统一的指标口径，内置回逆变换、正值投影、尺度自检与命中计数。

4. 多重比较校正 `holm_adjust(pvalues)` / `benjamini_hochberg_adjust(pvalues)`
   / `adjust_family(pvalues)`。multiple-testing protocol指出：DM 表做了 48 次检验
   而未做任何校正，只报告未校正显著性会夸大证据强度。三个函数给出 Holm（控制
   FWER，逐步向下）与 Benjamini--Hochberg（控制 FDR，逐步向上）的校正
   p 值，纯 numpy 实现，不依赖 scipy。

5. `set_seed(seed)` / `get_device()`
   确定性设置，保证同一 seed 可复现。

用法：
    from exp_common import (load_target_scaling, invert_target,
                            load_positivity_floor, compute_metrics,
                            adjust_family)
    scaling = load_target_scaling(DATA_DIR, h)
    floor = load_positivity_floor(DATA_DIR, h)      # 只依赖训练切分的目标
    metrics = compute_metrics(y_test, y_pred, scaling=scaling, floor=floor)
    metrics["MSE"], metrics["QLIKE"], metrics["n_projected"]
"""
import os
import pickle
import random

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:                                    # pragma: no cover
    _HAS_TORCH = False


# download_data.py 落盘的文件名。历史上此处曾用 "params.pkl"，
# 与 producer 不一致并导致回逆变换静默回退到恒等映射 —— 后果是
# 论文表格的 MSE 量纲错误。此处以 producer 为准，并保留旧名作为兜底。
PARAMS_FILENAME = "preprocess_params.pkl"
PARAMS_FILENAME_LEGACY = "params.pkl"

# compute_metrics 返回的键，按此顺序。各脚本用固定键列表收集结果，
# 额外的诊断键（命中计数等）不会破坏它们的累积循环。
METRIC_KEYS = ["MSE", "RMSE", "MAE", "MAPE", "R2", "QLIKE"]
DIAGNOSTIC_KEYS = ["n_projected", "frac_projected", "n_qlike", "floor",
                   "floor_source"]

# 正值投影地板的下限系数（相对训练集目标均值）。
# 见模块 docstring 第 2 条：训练集最小值在 h=1 上会退化为陈旧价格日，
# 需要一个同样与测试集无关的固定量兜底。取 1e-3 的量级依据是：目标以
# 百分比波动率计，训练集均值约 0.79--1.00，故 1e-3*mean 约 8e-4--1e-3，
# 比任何可解释的波动率预测小三个数量级，因此它只在"线性读出落到非正"
# 时起作用，不会改变正常值域上的预测。
EPS_FLOOR_REL = 1e-3


# ============================================================
# 目标标定参数
# ============================================================

def resolve_params_path(data_dir):
    """定位预处理参数文件；兼容旧的 `params.pkl` 命名。"""
    primary = os.path.join(data_dir, PARAMS_FILENAME)
    if os.path.exists(primary):
        return primary
    legacy = os.path.join(data_dir, PARAMS_FILENAME_LEGACY)
    if os.path.exists(legacy):
        return legacy
    return None


def load_target_scaling(data_dir, horizon):
    """读取目标 z-score 标定参数，返回 (mean, std)。

    参数缺失时回退为恒等映射 (0.0, 1.0) 并告警——此时指标落在标准化尺度
    上，与论文口径不可比。
    """
    params_path = resolve_params_path(data_dir)
    if params_path is None:
        print(f"[warn] 未找到 {os.path.join(data_dir, PARAMS_FILENAME)}；"
              f"指标将落在标准化尺度上，与论文口径不可比。"
              f"请先运行 download_data.py。")
        return 0.0, 1.0
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    tn = params.get("target_normalization", {})
    # 键可能是 int 或 str，两者都试。
    key = horizon if horizon in tn else str(horizon)
    if key not in tn:
        print(f"[warn] {PARAMS_FILENAME} 缺少 h={horizon} 的 "
              f"target_normalization；指标将落在标准化尺度上。")
        return 0.0, 1.0
    return float(tn[key]["mean"]), float(tn[key]["std"])


def inverse_target(y, scaling):
    """标准化尺度 -> 百分比波动率尺度： y * std + mean。"""
    mu, sd = scaling
    return np.asarray(y, dtype=np.float64) * sd + mu


# 论文章节里沿用了另一个写法 invert_target（method/experiments 的叙述），
# 这里给出同名别名，避免读者按论文符号搜索时找不到实现。
invert_target = inverse_target


def load_input_scaling(data_dir):
    """读取**输入**（标准化对数收益）的 z-score 常数，返回 (mean, std)。

    计量/朴素基线必须据此把输入窗口还原成收益率与已实现方差：
        r_t = z_t * std_in + mean_in
    否则只能把标准化输入直接当收益用，或在别处退化（历史版本即从 y 批次
    取"序列"，等于用目标自身作自变量）。缺键时返回 (0.0, 1.0) 并告警：
    此时 Persistence / HAR-RV 的绝对水平会错一个尺度因子，指标不可用，
    因此调用方应在正式运行前确认 preprocess_params.pkl 含该字段
    （download_data.py 自本轮起写入）。
    """
    params_path = resolve_params_path(data_dir)
    if params_path is None:
        print(f"[warn] 未找到 {os.path.join(data_dir, PARAMS_FILENAME)}；"
              f"输入标准化常数不可用，将按 (0.0, 1.0) 处理。")
        return 0.0, 1.0
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    inp = params.get("input_normalization")
    if not inp:
        print(f"[warn] {PARAMS_FILENAME} 缺少 input_normalization；"
              f"计量/朴素基线的绝对水平将不可比。请重跑 download_data.py。")
        return 0.0, 1.0
    return float(inp["mean"]), float(inp["std"])


# ============================================================
# 正值投影（eq:positivity_projection）
# ============================================================

def positivity_floor(y_train_ref):
    """训练切分口径的投影地板，输入需已在百分比波动率尺度上。

        y_min := max( min{ y_i : y_i > 0 , i ∈ train },
                      EPS_FLOOR_REL * mean{ y_i : i ∈ train } )

    两个分量都只依赖**训练切分**的目标，与被评分的数据无关，因此不构成
    泄漏。只传训练切分的目标进来是本函数的接口约定：函数名保留历史的
    `positivity_floor`，但语义已由reproducibility protocol 收紧。

    返回 float；训练切分全为非正值（不可能出现，仅作防御）时返回
    EPS_FLOOR_REL 对应的 0.0 下限并告警。
    """
    y = np.asarray(y_train_ref, dtype=np.float64)
    pos = y[y > 0]
    eps = EPS_FLOOR_REL * float(y.mean()) if y.size else 0.0
    if pos.size == 0:
        print("[warn] 训练切分没有正目标值；地板退化为固定 eps。")
        return float(eps)
    return float(max(float(pos.min()), eps))


def load_positivity_floor(data_dir, horizon, split="train"):
    """从**训练切分**读出投影地板（回逆变换到百分比尺度后按上式取）。

    为什么必须限定 split="train"
    ----------------------------
    地板出现在 scoring stage 并被称为 data-dependent floor。若它的最小值
    取自测试集真实 y_i，测试标签就参与了预测后处理：MSE、QLIKE 以及随后
    的 DM 检验都会被污染，而"测试统计量不进入 standardization / 特征构造
    / early stopping"的说明并不能覆盖 scoring 这一环。因此本函数默认且
    推荐只使用训练切分；`split` 参数保留给**审计脚本**用于对比不同切分的
    地板取值，任何非 train 的取值都会被
    警告——它不应出现在正式结果的生产路径上。
    """
    return positivity_floor_detail(data_dir, horizon, split=split)["floor"]


def positivity_floor_detail(data_dir, horizon, split="train"):
    """地板台账：把"来源切分 + 最小正目标 + 固定 eps + 最终地板"一并返回。

    供 `main_experiment.py` 落盘 `results/positivity_floor.csv`，使"地板只由
    训练切分确定"成为可核对的记录（`run_all.py` 的 X7 断言该文件里每一行的
    source_split 都是 train）。审计脚本也可用 split="test" 调用它，用来量化
    "若误用测试集地板"会差多少——那是审计用途，不是生产路径。
    """
    path = os.path.join(data_dir, f"h{horizon}", f"y_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"缺少 {path}，无法确定正值投影地板 y_min。"
            f"地板必须由训练切分确定；请先运行 download_data.py。"
            f"（不要用被评分目标的最小正值代替——那是测试集泄漏。）")
    if split != "train":
        print(f"[warn] 地板台账使用了 split={split!r}；"
              f"正式结果只应使用训练切分，此调用仅用于审计对比。")
    y = inverse_target(np.load(path), load_target_scaling(data_dir, horizon))
    pos = y[y > 0]
    mean_y = float(y.mean()) if y.size else 0.0
    eps = EPS_FLOOR_REL * mean_y
    min_pos = float(pos.min()) if pos.size else float("nan")
    return {
        "source_split": split,
        "n": int(y.size),
        "min_positive_target": min_pos,
        "mean_target": mean_y,
        "eps_rel": EPS_FLOOR_REL,
        "eps": float(eps),
        "floor": float(max(min_pos, eps)) if pos.size else float(eps),
    }


def apply_positivity_projection(y_pred_pct, floor):
    """yhat+ = max(yhat, y_min)。返回 (投影后预测, 命中次数)。

    floor 为 None 时原样返回，命中次数记 0（用于显式关闭投影的诊断）。
    """
    y = np.asarray(y_pred_pct, dtype=np.float64)
    if floor is None:
        return y, 0
    n_proj = int(np.count_nonzero(y < floor))
    return np.maximum(y, floor), n_proj


# ============================================================
# 指标
# ============================================================

def compute_metrics(y_true, y_pred, scaling=None, floor=None, project=True,
                    allow_evaluation_floor=False):
    """MSE / RMSE / MAE / MAPE / R2 / QLIKE，外加投影诊断量。

    流程（与论文 §4.3 一一对应）：
      1. scaling 非 None 时先做回逆变换到百分比波动率尺度；
      2. 若 project=True，用 floor 做正值投影 yhat+ = max(yhat, y_min)；
      3. 全部指标在投影后的预测 yhat+ 上计算，MSE 也不例外；
      4. 返回投影命中次数/比例、QLIKE 有效点数、地板值与地板来源。

    **地板缺失时不再静默退化。** 旧实现在 floor=None 时会取"被评分目标
    自身的最小正值"作地板，这等于让测试标签进入预测后处理（reproducibility protocol）。
    现在改为抛 ValueError；只有显式传 allow_evaluation_floor=True 才允许，
    且返回的 floor_source 会标注为 evaluation，便于审计时一眼看出该次运行
    的地板是否来自被评分数据。正式结果一律由 load_positivity_floor 提供
    训练切分地板。

    投影后 yhat+ 恒为正，QLIKE 的有效掩码退化为 y_true > 0，对同一
    horizon 的所有模型完全一致 —— 这正是论文所声称的可比性，由代码保证。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if scaling is not None:
        y_true = inverse_target(y_true, scaling)
        y_pred = inverse_target(y_pred, scaling)

    if (y_true < 0).any():
        print(f"[warn] compute_metrics 收到 {(y_true < 0).mean():.1%} 的负目标值，"
              f"说明未做回逆变换；MAPE / QLIKE 将不可用。")

    n_total = int(y_true.size)
    if project:
        if floor is None:
            if not allow_evaluation_floor:
                raise ValueError(
                    "正值投影地板缺失：y_min 必须由训练切分确定"
                    "（load_positivity_floor(data_dir, h)，split='train'）。"
                    "用被评分目标自身的最小正值作地板会让测试标签参与预测"
                    "后处理，构成数据泄漏。若确为诊断用途，请显式传 "
                    "allow_evaluation_floor=True。")
            floor = positivity_floor(y_true)
            floor_source = "evaluation"
        else:
            floor_source = "training"
        y_pred, n_projected = apply_positivity_projection(y_pred, floor)
    else:
        n_projected = 0
        floor_source = "disabled" if floor is None else "training"

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    mask = np.abs(y_true) > 1e-8
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.sum() > 0 else np.nan)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # QLIKE 在"方差尺度"上定义： r = y^2 / yhat^2,  loss = r - log r - 1。
    # 要求 y_true, y_pred > 0，这是已实现波动率（非负）的自然定义域。
    # 投影后 y_pred > 0 恒成立，故有效掩码 = (y_true > 0)，与模型无关。
    valid = (y_true > 1e-12) & (y_pred > 1e-12)
    if valid.sum() > 0:
        r = y_true[valid] ** 2 / y_pred[valid] ** 2
        qlike = float(np.mean(r - np.log(r) - 1.0))
    else:
        qlike = np.nan

    return {"MSE": mse, "RMSE": rmse, "MAE": mae,
            "MAPE": mape, "R2": r2, "QLIKE": qlike,
            "n_projected": n_projected,
            "frac_projected": (n_projected / n_total if n_total else np.nan),
            "n_qlike": int(valid.sum()),
            "floor": (np.nan if floor is None else float(floor)),
            "floor_source": floor_source}


# ============================================================
# 跨 seed 汇总
# ============================================================

def sample_std(values):
    """Sample standard deviation over finite values (ddof=1).

    The manuscript reports seed dispersion as the *sample* SD across the
    prespecified training seeds.  Returning NaN for fewer than two finite
    observations keeps quick/smoke modes honest instead of silently switching
    to a population-SD convention.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


# ============================================================
# 多重比较校正（multiple-testing protocol）
# ============================================================

def two_sided_p_from_z(z):
    """双侧正态 p 值。用 erfc 而不是 1-cdf，避免大 |z| 时的灾难性抵消。"""
    z = np.asarray(z, dtype=np.float64)
    p = np.array([_erfc_scaled(abs(float(v))) for v in np.atleast_1d(z)])
    return p if np.ndim(z) else float(p[0])


def _erfc_scaled(az):
    """P(|Z| > az) = erfc(az / sqrt(2))，用 math.erfc 的尾概率形式。"""
    import math
    return math.erfc(az / math.sqrt(2.0))


def holm_adjust(pvalues):
    """Holm 逐步向下校正（控制 FWER）。

    p_(1) <= ... <= p_(m) 为升序；调整量为
        p_adj_(i) = max_{j<=i} min(1, (m - j + 1) * p_(j)),
    单调不减。返回与输入同序的列表。
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        cand = min(1.0, (m - rank) * p[idx])
        running = max(running, cand)
        adj[idx] = running
    return [float(v) for v in adj]


def benjamini_hochberg_adjust(pvalues):
    """Benjamini--Hochberg 逐步向上校正（控制 FDR）。

        p_adj_(i) = min_{j>=i} min(1, m * p_(j) / j),
    单调不减。返回与输入同序的列表。
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    adj = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        cand = min(1.0, m * p[idx] / float(rank + 1))
        running = min(running, cand)
        adj[idx] = running
    return [float(v) for v in adj]


def adjust_family(pvalues, alpha=0.05):
    """一次性给出未校正 / Holm / BH 三套判定与计数。

    返回 dict：
        p_unadjusted, p_holm, p_bh               —— 三个同序列表
        n_tests                                  —— 检验族大小 m
        n_significant_unadjusted / _holm / _bh   —— 各口径下 p < alpha 的个数
        alpha                                    —— 所用水平

    注意：三套判定下"显著"的个数满足
        n_holm <= n_bh <= n_unadjusted
    这是校正的定义性质，`run_all.py` 的 X9 就断言这一条。
    """
    p = [float(v) for v in np.atleast_1d(np.asarray(pvalues, dtype=np.float64))]
    holm = holm_adjust(p)
    bh = benjamini_hochberg_adjust(p)
    return {
        "p_unadjusted": p,
        "p_holm": holm,
        "p_bh": bh,
        "n_tests": len(p),
        "n_significant_unadjusted": int(sum(v < alpha for v in p)),
        "n_significant_holm": int(sum(v < alpha for v in holm)),
        "n_significant_bh": int(sum(v < alpha for v in bh)),
        "alpha": float(alpha),
    }


# ============================================================
# 确定性
# ============================================================

def set_seed(seed):
    """固定全部随机源，保证同一 seed 下结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda=True):
    if _HAS_TORCH and prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"
