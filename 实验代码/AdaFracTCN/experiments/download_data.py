#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_data.py
================
获取并预处理实验数据，生成论文实验所需的滑动窗口数据集（输入窗口 L = 256）。

两种数据来源
------------
--offline
    从已缓存的原始行情 CSV 重建滑动窗口，**不联网**、只依赖 numpy。
    默认读取 data/sp500.csv。这是复现论文数据的推荐方式：原始行情是
    不可再生资源（Yahoo 的历史数据会小幅implementation update），缓存一份之后所有下游
    结果都可以被逐位复算。
（默认）
    通过 yfinance 联网下载；需要 pandas 与 yfinance。

用法
----
    python download_data.py --offline                     # 由缓存重建（L=256）
    python download_data.py --mode standard               # 联网下载完整实验
    python download_data.py --asset AAPL --proxy sqgk     # 指定资产与方差代理

输出（扁平布局）
----------------
    data/sp500.csv                 原始行情（离线模式的输入，也是它的产物）
    data/preprocess_params.pkl     预处理参数（window_len、target_normalization
                                   与 input_normalization）
    data/h{h}/X_train.npy, y_train.npy      训练集（每个 h 一个子目录）
    data/h{h}/X_val.npy,   y_val.npy        验证集
    data/h{h}/X_test.npy,  y_test.npy       测试集
    data/descriptive_stats.txt     描述性统计文本报告

布局说明：`main_experiment.py`、`ablation_experiment.py`、`regime_analysis.py`
等消费者统一按 `data/h{h}/` 与 `data/preprocess_params.pkl` 取数，因此**默认
标的（^GSPC）必须落在 data/ 根下**。非默认标的（--asset）落在
data/{slug}/ 下以免互相覆盖；此时下游脚本需相应指定 DATA_DIR。

预测目标（协议）
----------------
    y_t^(h) = 100 * sqrt( (1/h) * sum_{i=1..h} sigma_{t+i}^2 )   （百分比点）
其中 sigma_t^2 为每日方差代理：proxy='sq' 用平方收益 r_t^2，
proxy='sqgk' 用 Garman-Klass 区间估计量。

样本边界约定（与手稿 §4.2、§4.8 一致）
------------------------------------
预测日期属于哪个切分，由**未来标签窗口** {t+1,...,t+h} 决定；标签窗口必须
完整落在该切分内。输入历史 {t-L+1,...,t} 可以来自更早的切分，因为在预测时
这些收益已经可观测，这不会造成 leakage。这样 2021 年初的测试预测可以合法使用
2020 年及更早的历史，而不会人为丢掉 256 个测试日。训练切分因为没有更早的
对齐历史，仍需先积累 L 个输入位置。
"""

import argparse
import csv
import math
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 模式配置
# ============================================================

MODE_CONFIG = {
    "standard": {
        "start_date": "2000-01-03",
        "end_date": "2024-12-31",
        "train_end": "2017-12-29",
        "val_start": "2018-01-02",
        "val_end": "2020-12-31",
        "test_start": "2021-01-04",
        "test_end": "2024-12-31",
        "horizons": [1, 5, 10, 20],
        "label": "标准模式",
    },
    "quick": {
        "start_date": "2018-01-01",
        "end_date": "2024-12-31",
        "train_end": "2022-06-30",
        "val_start": "2022-07-01",
        "val_end": "2023-06-30",
        "test_start": "2023-07-01",
        "test_end": "2024-12-31",
        "horizons": [1, 10],
        "label": "快速模式",
    },
}

# 默认标的与窗口长度。WINDOW_LEN 与手稿 §4.1 / Table 1（protocol）一致。
DEFAULT_ASSET = "^GSPC"
WINDOW_LEN = 256

# 可下载标的（协议支持跨标的）
ASSETS = ["^GSPC", "^IXIC", "^DJI", "AAPL", "JPM", "XOM"]

# 替代波动率估计量：'sq' = 平方收益 r_t^2；'sqgk' = Garman-Klass 区间估计量
PROXIES = ["sq", "sqgk"]

# 默认标的的原始行情文件名。下游绘图脚本（plot_paper_figures.py）按
# data/sp500.csv 读取收盘价，故此处固定为 sp500.csv 而非 GSPC_raw.csv。
RAW_CSV_NAMES = {"^GSPC": "sp500.csv"}

RAW_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


# ============================================================
# 路径约定
# ============================================================

def asset_slug(asset):
    """把标的代码变成安全的目录名： '^GSPC' -> 'GSPC'。"""
    return asset.replace("^", "").replace("/", "_")


def resolve_output_dir(data_root, asset):
    """默认标的落在 data/ 根下（下游消费者的既有约定），其余落在 data/{slug}/。"""
    if asset == DEFAULT_ASSET:
        return data_root
    return os.path.join(data_root, asset_slug(asset))


def raw_csv_path(out_dir, asset):
    return os.path.join(out_dir, RAW_CSV_NAMES.get(asset, f"{asset_slug(asset)}_raw.csv"))


# ============================================================
# 原始行情读取 / 下载
# ============================================================

def read_raw_csv(path):
    """读取缓存的原始行情 CSV，返回 {列名: 数组}（不依赖 pandas）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到缓存的原始行情文件 {path}；离线模式需要它。"
            f"请先运行一次联网下载，或把原始 CSV 放到该路径。")

    cols = {c: [] for c in RAW_COLUMNS}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        # 表头大小写不敏感
        lut = {k.strip().lower(): k for k in (reader.fieldnames or [])}
        if "close" not in lut or "date" not in lut:
            raise ValueError(f"{path} 缺少 Date/Close 列，表头为 {reader.fieldnames}")
        for row in reader:
            try:
                close = float(row[lut["close"]])
            except (TypeError, ValueError):
                continue
            cols["Date"].append(row[lut["date"]])
            cols["Close"].append(close)
            for key in ("Open", "High", "Low", "Volume"):
                raw_key = lut.get(key.lower())
                try:
                    cols[key].append(float(row[raw_key]) if raw_key else np.nan)
                except (TypeError, ValueError):
                    cols[key].append(np.nan)

    if len(cols["Date"]) < 100:
        raise RuntimeError(f"{path} 有效行数过少（{len(cols['Date'])}）")

    out = {"Date": np.asarray(cols["Date"], dtype=object)}
    for key in ("Open", "High", "Low", "Close", "Volume"):
        out[key] = np.asarray(cols[key], dtype=float)
    return out


def download_asset(ticker, start_date, end_date):
    """从 Yahoo Finance 下载指定标的的日度数据（需要 pandas / yfinance）。"""
    import pandas as pd  # 延迟导入：离线模式不需要

    proxy = "http://127.0.0.1:12334"
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)
    print(f"  正在尝试通过代理连接: {proxy}")

    print(f"  正在从 Yahoo Finance 下载 {ticker} 数据（{start_date} ~ {end_date}）...")
    try:
        import yfinance as yf
    except ImportError:
        print("  [错误] 未安装 yfinance，正在尝试安装...")
        os.system(f"{sys.executable} -m pip install yfinance -q")
        import yfinance as yf

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise RuntimeError(f"标的数据为空，请检查网络连接、日期范围或代码 {ticker}。")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()

    print(f"  下载完成：共 {len(data)} 个交易日")
    # 统一为 'YYYY-MM-DD' 字符串，便于与配置中的边界日期做字典序比较
    date_col = data["Date"]
    if hasattr(date_col, "dt"):
        dates = date_col.dt.strftime("%Y-%m-%d")
    else:
        dates = date_col.astype(str).str.slice(0, 10)
    return {
        "Date": np.asarray(dates, dtype=object),
        "Open": data["Open"].to_numpy(dtype=float),
        "High": data["High"].to_numpy(dtype=float),
        "Low": data["Low"].to_numpy(dtype=float),
        "Close": data["Close"].to_numpy(dtype=float),
        "Volume": data["Volume"].to_numpy(dtype=float),
    }


def save_raw_csv(raw, path):
    """按统一列序回写原始行情，作为离线重建的缓存。"""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RAW_COLUMNS)
        for i in range(len(raw["Date"])):
            writer.writerow([
                raw["Date"][i],
                raw["Open"][i], raw["High"][i], raw["Low"][i],
                raw["Close"][i], raw["Volume"][i],
            ])


# ============================================================
# 特征工程（纯 numpy）
# ============================================================

def _rolling_std(x, window, ddof=1):
    """与 pandas .rolling(window).std(ddof=1) 等价；前 window-1 个为 NaN。"""
    n = x.size
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[i - window + 1:i + 1]
        if not np.isnan(seg).any():
            out[i] = np.std(seg, ddof=ddof)
    return out


def compute_features(raw):
    """计算对数收益率、Garman-Klass 区间方差等特征，并删除缺失行。

    返回等长（已对齐、已 dropna）的 numpy 数组字典。
    """
    close = np.asarray(raw["Close"], dtype=float)
    high = np.asarray(raw["High"], dtype=float)
    low = np.asarray(raw["Low"], dtype=float)
    open_ = np.asarray(raw["Open"], dtype=float)
    dates = np.asarray(raw["Date"], dtype=object)
    n = close.size

    logret = np.full(n, np.nan)
    logret[1:] = np.log(close[1:] / close[:-1])
    absret = np.abs(logret)
    sqret = logret ** 2

    # Garman-Klass（1980）日方差区间估计量（proxy='sqgk' 采用）：
    #   GK_t = 0.5 * ln(H_t/L_t)^2 - (2*ln2 - 1) * ln(C_t/O_t)^2
    # 该式在极端情况下可能为负，此处 clip 保证非负的"方差"口径。
    hl = np.log(high / low) ** 2
    co = np.log(close / open_) ** 2
    gk = np.clip(0.5 * hl - (2 * math.log(2) - 1) * co, 0.0, None)

    roll = _rolling_std(logret, 20, ddof=1)

    keep = ~np.isnan(logret) & ~np.isnan(roll)
    return {
        "Date": dates[keep],
        "LogReturn": logret[keep],
        "AbsReturn": absret[keep],
        "SqReturn": sqret[keep],
        "GKingVar": gk[keep],
        "RollVol20": roll[keep],
    }


# ============================================================
# 描述性统计（纯 numpy；ADF 依赖 statsmodels，缺失时记 NaN）
# ============================================================

def descriptive_statistics(returns):
    """基本统计量。偏度/超额峰度/Jarque-Bera 与 scipy 的默认口径一致
    （bias=True, fisher=True）；JB 的 p 值用 2 自由度卡方生存函数解析式
    P(X > x) = exp(-x/2)，因此不依赖 scipy。"""
    r = np.asarray(returns, dtype=float)
    dev = r - r.mean()
    m2 = np.mean(dev ** 2)
    m3 = np.mean(dev ** 3)
    m4 = np.mean(dev ** 4)
    skew = m3 / m2 ** 1.5 if m2 > 0 else np.nan
    exkurt = m4 / m2 ** 2 - 3.0 if m2 > 0 else np.nan
    jb = r.size / 6.0 * (skew ** 2 + exkurt ** 2 / 4.0)
    jb_pvalue = float(math.exp(-jb / 2.0)) if np.isfinite(jb) else np.nan

    # ADF：协议要求，但需要 statsmodels；缺失时记为 NaN 并在报告中体现。
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_pvalue = float(adfuller(r, autolag="AIC")[1])
    except Exception:
        adf_pvalue = np.nan

    return {
        "观测数": int(r.size),
        "均值": float(r.mean()),
        "标准差": float(r.std(ddof=1)),
        "偏度": float(skew),
        "超额峰度": float(exkurt),
        "Jarque-Bera 统计量": float(jb),
        "Jarque-Bera p值": jb_pvalue,
        "ADF p值": adf_pvalue,
    }


# ============================================================
# Hurst 指数估计
# ============================================================

def hurst_rs(series):
    """R/S 重标极差分析：拟合 log(R/S) vs log(n) 的斜率。"""
    s = np.asarray(series, dtype=float)
    N = len(s)
    if N < 32:
        return np.nan

    ns = []
    n = 16
    while n <= N // 2:
        ns.append(n)
        n = int(n * 1.5)
    if len(ns) < 3:
        ns = [N // 8, N // 4, N // 2]

    rs_values = []
    for n in ns:
        num_blocks = N // n
        if num_blocks < 1:
            continue
        rs_list = []
        for i in range(num_blocks):
            block = s[i * n:(i + 1) * n]
            mean_block = np.mean(block)
            cumdev = np.cumsum(block - mean_block)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((n, np.mean(rs_list)))

    if len(rs_values) < 3:
        return np.nan

    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    return np.polyfit(log_n, log_rs, 1)[0]


def hurst_mfdfa(series, q=2):
    """MF-DFA：对累积离差序列去趋势后拟合 log F_q(s) vs log s（q=2 即 H）。"""
    s = np.asarray(series, dtype=float)
    N = len(s)
    if N < 64:
        return np.nan

    Y = np.cumsum(s - np.mean(s))

    scales = []
    sc = 16
    while sc <= N // 4:
        scales.append(sc)
        sc = int(sc * 1.5)
    if len(scales) < 3:
        scales = [N // 16, N // 8, N // 4]

    fq_values = []
    for scale in scales:
        n_segments = N // scale
        if n_segments < 2:
            continue
        variances = []
        for i in range(n_segments):
            segment = Y[i * scale:(i + 1) * scale]
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            detrended = segment - np.polyval(coeffs, x)
            variances.append(np.mean(detrended ** 2))
        if variances:
            fq = np.mean(np.array(variances) ** (q / 2.0)) ** (1.0 / q)
            fq_values.append((scale, fq))

    if len(fq_values) < 3:
        return np.nan

    log_s = np.log([v[0] for v in fq_values])
    log_fq = np.log([v[1] for v in fq_values])
    return np.polyfit(log_s, log_fq, 1)[0]


def compute_hurst_all(returns):
    """对原始收益率、绝对收益率、平方收益率分别计算 Hurst 指数。"""
    r = np.asarray(returns, dtype=float)
    results = {}
    for name, data in [("原始收益率", r), ("绝对收益率", np.abs(r)),
                       ("平方收益率", r ** 2)]:
        results[name] = {"R/S": hurst_rs(data), "MF-DFA": hurst_mfdfa(data)}
    return results


# ============================================================
# 数据划分与预处理
# ============================================================

def split_masks(dates, config):
    """按时间顺序划分训练/验证/测试（返回布尔掩码）。"""
    d = np.asarray(dates, dtype=object)
    train = d <= config["train_end"]
    val = (d >= config["val_start"]) & (d <= config["val_end"])
    test = (d >= config["test_start"]) & (d <= config["test_end"])
    if not (train | val | test).all():
        raise ValueError("存在未被任何划分覆盖的观测；请检查日期配置。")
    return {"train": train, "val": val, "test": test}


def zscore_normalize(values, train_mask, tag):
    """用训练集统计量做 z-score 标准化（ddof=1），返回 (标准化数组, 参数)。"""
    mu = float(values[train_mask].mean())
    sd = float(values[train_mask].std(ddof=1))
    if sd <= 0:
        sd = 1.0
    print(f"  {tag} z-score：mu={mu:.8f}, sigma={sd:.8f}")
    return (values - mu) / sd, {"mean": mu, "std": sd}


# ============================================================
# 滑动窗口构建
# ============================================================

def create_windows(input_vals, var_vals, window_len, horizon):
    """Legacy all-within-one-array window builder (kept for small diagnostics)."""
    input_vals = np.asarray(input_vals, dtype=float)
    var_vals = np.asarray(var_vals, dtype=float)
    N = input_vals.size
    n_samples = N - window_len - (horizon - 1)
    if n_samples <= 0:
        raise ValueError(
            f"序列长度 {N} 不足以容纳 L={window_len} 与 h={horizon} 的窗口")
    X = np.empty((n_samples, window_len, 1), dtype=np.float32)
    y = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        X[i, :, 0] = input_vals[i:i + window_len]
        future_var = var_vals[i + window_len:i + window_len + horizon]
        y[i] = math.sqrt(float(np.mean(future_var))) * 100.0
    return X, y


def create_split_windows(input_vals, var_vals, dates, split_mask, window_len, horizon):
    """Build samples whose *label window* is wholly inside one split.

    Historical inputs may precede the split boundary. This is the correct
    forecasting chronology: at a 2021 forecast origin, observations from 2020
    are known information, not leakage. Only future targets are forbidden from
    crossing the split boundary.

    Returns ``X, y, origin_dates, target_end_dates``.
    """
    input_vals = np.asarray(input_vals, dtype=float)
    var_vals = np.asarray(var_vals, dtype=float)
    dates = np.asarray(dates, dtype=object)
    mask = np.asarray(split_mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("empty split")
    lo, hi = int(idx[0]), int(idx[-1])
    N = input_vals.size

    starts = []
    origins = []
    for origin in range(window_len - 1, N - horizon):
        target_first = origin + 1
        target_last = origin + horizon
        if target_first >= lo and target_last <= hi:
            starts.append(origin - window_len + 1)
            origins.append(origin)

    if not starts:
        raise ValueError(
            f"split too short for L={window_len}, h={horizon}: {dates[lo]}..{dates[hi]}")

    X = np.empty((len(starts), window_len, 1), dtype=np.float32)
    y = np.empty(len(starts), dtype=float)
    origin_dates = np.empty(len(starts), dtype='U10')
    target_end_dates = np.empty(len(starts), dtype='U10')
    for q, (st, origin) in enumerate(zip(starts, origins)):
        X[q, :, 0] = input_vals[st:origin + 1]
        future_var = var_vals[origin + 1:origin + 1 + horizon]
        y[q] = math.sqrt(float(np.mean(future_var))) * 100.0
        origin_dates[q] = str(dates[origin])
        target_end_dates[q] = str(dates[origin + horizon])

    return X, y, origin_dates, target_end_dates


def build_and_save_windows(feat, masks, config, out_dir, window_len, proxy):
    """Build chronological samples with label-only split membership.

    Input and target z-score parameters are estimated from the training split
    only. Validation/test samples may use pre-boundary historical inputs, but
    every target date must remain inside its own split. The per-sample forecast
    origin and target-end dates are saved so later audits can prove chronology.
    """
    input_vals, input_params = zscore_normalize(feat["LogReturn"], masks["train"], "输入")
    var_col = "SqReturn" if proxy == "sq" else "GKingVar"
    var_vals = feat[var_col]
    if not np.isfinite(var_vals).all():
        raise ValueError(
            f"方差代理 {var_col} 含 NaN（{np.isnan(var_vals).sum()} 个）；"
            f"proxy={proxy} 需要原始行情包含 Open/High/Low 列。")

    horizons = config["horizons"]
    all_shapes = {}
    target_norm = {}

    for h in horizons:
        h_dir = os.path.join(out_dir, f"h{h}")
        os.makedirs(h_dir, exist_ok=True)
        arrays = {}
        for split in ("train", "val", "test"):
            X, y, origin_dates, target_end_dates = create_split_windows(
                input_vals, var_vals, feat["Date"], masks[split], window_len, h)
            arrays[split] = (X, y, origin_dates, target_end_dates)

            # Chronology assertions: every target stays inside the split and
            # origin precedes target end. Validation/test inputs are allowed to
            # begin before split start by design.
            split_dates = feat["Date"][masks[split]]
            assert target_end_dates[0] >= str(split_dates[0])
            assert target_end_dates[-1] <= str(split_dates[-1])
            assert np.all(origin_dates < target_end_dates)

        mu_y = float(arrays["train"][1].mean())
        sd_y = float(arrays["train"][1].std(ddof=1))
        if sd_y <= 0:
            sd_y = 1.0
        target_norm[h] = {"mean": mu_y, "std": sd_y}

        for split in ("train", "val", "test"):
            X, y, origin_dates, target_end_dates = arrays[split]
            y_norm = ((y - mu_y) / sd_y).astype(np.float32)
            np.save(os.path.join(h_dir, f"X_{split}.npy"), X)
            np.save(os.path.join(h_dir, f"y_{split}.npy"), y_norm)
            np.save(os.path.join(h_dir, f"origin_dates_{split}.npy"), origin_dates)
            np.save(os.path.join(h_dir, f"target_end_dates_{split}.npy"), target_end_dates)

        all_shapes[h] = {
            "X_train": arrays["train"][0].shape, "y_train": arrays["train"][1].shape,
            "X_val": arrays["val"][0].shape, "y_val": arrays["val"][1].shape,
            "X_test": arrays["test"][0].shape, "y_test": arrays["test"][1].shape,
        }
        print(f"  h={h:2d}: train={arrays['train'][0].shape}, "
              f"val={arrays['val'][0].shape}, test={arrays['test'][0].shape} "
              f"(target mu={mu_y:.4f}, sd={sd_y:.4f})")

    return all_shapes, target_norm, input_params


# ============================================================
# 报告生成
# ============================================================

def generate_report(stats_dict, hurst_results, all_shapes, config, out_dir,
                    asset, proxy, window_len):
    """生成描述性统计文本报告。"""
    L = []
    add = L.append
    add("=" * 60)
    add(f"S&P 500 描述性统计报告（{config['label']}）")
    add("=" * 60)
    add("")

    add("【基本统计量（对数收益率）】")
    add("-" * 40)
    for key, val in stats_dict.items():
        if isinstance(val, float):
            if not np.isfinite(val):
                add(f"  {key:30s}: nan")
            elif "p值" in key:
                add(f"  {key:30s}: {val:.6f}")
            else:
                add(f"  {key:30s}: {val:.8f}")
        else:
            add(f"  {key:30s}: {val}")
    add("")

    add("【Hurst 指数】")
    add("-" * 40)
    add(f"  {'序列':15s} {'R/S 分析':>12s} {'MF-DFA':>12s}")
    for name, methods in hurst_results.items():
        h_rs, h_dfa = methods["R/S"], methods["MF-DFA"]
        add(f"  {name:15s} {('%.4f' % h_rs) if np.isfinite(h_rs) else 'N/A':>12s} "
            f"{('%.4f' % h_dfa) if np.isfinite(h_dfa) else 'N/A':>12s}")
    add("")

    h_abs_rs = hurst_results.get("绝对收益率", {}).get("R/S", np.nan)
    h_abs_dfa = hurst_results.get("绝对收益率", {}).get("MF-DFA", np.nan)
    if np.isfinite(h_abs_rs):
        add(f"  隐含分数差分参数 d (R/S)  = H - 0.5 = {h_abs_rs - 0.5:.4f}")
    if np.isfinite(h_abs_dfa):
        add(f"  隐含分数差分参数 d (DFA)  = H - 0.5 = {h_abs_dfa - 0.5:.4f}")
    add("")

    add("【滑动窗口数据形状】")
    add("-" * 40)
    for h, shapes in all_shapes.items():
        add(f"  h={h}:")
        for name, shape in shapes.items():
            add(f"    {name:12s}: {shape}")
    add("")

    add("【配置信息】")
    add("-" * 40)
    for key, val in config.items():
        add(f"  {key:15s}: {val}")
    add(f"  {'asset':15s}: {asset}")
    add(f"  {'proxy':15s}: {proxy}")
    add(f"  {'window_len':15s}: {window_len}")
    add("")

    text = "\n".join(L)
    with open(os.path.join(out_dir, "descriptive_stats.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)
    return text


# ============================================================
# 主函数
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="获取/重建收益率波动率实验数据")
    p.add_argument("--mode", type=str, default="standard",
                   choices=["standard", "quick"],
                   help="运行模式：standard（完整实验）或 quick（快速验证）")
    p.add_argument("--asset", type=str, default=DEFAULT_ASSET, choices=ASSETS,
                   help="标的代码（指数或个股）：默认 ^GSPC")
    p.add_argument("--proxy", type=str, default="sq", choices=PROXIES,
                   help="日方差代理：sq（平方收益）或 sqgk（Garman-Klass）")
    p.add_argument("--offline", action="store_true",
                   help="不联网：从缓存的原始行情 CSV 重建滑动窗口")
    p.add_argument("--window-len", type=int, default=WINDOW_LEN,
                   help=f"输入窗口长度（默认 {WINDOW_LEN}，与手稿一致）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = MODE_CONFIG[args.mode]
    asset, proxy, window_len = args.asset, args.proxy, args.window_len

    print("=" * 60)
    print(f"  数据获取与预处理 — {config['label']} "
          f"(asset={asset}, proxy={proxy}, L={window_len})")
    print(f"  来源：{'缓存重建（离线）' if args.offline else 'Yahoo Finance（联网）'}")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data")
    out_dir = resolve_output_dir(data_root, asset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  输出目录: {out_dir}")

    # Step 1: 原始行情
    raw_path = raw_csv_path(out_dir, asset)
    print("\n[1/6] 获取原始行情...")
    if args.offline:
        raw = read_raw_csv(raw_path)
        print(f"  已读取缓存 {raw_path}：{raw['Close'].size} 个交易日")
    else:
        raw = download_asset(asset, config["start_date"], config["end_date"])
        save_raw_csv(raw, raw_path)
        print(f"  原始数据已缓存至 {raw_path}")

    # Step 2: 特征工程
    print("\n[2/6] 计算特征（对数收益率、Garman-Klass 区间方差等）...")
    feat = compute_features(raw)
    print(f"  特征计算完成，有效数据 {feat['LogReturn'].size} 天 "
          f"({feat['Date'][0]} ~ {feat['Date'][-1]})")

    # Step 3: 描述性统计
    print("\n[3/6] 计算描述性统计量...")
    stats_dict = descriptive_statistics(feat["LogReturn"])
    for key, val in stats_dict.items():
        print(f"    {key:30s}: "
              f"{('%.8f' % val) if isinstance(val, float) and np.isfinite(val) else val}")

    # Step 4: Hurst 指数
    print("\n[4/6] 计算 Hurst 指数（R/S 分析 & MF-DFA）...")
    hurst_results = compute_hurst_all(feat["LogReturn"])
    for name, methods in hurst_results.items():
        h_rs, h_dfa = methods["R/S"], methods["MF-DFA"]
        print(f"    {name:15s}: R/S={('%.4f' % h_rs) if np.isfinite(h_rs) else 'N/A'}, "
              f"MF-DFA={('%.4f' % h_dfa) if np.isfinite(h_dfa) else 'N/A'}")

    # Step 5: 数据划分
    print("\n[5/6] 数据划分...")
    masks = split_masks(feat["Date"], config)
    for split in ("train", "val", "test"):
        sl = feat["Date"][masks[split]]
        print(f"    {split:5s}: {sl.size} 天 ({sl[0]} ~ {sl[-1]})")

    # Step 6: 滑动窗口
    print("\n[6/6] 构建滑动窗口数据...")
    all_shapes, target_norm, input_norm = build_and_save_windows(
        feat, masks, config, out_dir, window_len, proxy)

    params_path = os.path.join(out_dir, "preprocess_params.pkl")
    with open(params_path, "wb") as fh:
        pickle.dump({
            "asset": asset,
            "proxy": proxy,
            "window_len": window_len,
            "horizons": config["horizons"],
            "mode": args.mode,
            "data_config": config,
            "target_normalization": target_norm,
            # 输入（标准化对数收益）自己的 z-score 常数。
            # 计量/朴素基线据此把输入窗口还原成收益率 / 已实现方差序列；
            # 缺了它，baselines.py 只能从 y 批次取"序列"，即用目标当自变量。
            "input_normalization": {"mean": input_norm["mean"],
                                    "std": input_norm["std"]},
        }, fh)
    print(f"  预处理参数已保存至 {params_path}")

    generate_report(stats_dict, hurst_results, all_shapes, config, out_dir,
                    asset, proxy, window_len)

    print("\n" + "=" * 60)
    print("  数据预处理全部完成！")
    print(f"  输出目录: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
