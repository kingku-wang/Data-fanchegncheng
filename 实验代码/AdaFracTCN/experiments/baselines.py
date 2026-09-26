# -*- coding: utf-8 -*-
"""
baselines.py — 基线模型实现
============================
包含朴素/计量基线 (Persistence, Uncond mean, HAR-RV, GARCH, EGARCH) 与
深度学习基线 (ARIMA, LSTM, GRU, TCN, Transformer, Informer, Frac-LSTM)。
所有模型继承 BaseModel，统一 fit/predict 接口。

注意：协议将输入窗口定为 L=256，深度基线使用 param_budget.py 中逐模型容量匹配宽度；
Informer 与 Transformer 架构真正不同（Informer 使用 ProbSparse 自注意力 + distilling）。
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from scipy import stats

# 结构常量与训练协议常量从 adafractcn 引入，作为**唯一来源**：主表与消融
# 表必须共用同一个 TCN、同一套训练超参数，两处各存一份数值必然漂移
# （duplicate-constant drift 就是这么来的）。
from adafractcn import (TCN_HIDDEN, TCN_LAYERS, BASELINE_WIDTHS,
                        TRAIN_EPOCHS, TRAIN_PATIENCE, TRAIN_LR, TRAIN_ETA_MIN,
                        TRAIN_WEIGHT_DECAY, forecasting_loss)

warnings.filterwarnings("ignore")

# 默认输入窗口（协议：L=256）
DEFAULT_INPUT_LEN = 256


# ============================================================
# 基类
# ============================================================

class BaseModel(nn.Module):
    """所有基线模型的抽象基类。"""

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.mode = mode
        self._is_trained = False

    def fit(self, train_loader, val_loader, epochs=None, patience=None, device="cpu"):
        """通用训练循环。"""
        if epochs is None:
            epochs = TRAIN_EPOCHS if self.mode == "standard" else 20
        if patience is None:
            patience = TRAIN_PATIENCE if self.mode == "standard" else 3

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=TRAIN_LR,
                                    weight_decay=TRAIN_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=TRAIN_ETA_MIN)
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = self(X_batch)  # (B, horizon) or (B, 1)
                # 对齐 pred 和 y_batch 的形状
                if pred.dim() > 1 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)  # (B,)
                loss = forecasting_loss(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)
            scheduler.step()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = self(X_batch)
                    if pred.dim() > 1 and pred.size(-1) == 1:
                        pred = pred.squeeze(-1)
                    val_loss += forecasting_loss(pred, y_batch).item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # 恢复验证集最优权重。此前这里刻意"保留训练后期状态、不恢复最优"，
        # 而 AdaFracTCN.fit 一直恢复 best_state —— 同一张表里两种早停口径，
        # 基线因此被系统性削弱，与"matched training budget / protocol"的声明
        # 冲突。现在两边都恢复到最优验证状态，比较才成立。
        if best_state is not None:
            self.load_state_dict(best_state)
        self._is_trained = True
        self.to(device)
        return self

    def predict(self, test_loader, device="cpu"):
        """返回 shape (N,) 的 numpy 预测值。

        说明：直接返回确定性前向输出，不添加任何人为的预测不确定度扰动。
        （earlier implementation曾在本方法里加入 MC-Dropout 样式的 randn 噪声artificial output noise，已删除，
        protocol requires预测必须来自模型本身的确定性映射。）
        """
        self.eval()
        self.to(device)
        preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                pred = self(X_batch)
                if pred.dim() > 1 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                # 若输出仍为多列（horizon>1），取最后一列
                if pred.dim() > 1:
                    pred = pred[:, -1]
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return preds

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# ARIMA
# ============================================================

class ARIMAModel:
    """ARIMA 基线（非深度学习，独立接口）。"""

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        self.input_len = input_len
        self.horizon = horizon
        self.mode = mode
        self._model = None

    def fit(self, train_loader, val_loader=None, epochs=None, patience=None, device="cpu"):
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("请安装 pmdarima: pip install pmdarima")

        series = _extract_series(train_loader)
        self._model = auto_arima(series, seasonal=False, trace=False,
                                 suppress_warnings=True, error_action="ignore")
        return self

    def predict(self, test_loader, device="cpu"):
        if self._model is None:
            raise RuntimeError("模型尚未训练")
        series = _extract_series(test_loader)
        preds = self._model.predict(n_periods=len(series))
        return np.asarray(preds, dtype=np.float32)

    def count_parameters(self):
        return 0


def _extract_series(loader):
    """从 DataLoader 中提取一维序列（取最后时间步作为近似）。"""
    series = []
    for X_batch, y_batch in loader:
        series.extend(y_batch.numpy().tolist())
    return np.asarray(series, dtype=np.float64)


def _extract_input_series(loader):
    """从 DataLoader 中提取每个样本"当前日"的**原尺度**日收益。

    X[i] 的最后一个时间步 X[i][-1][0] 是样本 i 当前日的标准化对数收益，
    乘上输入的 z-score 常数即原尺度收益 r_t（见 _raw_returns）。
    该序列用于 GARCH/EGARCH 在"训练收益"上的极大似然估计。
    """
    return _raw_returns(_extract_input_windows(loader)[:, -1])


def _extract_input_windows(loader):
    """从 DataLoader 中提取整个输入窗口：返回 (N, L) 的标准化对数收益矩阵。

    计量/朴素基线的公式都定义在**可解释的收益率 / 已实现方差**上：
      * Persistence 用最近 h 日的已实现波动率；
      * HAR-RV 用日/周/月三个已实现方差分量；
      * 二者需要的滞后全部落在输入窗口之内（L = 256 ≫ h, 22）。
    因此只要拿到窗口矩阵与输入标准化常数，就能逐位复算它们的预测，
    不必也不应从 DataLoader 的 y 批次取"序列"——那样取到的是**目标**，
    与自变量同期，构成泄漏。
    """
    chunks = []
    for X_batch, _ in loader:
        chunks.append(X_batch.numpy()[:, :, 0].astype(np.float64))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0))


# ============================================================
# 协议标定常数（由驱动脚本按 (asset, horizon) 设置一次）
# ============================================================
#
# 输入窗口里的是 z_t = (r_t - mu_in) / sd_in，模型内部输出的是**标准化目标**
# 尺度上的预测。计量/朴素基线要按论文 §4.2 的公式工作，就必须同时知道
# (mu_in, sd_in)（还原收益率与已实现方差）和 (mu_tgt, sd_tgt)（把公式给出的
# 百分比波动率预测换算回标准化尺度）。两组常数来自
# data/preprocess_params.pkl，由 download_data.py 写入。
#
# Implementation note：PersistenceModel 与 HARRVModel 曾直接从
# DataLoader 的 y 批次取"序列"，即把**目标本身**当作已实现方差的自变量；
# 结果一是 Persistence 退化为"训练集最后一个目标值"的常数预测，与论文
# "最近 h 日已实现波动率"的公式不符；二是 HAR 的日分量与目标同期，构成泄漏。
_PROTOCOL_SCALING = {"input": (0.0, 1.0), "target": (0.0, 1.0)}


def set_protocol_scaling(input_scaling=None, target_scaling=None):
    """设置输入/目标的 z-score 常数；由驱动脚本在拿到某 (asset, horizon)
    的数据后调用一次。返回更新后的副本，便于日志记录。"""
    if input_scaling is not None:
        _PROTOCOL_SCALING["input"] = (float(input_scaling[0]),
                                      float(input_scaling[1]))
    if target_scaling is not None:
        _PROTOCOL_SCALING["target"] = (float(target_scaling[0]),
                                       float(target_scaling[1]))
    return dict(_PROTOCOL_SCALING)


def get_protocol_scaling():
    """当前协议标定常数，供驱动脚本落盘/审计使用。"""
    return dict(_PROTOCOL_SCALING)


def _raw_returns(windows):
    """标准化对数收益窗口 -> 原尺度日收益 r_t（逐位可复算）。"""
    mu_in, sd_in = _PROTOCOL_SCALING["input"]
    return windows * sd_in + mu_in


def _to_standardized_target(vol_pct):
    """百分比波动率预测 -> 标准化目标尺度（与回逆变换互为逆映射）。"""
    mu_tgt, sd_tgt = _PROTOCOL_SCALING["target"]
    return (np.asarray(vol_pct, dtype=np.float64) - mu_tgt) / sd_tgt


# ============================================================
# 朴素 / 计量基线（协议新增 5 个：Persistence、Uncond mean、HAR-RV、GARCH、EGARCH）
# ============================================================
#
# 说明：download_data 对目标 y^(h) 做了 z-score 标准化。因此下列计量基线
# 所拟合/外推的"目标序列"即 loader 中的 y（标准化 h 日平均已实现波动率）。
# 各公式按协议书写；指标在实验脚本中按需回逆变换。

class PersistenceModel:
    """Persistence / 随机游走基线。

    协议公式：y_hat^(h)_t = 100 * sqrt( (1/h) * sum_{i=0..h-1} RV_{t-i} )
    即"把最近 h 日的已实现波动率水平当作未来 h 日的预测"。

    实现说明（Implementation note）
    --------------------
    所需的 RV_{t-i} 全部落在输入窗口的最后 h 个位置上，因此每个测试样本
    的预测可以**逐日**算出来，而不是像旧实现那样取"训练集最后一个目标值"
    当作整段测试期的常数。那种常数预测既不是随机游走规则，也不可能击败
    最佳常数预测（无条件均值）；而论文报告 persistence 在 h=1 上是全表
    最弱的一行（MSE 1.005 对常数 0.556），两件事不可能同时成立。
    h=1 时退化为 y_hat = 100*|r_t|，与论文 §4.4 的叙述一致。
    """

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        self.input_len = input_len
        self.horizon = horizon
        self.mode = mode

    def fit(self, train_loader, val_loader=None, epochs=None, patience=None, device="cpu"):
        # 随机游走无需估计参数：预测完全由输入窗口决定。
        return self

    def predict(self, test_loader, device="cpu"):
        windows = _extract_input_windows(test_loader)           # (N, L)
        h = min(self.horizon, windows.shape[1])
        ret = _raw_returns(windows[:, -h:])                     # 最近 h 日收益
        vol_pct = 100.0 * np.sqrt(np.mean(ret ** 2, axis=1))
        return _to_standardized_target(vol_pct).astype(np.float32)

    def count_parameters(self):
        return 0


class UnconditionalMeanModel:
    """无条件均值基线：预测为训练集目标 y^(h) 的均值（常数预测）。

    它是"最佳常数预测"，也是 persistence 在 h=1 上的绑定对照：两者一起
    把可实现 R^2 的两端夹住（见论文 §4.4 与 §4.8）。
    """

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        self.input_len = input_len
        self.horizon = horizon
        self.mode = mode
        self._mean = None

    def fit(self, train_loader, val_loader=None, epochs=None, patience=None, device="cpu"):
        series = _extract_series(train_loader)
        self._mean = float(np.mean(series))
        return self

    def predict(self, test_loader, device="cpu"):
        if self._mean is None:
            raise RuntimeError("模型尚未训练")
        n = sum(len(y) for _, y in test_loader)
        return np.full(n, self._mean, dtype=np.float32)

    def count_parameters(self):
        return 0


class HARRVModel:
    """HAR-RV 基线（Corsi, 2009）。

    以日/周/月已实现方差为自变量，对 h 日平均已实现波动率的**标准化目标**
    做 OLS；测试期使用固定系数（训练期估计，测试期不再更新）。

       daily_t   = RV_t
       weekly_t  = (1/5)  * sum_{i=0..4}  RV_{t-i}
       monthly_t = (1/22) * sum_{i=0..21} RV_{t-i}

    实现说明（Implementation note）
    --------------------
    三个分量必须由**当前日及其之前**的已实现方差构成，而这些滞后全部落在
    输入窗口的最后 22 个位置上（L = 256）。旧实现从 DataLoader 的 y 批次
    取序列当自变量，取到的其实是**目标本身**：于是日分量 RV_t 与目标 y_t
    同期，测试期的 HAR 预测直接读入了它要预测的量——这是本仓库里最严重的
    一处泄漏。现在改为由输入窗口还原：RV_t = (z_t * sd_in + mu_in)^2，
    与 download_data.py 写出的 input_normalization 逐位一致。
    """

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        self.input_len = input_len
        self.horizon = horizon
        self.mode = mode
        self._coef = None

    def _design(self, windows):
        """输入窗口矩阵 -> HAR 设计阵（取每个窗口的最后一个位置作为 t）。

        每个样本的日/周/月分量都只用到该样本输入窗口内的已实现方差，
        因此没有任何前视信息。
        """
        ret = _raw_returns(windows)
        rv = ret ** 2
        n = rv.shape[0]
        daily = rv[:, -1]
        weekly = rv[:, -5:].mean(axis=1) if rv.shape[1] >= 5 else daily
        monthly = rv[:, -22:].mean(axis=1) if rv.shape[1] >= 22 else weekly
        return np.column_stack([np.ones(n), daily, weekly, monthly])

    @staticmethod
    def _fit_ols(X, y):
        # coef = (X'X)^{-1} X'y；任意一行含 NaN 即整行剔除。
        valid = ~np.any(np.isnan(X), axis=1) & np.isfinite(y)
        Xv, yv = X[valid], y[valid]
        if len(yv) <= Xv.shape[1]:
            raise RuntimeError("HAR-RV：有效样本不足")
        return np.linalg.pinv(Xv.T @ Xv) @ (Xv.T @ yv)

    def fit(self, train_loader, val_loader=None, epochs=None, patience=None, device="cpu"):
        windows = _extract_input_windows(train_loader)
        if windows.shape[1] < 22:
            raise RuntimeError("HAR-RV：输入窗口不足以构造月度分量（需 L >= 22）")
        X = self._design(windows)
        y = _extract_series(train_loader)
        if y.size != X.shape[0]:
            raise RuntimeError(f"HAR-RV：设计阵与目标长度不一致 {X.shape[0]} vs {y.size}")
        self._coef = self._fit_ols(X, y)
        return self

    def predict(self, test_loader, device="cpu"):
        if self._coef is None:
            raise RuntimeError("模型尚未训练")
        X = self._design(_extract_input_windows(test_loader))
        return (X @ self._coef).astype(np.float32)

    def count_parameters(self):
        return 0


# ============================================================
# GARCH / EGARCH
# ============================================================
#
# 协议：参数在**训练收益**上做极大似然估计，测试期系数固定不再更新；在每个
# 测试日 t 上，用该参数把条件方差递推**重新过滤**到 t，再向前迭代 h 步、
# 取均值后开方，得到该日的预测。这与协议"所有模型共享同一信息集与同一输入
# 窗口"的约定一致：过滤所需的收益全部落在输入窗口之内（L=256 ≫ 预热长度）。
#
# Implementation note修掉两处口径错误：
#   (1) 旧实现只在**训练期末**做一次 h 步预报，把结果当作整段测试期的常数，
#       于是 GARCH 行与常数预测无异，不可能优于无条件均值——而论文报告的
#       GARCH/EGARCH 却优于全部常数预测，两者不可能同时成立；
#   (2) 旧实现的返回值落在**百分比波动率**尺度上，而评分管线会对模型输出
#       再施加一次回逆变换，等于对 GARCH 的预测二次换尺度（量纲错误）。
#       现在统一返回标准化目标尺度，与其余模型一致。

def _garch_params(res, vol):
    """从 arch 的拟合结果里取参数，返回按名称索引的 dict。"""
    p = {k: float(v) for k, v in res.params.items()}
    need = (["omega", "alpha[1]"] if vol == "GARCH"
            else ["omega", "alpha[1]", "gamma[1]"])
    for k in need:
        if k not in p:
            raise RuntimeError(f"{vol} 拟合结果缺少参数 {k}：{list(p)}")
    p.setdefault("beta[1]", 0.0)
    return p


def _garch_forecast_vol(windows, horizon, params, burn_in=32):
    """逐样本重过滤 + h 步迭代，返回百分比波动率预测 (N,)。

    GARCH(1,1)，零均值新息： sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}
    EGARCH(1,1)（Nelson，arch 的参数化）：
        ln sigma2_t = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1}
                      + beta*ln sigma2_{t-1}
    其中 z = r/sigma，E|z| = sqrt(2/pi)（正态新息）。
    迭代期（j >= 2）取新息的条件期望：GARCH 下 E[r^2] = sigma^2，
    EGARCH 下 E|z| = sqrt(2/pi)、E[z] = 0。
    """
    ret = _raw_returns(windows)                          # (N, L)
    n, L = ret.shape
    out = np.empty(n, dtype=np.float64)
    om = params["omega"]
    al = params["alpha[1]"]
    be = params.get("beta[1]", 0.0)
    ga = params.get("gamma[1]")
    e_abs = float(np.sqrt(2.0 / np.pi))
    horizon = max(1, int(horizon))

    for i in range(n):
        r = ret[i]
        v = float(np.var(r, ddof=1)) if r.size > 1 else 1e-8
        if not np.isfinite(v) or v <= 0:
            v = 1e-12
        # 预热：用窗口前段收益把 sigma2 推到 burn_in 之后，初值影响可忽略。
        start = max(1, min(burn_in, L - 1))
        for t in range(1, start):
            if ga is None:
                v = om + al * r[t - 1] ** 2 + be * v
            else:
                z = r[t - 1] / np.sqrt(max(v, 1e-12))
                v = float(np.exp(om + al * (abs(z) - e_abs) + ga * z
                                 + be * np.log(max(v, 1e-12))))
        for t in range(start, L):
            if ga is None:
                v = om + al * r[t - 1] ** 2 + be * v
            else:
                z = r[t - 1] / np.sqrt(max(v, 1e-12))
                v = float(np.exp(om + al * (abs(z) - e_abs) + ga * z
                                 + be * np.log(max(v, 1e-12))))

        # 从最后一个观测出发，向前迭代 h 步并取均值
        v = max(float(v), 1e-12)
        acc = []
        for j in range(horizon):
            if j == 0:
                if ga is None:
                    v_next = om + al * r[-1] ** 2 + be * v
                else:
                    z = r[-1] / np.sqrt(v)
                    v_next = float(np.exp(om + al * (abs(z) - e_abs) + ga * z
                                          + be * np.log(v)))
            else:
                v_next = (om + (al + be) * v) if ga is None else \
                    float(np.exp(om + be * np.log(v)))
            v = max(float(v_next), 1e-12)
            acc.append(v)
        out[i] = 100.0 * np.sqrt(float(np.mean(acc)))
    return out


class GARCHModel:
    """GARCH(1,1) 基线：训练收益上 MLE，测试期逐日重过滤 + h 步迭代。

    y_hat^(h)_t = 100 * sqrt( (1/h) * sum_{j=1..h} sigma^2_{t+j} )
    sigma^2_{t+j} 由 GARCH(1,1) 在 t 时刻条件方差的 h 步迭代预报给出。
    """

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        self.input_len = input_len
        self.horizon = horizon
        self.mode = mode
        self._params = None

    def fit(self, train_loader, val_loader=None, epochs=None, patience=None, device="cpu"):
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("请安装 arch 库: pip install arch")

        returns = _extract_input_series(train_loader)      # 训练收益（原尺度）
        if len(returns) < 20:
            raise RuntimeError("训练序列过短，无法估计 GARCH")
        res = arch_model(returns, vol="GARCH", p=1, o=0, q=1,
                         dist="normal").fit(disp="off", show_warning=False)
        self._params = _garch_params(res, "GARCH")
        return self

    def predict(self, test_loader, device="cpu"):
        if self._params is None:
            raise RuntimeError("模型尚未训练")
        vol_pct = _garch_forecast_vol(_extract_input_windows(test_loader),
                                      self.horizon, self._params)
        return _to_standardized_target(vol_pct).astype(np.float32)

    def count_parameters(self):
        return 0


class EGARCHModel:
    """EGARCH(1,1) 基线：对数条件方差规格，捕捉杠杆效应；其余同 GARCHModel。"""

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        self.input_len = input_len
        self.horizon = horizon
        self.mode = mode
        self._params = None

    def fit(self, train_loader, val_loader=None, epochs=None, patience=None, device="cpu"):
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("请安装 arch 库: pip install arch")

        returns = _extract_input_series(train_loader)
        if len(returns) < 20:
            raise RuntimeError("训练序列过短，无法估计 EGARCH")
        res = arch_model(returns, vol="EGARCH", p=1, o=1, q=1,
                         dist="normal").fit(disp="off", show_warning=False)
        self._params = _garch_params(res, "EGARCH")
        return self

    def predict(self, test_loader, device="cpu"):
        if self._params is None:
            raise RuntimeError("模型尚未训练")
        vol_pct = _garch_forecast_vol(_extract_input_windows(test_loader),
                                      self.horizon, self._params)
        return _to_standardized_target(vol_pct).astype(np.float32)

    def count_parameters(self):
        return 0


# ============================================================
# LSTM — capacity-matched hidden size, 1 layer
# ============================================================

class LSTMModel(BaseModel):
    """LSTM baseline; protocol width is supplied by get_model()."""

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        super().__init__(input_len, horizon, hidden_dim, mode)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_dim, 1)  # 始终输出单步预测

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, L, H)
        out = self.fc(out[:, -1, :])  # 取最后时间步
        return out


# ============================================================
# GRU — capacity-matched hidden size, 1 layer
# ============================================================

class GRUModel(BaseModel):
    """GRU baseline; protocol width is supplied by get_model()."""

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard"):
        super().__init__(input_len, horizon, hidden_dim, mode)
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=1,
                          batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_dim, 1)  # 始终输出单步预测

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


# ============================================================
# TCN —— 转发到 adafractcn 的 "TCN" 变体（single shared TCN implementation）
# ============================================================

class TCNModel(BaseModel):
    """标准 TCN 基线。

    **本类不再自带一份卷积栈。** 稿件 §4.1 声明 "the TCN baseline has a
    single score at each horizon, shared across the main results, the
    ablation and the parameter-count tables"，而早期版本有两个不同的
    TCN：本类（2 块、dilation [1, 2]、R = 13、lr 1e-3）与
    adafractcn 的 "TCN" 变体（4 块、[1, 2, 4, 8]、R = 61、双速优化器）。
    两者既不是同一个网络，也不是同一次训练，于是"同一个 TCN"这句话
    以及随后的感受野叙述都不可能成立。

    现在本类只做转发，结构与训练循环全部来自
    ``adafractcn.get_adafractcn("TCN")``：
      * L_net = TCN_LAYERS = 4，K = 5，dilation = [1, 2, 4, 8]，按正文感受野约定 R = 61；
      * 宽度 TCN_HIDDEN = 96，使自由参数数与 AdaFracTCN 同预算（±10%）；
      * 训练循环与 AdaFracTCN.family 一致（同一 optimizer / 日程 / 轮数 /
        早停 / 种子集）。

    ``hidden_dim`` 形参被刻意忽略：宽度由 TCN_HIDDEN 决定，否则调用方
    （``get_model`` 的默认 hidden_dim=64）会把这张表悄悄改回未匹配的宽度。
    """

    def __new__(cls, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=None,
                mode="standard"):
        from adafractcn import get_adafractcn
        if hidden_dim is not None and hidden_dim != TCN_HIDDEN:
            warnings.warn(
                "TCNModel 忽略 hidden_dim=%r；宽度由 adafractcn.TCN_HIDDEN"
                "（=%d）统一决定，以保证主表与消融共用同一个 TCN。"
                % (hidden_dim, TCN_HIDDEN))
        return get_adafractcn("TCN", input_len=input_len, horizon=horizon,
                              mode=mode)

    def receptive_field_note(self):
        """供 Table 2/§4.5 对账。"""
        return ("standard TCN (manuscript RF convention), L_net=%d, K=5, R=1+(K-1)(2^L_net-1)=%d"
                % (TCN_LAYERS, 1 + 4 * (2 ** TCN_LAYERS - 1)))


# ============================================================
# Transformer — 轻量版：d_model=32, 1层
# ============================================================

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=DEFAULT_INPUT_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"sequence length {x.size(1)} exceeds positional-encoding capacity {self.pe.size(1)}"
            )
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(BaseModel):
    """Transformer Encoder baseline; protocol d_model is capacity matched."""

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64, mode="standard",
                 n_heads=4, num_layers=1, ffn_mult=2.0, dropout=0.2):
        super().__init__(input_len, horizon, hidden_dim, mode)
        if hidden_dim % n_heads != 0:
            raise ValueError("Transformer hidden_dim must be divisible by n_heads")
        self.input_proj = nn.Linear(1, hidden_dim)
        self.pos_enc = _PositionalEncoding(hidden_dim, max_len=input_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=max(4, int(round(ffn_mult * hidden_dim))),
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)  # 始终输出单步预测

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.encoder(x)
        out = self.fc(out[:, -1, :])
        return out


# ============================================================
# Informer（简化版）— ProbSparse 自注意力 + encoder distilling
# ============================================================

class _ProbSparseAttention(nn.Module):
    """Informer 的判别性组件：ProbSparse 自注意力（简化版）。

    与 Transformer 的密集自注意力不同，这里只对每个 query 保留
    稀疏度得分最高的 c_top 个 key 做点积（top-k 稀疏 QK 点积），
    从而在长序列上降低复杂度并突出"主导点"（dominant keys）。
    这是 Informer 相对 Transformer 的核心差异，而不是第二份拷贝。
    """
    def __init__(self, d_model, n_heads, topk=4, dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.topk = topk
        self.scaling = self.head_dim ** -0.5
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.w_q(x).view(B, L, H, D).transpose(1, 2)   # (B,H,L,D)
        k = self.w_k(x).view(B, L, H, D).transpose(1, 2)
        v = self.w_v(x).view(B, L, H, D).transpose(1, 2)

        # top-k 稀疏：每个 query 只保留点积得分最高的 topk_t 个 key，
        # 其余置 -inf（softmax 后为 0），从而在长序列上突出"主导点"
        # 并降低注意力复杂度。这是 ProbSparse 自注意力的判别性特征。
        topk_t = min(self.topk, L)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # (B,H,L,L)
        thresh = scores.topk(topk_t, dim=-1).values[..., -1:]         # 每行 topk 阈值
        mask = scores >= thresh
        inf = torch.ones_like(scores) * float("-inf")
        attn = F.softmax(torch.where(mask, scores, inf), dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,L,D)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class _EncoderLayer(nn.Module):
    """单层 encoder：ProbSparse 注意力 + 前馈。"""
    def __init__(self, d_model, n_heads, topk, dropout=0.2):
        super().__init__()
        self.attn = _ProbSparseAttention(d_model, n_heads, topk, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class _DistillingLayer(nn.Module):
    """Informer 的 encoder distilling：对序列维度下采样，增强主导特征。

    使用 stride=2、核=3 的因果 1D 卷积 + MaxPool，序列长度减半。
    这是 Transformer encoder 所没有的判别性组件。
    """
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = F.pad(x, (2, 0))  # 因果填充，保持序列左对齐
        x = self.dropout(self.conv(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)  # 序列长度减半
        return x.transpose(1, 2)


class InformerModel(BaseModel):
    """简化版 Informer：ProbSparse 自注意力 + encoder distilling。

    注意：这不是 Transformer 的第二份拷贝。与 Transformer 的区别在于
      (1) 使用 _ProbSparseAttention（top-k 稀疏点积）而非全注意力；
      (2) 在 encoder 之后/中间加入了 distilling 下采样层。
    因此两模型的参数与行为显著不同。
    """

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64,
                 mode="standard", n_heads=4, topk=4, dropout=0.2):
        super().__init__(input_len, horizon, hidden_dim, mode)
        self.input_proj = nn.Linear(1, hidden_dim)
        self.pos_enc = _PositionalEncoding(hidden_dim, max_len=input_len)
        # 两层 encoder + 中间的 distilling 下采样（Informer 判别性结构）
        self.encoder = nn.Sequential(
            _EncoderLayer(hidden_dim, n_heads, topk, dropout=dropout),
            _DistillingLayer(hidden_dim, dropout=dropout),
            _EncoderLayer(hidden_dim, n_heads, topk, dropout=dropout),
        )
        self.fc = nn.Linear(hidden_dim, 1)  # 始终输出单步预测

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.encoder(x)
        out = self.fc(out[:, -1, :])
        return out


# ============================================================
# Frac-LSTM（分数阶 LSTM）— truncation=64, α=0.5（固定）
# ============================================================

class FracLSTMModel(BaseModel):
    """分数阶 LSTM：在输入前乘以固定 GL 系数，α=0.5（固定），truncation=64。"""

    def __init__(self, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=64,
                 mode="standard", alpha=0.5, truncation=64):
        super().__init__(input_len, horizon, hidden_dim, mode)
        self.alpha = alpha
        self.truncation = int(truncation)
        # 预计算 GL 系数（α=0.5 固定）
        coeffs = self._gl_coeffs(alpha, self.truncation)
        self.register_buffer("gl_coeffs", coeffs.float().view(1, -1, 1))

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_dim, 1)  # 始终输出单步预测

    def _gl_coeffs(self, alpha, M):
        c = torch.ones(M)
        for k in range(1, M):
            c[k] = c[k - 1] * (k - 1 - alpha) / k
        return c

    def forward(self, x):
        # 用 GL 系数加权输入序列（简化版分数阶操作）
        # 对齐长度：GL 系数截断至 input_len 或对超出部分填0
        L = x.size(1)
        if L <= self.truncation:
            w = self.gl_coeffs[:, :L, :]
        else:
            pad = torch.zeros(1, L - self.truncation, 1, device=x.device)
            w = torch.cat([self.gl_coeffs, pad], dim=1)
        x = x * w
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY = {
    # 朴素 / 计量基线（协议：11 基线中的 5 个朴素/计量层）
    "Persistence": PersistenceModel,
    "Uncond mean": UnconditionalMeanModel,
    "HAR-RV": HARRVModel,
    "GARCH": GARCHModel,
    "EGARCH": EGARCHModel,
    # 统计学 / 深度学习基线
    "ARIMA": ARIMAModel,
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "TCN": TCNModel,
    "Transformer": TransformerModel,
    "Informer": InformerModel,
    "Frac-LSTM": FracLSTMModel,
}


def get_model(name, input_len=DEFAULT_INPUT_LEN, horizon=1, hidden_dim=None, mode="standard"):
    """根据名称获取模型实例，并默认使用论文的容量匹配宽度。

    早期实现虽然 Table 2 写了容量匹配宽度，主实验却通过默认
    ``hidden_dim=64`` 实例化 LSTM/GRU/Transformer/Informer/Frac-LSTM。
    现在 ``hidden_dim=None`` 表示从 ``param_budget.BASELINE_WIDTHS`` 读取
    唯一协议值；只有显式传入 hidden_dim 才构造非协议宽度（用于诊断）。
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}，可选: {list(MODEL_REGISTRY.keys())}")
    if name == "TCN":
        return MODEL_REGISTRY[name](input_len=input_len, horizon=horizon, mode=mode)
    if hidden_dim is None:
        hidden_dim = BASELINE_WIDTHS.get(name, 64)
    return MODEL_REGISTRY[name](input_len=input_len, horizon=horizon,
                                hidden_dim=hidden_dim, mode=mode)


# ============================================================
# 快速验证
# ============================================================

if __name__ == "__main__":
    print("=== 基线模型快速验证 ===\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 构造假数据
    X = torch.randn(32, 60, 1).to(device)
    y = torch.randn(32).to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    for name in MODEL_REGISTRY:
        if name in ("ARIMA", "Persistence", "Uncond mean", "HAR-RV", "GARCH", "EGARCH"):
            print(f"\n{name}: 跳过（朴素/计量基线，不需要前向网络）")
            continue
        try:
            model = get_model(name, horizon=1, mode="quick").to(device)
            out = model(X)
            n_params = model.count_parameters()
            print(f"{name:15s}: 输出 shape={list(out.shape)}, 参数量={n_params:,}")
        except Exception as e:
            print(f"{name:15s}: ERROR - {e}")

    print("\n=== 验证完成 ===")
