# -*- coding: utf-8 -*-
"""
adafractcn.py — AdaFracTCN 模型实现
=====================================
包含 FractionalConv1d、PowerLawConv1d、FracTCNBlock、DualChannelFracTCNBlock
和 AdaFracTCN 完整模型。

消融/对照变体（`get_adafractcn`）：

  TCN                  标准 TCN baseline，R = 1+(K-1)(2^L_net-1) = 61
  TCN-RFmatched        E1 对照：感受野匹配的标准 TCN，L_net=7 -> R = 509 >= 256，
                       宽度降到 48 以保持自由参数数与 TCN baseline 相当
  PowerLaw             E2 对照：同包络非分数阶——显式幂律包络 (k+1)^{-(1+α)}
                       配同一个短可学习滤波器，不用 GL 递推，每层跨度 67
                       （R_eff 与分数阶一致）
  FracTCN-fixed        单通道、固定阶 α = 0.5
  FracTCN-learnable    单通道、可学习阶
  FracTCN-dual-fixed   双通道、两分支各自固定阶（0.2 / 0.8）
  FracTCN-dual-indep   双通道、两分支各自可学习阶（输入无关）
  FracTCN-dual         双通道、共享一个可学习阶（论文 §4.6 的 dual 变体）
  AdaFracTCN (full)    双通道、两分支各自自适应阶 α(t)

后四者构成完整的"阶的处理方式"阶梯：单个固定 → 单个可学习 → 双通道各自
固定 → 双通道各自可学习 → 双通道共享 → 双通道自适应，把原来缺的两个中间
格子补齐，避免消融被读成一条跳跃的链。
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


def forecasting_loss(pred, target):
    """Training/validation objective used by every neural model.

    The manuscript specifies an equal-weight combination of MSE and Huber
    loss with delta=1 on the standardized target. Keeping this helper in the
    model module gives AdaFracTCN and all neural baselines one executable
    definition of the objective.
    """
    mse = F.mse_loss(pred, target)
    huber = F.smooth_l1_loss(pred, target, beta=TRAIN_HUBER_BETA)
    return 0.5 * mse + 0.5 * huber


# ============================================================
# 基类（与 baselines.py 保持一致）
# ============================================================

class BaseModel(nn.Module):
    """所有模型的抽象基类（与 baselines.py 中完全一致）。"""

    def __init__(self, input_len=256, horizon=1, hidden_dim=64, mode="standard"):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.mode = mode
        self._is_trained = False

    def fit(self, train_loader, val_loader, epochs=None, patience=None, device="cpu"):
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
        pc = 0
        for epoch in range(epochs):
            self.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(Xb)
                if pred.dim() > 1 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                loss = forecasting_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            self.eval()
            vl = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = self(Xb)
                    if pred.dim() > 1 and pred.size(-1) == 1:
                        pred = pred.squeeze(-1)
                    vl += forecasting_loss(pred, yb).item() * Xb.size(0)
            vl /= len(val_loader.dataset)
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                pc = 0
            else:
                pc += 1
                if pc >= patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self._is_trained = True
        self.to(device)
        return self

    def predict(self, test_loader, device="cpu"):
        self.eval()
        self.to(device)
        preds = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(device)
                pred = self(Xb)
                if pred.dim() > 1 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                if pred.dim() > 1:
                    pred = pred[:, -1]
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# 分数阶因果卷积层
# ============================================================

class FractionalConv1d(nn.Module):
    """分数阶因果卷积层。

    核心思想：用 Grünwald-Letnikov 分数阶系数序列 c_k(α) 与一个短可学习
    滤波器 w 做"离散卷积"，得到有效卷积核：

        a_k = sum_{j=0..K-1} w_j * c_{k-j}(α),   k = 0..M+K-2

    其中 c 的长度为截断长度 M（跨 M 个滞后），w 只有 K 个自由参数。
    有效核长度 = M + K - 1，随后做因果填充。这不同于earlier implementation的逐点乘积
    weight*c（earlier implementation used an effective kernel of length M with masked full-length weights）。

    支持三种 alpha 模式:
      - fixed:       α = 固定值 (如 0.5)
      - learnable:   α = sigmoid(α_raw) 可梯度优化
      - adaptive:    α(t) = sigmoid(W_α · h_t + b_α) 输入自适应
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 truncation=128, init_alpha=0.5,
                 learnable_alpha=True, adaptive_alpha=False,
                 n_alpha=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size      # K：可学习滤波器的自由参数个数
        self.dilation = dilation
        self.truncation = truncation        # M：GL 系数截断长度（滞后数）
        self.learnable_alpha = learnable_alpha
        self.adaptive_alpha = adaptive_alpha
        self.n_alpha = n_alpha              # 通道数：1 = 单通道，2 = 双通道

        # 可学习短滤波器 w：只有 K 个自由参数（每个输出/输入通道一组）
        self.w = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Alpha 参数
        if adaptive_alpha:
            # 自适应 α(t) = sigmoid(W_α h_t + b_α)。
            # n_alpha 维输出：双通道时一次前向即产出 (α_1(t), α_2(t))，
            # 与论文式(dual_alpha_readout)一致。
            self.alpha_net = nn.Linear(in_channels, n_alpha)
            self.alpha_bias = nn.Parameter(torch.zeros(n_alpha))
            # 初始化：long 通道偏小、short 通道偏大（见 rem:dual_rationale）
            with torch.no_grad():
                if n_alpha == 2:
                    self.alpha_bias[0] = np.log(0.2 / 0.8)   # -> α_1 ≈ 0.2
                    self.alpha_bias[1] = np.log(0.8 / 0.2)   # -> α_2 ≈ 0.8
                else:
                    self.alpha_bias[0] = np.log(init_alpha / (1.0 - init_alpha + 1e-8))
            self._alpha_registered = None
        elif learnable_alpha:
            # 可学习全局 α = sigmoid(α_raw), 初始化为 init_alpha
            init_logit = np.log(init_alpha / (1.0 - init_alpha + 1e-8))
            self.alpha_raw = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
            self._alpha_registered = "learnable"
        else:
            # 固定 α
            self.register_buffer("fixed_alpha", torch.tensor(init_alpha, dtype=torch.float32))
            self._alpha_registered = "fixed"

    @property
    def alpha(self):
        if self.adaptive_alpha:
            raise RuntimeError("adaptive alpha 需要输入数据，请使用 forward(x) 调用")
        if self._alpha_registered == "learnable":
            return torch.sigmoid(self.alpha_raw)
        return self.fixed_alpha

    def adaptive_alpha_vector(self, h):
        """由隐藏状态 ``h`` (B, C, L) 计算逐样本自适应阶。

        返回 ``(B, n_alpha)``。时间维可以汇聚，因为每个样本代表一个
        forecast origin；**batch 维绝不能汇聚**，否则同一条样本的预测会依赖
        同一个 mini-batch 中的其它样本，且论文中的 alpha(t) 会退化成
        batch-level alpha。
        """
        pooled = h.mean(dim=-1)                              # (B, C)
        logits = self.alpha_net(pooled) + self.alpha_bias    # (B, n_alpha)
        return torch.sigmoid(logits)                         # (B, n_alpha)

    def _compute_gl_coeffs(self, M, alpha):
        """向量化计算 GL 系数 c_k(α)（避免 inplace 操作，支持自动微分）。

        alpha 为标量张量 -> 返回 (M,)；若为 (n,)，则返回 (n, M)。
        """
        k = torch.arange(1, M, device=self.w.device, dtype=self.w.dtype)
        if alpha.dim() == 0:
            factors = (k - 1 - alpha) / k
            return torch.cat([torch.ones(1, device=self.w.device,
                                         dtype=self.w.dtype),
                              torch.cumprod(factors, dim=0)])
        # (n, M-1) 的逐通道递推
        factors = (k.view(1, -1) - 1 - alpha.view(-1, 1)) / k.view(1, -1)
        c0 = torch.ones(alpha.size(0), 1, device=self.w.device,
                        dtype=self.w.dtype)
        return torch.cat([c0, torch.cumprod(factors, dim=1)], dim=1)

    def materialize_effective_kernel(self, alpha):
        """Return the mathematical effective kernel ``a = w * c_alpha``.

        This helper is used only for diagnostics.  It mirrors Equation (6) in
        the manuscript and therefore returns taps in lag order: tap 0 is the
        contemporaneous coefficient and tap k multiplies lag k before dilation.
        ``alpha`` may be a scalar or a vector of length B.  Shapes are
        ``(C_out,C_in,M+K-1)`` for a scalar and
        ``(B,C_out,C_in,M+K-1)`` for a vector.
        """
        if not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha, dtype=self.w.dtype, device=self.w.device)
        alpha = alpha.to(device=self.w.device, dtype=self.w.dtype)
        scalar = alpha.dim() == 0
        if scalar:
            coeff = self._compute_gl_coeffs(self.truncation, alpha).unsqueeze(0)
        else:
            coeff = self._compute_gl_coeffs(self.truncation, alpha.reshape(-1))
        B = coeff.size(0)
        N = self.truncation + self.kernel_size - 1
        out = torch.zeros(B, self.out_channels, self.in_channels, N,
                          dtype=self.w.dtype, device=self.w.device)
        for j in range(self.kernel_size):
            out[..., j:j + self.truncation] += (
                self.w[..., j].unsqueeze(0).unsqueeze(-1)
                * coeff[:, None, None, :])
        return out[0] if scalar else out

    def forward(self, x, alpha_override=None):
        """Apply the GL envelope followed by the learned short filter.

        ``x`` has shape ``(B,C,L)``. For adaptive models ``alpha`` is
        sample-specific, shape ``(B,)``; for fixed/global-learnable models it
        is a scalar. The implementation uses associativity of convolution:

            (w * c_alpha) * x = w * (c_alpha * x),

        which avoids materialising a separate ``C_out x C_in x (M+K-1)``
        kernel for every sample. Both stages use the same dilation, so their
        combined lag is exactly the effective-kernel lag in the manuscript.
        PyTorch Conv1d is cross-correlation, hence the tap vectors are flipped
        before application; this makes coefficient index 0 multiply the
        contemporaneous input and index k multiply lag k.
        """
        M = self.truncation
        K = self.kernel_size
        B, C, L = x.shape

        if alpha_override is not None:
            alpha = alpha_override
        elif self.adaptive_alpha:
            av = self.adaptive_alpha_vector(x)               # (B,n_alpha)
            if av.size(-1) != 1:
                raise RuntimeError(
                    "FractionalConv1d with n_alpha>1 must be split by the "
                    "dual-channel block before convolution")
            alpha = av[:, 0]                                 # (B,)
        else:
            alpha = self.alpha                               # scalar

        if not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
        alpha = alpha.to(dtype=x.dtype, device=x.device)
        c = self._compute_gl_coeffs(M, alpha)                # (M,) or (B,M)

        # Stage 1: causal GL filtering, shared across feature channels but
        # sample-specific when alpha is adaptive.
        gl_pad = (M - 1) * self.dilation
        x_pad = F.pad(x, (gl_pad, 0))
        if c.dim() == 1:
            gl_weight = c.flip(-1).view(1, 1, M).expand(C, 1, M)
            z = F.conv1d(x_pad, gl_weight, bias=None,
                         dilation=self.dilation, groups=C)
        elif c.dim() == 2 and c.size(0) == B:
            # Group by sample * input channel. No sample can affect another.
            gl_weight = (c[:, None, :].expand(B, C, M)
                         .reshape(B * C, 1, M).flip(-1))
            x_grouped = x_pad.reshape(1, B * C, x_pad.size(-1))
            z = F.conv1d(x_grouped, gl_weight, bias=None,
                         dilation=self.dilation, groups=B * C)
            z = z.reshape(B, C, L)
        else:
            raise ValueError(
                f"alpha shape {tuple(alpha.shape)} is incompatible with batch B={B}")

        # Stage 2: learned K-tap causal filter. Conv1d is cross-correlation,
        # so flip w to implement sum_j w_j z_{t-j*d}.
        short_pad = (K - 1) * self.dilation
        z_pad = F.pad(z, (short_pad, 0))
        return F.conv1d(z_pad, self.w.flip(-1), self.bias,
                        dilation=self.dilation)


# ============================================================
# 同包络非分数阶对照（E2）
# ============================================================

class PowerLawConv1d(FractionalConv1d):
    """幂律包络因果卷积：与 FractionalConv1d 只差"包络如何生成"。

    The envelope control uses the same envelope with a non-fractional convolution，即权重
    $w_k \\propto (k+1)^{-(1+\\alpha)}$ 而**没有** GL 递推。这里不去另写
    一套卷积，而是继承 FractionalConv1d 并只覆盖系数生成：

      FractionalConv1d : c_k(α) = Π_{j<=k} (j-1-α)/j        （GL 递推）
      PowerLawConv1d   : env_k(α) = (k+1)^{-(1+α)}           （显式幂律）

    其余部分——短可学习滤波器 w 与包络的离散卷积、因果填充、膨胀、以及
    fixed / learnable / adaptive 三种 α 处理——**逐字继承**。因此两个模型
    的自由参数个数、感受野、参数量级完全相同，差别只在系数序列：

      * GL 递推的 c_k 渐近等价于 k^{-(α+1)}/|Γ(-α)|，与显式幂律同指数；
      * 但 c_k 的前若干项带有递推造成的符号/幅度结构，而 env_k 是纯幂律。

    于是这个对照恰好回答"增益来自幂律包络这一点，还是来自 GL 递推这条
    实现路径"：若两者精度无显著差别，则机制是包络而不是递推。
    """

    def _compute_gl_coeffs(self, M, alpha):
        """覆盖父类：用显式幂律包络取代乘积累积的 GL 递推。

        alpha 为标量张量 -> 返回 (M,)；若为 (n,)，则返回 (n, M)。
        返回值的接口与父类一致，因此其后的"包络 ⊛ 短滤波器"流程无需改动。
        """
        k = torch.arange(1, M + 1, device=self.w.device, dtype=self.w.dtype)
        if alpha.dim() == 0:
            return k ** (-(1.0 + alpha))
        return k.view(1, -1) ** (-(1.0 + alpha.view(-1, 1)))


# ============================================================
# 标准 TCN 残差块（用于 use_fractional=False 变体）
# ============================================================

class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        nn.utils.weight_norm(self.conv)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _StandardTCNBlock(nn.Module):
    """标准 TCN 残差块。"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn(out + residual)
        return F.relu(out)


# ============================================================
# 单通道分数阶 TCN 残差块
# ============================================================

class _FracTCNBlock(nn.Module):
    """单通道分数阶 TCN 残差块（用于 FracTCN-fixed 和 FracTCN-learnable）。

    conv_cls 控制卷积层的实现：默认 FractionalConv1d（GL 递推），传入
    PowerLawConv1d 即得 E2 的同包络非分数阶对照。两种情况下本块的其余
    结构（残差、BN、dropout、膨胀）完全一致，所以对照只改变包络。
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, truncation,
                 init_alpha=0.5, learnable_alpha=True, adaptive_alpha=False,
                 dropout=0.1, conv_cls=None):
        super().__init__()
        conv_cls = FractionalConv1d if conv_cls is None else conv_cls
        self.conv_cls = conv_cls
        self.conv1 = conv_cls(in_ch, out_ch, kernel_size, dilation,
                              truncation, init_alpha,
                              learnable_alpha, adaptive_alpha)
        self.conv2 = conv_cls(out_ch, out_ch, kernel_size, dilation,
                              truncation, init_alpha,
                              learnable_alpha, adaptive_alpha)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x, alpha_override=None):
        """Two-stage fractional operator followed by the manuscript block map.

        The two fractional stages (with the ReLU between them) constitute the
        DFC operator diagnosed separately in Section 4.7.  The *block output*
        then follows Eq. (forward_block) exactly:

            Dropout(ReLU(BN(DFC(x)))) + Res(x).
        """
        residual = self.downsample(x)
        out = F.relu(self.conv1(x, alpha_override))
        out = self.conv2(out, alpha_override)
        out = self.dropout(F.relu(self.bn(out)))
        return out + residual


# ============================================================
# 双通道分数阶 TCN 残差块（门控融合）
# ============================================================

class DualChannelFracTCNBlock(nn.Module):
    """双通道分数阶 TCN 残差块。

    长记忆通道（α 小，收敛后记忆长）+ 短记忆通道（α 大，记忆短），
    通过门控融合自适应组合: y = g · y_long + (1-g) · y_short。
    按收敛后的记忆行为命名：α 越小 ⇒ GL 系数衰减越慢 ⇒ 记忆越长，
    故 long-memory 通道用 init_alpha=0.2，short-memory 通道用 init_alpha=0.8
    （若默认可学习 α 收敛，则 long 通道 α 趋小、short 通道 α 趋大）。

    dual-order protocol 的修复：两分支各自持有独立的 α。
      * alpha_override 支持 (a1, a2) 元组 —— 分别喂给 long / short 分支；
      * 自适应模式下由 long1 的 alpha_net 对每个样本产出 2 维 α，再拆分复用；
      * 不对 batch 求均值，因此同一样本的 α 与预测不依赖 batch 伙伴。

    另有 shared_alpha=True 的共享阶模式（论文 §4.6 的 FracTCN-dual 变体）：
    两分支显式共用一个阶参数，短分支根本不注册自己的阶，用于把"双通道
    各自持阶"与"双通道共享阶"分开检验。
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, truncation,
                 adaptive_alpha=False, dropout=0.1, learnable_alpha=True,
                 shared_alpha=False):
        super().__init__()
        if adaptive_alpha and shared_alpha:
            raise ValueError("shared_alpha 与 adaptive_alpha 互斥：共享阶的变体"
                             "按定义只有一个阶参数，而 full 模型的两个分支"
                             "各自持有输入自适应的 α(t)。")
        self.shared_alpha = shared_alpha
        # 共享阶模式下短分支不再持有自己的阶参数，"共享"是结构性的，
        # 而不是靠把两个参数初始化成同一个值来假装。
        short_learnable = bool(learnable_alpha) and not shared_alpha
        short_init = 0.5 if shared_alpha else 0.8
        # 长记忆通道：α 小（init_alpha=0.2），记忆持久。
        # 自适应模式下由 long1 承担 2 维 α 的产生，long2 复用其通道 0。
        self.conv_long1 = FractionalConv1d(in_ch, out_ch, kernel_size, dilation,
                                            truncation, init_alpha=0.2,
                                            learnable_alpha=learnable_alpha,
                                            adaptive_alpha=adaptive_alpha,
                                            n_alpha=2 if adaptive_alpha else 1)
        self.conv_long2 = FractionalConv1d(out_ch, out_ch, kernel_size, dilation,
                                            truncation, init_alpha=0.2,
                                            learnable_alpha=learnable_alpha,
                                            adaptive_alpha=False)
        # 短记忆通道：α 大（init_alpha=0.8），记忆短促；
        # 共享阶模式下它的 α 由 alpha_pair 从长分支复制过来。
        self.conv_short1 = FractionalConv1d(in_ch, out_ch, kernel_size, dilation,
                                             truncation, init_alpha=short_init,
                                             learnable_alpha=short_learnable,
                                             adaptive_alpha=False)
        self.conv_short2 = FractionalConv1d(out_ch, out_ch, kernel_size, dilation,
                                             truncation, init_alpha=short_init,
                                             learnable_alpha=short_learnable,
                                             adaptive_alpha=False)

        # 门控融合
        self.gate = nn.Linear(out_ch * 2, out_ch)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_ch)

    def alpha_pair(self, x, alpha_override=None):
        """解析出 (alpha_long, alpha_short) 两个独立标量张量。

        优先级：显式 override 元组 > 共享阶 > 自适应 2 维输出 > 各分支
        自身的可学习 α。
        """
        if alpha_override is not None:
            if isinstance(alpha_override, (tuple, list)):
                return alpha_override[0], alpha_override[1]
            a = alpha_override
            # 允许传入形状 (2,) 的张量，同样按通道拆分
            if torch.is_tensor(a) and a.numel() == 2:
                return a[0], a[1]
            return a, a

        # 共享阶：两分支用同一个阶，短分支的 α 由长分支复制而来。
        if getattr(self, "shared_alpha", False):
            if self.conv_long1.adaptive_alpha:
                av = self.conv_long1.adaptive_alpha_vector(x)
                a = av[:, 0]
                return a, a
            a = self.conv_long1.alpha
            return a, a

        if self.conv_long1.adaptive_alpha:
            # long1 的 alpha_net 对每个 forecast origin 输出两列：
            # (B,2) = (alpha_long(t), alpha_short(t)).
            av = self.conv_long1.adaptive_alpha_vector(x)
            if av.size(-1) == 2:
                return av[:, 0], av[:, 1]
            return av[:, 0], av[:, 0]

        # 非自适应：两分支各自的标量 α（可学习或固定）
        return self.conv_long1.alpha, self.conv_short1.alpha

    def forward(self, x, alpha_override=None):
        """Forward pass matching the manuscript's block-level equation.

        Each branch contains two fractional-convolution stages with a ReLU
        between them (the two stages are the objects analysed by the fitted-
        kernel diagnostic).  Their gated fusion is DFC_d(x).  The enclosing
        residual block is exactly

            Dropout(ReLU(BN(DFC_d(x)))) + Res(x).
        """
        residual = self.downsample(x)
        a_long, a_short = self.alpha_pair(x, alpha_override)

        # DFC internal stage 1 -> nonlinearity -> stage 2.
        y_long = F.relu(self.conv_long1(x, a_long))
        y_long = self.conv_long2(y_long, a_long)
        y_short = F.relu(self.conv_short1(x, a_short))
        y_short = self.conv_short2(y_short, a_short)

        # Learned gate = DFC branch fusion.
        cat = torch.cat([y_long.transpose(1, 2), y_short.transpose(1, 2)], dim=-1)
        g = torch.sigmoid(self.gate(cat)).transpose(1, 2)
        fused = g * y_long + (1 - g) * y_short

        block = self.dropout(F.relu(self.bn(fused)))
        return block + residual

    def forward_with_alpha(self, x, alpha_override=None):
        """Identical block forward pass, additionally returning operative alpha.

        The returned tensor has shape ``(B,2)`` and contains the exact orders
        used to build this block's two GL kernels for each sample.
        """
        residual = self.downsample(x)
        a_long, a_short = self.alpha_pair(x, alpha_override)
        B = x.size(0)

        def _as_batch(a):
            if not torch.is_tensor(a):
                a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
            if a.dim() == 0:
                return a.expand(B)
            return a.reshape(B)

        a_long_b, a_short_b = _as_batch(a_long), _as_batch(a_short)
        y_long = F.relu(self.conv_long1(x, a_long_b))
        y_long = self.conv_long2(y_long, a_long_b)
        y_short = F.relu(self.conv_short1(x, a_short_b))
        y_short = self.conv_short2(y_short, a_short_b)
        cat = torch.cat([y_long.transpose(1, 2), y_short.transpose(1, 2)], dim=-1)
        g = torch.sigmoid(self.gate(cat)).transpose(1, 2)
        fused = g * y_long + (1 - g) * y_short
        out = self.dropout(F.relu(self.bn(fused))) + residual
        alpha_pair = torch.stack([a_long_b, a_short_b], dim=-1)
        return out, alpha_pair


# ============================================================
# AdaFracTCN 完整模型
# ============================================================

class AdaFracTCN(BaseModel):
    """自适应分数阶时序卷积网络，支持全部消融/对照变体。

    消融变体配置（由 `get_adafractcn` 装配）:
      - TCN (baseline):       use_fractional=False
      - TCN-RFmatched:        use_fractional=False, rf_matched=True, L_net=7
      - PowerLaw:             envelope_only=True（幂律包络、无 GL 递推）
      - FracTCN-fixed:        use_fractional=True, learnable_alpha=False
      - FracTCN-learnable:    use_fractional=True, learnable_alpha=True, dual_channel=False
      - FracTCN-dual-fixed:   dual_channel=True, learnable_alpha=False
      - FracTCN-dual-indep:   dual_channel=True, learnable_alpha=True, adaptive_alpha=False
      - FracTCN-dual:         dual_channel=True, learnable_alpha=True, shared_alpha=True
      - AdaFracTCN (full):    dual_channel=True, adaptive_alpha=True

    感受野记账（论文 §4.5 的 rem:receptive_field）在构造时算好并挂在实例上：
      standard_receptive_field   = 1 + (K-1)(2^L_net - 1)
      fractional_receptive_field = 1 + (M+K-2)(2^L_net - 1)
    前者对标准 TCN 用，后者对任何带分数阶包的堆叠用；两者都只由结构决定，
    不含任何学习量。
    """

    def __init__(self, input_len=256, horizon=1, hidden_dim=64, mode="standard",
                 use_fractional=True, learnable_alpha=True, dual_channel=False,
                 adaptive_alpha=False, fixed_alpha=0.5,
                 truncation=None, num_layers=None, kernel_size=None,
                 envelope_only=False, shared_alpha=False, rf_matched=False):
        super().__init__(input_len, horizon, hidden_dim, mode)

        self.use_fractional = use_fractional
        self.learnable_alpha = learnable_alpha
        self.dual_channel = dual_channel
        self.adaptive_alpha = adaptive_alpha
        self.envelope_only = envelope_only
        self.shared_alpha = shared_alpha
        self.rf_matched = rf_matched

        # 超参数：支持外部传入（用于超参数敏感性分析）；默认按协议 M=64, L_net=4, K=5
        if truncation is not None:
            self.truncation = truncation
        else:
            self.truncation = 16 if mode == "quick" else 64
        if num_layers is not None:
            self.num_layers = num_layers
        else:
            self.num_layers = 2 if mode == "quick" else 4
        if kernel_size is not None:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = 5
        dilations = [2 ** i for i in range(self.num_layers)]

        # Receptive-field bookkeeping follows the manuscript convention
        # (Eq. receptive_field / frac_receptive_field): one dilation
        # contribution per named TCN block.  E1 then deepens the standard TCN
        # until this structural reach covers the common L=256 input window.
        if use_fractional or envelope_only:
            k_span = self.truncation + self.kernel_size - 2
        else:
            k_span = self.kernel_size - 1
        self.receptive_field = 1 + k_span * (2 ** self.num_layers - 1)
        self.standard_receptive_field = (
            1 + (self.kernel_size - 1) * (2 ** self.num_layers - 1))
        self.fractional_receptive_field = (
            1 + (self.truncation + self.kernel_size - 2) * (2 ** self.num_layers - 1))
        self.usable_receptive_field = min(self.receptive_field, self.input_len)

        if rf_matched:
            if use_fractional or envelope_only:
                raise ValueError("rf_matched 只用于标准 TCN 对照；带包络的堆叠"
                                 "请直接读 fractional_receptive_field。")
            if self.standard_receptive_field < input_len:
                raise ValueError(
                    f"感受野匹配未达成：R={self.standard_receptive_field} < "
                    f"L={input_len}，请增大 num_layers 或 kernel_size。")

        # 输入投影
        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)

        # 构建主网络
        blocks = []
        in_ch = hidden_dim
        for d in dilations:
            if envelope_only:
                block = _FracTCNBlock(
                    in_ch, hidden_dim, self.kernel_size, d,
                    truncation=self.truncation,
                    init_alpha=fixed_alpha,
                    learnable_alpha=learnable_alpha,
                    adaptive_alpha=adaptive_alpha,
                    dropout=0.2,
                    conv_cls=PowerLawConv1d,
                )
            elif use_fractional and dual_channel:
                block = DualChannelFracTCNBlock(
                    in_ch, hidden_dim, self.kernel_size, d,
                    truncation=self.truncation, adaptive_alpha=adaptive_alpha,
                    dropout=0.2, learnable_alpha=learnable_alpha,
                    shared_alpha=shared_alpha,
                )
            elif use_fractional:
                block = _FracTCNBlock(
                    in_ch, hidden_dim, self.kernel_size, d,
                    truncation=self.truncation,
                    init_alpha=fixed_alpha,
                    learnable_alpha=learnable_alpha,
                    adaptive_alpha=adaptive_alpha,
                    dropout=0.2
                )
            else:
                block = _StandardTCNBlock(
                    in_ch, hidden_dim, self.kernel_size, d, dropout=0.2
                )
            blocks.append(block)
            in_ch = hidden_dim

        self.net = nn.ModuleList(blocks)

        # 输出层：2层 MLP + dropout 正则化
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # 跳过连接：从最后一个输入值直接映射到输出（零初始化，仅在有用时激活）
        self.skip = nn.Linear(1, 1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def fit(self, train_loader, val_loader, epochs=None, patience=None, device="cpu"):
        """自定义训练循环：alpha 参数使用更高学习率，训练更充分。"""
        if epochs is None:
            epochs = TRAIN_EPOCHS if self.mode == "standard" else 80
        if patience is None:
            patience = TRAIN_PATIENCE if self.mode == "standard" else 12

        self.to(device)

        # 参数分组：非阶参数与六个神经基线**逐项一致**（Adam, lr =
        # TRAIN_LR, weight_decay = TRAIN_WEIGHT_DECAY），这样 Table 2 的
        # 公平性声明（相同优化器 / 学习率日程 / 轮数 / 早停 / 种子集）
        # 对共享参数成立；阶参数另成一组，因为 sigmoid 门控的 logit 需要
        # 更高的步长才能在 300 轮内离开初始化点，这属于方法本身而非容量
        # 优势，已在协议表（Table 1）中显式列出。
        alpha_params = []
        other_params = []
        for pname, param in self.named_parameters():
            if "alpha" in pname.lower():
                alpha_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": TRAIN_LR,
             "weight_decay": TRAIN_WEIGHT_DECAY},
        ]
        if alpha_params:
            param_groups.append({"params": alpha_params,
                                 "lr": TRAIN_ALPHA_LR, "weight_decay": 0.0})

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=TRAIN_ETA_MIN)
        best_val_loss = float("inf")
        best_state = None
        pc = 0
        for epoch in range(epochs):
            self.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(Xb)
                if pred.dim() > 1 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                loss = forecasting_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            self.eval()
            vl = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = self(Xb)
                    if pred.dim() > 1 and pred.size(-1) == 1:
                        pred = pred.squeeze(-1)
                    vl += forecasting_loss(pred, yb).item() * Xb.size(0)
            vl /= len(val_loader.dataset)
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                pc = 0
            else:
                pc += 1
                if pc >= patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self._is_trained = True
        self.to(device)
        return self

    def _forward_impl(self, x, record_alpha=False):
        """Shared forward path; optionally records the alpha actually used."""
        x_orig = x
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        alpha_trace = []

        for block in self.net:
            if record_alpha and hasattr(block, "forward_with_alpha"):
                x, a = block.forward_with_alpha(x)
                alpha_trace.append(a)
            else:
                x = block(x)

        # Eq. (forward_pool)--(forward_output): global mean pooling is fed
        # directly to the two-layer MLP; dropout belongs to the residual blocks.
        x = x.mean(dim=-1)
        out = self.output_fc(x) + self.skip(x_orig[:, -1, :])
        if not record_alpha:
            return out
        if alpha_trace:
            trace = torch.stack(alpha_trace, dim=1)  # (B, blocks, 2)
        else:
            trace = torch.empty(x_orig.size(0), 0, 0, device=x_orig.device,
                                dtype=x_orig.dtype)
        return out, trace

    def forward(self, x):
        """x: (B, L, 1) -> (B, 1)."""
        return self._forward_impl(x, record_alpha=False)

    def forward_with_alpha(self, x):
        """Return predictions and the per-sample operative alpha trace.

        For the full AdaFracTCN the trace is ``(B,L_net,2)``. These are not
        post-hoc re-evaluations: they are the exact orders used in each block
        during the same forward pass that produced the prediction.
        """
        return self._forward_impl(x, record_alpha=True)


# ============================================================
# 模型注册表
# ============================================================

# ------------------------------------------------------------
# 结构常量、变体配置与参数预算（唯一来源：param_budget）
# ------------------------------------------------------------
#
# 常量与变体配置全部来自 param_budget.py —— 那是**纯标准库**模块，因此
# 即使在没有 torch 的机器上，论文 Table 2 / Table 5 的参数量与"±10%
# 参数预算"声明也能被独立复算（run_all.py 的 X10 就在做这件事）。
# 早期版本把宽度、感受野与参数量分别写在三处注释里，彼此矛盾（注释说
# 164k、实算 365k），这正是单一来源要解决的问题。
#
# 本段只做 re-export，便于既有代码继续 `from adafractcn import TCN_HIDDEN`。
from param_budget import (  # noqa: F401  (re-export)
    MODEL_WIDTH, TRUNCATION, KERNEL_SIZE, TCN_LAYERS, TCN_HIDDEN,
    ABLATION_SINGLE_CHANNEL_HIDDEN, ABLATION_DUAL_CHANNEL_HIDDEN,
    RF_MATCHED_TCN_LAYERS, RF_MATCHED_TCN_HIDDEN,
    ENVELOPE_CONTROL_TRUNCATION, ENVELOPE_CONTROL_SPAN,
    BASELINE_WIDTHS, VARIANT_FLAGS, ABLATION_VARIANTS, BUDGETED_MODELS,
    TRAIN_EPOCHS, TRAIN_PATIENCE, TRAIN_LR, TRAIN_ETA_MIN, TRAIN_WEIGHT_DECAY,
    TRAIN_ALPHA_LR, TRAIN_HUBER_BETA,
    resolve_variant, variant_width, analytic_param_count,
    analytic_receptive_field, baseline_analytic_param_count,
    param_budget_report,
)


def get_adafractcn(variant_name, input_len=256, horizon=1,
                   mode="standard", **overrides):
    """根据变体名称获取 AdaFracTCN 实例。

    overrides 供超参数敏感性分析使用，按关键字覆盖配置项
    （truncation / num_layers / kernel_size / hidden_dim）。
    """
    hidden, cfg = resolve_variant(variant_name, **overrides)
    return AdaFracTCN(input_len=input_len, horizon=horizon, hidden_dim=hidden,
                      mode=mode, **cfg)


def receptive_field_report(input_len=256, mode="standard", strict=True):
    """打印各变体的感受野与自由参数数，并与解析计数逐项对账。

    这张表是论文 Table 2 与 Table 5 的记账来源。**实例计数与解析计数
    必须相等**（``param_budget.analytic_param_count`` 与
    ``model.count_parameters()``），否则说明结构被改动而计数没跟上——
    这正是早期版本"注释说 164k、实算 365k"的失效模式，所以在这里断言。
    """
    rows = []
    bad = []
    for name in ABLATION_VARIANTS:
        m = get_adafractcn(name, input_len=input_len, mode=mode)
        actual = m.count_parameters()
        expect = analytic_param_count(name)
        rows.append((name, m.receptive_field, actual, expect))
        if actual != expect:
            bad.append((name, actual, expect))
    print(f"{'Variant':<20} {'R':>6} {'Params (torch)':>16} "
          f"{'Params (analytic)':>18}")
    for name, rf, actual, expect in rows:
        mark = "" if actual == expect else "   <-- MISMATCH"
        print(f"{name:<20} {rf:>6d} {actual:>16,} {expect:>18,}{mark}")
    if strict:
        assert not bad, ("解析参数计数与实例不符（结构改动后必须同步 "
                         "param_budget）：%s" % bad)
    return rows


# ============================================================
# 快速验证
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("AdaFracTCN 模型快速验证")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 假数据
    X = torch.randn(32, 60, 1).to(device)  # (B, L, 1)
    y = torch.randn(32).to(device)

    print(f"\n输入形状: {X.shape}")

    for name in ABLATION_VARIANTS:
        try:
            model = get_adafractcn(name, horizon=1, mode="quick").to(device)
            out = model(X)
            n_params = model.count_parameters()
            print(f"\n{name:20s}: 输出 shape={list(out.shape)}, 参数量={n_params:,}, "
                  f"感受野={model.receptive_field}")
            print(f"  {'':20s} 架构: {model.use_fractional=}, {model.learnable_alpha=}, "
                  f"{model.dual_channel=}, {model.adaptive_alpha=}, "
                  f"{model.envelope_only=}, {model.shared_alpha=}")
        except Exception as e:
            print(f"\n{name:20s}: ERROR - {e}")

    print("\n" + "=" * 50)
    print("前向传播验证通过")
    print("=" * 50)

    # 测试完整训练流程
    print("\n--- 完整训练流程测试 (quick mode) ---")
    train_loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[:8], y[:8]), batch_size=8)

    model = get_adafractcn("AdaFracTCN", horizon=1, mode="quick").to(device)
    model.fit(train_loader, val_loader, epochs=3, patience=2, device=device)
    preds = model.predict(val_loader, device=device)
    print(f"AdaFracTCN: pred shape={preds.shape}, mean={preds.mean():.4f}")
    print("训练流程验证通过")