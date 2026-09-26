# -*- coding: utf-8 -*-
"""param_budget.py —— 结构常量、变体配置与自由参数预算
=====================================================

本模块**只用标准库**，不 import torch。这样做的理由有两条：

1. 论文 Table 2 / Table 5 的参数量与"±10% 参数预算"声明，必须能在
   没有 torch 的机器上被独立复算（``run_all.py`` 的 X10 就在做这件事）；
2. 变体配置（哪一路用不用分数阶、是否双通道、宽度多少）此前分散在
   ``adafractcn.get_adafractcn`` 与几处注释里，属于同一事实的多份副本，
   已经出现过"注释说 64k、代码其实是 365k"的漂移。现在这里是**唯一来源**，
   ``adafractcn`` 与 ``baselines`` 都从这里取。

命名口径：论文称"自由参数"（free parameters），对应
``sum(p.numel() for p in model.parameters() if p.requires_grad)``。
注意 ``nn.utils.weight_norm`` 会把 ``weight`` 拆成 ``(weight_g, weight_v)``，
``weight_g`` 的形状是 ``(out_ch, 1, 1)``，因此比未加 weight_norm 多 ``out_ch``
个参数——本模块的计数把这一点算进去了，早期版本漏算，于是低估了 TCN。
"""

# ------------------------------------------------------------
# 结构常量
# ------------------------------------------------------------

#: AdaFracTCN 自身的默认宽度。论文 Figure 1 的读出模块是 FC(32 -> 1)，
#: 对应 output_fc = Linear(hidden, hidden // 2) -> Linear(hidden // 2, 1)，
#: 所以 hidden 必须保持 64：改宽度会让该图与正文同时失效。
MODEL_WIDTH = 64

#: GL 系数截断长度 M（协议值）。
TRUNCATION = 64

#: 短可学习滤波器的抽头数 K。
KERNEL_SIZE = 5

#: 标准 TCN 基线的结构：L_net 层膨胀残差块，dilation = [1, 2, 4, ...]。
#: 感受野 R = 1 + (K-1)(2^L_net - 1)：L_net = 4 -> R = 61。
#:
#: 主表（Table 3）、消融表（Table 5）与参数表（Table 2）必须共用这**一个**
#: TCN —— 稿件 §4.1 就是这么写的。规格由本模块给出，实现只此一处。
TCN_LAYERS = 4
TCN_HIDDEN = 96

#: 消融阶梯的容量对齐。双通道块含两条分支、四条分数阶卷积，同一宽度下
#: 自由参数约为单通道块的 2.2 倍；若两类变体取同一宽度，阶梯就会把
#: "机制"与"容量"混在一起，Table 5 的增益便无法归因于机制。
ABLATION_SINGLE_CHANNEL_HIDDEN = 96
ABLATION_DUAL_CHANNEL_HIDDEN = 64

#: 感受野匹配的标准 TCN 对照：L_net = 7 -> R = 509 >= L = 256。
#: 深度由 4 提到 7 使参数量约增至 1.75 倍，故把宽度由 96 降到 72，
#: 使该对照同时满足"感受野 >= 输入窗口"与"参数预算匹配"。
RF_MATCHED_TCN_LAYERS = 7
RF_MATCHED_TCN_HIDDEN = 72

#: 同包络非分数阶对照（PowerLaw）的截断长度。取 M = 64 而不是 67，是为了
#: 让它每层的跨度 M + K - 2 = 67 与分数阶堆叠**完整相同**（早期版本把
#: "跨度 67"写成了"截断 67"，于是实际跨度是 70、感受野 1051 而非 1006）。
ENVELOPE_CONTROL_TRUNCATION = 64
ENVELOPE_CONTROL_SPAN = ENVELOPE_CONTROL_TRUNCATION + KERNEL_SIZE - 2  # = 67

#: 六个可训练神经基线的宽度，逐个调到与 AdaFracTCN 同预算（见
#: param_budget_report）。宽度是本模块的结构事实，正文的 Table 2 由它产出。
BASELINE_WIDTHS = {
    "LSTM": 300,
    "GRU": 350,
    "TCN": TCN_HIDDEN,
    "Transformer": 212,
    "Informer": 116,
    "Frac-LSTM": 300,
}

# ------------------------------------------------------------
# 训练协议常量（论文 Table 1；adafractcn 与 baselines 共用）
# ------------------------------------------------------------
TRAIN_EPOCHS = 300
TRAIN_PATIENCE = 15
TRAIN_LR = 1e-3
TRAIN_ETA_MIN = 1e-6
TRAIN_WEIGHT_DECAY = 1e-5
#: 阶参数单独一组：sigmoid 门控的 logit 需要更大步长才能在 300 轮内离开
#: 初始化点。该组只存在于带阶参数的模型，属方法本身而非容量优势。
TRAIN_ALPHA_LR = 2e-3
#: Equal-weight MSE + Huber objective; Huber delta/beta follows Eq. (loss).
TRAIN_HUBER_BETA = 1.0


# ------------------------------------------------------------
# 变体配置
# ------------------------------------------------------------

#: 变体 -> 结构开关。**不含宽度**，宽度由 variant_width 统一决定。
VARIANT_FLAGS = {
    # 标准 TCN 基线（主表与消融同一个模型）
    "TCN": dict(use_fractional=False, learnable_alpha=False,
                dual_channel=False, adaptive_alpha=False,
                num_layers=TCN_LAYERS),
    # 感受野匹配的标准 TCN 对照
    "TCN-RFmatched": dict(use_fractional=False, learnable_alpha=False,
                          dual_channel=False, adaptive_alpha=False,
                          rf_matched=True, num_layers=RF_MATCHED_TCN_LAYERS),
    # 同包络非分数阶对照（显式幂律包络、无 GL 递推）
    "PowerLaw": dict(use_fractional=True, learnable_alpha=True,
                     dual_channel=False, adaptive_alpha=False,
                     envelope_only=True,
                     truncation=ENVELOPE_CONTROL_TRUNCATION),
    # 阶梯：单通道，固定阶 / 可学习阶
    "FracTCN-fixed": dict(use_fractional=True, learnable_alpha=False,
                          dual_channel=False, adaptive_alpha=False,
                          fixed_alpha=0.5),
    "FracTCN-learnable": dict(use_fractional=True, learnable_alpha=True,
                              dual_channel=False, adaptive_alpha=False),
    # 阶梯：双通道，共享可学习阶 / 两个独立的固定阶 / 两个独立的可学习阶
    "FracTCN-dual": dict(use_fractional=True, learnable_alpha=True,
                         dual_channel=True, adaptive_alpha=False,
                         shared_alpha=True),
    "FracTCN-dual-fixed": dict(use_fractional=True, learnable_alpha=False,
                               dual_channel=True, adaptive_alpha=False,
                               shared_alpha=False),
    "FracTCN-dual-indep": dict(use_fractional=True, learnable_alpha=True,
                               dual_channel=True, adaptive_alpha=False,
                               shared_alpha=False),
    # 完整模型：双通道 + 两个各自输入自适应的阶
    "AdaFracTCN": dict(use_fractional=True, learnable_alpha=True,
                       dual_channel=True, adaptive_alpha=True),
}

#: 消融表（Table 5）的行序，与代码里实际构造变体的顺序逐字一致。
ABLATION_VARIANTS = [
    "TCN",
    "TCN-RFmatched",
    "PowerLaw",
    "FracTCN-fixed",
    "FracTCN-learnable",
    "FracTCN-dual",
    "FracTCN-dual-fixed",
    "FracTCN-dual-indep",
    "AdaFracTCN",
]

#: Table 2 的七个可训练模型（六个神经基线 + AdaFracTCN），即
#: "±10% 参数预算"覆盖的集合。
BUDGETED_MODELS = ["LSTM", "GRU", "TCN", "Transformer", "Informer",
                   "Frac-LSTM", "AdaFracTCN"]


def variant_width(variant_name):
    """变体的通道宽度。宽度是结构事实，不接受调用方覆盖。"""
    if variant_name == "TCN-RFmatched":
        return RF_MATCHED_TCN_HIDDEN
    if variant_name == "TCN":
        return TCN_HIDDEN
    flags = VARIANT_FLAGS[variant_name]
    if flags.get("dual_channel"):
        return ABLATION_DUAL_CHANNEL_HIDDEN
    return ABLATION_SINGLE_CHANNEL_HIDDEN


def resolve_variant(variant_name, **overrides):
    """变体名 -> (hidden_dim, cfg)。变体结构的**唯一来源**。

    ``adafractcn.get_adafractcn``（建模型）与 ``analytic_param_count``
    （数参数）都从这里取配置，避免"建模型用一张表、数参数用另一张表"。
    """
    if variant_name not in VARIANT_FLAGS:
        raise ValueError("未知变体: %s，可选: %s"
                         % (variant_name, sorted(VARIANT_FLAGS)))
    cfg = dict(VARIANT_FLAGS[variant_name])
    hidden = variant_width(variant_name)
    cfg.update(overrides)
    hidden = cfg.pop("hidden_dim", hidden)
    return hidden, cfg


# ------------------------------------------------------------
# 重量级模块的参数个数（解析式）
# ------------------------------------------------------------

def _conv1d(in_ch, out_ch, k, weight_norm=False):
    """nn.Conv1d：weight（+ weight_g，若 weight_norm）+ bias。"""
    w = out_ch * in_ch * k
    if weight_norm:
        w += out_ch          # weight_g: shape (out_ch, 1, 1)
    return w + out_ch


def _linear(in_ch, out_ch):
    return in_ch * out_ch + out_ch


def _fractional_conv(in_ch, out_ch, k, *, learnable, adaptive, n_alpha=1):
    """FractionalConv1d：短滤波器 w + bias + 阶参数（固定阶不占参数）。"""
    p = out_ch * in_ch * k + out_ch
    if adaptive:
        p += _linear(in_ch, n_alpha) + n_alpha
    elif learnable:
        p += 1
    return p


def _standard_block(n, k, weight_norm=True):
    """_StandardTCNBlock：两条因果卷积 + BatchNorm1d。"""
    return 2 * _conv1d(n, n, k, weight_norm=weight_norm) + 2 * n


def _frac_block(n, k, *, learnable, adaptive):
    """_FracTCNBlock：两条分数阶卷积 + BatchNorm1d。"""
    return 2 * _fractional_conv(n, n, k, learnable=learnable,
                               adaptive=adaptive) + 2 * n


def _dual_block(n, k, *, learnable, shared, adaptive):
    """DualChannelFracTCNBlock：四条分数阶卷积 + 门控 + BatchNorm1d。"""
    short_learnable = bool(learnable) and not shared
    p = _fractional_conv(n, n, k, learnable=learnable, adaptive=adaptive,
                         n_alpha=2 if adaptive else 1)
    p += _fractional_conv(n, n, k, learnable=learnable, adaptive=False)
    p += _fractional_conv(n, n, k, learnable=short_learnable, adaptive=False)
    p += _fractional_conv(n, n, k, learnable=short_learnable, adaptive=False)
    p += _linear(2 * n, n)      # gate
    p += 2 * n                  # BatchNorm1d
    return p


def _head(n):
    """input_proj + output_fc（两层）+ skip。"""
    return (_conv1d(1, n, 1)
            + _linear(n, n // 2)
            + _linear(n // 2, 1)
            + _linear(1, 1))


def analytic_param_count(variant_name, kernel_size=KERNEL_SIZE, **overrides):
    """按结构解析计算某变体的自由参数个数（与 torch 实例一一对应）。

    ``audit_semantics.py`` verifies that this analytic count matches
    ``sum(p.numel() for p in model.parameters() if p.requires_grad)`` for the
    instantiated PyTorch models.
    """
    n, cfg = resolve_variant(variant_name, **overrides)
    layers = cfg.get("num_layers", 4)
    k = cfg.get("kernel_size") or kernel_size
    use_frac = cfg.get("use_fractional", True)
    envelope_only = cfg.get("envelope_only", False)
    learnable = cfg.get("learnable_alpha", True)
    dual = cfg.get("dual_channel", False)
    adaptive = cfg.get("adaptive_alpha", False)
    shared = cfg.get("shared_alpha", False)

    total = _head(n)
    if dual:
        total += layers * _dual_block(n, k, learnable=learnable,
                                      shared=shared, adaptive=adaptive)
    elif use_frac or envelope_only:
        total += layers * _frac_block(n, k, learnable=learnable,
                                      adaptive=adaptive)
    else:
        total += layers * _standard_block(n, k, weight_norm=True)
    return total


def baseline_analytic_param_count(name):
    """六个神经基线的解析参数个数（结构与 baselines.py 逐个对应）。

    * LSTM / Frac-LSTM：nn.LSTM(1, H, 1) -> 4H(H+1) + 8H，加 fc 的 H + 1；
    * GRU：nn.GRU(1, H, 1) -> 3H(H+1) + 6H，加 fc 的 H + 1；
    * TCN：转发到 adafractcn 的 "TCN" 变体；
    * Transformer：input_proj + 1 层 TransformerEncoderLayer
      (d_model = D, nhead = 4, dim_feedforward = 2D) + fc；
    * Informer：input_proj + 2 层 ProbSparse encoder + distilling 卷积 + fc。
    """
    if name == "TCN":
        return analytic_param_count("TCN")
    if name == "AdaFracTCN":
        return analytic_param_count("AdaFracTCN")
    w = BASELINE_WIDTHS[name]
    if name in ("LSTM", "Frac-LSTM"):
        return 4 * w * (w + 1) + 8 * w + w + 1
    if name == "GRU":
        return 3 * w * (w + 1) + 6 * w + w + 1
    if name == "Transformer":
        d, f = w, 2 * w
        layer = 4 * d * d + 2 * d * f + 9 * d + f   # attn + ff + 2 x LayerNorm
        return _linear(1, d) + layer + _linear(d, 1)
    if name == "Informer":
        d = w
        attn = 4 * _linear(d, d)                    # w_q, w_k, w_v, out_proj
        encoder_layer = attn + 2 * 2 * d + 8 * d * d + 5 * d  # + ff + 2 x LN
        encoder = 2 * encoder_layer
        distill = _conv1d(d, d, 3)
        return _linear(1, d) + encoder + distill + _linear(d, 1)
    raise ValueError("未知基线: %s" % name)


def manuscript_receptive_field(variant_name, kernel_size=KERNEL_SIZE, **overrides):
    """Return the structural receptive field under the manuscript convention.

    Equations (2) and (17) in the manuscript define one dilation contribution
    per TCN block:

      standard TCN:  R = 1 + (K-1) (2^L_net - 1)
      fractional:    R = 1 + (M+K-2) (2^L_net - 1)

    This helper is the *single executable source* for every RF value printed by
    the experiment code.  In particular, the RF-matched TCN is selected using
    this convention and must cover the common L=256 observed window.
    """
    n, cfg = resolve_variant(variant_name, **overrides)
    layers = cfg.get("num_layers", 4)
    k = cfg.get("kernel_size") or kernel_size
    if cfg.get("use_fractional", True) or cfg.get("envelope_only", False):
        span = cfg.get("truncation", TRUNCATION) + k - 2
    else:
        span = k - 1
    return 1 + span * (2 ** layers - 1), n


def usable_receptive_field(variant_name, input_len=256, kernel_size=KERNEL_SIZE,
                           **overrides):
    """Manuscript structural RF capped by the observed input window length."""
    r, n = manuscript_receptive_field(variant_name, kernel_size=kernel_size,
                                      **overrides)
    return min(int(r), int(input_len)), n


# Backward-compatible name used by existing scripts.
def analytic_receptive_field(variant_name, kernel_size=KERNEL_SIZE, **overrides):
    return manuscript_receptive_field(variant_name, kernel_size=kernel_size,
                                      **overrides)


def param_budget_report(tolerance=0.10, verbose=True):
    """核对"±10% 参数预算"声明。

    返回 (budget, rows)；rows 为 (name, params, ratio, kind)。
    """
    budget = analytic_param_count("AdaFracTCN")
    rows = []
    for name in BUDGETED_MODELS:
        p = baseline_analytic_param_count(name)
        rows.append((name, p, p / budget, "Table 2"))
    for name in ABLATION_VARIANTS:
        if name in ("TCN", "AdaFracTCN"):
            continue
        p = analytic_param_count(name)
        rows.append((name, p, p / budget, "Table 5"))
    if verbose:
        print("%-20s%12s%9s%10s" % ("Model", "Params", "Ratio", "Used in"))
        for name, p, r, kind in rows:
            flag = "" if abs(r - 1.0) <= tolerance else "   <-- OUT OF BUDGET"
            print("%-20s%12s%9.3f%10s%s" % (name, format(p, ","), r, kind, flag))
    bad = [(n, r) for n, _, r, _ in rows if abs(r - 1.0) > tolerance]
    assert not bad, "参数预算未对齐（超出 ±%.0f%%）：%s" % (tolerance * 100, bad)
    return budget, rows


if __name__ == "__main__":
    budget, rows = param_budget_report()
    print()
    print("AdaFracTCN budget = %s free parameters" % format(budget, ","))
    print("band (%.0f%%)      = [%s, %s]"
          % (10, format(int(budget * 0.9), ","), format(int(budget * 1.1), ",")))
    print()
    print("%-20s%8s%10s%12s" % ("Variant", "R", "Width", "Params"))
    for name in ABLATION_VARIANTS:
        rf, w = analytic_receptive_field(name)
        print("%-20s%8d%10d%12s"
              % (name, rf, w, format(analytic_param_count(name), ",")))
