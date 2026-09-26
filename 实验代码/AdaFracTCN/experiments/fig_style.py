# -*- coding: utf-8 -*-
"""Shared figure style and output paths for AdaFracTCN experiments.

All experiment-generated PDFs are written to ``experiments/figures/``.
The module centralizes fonts, line widths, DPI, and save behavior so figures
produced by different analysis scripts use the same visual conventions.

Used by the manuscript-facing figure generator, hyperparameter sensitivity,
regime diagnostics, residual diagnostics, ablation plotting, and fitted-kernel
analysis.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# 路径
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# ======================================================================
# 样式契约
# ======================================================================
# 字号阶梯（pt）：论文正文字号约 10pt，插图内文字不得小于 9pt，
# 否则缩印到双栏后不可读。此处的 11--17 区间是缩印后仍清晰的下限区间。
FONT_SIZE = 11
AXES_LABELSIZE = 12
TICK_LABELSIZE = 10.5
LEGEND_FONTSIZE = 10
TITLE_FONTSIZE = 12

# 线宽（pt）
LINEWIDTH = 1.6
LINEWIDTH_EMPH = 2.6
MARKERSIZE = 4.5

# 分辨率
FIG_DPI = 150
SAVE_DPI = 300

# 字体：Times 系（与 MDPI 正文的 Times 一致），数学字体用 stix 匹配。
# 若系统无 Times New Roman，matplotlib 自动回退到 DejaVu Serif，不报错。
_SERIF = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
_SANS = ["Arial", "Helvetica", "DejaVu Sans"]


def apply(serif: bool = True) -> None:
    """应用统一的 rcParams。

    Parameters
    ----------
    serif : bool
        True  -> 字体族 serif（Times），用于与论文正文一致的插图。
        False -> 字体族 sans-serif，用于需要屏幕可读性的辅助图。
    """
    family = "serif" if serif else "sans-serif"
    plt.rcParams.update({
        "font.family": family,
        "font.serif": _SERIF,
        "font.sans-serif": _SANS,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        "font.size": FONT_SIZE,
        "axes.labelsize": AXES_LABELSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,

        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,

        "figure.dpi": FIG_DPI,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Type-42 内嵌字体：MDPI 要求 PDF 中字体可嵌入、可检索。
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
    })


def fig_path(name: str, outdir: str | None = None) -> str:
    """返回插图输出路径（不创建目录）。"""
    outdir = outdir or FIGURES_DIR
    return os.path.join(outdir, name)


def save(fig, name: str, outdir: str | None = None, close: bool = True) -> str:
    """保存为 PDF 矢量图，返回路径。"""
    outdir = outdir or FIGURES_DIR
    os.makedirs(outdir, exist_ok=True)
    path = fig_path(name, outdir)
    fig.savefig(path, format="pdf")
    if close:
        plt.close(fig)
    print("  [saved] %s" % path)
    return path


# 颜色：主模型用红色（与正文的强调色一致），基线用中性色阶。
COLOR_MAIN = "#d62728"
COLOR_BASELINE = "#4c72b0"
COLOR_ALT = "#dd8452"
COLOR_GREY = "#7f7f7f"
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52",
           "#8172b2", "#937860", "#da8bc3", "#8c8c8c",
           "#ccb974", "#64b5cd"]


if __name__ == "__main__":
    apply()
    print("FIGURES_DIR = %s" % FIGURES_DIR)
    print("font.size   = %s" % plt.rcParams["font.size"])
