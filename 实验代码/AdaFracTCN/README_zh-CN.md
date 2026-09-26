# AdaFracTCN

本仓库整理了论文 **Adaptive Fractional-Order Temporal Convolutional Network with Learnable Memory for Multi-Horizon Volatility Forecasting** 的实验代码与可复现材料。

当前代码已按论文正文中的实验协议和模型定义整理：S&P 500 日频数据、输入长度 `L=256`、预测跨度 `1/5/10/20`、容量匹配的神经网络基线、多随机种子比较、稳健性检验与解释性分析。

## 快速开始

推荐 Python 3.11：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

离线重建 S&P 500 预处理数据：

```bash
cd experiments
python download_data.py --mode standard --offline
```

先查看完整复现流程而不真正训练：

```bash
python run_all.py --dry-run
```

轻量检查：

```bash
cd ..
python -m unittest discover -s tests -v
cd experiments
python audit_semantics.py
python robustness_experiment.py --mode standard --check-only --prepare-gk
```

完整复现：

```bash
python run_all.py
```

完整说明见：

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)：复现实验协议与运行层级
- [`docs/MANUSCRIPT_CODE_MAP.md`](docs/MANUSCRIPT_CODE_MAP.md)：正文与代码文件逐项对应
- [`docs/PUBLIC_RELEASE_CHECKLIST.md`](docs/PUBLIC_RELEASE_CHECKLIST.md)：正式公开前检查清单

## 目录原则

- `experiments/data/sp500.csv`：用于离线确定性重建的 S&P 500 缓存；其余预处理数组均由代码生成，不纳入 Git。
- `experiments/results/reference/`：论文侧的紧凑参考结果；完整预测数组、checkpoint、运行日志和生成图不纳入 Git。
- `tests/`：只做快速结构检查，不启动长训练。
- `.github/workflows/`：GitHub Actions 的轻量 smoke CI。

## License

代码默认使用 MIT License。金融行情数据仍受原始数据提供方条款约束。
