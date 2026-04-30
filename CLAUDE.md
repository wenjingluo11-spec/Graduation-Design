# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

毕业设计仓库，基于 PyTorch 实现开题报告中的三类基础 NLP 任务：IMDB 情感二分类、Reuters-46 新闻多分类、西班牙语→英语机器翻译。模型范围严格限定在 RNN/CNN/Transformer 等基础结构（**不依赖大语言模型**），并配备传统基线（Naive Bayes + TF-IDF）。每个任务同时提供主脚本（`.py`）和 Notebook（`.ipynb`），主脚本是当前的主要入口。

## 环境与依赖

依赖锁定在 `requirements.txt`：`torch>=2.2`、`numpy`、`pandas`、`matplotlib`、`seaborn`、`scikit-learn`、`nltk`、`jupyter`。

```bash
# Mac (一次性初始化)
./setup_mac.sh                     # 创建 .venv-mac，安装依赖，下载 NLTK punkt
source .venv-mac/bin/activate

# Windows (已有 .venv/)
.venv\Scripts\activate
```

仓库同时存在 `.venv/`（Windows）与 `.venv-mac/`（macOS）；`.gitignore` 已忽略两者。**不要在仓库中提交模型权重**（`*.pt/*.pth/*.ckpt`）或 PNG 图表。

## 常用命令

### 主实验（每任务一个 `.py`，必须在该任务目录下运行）

```bash
# 快速冒烟（约几分钟，用于回归校验）
cd 情感二分类     && python sentiment_analysis.py --quick
cd ../新闻多分类  && python reuters_multiclass.py --quick
cd ../机器翻译    && python machine_translation.py --quick

# 完整实验
cd 情感二分类     && python sentiment_analysis.py --device cuda
cd ../新闻多分类  && python reuters_multiclass.py --device cuda
cd ../机器翻译    && python machine_translation.py --device cuda
```

三个主脚本都通过 `argparse` 暴露完整参数（`--epochs`、`--batch-size`、`--seed`、`--patience`、`--device`、`--quick`、`--include-hybrid`、`--output-suffix` 等）。`--device` 接受 `cpu|cuda|mps`，默认按 `mps → cuda → cpu` 三级 fallback 自动选择。

任务特定开关：
- `reuters_multiclass.py`：`--loss {ce,focal}`、`--focal-gamma`
- `machine_translation.py`：`--label-smoothing`、`--epochs-seq2seq`、`--epochs-transformer`、`--patience-seq2seq`、`--patience-transformer`

`--output-suffix` 可避免不同消融配置（如 γ-scan、不同 label-smoothing）覆盖彼此的 `*_results.csv`。

### 一键全量 + 汇总 + 校验（PowerShell）

```powershell
.\run_full_experiments.ps1 -Device cuda
# 内部依次：sentiment → reuters → translation → summarize_final_results.py → audit_outputs.py
```

### 补充实验（5 个 package，由根目录的 `supplementary_experiments.py` 调度）

```bash
# 单跑：误差分析 + 效率（最便宜，几分钟级别）
python supplementary_experiments.py --run-error-analysis --run-efficiency

# 全跑（数小时级，建议 cuda）
python supplementary_experiments.py --all --device cuda --fast

# 仅做种子稳定性
python supplementary_experiments.py --run-stability --seeds 42 2024 3407

# PowerShell 入口
.\run_supplementary_experiments.ps1 -All -Device cuda
```

支持的 package：`--run-stability`、`--run-ablation`、`--run-error-analysis`、`--run-efficiency`、`--run-data-scale`。该脚本会 `subprocess` 调用三个主脚本并聚合产物到 `supplementary_outputs/`。

### 进阶分析（`analyses/` 包，按需运行）

```bash
python analyses/learning_curve.py --tasks sentiment,reuters,translation
python analyses/decode_grid.py    --beam-sizes 1,3,5
python analyses/robustness.py     # 词级扰动鲁棒性
```

### 出图与最终汇总

```bash
python make_figures.py                 # 从 outputs/ + supplementary_outputs/ 生成 figures/*.png
python summarize_final_results.py      # 写 final_results_summary.csv
python audit_outputs.py                # 写 outputs_audit_report.txt（指标 ∈ [0,1]、best_epoch ≤ trained_epochs 等约束）
```

### 测试

测试在仓库根目录通过 pytest 运行（`.pytest_cache/` 已存在）：

```bash
pytest tests/                          # 全部
pytest tests/test_focal_loss.py -v     # 单文件
pytest tests/test_focal_loss.py::test_focal_loss_gamma_zero_equals_ce  # 单测
```

`tests/` 通过把 `情感二分类/`、`新闻多分类/`、`机器翻译/` 注入到 `sys.path` 的方式直接 import 主脚本中的类（如 `FocalLoss`、`LabelSmoothingCE`、`CNNBiGRU`），所以**测试与脚本耦合于函数/类名**——重命名时必须同步更新。

## 架构与关键设计

### 入口形态

每个任务目录下既有 `.py` 又有 `.ipynb`，**`.py` 是当前主入口，Notebook 用于探索与展示**。主脚本结构一致：顶部 `Config` dataclass + `parse_args()` → `main()`，输出统一写入本任务的 `outputs/` 子目录（CSV、JSON、PNG）；`outputs_smoke/` 保存 `--quick` 模式产物。

### 路径约定

- 数据：`aclImdb_v1/`、`reuters.npz`、`spa.txt` 在仓库根。
- 主脚本默认数据路径：sentiment 是 `aclImdb`（任务目录内的链接/拷贝），reuters 是 `../reuters.npz`，translation 是 `../spa.txt`。**主脚本必须在自己的任务目录下执行**，否则相对路径失效。
- 补充脚本（`supplementary_experiments.py`、`analyses/*.py`、`audit_outputs.py`、`summarize_final_results.py`、`make_figures.py`）从仓库根运行，会用 `Path.iterdir()` 探测含特定 `.py` 的子目录，因此**任务目录的中文名 + 主脚本文件名构成隐式契约**，不要随意重命名。

### 设备 fallback（重要）

主脚本顶部都设置：

```python
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # 必须在 import torch 之前
```

否则 `nn.TransformerEncoder` 在 `__init__` 时会锁定 `use_nested_tensor` 路径，导致 forward 时 MPS 后端报未实现算子。新增 Transformer 系模型或调整 import 顺序时务必保留此行。

### 各任务关键差异

| 方面 | 情感二分类 | 新闻多分类 | 机器翻译 |
|------|-----------|-----------|----------|
| 输出层 | `Linear(n, 1)` + Sigmoid | `Linear(n, 46)` | 每步 `Linear(d, vocab_size)` |
| 损失函数 | `BCELoss` | `CrossEntropyLoss` 或 `FocalLoss` | `CrossEntropyLoss(ignore_index=PAD)` 或 `LabelSmoothingCE` |
| 解码器 | 无 | 无 | 自回归（贪心 / beam，由 `analyses/decode_grid.py` 比较） |
| 特殊标记 | 仅 PAD | 仅 PAD | PAD、SOS、EOS |
| 主指标 | F1、Accuracy | F1 macro、F1 weighted、Accuracy | BLEU-4（自实现，无外部依赖） |

**Seq2Seq 训练**：Teacher Forcing（`TEACHER_FORCING_RATIO = 0.5`）+ 梯度裁剪（`max_norm=1.0`）。

**翻译用 Transformer**：解码器使用 `nn.Transformer.generate_square_subsequent_mask()` 作因果掩码；词嵌入按 `sqrt(d_model)` 缩放。

**FocalLoss / LabelSmoothingCE**：定义在各自主脚本内，受 CLI 控制；γ=0 时 FocalLoss 必须与 CE 数值等价（被 `tests/test_focal_loss.py` 覆盖）。

### 输出契约（被 `audit_outputs.py` 强校验）

- 所有概率/F1/准确率列必须落在 `[0, 1]`。
- 深度模型行必须满足 `1 <= best_epoch <= trained_epochs`。
- 每个任务的 `outputs/` 必须包含成对的 `*_config.json` + `*_results.csv` 以及 `checkpoints/` 目录。
- `summarize_final_results.py` 假定每任务目录下 `outputs/` 中存在 `{sentiment,reuters,translation}_{config.json,results.csv}`，且按主指标（`f1` / `f1_macro` / 翻译用 BLEU 列）能排序选 best 行。

新增模型或改名指标列时，**必须同步更新 `summarize_final_results.py` 与 `audit_outputs.py`**，否则 `run_full_experiments.ps1` 的最后两步会失败。

### 中文 matplotlib

所有出图代码均设置：

```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
```

跨平台时若中文渲染为方块，请确认系统已安装上述任一字体。

## 修改时的常见陷阱

- **主脚本是补充实验的依赖**：改动主脚本的 CLI 参数、输出列名或 `outputs/` 文件名时，先在 `supplementary_experiments.py` 的 `TASK_SPECS` 与 `audit_outputs.py` / `summarize_final_results.py` 中确认是否需要同步。
- **`tests/` 通过 `sys.path.insert` 直接 import 主脚本类**：重命名 `FocalLoss`、`LabelSmoothingCE`、`CNNBiGRU` 等会立刻让 pytest 红。
- **`--quick` 写到 `outputs_smoke/`，正式跑写到 `outputs/`**：在跑正式实验前确认 `output-dir` 没被 smoke 产物污染。
- **不要把 `*.pt`/`*.png` 提交进仓**：`.gitignore` 已配置；如确需保留某图，单独 `git add -f`。
