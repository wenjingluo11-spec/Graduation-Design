# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导说明。

## 项目概述

这是一个毕业设计项目，使用 PyTorch 探索 NLP 任务。项目包含三个独立的子实验，每个实验位于独立目录中，均以 Jupyter Notebook 的形式实现。

## 运行实验

所有实验均为 Jupyter Notebook。从项目根目录启动：

```bash
# 首先激活虚拟环境
.venv\Scripts\activate

# 启动 Jupyter
jupyter notebook
```

每个 Notebook 均为自包含——从上到下依次运行所有单元格。单元格必须按顺序执行，因为它们共享状态（前面单元格定义的变量会在后面的单元格中使用）。

运行独立脚本（仅限 `情感二分类/` 目录）：
```bash
python "情感二分类/binary classification.py"
```

## 子实验

### 情感二分类/ — IMDB 情感二分类
- **Notebooks**: `binary classification.ipynb`、`binary_classification_modular.ipynb`、`sentiment_analysis_experiment.ipynb`
- **数据**: `aclImdb_v1/` 目录（在 Notebook 中以 `../aclImdb_v1/` 路径加载）
- **模型**: 朴素贝叶斯 + TF-IDF、TextCNN、BiLSTM、Transformer Encoder
- **损失函数**: `BCELoss`，Sigmoid 输出（1 个输出节点）
- **评估指标**: 二分类 F1、准确率

### 新闻多分类/ — Reuters 46 类新闻分类
- **Notebook**: `reuters_multiclass.ipynb`
- **数据**: 项目根目录下的 `reuters.npz`（以 `../reuters.npz` 路径加载）
- **模型**: 朴素贝叶斯 + TF-IDF、TextCNN、BiLSTM、Transformer Encoder
- **损失函数**: `CrossEntropyLoss`（46 个输出节点，隐式 Softmax）
- **评估指标**: 准确率、F1 macro、F1 weighted

### 机器翻译/ — 西班牙语-英语机器翻译
- **Notebook**: `machine_translation.ipynb`
- **数据**: 项目根目录下的 `spa.txt`（以 `../spa.txt` 路径加载）
- **模型**: Seq2Seq + GRU + Bahdanau Attention、完整 Transformer（Encoder-Decoder）
- **评估指标**: BLEU-4 分数（自定义实现，不依赖外部库）

## 架构模式

**超参数配置**：每个 Notebook 顶部附近都有专门的"全局超参数配置"单元格，集中管理所有设置（批大小、训练轮数、词嵌入维度、学习率等）。运行前在此处修改参数。

**设备检测**：所有 Notebook 自动检测 CUDA，若不可用则回退到 CPU：
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**数据相对路径**：Notebook 使用 `../` 相对路径从项目根目录加载数据。请在各子目录中运行 Notebook，或在工作目录不同时调整路径。

**中文 matplotlib 支持**：所有 Notebook 均配置了 `plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']`。如果图表中文显示乱码，请确保已安装 SimHei 或 Microsoft YaHei 字体。

**输出文件**：每个 Notebook 将 PNG 图表保存到其所在目录（例如 `eda_class_distribution.png`、`model_comparison_bar.png`）。

## 各任务关键设计差异

| 方面 | 二分类 | 多分类 | 机器翻译 |
|------|--------|--------|----------|
| 输出层 | `Linear(n, 1)` + Sigmoid | `Linear(n, 46)` | 每步 `Linear(d, vocab_size)` |
| 损失函数 | BCELoss | CrossEntropyLoss | CrossEntropyLoss（忽略 PAD） |
| 解码器 | 无 | 无 | 自回归（贪心解码） |
| 特殊标记 | 仅 PAD | 仅 PAD | PAD、SOS、EOS |

**Seq2Seq 特性**：训练时使用 Teacher Forcing（`TEACHER_FORCING_RATIO = 0.5`）。应用梯度裁剪（`max_norm=1.0`）以防止 RNN 梯度爆炸。

**翻译用 Transformer**：在解码器上使用 `nn.Transformer.generate_square_subsequent_mask()` 作为因果掩码，防止注意力机制关注未来的词。词嵌入按 `sqrt(d_model)` 缩放。

## 依赖项

`.venv/` 虚拟环境已预先配置。关键包：`torch`、`numpy`、`pandas`、`matplotlib`、`seaborn`、`scikit-learn`、`jupyter`。

项目不包含 `requirements.txt` 文件——如需复现环境，请检查 `.venv/Lib/site-packages/` 目录。
