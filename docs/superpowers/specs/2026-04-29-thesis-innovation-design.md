# 毕业设计创新点扩展 + Mac 环境适配 设计文档

**日期**: 2026-04-29
**作者**: 罗文景
**状态**: Draft (待用户 review)

---

## 1. 背景与动机

### 1.1 当前痛点

开题报告"五、创新点"列出了 3 条创新点：

1. RNN/CNN/Transformer 跨三任务系统对比 — **已完成**
2. 轻量化实验流程 + 完整注释代码 — **已完成**
3. 针对中文情感分类的预处理优化 — **悬空且与实际项目矛盾**：本项目使用的是 IMDB 英文数据集，并非中文。

因此实质有效的创新点只剩 2 条，且都偏工程，无方法论或实证分析层面的贡献。需要补充能扛住答辩、能写进论文 4.x 章节的实质内容。

### 1.2 当前环境痛点

仓库内 `.venv/` 是 Windows 虚拟环境（含 `Scripts/*.exe`），无法直接在 Mac 上运行。需要建立 Mac 适配的环境与代码路径。

### 1.3 既有约束

- 不引入预训练模型 / 大语言模型 (开题报告明确范围)
- 仅在 RNN/CNN/Transformer 三类基础模型框架内做扩展
- 已有的 Windows `.venv/` 与历史训练输出 (`outputs/`) 保持不动 (作为对照基线)

---

## 2. 范围 (Scope)

本设计涵盖 6 项工作，按主题归类：

### 2.1 训练改进 (P1 中期已承诺)

| 项 | 内容 | 任务 |
|---|------|------|
| T1 | **Focal loss** + γ 网格扫描 (γ ∈ {0.5, 1.0, 2.0, 5.0}) | Reuters-46 |
| T2 | **Label smoothing** (ε=0.1) + 重训 Transformer | 翻译 |
| T3 | **解码网格**: beam ∈ {1,3,5} × length_penalty ∈ {0.6, 0.8, 1.0, 1.2}, 不重训, 仅复用 checkpoint 重新推理 | 翻译 |

### 2.2 架构创新 (新)

| 项 | 内容 |
|---|------|
| A1 | **CNN-BiGRU 混合模型** (Stacked) 加入三任务对比 |

### 2.3 分析创新 (新)

| 项 | 内容 | 是否重训 |
|---|------|---------|
| C1 | **学习曲线** (Data efficiency)：data ratio ∈ {25%, 50%, 75%, 100%} × 5 模型 × 3 任务（翻译只跑 50%/100%） | 是 |
| C2 | **鲁棒性测试**：测试集词级扰动 (随机删 / 替换为 unk) at {5, 10, 15, 20}% × 5 模型 | 否 |
| C3 | **Transformer 注意力可视化**：5–10 个错例的 self-attention（翻译加 cross-attention）热图 | 否 |

### 2.4 论文创新点章节最终版本

> 1. **架构层面**：提出 CNN-BiGRU 混合架构，融合卷积局部 n-gram 与门控循环时序建模能力，在三类任务上系统验证其适配边界。
> 2. **训练层面**：针对 Reuters-46 长尾问题引入 focal loss 并做 γ 网格分析；针对翻译系统比较 beam × length-penalty × label smoothing 的影响。
> 3. **分析层面**：补充数据规模学习曲线、词级扰动鲁棒性、Transformer 注意力可视化三类深度分析，刻画基础模型在小数据 / 噪声 / 长尾场景下的能力边界。

原悬空的"中文预处理"创新点用此 3 条替换。

---

## 3. CNN-BiGRU 混合模型详细设计

### 3.1 架构 (Stacked)

```
输入 token ids [B, L]
     ↓
词嵌入层 Embedding(V, E)        → [B, L, E]
     ↓
Conv1d(E → C, kernel=3, pad=1)  → [B, C, L]   保留长度
     ↓ ReLU + Dropout
转置为 [B, L, C]
     ↓
BiGRU(C → H)                    → [B, L, 2H]
     ↓
拼接 MaxPool + MeanPool over L  → [B, 4H]
     ↓
Dropout + Linear → 输出层
```

### 3.2 三任务输出层适配

| 任务 | 输出层 | 损失 |
|------|-------|------|
| IMDB 二分类 | `Linear(4H, 1)` + Sigmoid | BCELoss |
| Reuters-46 | `Linear(4H, 46)` | CrossEntropyLoss / FocalLoss |
| 翻译 | 把 CNN-BiGRU 整体当 **encoder**, decoder 复用现有 GRU + Bahdanau Attention | CrossEntropyLoss(ignore PAD) |

### 3.3 参数量目标

控制在 ~2.7M, 与 TextCNN (2.76M) / BiGRU (2.76M) 同量级，保证公平对比。具体配置：
- Embedding dim E = 128
- Conv channels C = 128
- GRU hidden H = 128

### 3.4 命名

类名 `CNNBiGRU`，在 results.csv 的 model 列以 `"CNNBiGRU"` 出现。

---

## 4. Focal Loss 与 Label Smoothing

### 4.1 Focal Loss (Reuters)

公式: `FL(p_t) = -α (1 - p_t)^γ log(p_t)`

实现：在 `reuters_multiclass.py` 内联 `class FocalLoss(nn.Module)`，作为 `--loss {ce, focal}` 选项。

γ 网格执行方式（明确分两阶段）：
- **阶段 A** — γ 选优：仅在 **TextCNN**（baseline 最优模型）上跑 γ ∈ {0.5, 1.0, 2.0, 5.0} 共 4 次，挑选最佳 γ*。
- **阶段 B** — 主对比扩展：在所有 **4 个深度模型** (TextCNN, BiGRU, Transformer, CNNBiGRU) 上用 γ* 跑一次 (4 次)。NaiveBayes 走 sklearn，不参与。
- 总计 8 次完整训练。阶段 A 的 4 组结果做成 ablation 表 (γ vs Macro-F1)；阶段 B 结果合并进主对比表 results.csv。

### 4.2 Label Smoothing (翻译)

实现：在 `machine_translation.py` 内联 `LabelSmoothingCrossEntropy(epsilon=0.1, ignore_index=PAD)`。

只对 Transformer 重训一组（Seq2Seq+Attn 已是较优，不重训）。

### 4.3 解码网格 (翻译)

实现：新文件 `analyses/decode_grid.py`，加载已训好的 best Transformer checkpoint 与 best Seq2Seq checkpoint，用全部 3200 句测试集对所有 (beam, lp) 组合算 BLEU-1/2/4。输出到 `机器翻译/outputs/translation_decode_grid.csv`。

依赖：训练脚本必须保存最优 checkpoint。**前置补丁**（在主实验 Step 1 跑之前必须先打）：
- 检查三个 `.py` 是否在训练后 `torch.save(model.state_dict(), ...)`；若没有，添加保存逻辑到 `outputs/{task}_{model}_best.pt`。
- 同时为 Transformer 与 Seq2Seq 模型添加 `return_attention=True` 模式（也是 C3 注意力可视化的依赖），避免后期再改一遍。

---

## 5. 三大分析创新

### 5.1 学习曲线 `analyses/learning_curve.py`

- 接收参数：`--task {sentiment, reuters, translation}`、`--ratios 25,50,75,100`、`--models all|...`
- 实现：通过修改 max_train_samples 配置项实现 data ratio
- 翻译任务硬编码 ratios=[50, 100]
- 输出：`supplementary_outputs/learning_curve/{task}/curve_results.csv`，列为 model × ratio × score

### 5.2 鲁棒性 `analyses/robustness.py`

- 不重训。加载已训好的所有 best checkpoint（含新增 CNN-BiGRU）
- 扰动方式：
  - `delete`: 随机删除 p% 比例的 token
  - `unk`: 随机将 p% 比例的 token 替换为 \<unk\>
- 扰动率 p ∈ {0, 5, 10, 15, 20}（0 即原测试集）
- 输出：`supplementary_outputs/robustness/{task}/robustness_results.csv`，列为 model × perturbation_type × p × score

### 5.3 注意力可视化 `analyses/attention_viz.py`

- 修改 Transformer 模型，新增 `return_attention=True` 模式
- 三任务各挑 5–10 条样本（优先选错例）
- 翻译额外可视化 cross-attention（spa-en 对齐）
- 输出：`figures/attn_<task>_<sample_id>.png`

---

## 6. Mac 环境适配

### 6.1 设备检测改造

每个 `.py` 顶部统一替换为：

```python
def _get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _get_device()
```

### 6.2 依赖管理

新建 `requirements.txt`（去 GPU 锁版本）：

```
torch>=2.2
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
nltk>=3.8
jupyter>=1.0
```

### 6.3 一键 Mac 安装脚本 `setup_mac.sh`

```bash
#!/bin/bash
set -e
PY=python3
$PY -m venv .venv-mac
source .venv-mac/bin/activate
pip install -U pip
pip install -r requirements.txt
$PY -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ Mac 环境就绪。 source .venv-mac/bin/activate 即可用。"
```

`.venv-mac/` 与现有 `.venv/` (Windows) 共存。`.gitignore` 已含 `.venv*`，无需改。

### 6.4 字体修复

三个 `.py` 与 `make_figures.py` 共享同一字体加载逻辑：通过路径 `addfont` 加载 `/System/Library/Fonts/Supplemental/Arial Unicode.ttf`（Mac）或 `C:/Windows/Fonts/msyh.ttc`（Win）；列表中 Arial Unicode MS 优先。该方案已在 `make_figures.py` 中验证。

### 6.5 已知 MPS 限制

PyTorch MPS 在某些算子上仍会回退或报错。处理：
- 训练循环最外层 `try ... except RuntimeError as e: if 'mps' in str(e): fallback_to_cpu()`
- 警告打印 `[WARN] MPS 算子异常, fallback CPU` 即可

不需要也不应当为此重写训练循环。

---

## 7. 文件组织

```
Graduation-Design/
├── 情感二分类/
│   └── sentiment_analysis.py            [改] +CNNBiGRU, +MPS device
├── 新闻多分类/
│   └── reuters_multiclass.py            [改] +CNNBiGRU, +FocalLoss, +γ-scan CLI, +MPS
├── 机器翻译/
│   └── machine_translation.py           [改] +CNNBiGRU encoder, +LabelSmoothing CLI, +MPS
├── analyses/                            [新]
│   ├── __init__.py
│   ├── decode_grid.py                   # P1 翻译解码网格 (T3)
│   ├── learning_curve.py                # 数据规模学习曲线 (C1)
│   ├── robustness.py                    # 扰动鲁棒性 (C2)
│   └── attention_viz.py                 # 注意力可视化 (C3)
├── make_figures.py                      [扩展] 增 6 张图
├── requirements.txt                     [新]
├── setup_mac.sh                         [新]
├── README_MAC.md                        [新] Mac 跑法说明 + 重跑命令
└── docs/superpowers/specs/2026-04-29-thesis-innovation-design.md  [本文件]
```

历史输出 `outputs/` 与 `supplementary_outputs/stability,error_analysis,efficiency` 不动。新输出走 `supplementary_outputs/learning_curve/`、`supplementary_outputs/robustness/` 等子目录。

---

## 8. 重跑命令清单 (按依赖顺序)

```bash
# Step 0: 一次性 Mac 环境
bash setup_mac.sh && source .venv-mac/bin/activate

# Step 1: 主实验 — 5 模型 (含新 CNN-BiGRU) × 3 任务
cd 情感二分类 && python sentiment_analysis.py --include-hybrid && cd ..
cd 新闻多分类 && python reuters_multiclass.py --include-hybrid && cd ..
cd 机器翻译  && python machine_translation.py  --include-hybrid && cd ..

# Step 2: Focal loss γ 扫描 (Reuters)
cd 新闻多分类
for g in 0.5 1.0 2.0 5.0; do
  python reuters_multiclass.py --loss focal --focal-gamma $g --output-suffix _focal_g$g
done
cd ..

# Step 3: Label smoothing + 解码网格 (翻译)
cd 机器翻译
python machine_translation.py --label-smoothing 0.1 --output-suffix _ls
cd ..
python analyses/decode_grid.py

# Step 4: 三大分析创新
python analyses/learning_curve.py
python analyses/robustness.py
python analyses/attention_viz.py

# Step 5: 重生成图表
python make_figures.py
```

---

## 9. 测试 / 验收标准

- 每个 `.py` 在 `--quick` 模式下能在 Mac (MPS) 上无报错跑通完整流程
- `requirements.txt` 在干净 venv 下 `pip install -r` 后能直接 `import torch` 无报错
- `make_figures.py` 在新增数据后输出 ≥ 22 张图（原 16 + 新增 6）
- 新增的 5 个 `analyses/*.py` 各自能独立运行（不依赖临时变量）
- README_MAC.md 中的所有命令复制粘贴到终端能跑通

---

## 10. 不在范围 (Out of Scope)

- 不重训历史 Windows CUDA 上跑过的 baseline 实验 (sentiment / reuters baseline / 翻译 baseline)，避免重复劳动
- 不引入预训练词向量 (Word2Vec/GloVe) — 保持开题报告"端到端学习"基调
- 不做模型量化 / 剪枝 / 蒸馏 — 留待 future work
- 不修改开题报告内容（开题已交，不可改），但论文正文里"创新点"章节按本设计写

---

## 11. 实施顺序 (供后续 writing-plans 参考)

1. **Mac 环境 + device 适配**：`requirements.txt`、`setup_mac.sh`、三个 `.py` 的 device 三级 fallback。`--quick` 模式 smoke 验证。
2. **前置补丁**：三个 `.py` 加 best checkpoint 保存逻辑；Transformer / Seq2Seq 加 `return_attention` 模式。
3. **CNN-BiGRU 模型类**：在三个 `.py` 加 `CNNBiGRU`，`--include-hybrid` 启用，`--quick` 验证。
4. **训练改进**：Reuters 加 `FocalLoss`；翻译加 `LabelSmoothingCE`。
5. `analyses/decode_grid.py`
6. `analyses/learning_curve.py`
7. `analyses/robustness.py`
8. `analyses/attention_viz.py`
9. `make_figures.py` 扩展 6 张图
10. `README_MAC.md` 总入口 + 全套重跑命令验证

每完成一项做一次 commit，便于回滚。
