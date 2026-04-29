# 毕业设计创新点扩展实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在三个 NLP 任务（IMDB 情感二分类、Reuters-46 多分类、西英翻译）上实施 CNN-BiGRU 混合架构、focal loss、label smoothing 三类训练改进，加入学习曲线/鲁棒性/注意力可视化三类深度分析，并完成 Mac 环境适配。

**Architecture:** 三个现有 `.py` 增加新模型类与 CLI 选项；新建 `analyses/` 目录承载 4 个分析脚本；扩展 `make_figures.py` 支持新图表；保持 Windows `.venv/` 与历史 `outputs/` 不动以作对照基线。

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.2 (with MPS for Apple Silicon), pandas, numpy, matplotlib, seaborn, scikit-learn, nltk

**Spec Reference:** `docs/superpowers/specs/2026-04-29-thesis-innovation-design.md`

---

## 一项发现 (相对 spec 的修正)

通过代码审查发现：**三个 `.py` 已经全部实现 checkpoint 保存**（`sentiment_analysis.py:404`, `reuters_multiclass.py:356`, `machine_translation.py:905,975`）。spec 第 4.3 节里的"前置补丁: 加 checkpoint 保存"任务**不需要**。但 `return_attention` 模式仍然没有，作为 Task 8 / 9 处理。

---

## Phase 1: Mac 环境适配

### Task 1: 创建 `requirements.txt`

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: 写文件**

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

- [ ] **Step 2: 验证文件存在且非空**

Run: `wc -l requirements.txt`
Expected: 输出行数 ≥ 8

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt for cross-platform installation"
```

---

### Task 2: 创建 `setup_mac.sh`

**Files:**
- Create: `setup_mac.sh`

- [ ] **Step 1: 写文件**

```bash
#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON:-python3}
echo "==> 创建 .venv-mac 虚拟环境"
$PY -m venv .venv-mac
# shellcheck disable=SC1091
source .venv-mac/bin/activate
echo "==> 升级 pip"
pip install -U pip
echo "==> 安装依赖"
pip install -r requirements.txt
echo "==> 下载 NLTK punkt"
python -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ Mac 环境就绪。 source .venv-mac/bin/activate 即可使用。"
```

- [ ] **Step 2: 加可执行权限**

```bash
chmod +x setup_mac.sh
```

- [ ] **Step 3: Commit**

```bash
git add setup_mac.sh
git commit -m "chore: add Mac venv setup script"
```

---

### Task 3: 验证 Mac venv 安装能跑通

**Files:**
- 仅运行验证

- [ ] **Step 1: 跑 setup 脚本**

Run: `bash setup_mac.sh 2>&1 | tail -20`
Expected: 最后一行包含 `✓ Mac 环境就绪`，无 `ERROR` 或 `error:`

- [ ] **Step 2: 验证 PyTorch + MPS 可用**

```bash
source .venv-mac/bin/activate
python -c "import torch; print('torch', torch.__version__, 'MPS', torch.backends.mps.is_available())"
```
Expected: 输出包含 `MPS True`（Apple Silicon）或 `MPS False`（Intel Mac）。

- [ ] **Step 3: 验证依赖**

```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, nltk; print('all ok')"
```
Expected: 输出 `all ok`

- [ ] **Step 4: Commit (如有 .gitignore 调整)**

```bash
# 检查 .gitignore 是否已忽略 .venv-mac
grep -q ".venv" .gitignore || echo ".venv*/" >> .gitignore
git diff --quiet .gitignore || (git add .gitignore && git commit -m "chore: ignore .venv-mac")
```

---

### Task 4: `sentiment_analysis.py` 加 MPS 设备检测

**Files:**
- Modify: `情感二分类/sentiment_analysis.py:74` (Config.device 默认值)
- Modify: `情感二分类/sentiment_analysis.py:104` (parse_args 的 --device choices)
- Modify: `情感二分类/sentiment_analysis.py:113` (parse_args 里 device 处理逻辑)

- [ ] **Step 1: 在 Config 之前加 helper 函数**

在 `class Config` 上方插入：

```python
def _detect_device() -> str:
    """三级设备 fallback: mps → cuda → cpu"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

- [ ] **Step 2: 改 Config.device 默认值**

把这一行：
```python
device: str = "cuda" if torch.cuda.is_available() else "cpu"
```
改为：
```python
device: str = field(default_factory=_detect_device)
```

记得在文件顶部 `from dataclasses import asdict, dataclass` 改成 `from dataclasses import asdict, dataclass, field`。

- [ ] **Step 3: 改 argparse 的 device 参数**

把：
```python
parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
```
改为：
```python
parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
```

把：
```python
if args.device is not None:
    cfg.device = args.device if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
```
改为：
```python
if args.device is not None:
    cfg.device = args.device  # 用户显式指定即生效，运行期出错可 fallback
```

- [ ] **Step 4: smoke test**

```bash
cd 情感二分类
python -c "from sentiment_analysis import _detect_device, Config; c = Config(); print(c.device); assert c.device in ('mps','cuda','cpu')"
cd ..
```
Expected: 输出 `mps`/`cuda`/`cpu` 之一，断言不抛错

- [ ] **Step 5: Commit**

```bash
git add 情感二分类/sentiment_analysis.py
git commit -m "feat(sentiment): three-tier device fallback (mps/cuda/cpu)"
```

---

### Task 5: `reuters_multiclass.py` 加 MPS 设备检测

**Files:**
- Modify: `新闻多分类/reuters_multiclass.py:65` (Config.device 默认值)
- Modify: `新闻多分类/reuters_multiclass.py:78,90` (parse_args)

- [ ] **Step 1: 在 Config 之前加 helper 函数（同 Task 4 Step 1 代码）**

```python
def _detect_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

- [ ] **Step 2: 改 Config.device 默认值**

把：
```python
device: str = "cuda" if torch.cuda.is_available() else "cpu"
```
改为：
```python
device: str = field(default_factory=_detect_device)
```

import 同步加 `field`。

- [ ] **Step 3: 改 argparse**

把 `choices=[None, "cpu", "cuda"]` 改为 `choices=[None, "cpu", "cuda", "mps"]`，把 device 处理逻辑改为：
```python
if args.device is not None:
    cfg.device = args.device
```

- [ ] **Step 4: smoke test**

```bash
cd 新闻多分类
python -c "from reuters_multiclass import _detect_device, Config; c = Config(); print(c.device); assert c.device in ('mps','cuda','cpu')"
cd ..
```
Expected: 输出 `mps`/`cuda`/`cpu` 之一

- [ ] **Step 5: Commit**

```bash
git add 新闻多分类/reuters_multiclass.py
git commit -m "feat(reuters): three-tier device fallback (mps/cuda/cpu)"
```

---

### Task 6: `machine_translation.py` 加 MPS 设备检测

**Files:**
- Modify: `机器翻译/machine_translation.py:74` (Config.device)
- Modify: `机器翻译/machine_translation.py:88,124` (parse_args)

- [ ] **Step 1: 在 Config 之前加 helper 函数（同上）**

```python
def _detect_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

- [ ] **Step 2: 改 Config.device 默认值（同上模式）**

```python
device: str = field(default_factory=_detect_device)
```

import 加 `field`。

- [ ] **Step 3: 改 argparse**

`choices=[None, "cpu", "cuda", "mps"]`；device 处理改为 `cfg.device = args.device`。

- [ ] **Step 4: smoke test**

```bash
cd 机器翻译
python -c "from machine_translation import _detect_device, Config; c = Config(); print(c.device); assert c.device in ('mps','cuda','cpu')"
cd ..
```

- [ ] **Step 5: Commit**

```bash
git add 机器翻译/machine_translation.py
git commit -m "feat(translation): three-tier device fallback (mps/cuda/cpu)"
```

---

### Task 7: 三个脚本 `--quick` smoke test (Mac MPS)

- [ ] **Step 1: sentiment quick**

```bash
source .venv-mac/bin/activate
cd 情感二分类
python sentiment_analysis.py --quick 2>&1 | tail -20
cd ..
```
Expected: 退出码 0；输出包含 `测试集指标` 与 4 个模型名 (NaiveBayes/TextCNN/BiGRU/Transformer)

- [ ] **Step 2: reuters quick**

```bash
cd 新闻多分类
python reuters_multiclass.py --quick 2>&1 | tail -20
cd ..
```
Expected: 退出码 0

- [ ] **Step 3: translation quick**

```bash
cd 机器翻译
python machine_translation.py --quick 2>&1 | tail -10
cd ..
```
Expected: 退出码 0

- [ ] **Step 4: 如出现 MPS 算子错误,记录并切到 CPU 重试**

如某个脚本报 `MPSNDArray ... not implemented`,则用 `--device cpu` 重跑该脚本,在 README_MAC.md (Task 21) 中标注。

- [ ] **Step 5: Commit (如有问题修复)**

```bash
git diff --quiet || (git add -A && git commit -m "fix: address MPS-specific runtime issues found in smoke test")
```

---

## Phase 2: 前置补丁 — return_attention 模式

### Task 8: 分类任务 Transformer 加 return_attention

**Files:**
- Modify: `情感二分类/sentiment_analysis.py:266-303` (TransformerClassifier)
- Modify: `新闻多分类/reuters_multiclass.py:228-264` (TransformerMulti)

- [ ] **Step 1: 修改 sentiment 的 `TransformerClassifier.forward` 签名**

把 forward 改成接受 `return_attention=False`：
```python
def forward(self, x: torch.Tensor, return_attention: bool = False):
    # 现有 embedding + pos_encoding 不变
    emb = self.pos_encoding(self.embedding(x) * math.sqrt(self.emb_dim))
    src_key_padding_mask = x.eq(0)
    if return_attention:
        # 手动逐层调用以获取注意力
        attns = []
        h = emb
        for layer in self.encoder.layers:
            h, attn = layer.self_attn(
                h, h, h,
                key_padding_mask=src_key_padding_mask,
                need_weights=True, average_attn_weights=False,
            )
            attns.append(attn.detach().cpu())
            h = layer.norm1(layer.dropout1(h) + emb)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(layer.dropout2(ff) + h)
        pooled = h.mean(dim=1)
        return self.classifier(self.dropout(pooled)).squeeze(1), attns
    out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
    pooled = out.mean(dim=1)
    return self.classifier(self.dropout(pooled)).squeeze(1)
```

注：上述手动 forward 与 nn.TransformerEncoderLayer 内部行为可能略有差异。简化方案：仅返回最后一层 attention：

```python
def forward(self, x: torch.Tensor, return_attention: bool = False):
    emb = self.pos_encoding(self.embedding(x) * math.sqrt(self.emb_dim))
    src_key_padding_mask = x.eq(0)
    out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
    pooled = out.mean(dim=1)
    logits = self.classifier(self.dropout(pooled)).squeeze(1)
    if return_attention:
        # 取最后一层 self-attn 单独跑一次以获取权重
        last_layer = self.encoder.layers[-1]
        with torch.no_grad():
            _, attn = last_layer.self_attn(
                out, out, out,
                key_padding_mask=src_key_padding_mask,
                need_weights=True, average_attn_weights=False,
            )
        return logits, attn  # attn shape: [B, num_heads, L, L]
    return logits
```

采用第二种简化方案（避免重写整个 forward 路径）。

- [ ] **Step 2: 同样改 reuters 的 `TransformerMulti.forward`**

逻辑相同，只是 logits 维度是 46。

- [ ] **Step 3: 写一个临时 smoke test**

新建 `tests/test_return_attention.py`（如 tests/ 目录不存在则创建）：
```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "情感二分类"))
sys.path.insert(0, str(ROOT / "新闻多分类"))

import torch
from sentiment_analysis import TransformerClassifier as SentTrans
from reuters_multiclass import TransformerMulti as ReutTrans

def test_sentiment_return_attention():
    m = SentTrans(vocab_size=100, emb_dim=32, num_heads=2, ff_dim=64, num_layers=2, dropout=0.1)
    x = torch.randint(1, 100, (3, 20))
    logits, attn = m(x, return_attention=True)
    assert logits.shape == (3,)
    assert attn.shape == (3, 2, 20, 20)  # B, heads, L, L

def test_reuters_return_attention():
    m = ReutTrans(vocab_size=100, num_classes=46, emb_dim=32, num_heads=2, ff_dim=64, num_layers=2, dropout=0.1)
    x = torch.randint(1, 100, (3, 20))
    logits, attn = m(x, return_attention=True)
    assert logits.shape == (3, 46)
    assert attn.shape == (3, 2, 20, 20)
```

- [ ] **Step 4: 运行 smoke test**

```bash
source .venv-mac/bin/activate
python -m pytest tests/test_return_attention.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add 情感二分类/sentiment_analysis.py 新闻多分类/reuters_multiclass.py tests/test_return_attention.py
git commit -m "feat(transformer): add return_attention mode for classification tasks"
```

---

### Task 9: 翻译 Transformer + Seq2Seq 加 return_attention

**Files:**
- Modify: `机器翻译/machine_translation.py:393-414` (TransformerTranslator.forward)
- Modify: `机器翻译/machine_translation.py:301-326` (Decoder.forward — Bahdanau attention)

- [ ] **Step 1: 修改 `TransformerTranslator.forward`**

```python
def forward(self, src: torch.Tensor, tgt_in: torch.Tensor, return_attention: bool = False):
    # 现有 mask + embedding 逻辑保持不变
    src_mask = None
    tgt_len = tgt_in.size(1)
    tgt_causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=src.device, dtype=torch.bool),
        diagonal=1,
    )
    src_key_padding_mask = src.eq(PAD_IDX)
    tgt_key_padding_mask = tgt_in.eq(PAD_IDX)
    src_emb = self.pos(self.src_embedding(src) * math.sqrt(self.d_model))
    tgt_emb = self.pos(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))

    out = self.transformer(
        src_emb, tgt_emb,
        src_mask=src_mask, tgt_mask=tgt_causal_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )
    logits = self.fc(out)
    if return_attention:
        # 单独跑最后一层 decoder 以提取 cross-attention
        memory = self.transformer.encoder(
            src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
        )
        last_dec = self.transformer.decoder.layers[-1]
        with torch.no_grad():
            # cross-attention: query=tgt, key/value=memory
            _, cross_attn = last_dec.multihead_attn(
                tgt_emb, memory, memory,
                key_padding_mask=src_key_padding_mask,
                need_weights=True, average_attn_weights=False,
            )
        return logits, cross_attn  # [B, heads, T_tgt, T_src]
    return logits
```

- [ ] **Step 2: Bahdanau Decoder 已经计算 attention weights，只需要返回**

查看 `Decoder.forward` 的现有实现（Line 301-326），它内部用 `BahdanauAttention` 算 alpha。把这个 alpha 加到返回值。

```python
# 在 Decoder.forward 末尾把 attn_weights 一起返回
def forward(self, ...):
    # ... 现有逻辑 ...
    # 在 attention 模块返回的地方多保存一份
    attn_weights = self.attention(hidden, encoder_outputs, mask=src_mask)  # 已存在
    # ... 后续解码步骤 ...
    return output, hidden, attn_weights  # 把 attn_weights 加到现有返回元组
```

注: 调用 Decoder 的地方（`Seq2Seq.forward`, `decode_seq2seq_greedy`, `decode_seq2seq_beam`）需要同步解包多一个返回值。**改最小化方案**: 让 Decoder.forward 始终返回 attn，调用方用 `_` 忽略；attention_viz.py 显式接收。

- [ ] **Step 3: 同步修改 Seq2Seq 调用**

在 `Seq2Seq.forward`、`decode_seq2seq_greedy`、`decode_seq2seq_beam` 三处把 `output, hidden = self.decoder(...)` 改为 `output, hidden, _ = self.decoder(...)`。

- [ ] **Step 4: 加 smoke test**

新增到 `tests/test_return_attention.py`:
```python
sys.path.insert(0, str(ROOT / "机器翻译"))

def test_translation_transformer_attention():
    from machine_translation import TransformerTranslator
    m = TransformerTranslator(src_vocab_size=100, tgt_vocab_size=120, d_model=32, nhead=2, num_layers=2, ff_dim=64, dropout=0.1)
    src = torch.randint(1, 100, (2, 8))
    tgt = torch.randint(1, 120, (2, 6))
    logits, attn = m(src, tgt, return_attention=True)
    assert logits.shape == (2, 6, 120)
    assert attn.shape == (2, 2, 6, 8)

def test_translation_seq2seq_attention():
    from machine_translation import Encoder, Decoder, BahdanauAttention, Seq2Seq
    enc = Encoder(vocab_size=100, emb_dim=16, hid_dim=24, dropout=0.1)
    attn = BahdanauAttention(hid_dim=24)
    dec = Decoder(vocab_size=120, emb_dim=16, hid_dim=24, attention=attn, dropout=0.1)
    s2s = Seq2Seq(enc, dec, device=torch.device("cpu"))
    src = torch.randint(1, 100, (2, 8))
    tgt = torch.randint(1, 120, (2, 6))
    out = s2s(src, tgt, teacher_forcing_ratio=1.0)
    assert out.shape == (2, 6, 120)
```

- [ ] **Step 5: 运行 test**

```bash
python -m pytest tests/test_return_attention.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add 机器翻译/machine_translation.py tests/test_return_attention.py
git commit -m "feat(translation): add return_attention mode for Transformer + propagate Seq2Seq alphas"
```

---

## Phase 3: CNN-BiGRU 混合模型

### Task 10: `sentiment_analysis.py` 加 CNNBiGRU 类

**Files:**
- Modify: `情感二分类/sentiment_analysis.py:209-249` (在 BiGRUClassifier 之后插入)
- Modify: `情感二分类/sentiment_analysis.py:78-117` (parse_args 加 --include-hybrid)
- Modify: `情感二分类/sentiment_analysis.py:577-720` (run_experiment 调度新模型)

- [ ] **Step 1: 加 CNNBiGRU 类**

在 `BiGRUClassifier` 之后插入：

```python
class CNNBiGRU(nn.Module):
    """Stacked CNN-BiGRU 混合架构: 卷积提局部 n-gram → BiGRU 建模时序 → max+mean pool → 分类头"""
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.bigru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2 * 2, 1)  # bi*hidden, max+mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                           # [B, L, E]
        conv_in = emb.transpose(1, 2)                     # [B, E, L]
        conv_out = torch.relu(self.conv(conv_in))         # [B, E, L]
        gru_in = conv_out.transpose(1, 2)                 # [B, L, E]
        gru_out, _ = self.bigru(gru_in)                   # [B, L, 2H]
        max_pool = gru_out.max(dim=1)[0]                  # [B, 2H]
        mean_pool = gru_out.mean(dim=1)                   # [B, 2H]
        pooled = torch.cat([max_pool, mean_pool], dim=1)  # [B, 4H]
        return self.classifier(self.dropout(pooled)).squeeze(1)
```

- [ ] **Step 2: 加 CLI flag --include-hybrid**

在 parse_args 末尾、`args = parser.parse_args()` 之前加：
```python
parser.add_argument("--include-hybrid", action="store_true", help="加入 CNN-BiGRU 混合模型对比")
```
Config dataclass 加字段:
```python
include_hybrid: bool = False
```
parse_args 末尾构造 cfg 时加 `include_hybrid=args.include_hybrid`。

- [ ] **Step 3: run_experiment 末尾加 CNNBiGRU 调度**

在 Transformer 训练块之后（即 `# Transformer` 块结束 + `results.append({"model": "Transformer", ...})` 之后）加：

```python
if cfg.include_hybrid:
    hybrid = CNNBiGRU(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    h_metrics, h_hist, h_y, h_pred, h_prob = run_dl_model(
        "CNNBiGRU", hybrid,
        train_loader, val_loader, test_loader,
        ckpt_dir, cfg,
    )
    plot_history(h_hist, "CNNBiGRU", out_dir)
    plot_confusion_matrix(
        confusion_matrix(h_y, h_pred),
        ["negative", "positive"],
        "CNNBiGRU Confusion Matrix",
        out_dir / "cnnbigru_confusion_matrix.png",
    )
    results.append({"model": "CNNBiGRU", **h_metrics})
    pred_by_model["CNNBiGRU"] = h_pred
    prob_by_model["CNNBiGRU"] = h_prob
```

- [ ] **Step 4: smoke test**

新建 `tests/test_cnnbigru.py`：
```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "情感二分类"))
import torch
from sentiment_analysis import CNNBiGRU

def test_cnnbigru_sentiment_forward():
    m = CNNBiGRU(vocab_size=200, emb_dim=32, hidden_dim=24, dropout=0.1)
    x = torch.randint(1, 200, (4, 50))
    out = m(x)
    assert out.shape == (4,)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"CNNBiGRU sentiment params: {n_params}")
```

```bash
python -m pytest tests/test_cnnbigru.py::test_cnnbigru_sentiment_forward -v -s
```
Expected: PASS, 输出参数量

- [ ] **Step 5: --quick 端到端验证**

```bash
cd 情感二分类
python sentiment_analysis.py --quick --include-hybrid 2>&1 | tail -20
cd ..
```
Expected: 退出码 0；最终 results.csv 含 5 行模型 (NB, TextCNN, BiGRU, Transformer, CNNBiGRU)

- [ ] **Step 6: Commit**

```bash
git add 情感二分类/sentiment_analysis.py tests/test_cnnbigru.py
git commit -m "feat(sentiment): add CNN-BiGRU hybrid model with --include-hybrid flag"
```

---

### Task 11: `reuters_multiclass.py` 加 CNNBiGRU 类

**Files:**
- Modify: `新闻多分类/reuters_multiclass.py:198-213` (在 BiGRUMulti 之后插入)
- Modify: `新闻多分类/reuters_multiclass.py:68-105` (parse_args + Config 加 --include-hybrid)
- Modify: `新闻多分类/reuters_multiclass.py:539-647` (run_experiment 调度)

- [ ] **Step 1: 加 CNNBiGRUMulti 类**

```python
class CNNBiGRUMulti(nn.Module):
    """Stacked CNN-BiGRU 用于 46 类分类"""
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int, hidden_dim: int, dropout: float, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.bigru = nn.GRU(emb_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        conv_out = torch.relu(self.conv(emb.transpose(1, 2)))
        gru_in = conv_out.transpose(1, 2)
        gru_out, _ = self.bigru(gru_in)
        max_pool = gru_out.max(dim=1)[0]
        mean_pool = gru_out.mean(dim=1)
        pooled = torch.cat([max_pool, mean_pool], dim=1)
        return self.classifier(self.dropout(pooled))
```

- [ ] **Step 2: 加 --include-hybrid CLI 与 Config 字段（同 Task 10 Step 2 模式）**

- [ ] **Step 3: run_experiment 末尾加 CNNBiGRU 调度**

在 Transformer 块之后插入：
```python
if cfg.include_hybrid:
    hybrid = CNNBiGRUMulti(num_words, num_classes, cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    h_metrics, h_hist, h_y, h_pred = run_dl_model(
        "CNNBiGRU", hybrid,
        train_loader, val_loader, test_loader,
        ckpt_dir, cfg,
    )
    plot_history(h_hist, "CNNBiGRU", out_dir)
    plot_conf_matrix(h_y, h_pred, "CNNBiGRU Confusion Matrix", out_dir / "cnnbigru_confusion_matrix.png")
    results.append({"model": "CNNBiGRU", **h_metrics})
    pred_by_model["CNNBiGRU"] = h_pred
```

- [ ] **Step 4: smoke test**

加到 `tests/test_cnnbigru.py`:
```python
sys.path.insert(0, str(ROOT / "新闻多分类"))

def test_cnnbigru_reuters_forward():
    from reuters_multiclass import CNNBiGRUMulti
    m = CNNBiGRUMulti(vocab_size=200, num_classes=46, emb_dim=32, hidden_dim=24, dropout=0.1)
    x = torch.randint(1, 200, (4, 50))
    out = m(x)
    assert out.shape == (4, 46)
```

```bash
python -m pytest tests/test_cnnbigru.py::test_cnnbigru_reuters_forward -v
```
Expected: PASS

- [ ] **Step 5: --quick 端到端**

```bash
cd 新闻多分类
python reuters_multiclass.py --quick --include-hybrid 2>&1 | tail -15
cd ..
```
Expected: 退出码 0；results.csv 含 5 行

- [ ] **Step 6: Commit**

```bash
git add 新闻多分类/reuters_multiclass.py tests/test_cnnbigru.py
git commit -m "feat(reuters): add CNN-BiGRU hybrid model"
```

---

### Task 12: `machine_translation.py` 加 CNNBiGRU encoder

**Files:**
- Modify: `机器翻译/machine_translation.py:272-283` (Encoder 之后加 CNNBiGRUEncoder)
- Modify: `机器翻译/machine_translation.py:78-125` (parse_args 加 --include-hybrid)
- Modify: `机器翻译/machine_translation.py:817-1124` (run_experiment 加分支)

- [ ] **Step 1: 加 CNNBiGRUEncoder 类**

在 `class Encoder` 之后：
```python
class CNNBiGRUEncoder(nn.Module):
    """混合 encoder: Conv1d (k=3) + BiGRU, 用于 Seq2Seq 替换原 Encoder"""
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.bigru = nn.GRU(emb_dim, hid_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        emb = self.dropout(self.embedding(src))
        conv = torch.relu(self.conv(emb.transpose(1, 2))).transpose(1, 2)
        outputs, hidden = self.bigru(conv)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        decoder_init = torch.tanh(self.fc(hidden_cat)).unsqueeze(0)
        return outputs, decoder_init
```

接口与现有 `Encoder.forward` 保持一致 (`outputs, hidden`)，方便 Seq2Seq 直接替换。

- [ ] **Step 2: 加 --include-hybrid CLI + Config**

在 parse_args 加 `--include-hybrid`，Config 加 `include_hybrid: bool = False`。

- [ ] **Step 3: run_experiment 加 hybrid Seq2Seq 训练分支**

找到 Seq2Seq 训练块结束后（即 `seq2seq.load_state_dict(...)` 之后、Transformer 训练之前），插入：

```python
if cfg.include_hybrid:
    print("\n" + "=" * 70 + "\n训练 Seq2Seq+Hybrid (CNN-BiGRU encoder)\n" + "=" * 70)
    hyb_encoder = CNNBiGRUEncoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    hyb_attn = BahdanauAttention(cfg.hidden_dim)
    hyb_decoder = Decoder(len(tgt_vocab), cfg.embedding_dim, cfg.hidden_dim, hyb_attn, cfg.dropout)
    hyb_s2s = Seq2Seq(hyb_encoder, hyb_decoder, device).to(device)
    # 复用 train_seq2seq_epoch / decode_seq2seq_* / evaluate_seq2seq_bleu
    # ... (训练循环复制 Seq2Seq 块,把 seq2seq 改成 hyb_s2s,checkpoint 路径 hybrid_best.pt)
```

由于 Seq2Seq 训练循环代码量较大（~80 行），实际改动是把现有 `seq2seq` 块复制一份，把变量名替换为 `hyb_s2s`，把 checkpoint 路径与 model 名称改为 "CNNBiGRU"，最后在 results 列表加一行。

- [ ] **Step 4: smoke test**

加到 `tests/test_cnnbigru.py`:
```python
sys.path.insert(0, str(ROOT / "机器翻译"))

def test_cnnbigru_translation_encoder():
    from machine_translation import CNNBiGRUEncoder
    m = CNNBiGRUEncoder(vocab_size=200, emb_dim=32, hid_dim=24, dropout=0.1)
    x = torch.randint(1, 200, (4, 12))
    outputs, hidden = m(x)
    assert outputs.shape == (4, 12, 48)  # 2*hid_dim
    assert hidden.shape == (1, 4, 24)
```

```bash
python -m pytest tests/test_cnnbigru.py::test_cnnbigru_translation_encoder -v
```
Expected: PASS

- [ ] **Step 5: --quick 端到端**

```bash
cd 机器翻译
python machine_translation.py --quick --include-hybrid 2>&1 | tail -10
cd ..
```
Expected: 退出码 0；translation_results.csv 含 3 行 (Seq2Seq+Attention, Transformer, CNNBiGRU)

- [ ] **Step 6: Commit**

```bash
git add 机器翻译/machine_translation.py tests/test_cnnbigru.py
git commit -m "feat(translation): add CNN-BiGRU hybrid encoder for Seq2Seq"
```

---

### Task 13: 三任务 `--include-hybrid --quick` 联合 smoke

- [ ] **Step 1: 三个连跑**

```bash
source .venv-mac/bin/activate
cd 情感二分类 && python sentiment_analysis.py --quick --include-hybrid 2>&1 | tail -3 && cd ..
cd 新闻多分类 && python reuters_multiclass.py --quick --include-hybrid 2>&1 | tail -3 && cd ..
cd 机器翻译  && python machine_translation.py  --quick --include-hybrid 2>&1 | tail -3 && cd ..
```
Expected: 三段都退出码 0

- [ ] **Step 2: 验证三个 results.csv 都含 CNNBiGRU 行**

```bash
grep -l CNNBiGRU 情感二分类/outputs/sentiment_results.csv 新闻多分类/outputs/reuters_results.csv 机器翻译/outputs/translation_results.csv
```
Expected: 三个文件路径全部输出

- [ ] **Step 3: 无 commit (纯验证)**

---

## Phase 4: 训练改进

### Task 14: `reuters_multiclass.py` 加 FocalLoss + γ 扫描 CLI

**Files:**
- Modify: `新闻多分类/reuters_multiclass.py:230-280` (在 train_one_epoch 之前加 FocalLoss 类)
- Modify: `新闻多分类/reuters_multiclass.py:68-105` (parse_args 加 --loss --focal-gamma --output-suffix)
- Modify: `新闻多分类/reuters_multiclass.py:310-380` (run_dl_model 接受可选 criterion)

- [ ] **Step 1: 加 FocalLoss 类**

在 `train_one_epoch` 上方插入：
```python
class FocalLoss(nn.Module):
    """多分类 focal loss: FL = -(1-p_t)^gamma * log(p_t)"""
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()
```

- [ ] **Step 2: 加 CLI**

```python
parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
parser.add_argument("--focal-gamma", type=float, default=2.0)
parser.add_argument("--output-suffix", type=str, default="", help="附加到所有输出文件名后")
```

Config 加字段：
```python
loss: str = "ce"
focal_gamma: float = 2.0
output_suffix: str = ""
```

- [ ] **Step 3: 改 run_dl_model 接受 criterion 参数**

把 `def run_dl_model(model_name, model, train_loader, val_loader, test_loader, checkpoint_dir, cfg)` 内部的：
```python
criterion = nn.CrossEntropyLoss()
```
改为：
```python
if cfg.loss == "focal":
    criterion = FocalLoss(gamma=cfg.focal_gamma)
else:
    criterion = nn.CrossEntropyLoss()
```

- [ ] **Step 4: 改输出路径加 suffix**

修改 `run_experiment` 末尾保存 csv/json 的位置：
```python
suffix = cfg.output_suffix
results_df.to_csv(out_dir / f"reuters_results{suffix}.csv", index=False, encoding="utf-8-sig")
# 同样处理 reuters_efficiency.csv 等所有输出文件
```

- [ ] **Step 5: 单元测试 FocalLoss**

`tests/test_focal_loss.py`:
```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "新闻多分类"))
import torch
from reuters_multiclass import FocalLoss

def test_focal_loss_gamma_zero_equals_ce():
    fl = FocalLoss(gamma=0.0)
    ce = torch.nn.CrossEntropyLoss()
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    assert torch.allclose(fl(logits, target), ce(logits, target), atol=1e-5)

def test_focal_loss_gamma_positive_smaller_than_ce_for_easy():
    fl = FocalLoss(gamma=2.0)
    # 制造高置信度正确预测
    logits = torch.zeros(4, 3); logits[range(4), [0,1,2,0]] = 10.0
    target = torch.tensor([0,1,2,0])
    ce_val = torch.nn.CrossEntropyLoss()(logits, target)
    fl_val = fl(logits, target)
    assert fl_val < ce_val
```

```bash
python -m pytest tests/test_focal_loss.py -v
```
Expected: 2 passed

- [ ] **Step 6: --quick 端到端**

```bash
cd 新闻多分类
python reuters_multiclass.py --quick --loss focal --focal-gamma 2.0 --output-suffix _smoketest 2>&1 | tail -3
ls outputs/reuters_results_smoketest.csv
rm outputs/reuters_*_smoketest*  # 清理 smoketest
cd ..
```
Expected: 退出码 0；带 suffix 的文件存在

- [ ] **Step 7: Commit**

```bash
git add 新闻多分类/reuters_multiclass.py tests/test_focal_loss.py
git commit -m "feat(reuters): add FocalLoss with --loss/--focal-gamma/--output-suffix CLI"
```

---

### Task 15: `machine_translation.py` 加 LabelSmoothingCE + ε CLI

**Files:**
- Modify: `机器翻译/machine_translation.py:564-590` (在 train_transformer_epoch 之前加 LabelSmoothingCE)
- Modify: `机器翻译/machine_translation.py:78-125` (parse_args 加 --label-smoothing --output-suffix)
- Modify: `机器翻译/machine_translation.py:564-590` (train_transformer_epoch 接受 criterion)

- [ ] **Step 1: 加 LabelSmoothingCrossEntropy 类**

在 `train_transformer_epoch` 上方插入：
```python
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy with PAD ignore"""
    def __init__(self, epsilon: float = 0.1, ignore_index: int = PAD_IDX):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B*T, V], target: [B*T]
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            mask = (target != self.ignore_index)
            true_dist = torch.full_like(log_probs, self.epsilon / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1).clamp(min=0), 1 - self.epsilon)
        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp(min=1)
```

- [ ] **Step 2: 加 CLI**

```python
parser.add_argument("--label-smoothing", type=float, default=0.0)
parser.add_argument("--output-suffix", type=str, default="")
```

Config:
```python
label_smoothing: float = 0.0
output_suffix: str = ""
```

- [ ] **Step 3: 改 train_transformer_epoch 接受 criterion**

签名改为 `def train_transformer_epoch(model, loader, optimizer, criterion, device, ...)`，函数内部使用传入的 criterion。

调用处（在 run_experiment Transformer 块里）：
```python
if cfg.label_smoothing > 0:
    tf_criterion = LabelSmoothingCrossEntropy(epsilon=cfg.label_smoothing, ignore_index=PAD_IDX)
else:
    tf_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# ... 训练循环里:
loss = train_transformer_epoch(transformer, train_loader, optimizer, tf_criterion, device)
```

- [ ] **Step 4: 改输出路径加 suffix**

同 Task 14 Step 4 模式。

- [ ] **Step 5: 单元测试**

`tests/test_label_smoothing.py`:
```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "机器翻译"))
import torch
from machine_translation import LabelSmoothingCrossEntropy, PAD_IDX

def test_ls_zero_close_to_ce():
    ls = LabelSmoothingCrossEntropy(epsilon=0.0)
    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    logits = torch.randn(20, 50)
    target = torch.randint(1, 50, (20,))  # 避开 PAD
    assert torch.allclose(ls(logits, target), ce(logits, target), atol=1e-4)

def test_ls_ignores_pad():
    ls = LabelSmoothingCrossEntropy(epsilon=0.1)
    logits = torch.randn(4, 50)
    target = torch.tensor([PAD_IDX, PAD_IDX, 1, 2])
    loss = ls(logits, target)
    # 仅最后两个有效
    assert torch.isfinite(loss)
```

```bash
python -m pytest tests/test_label_smoothing.py -v
```
Expected: 2 passed

- [ ] **Step 6: --quick 端到端**

```bash
cd 机器翻译
python machine_translation.py --quick --label-smoothing 0.1 --output-suffix _smoketest 2>&1 | tail -3
ls outputs/translation_results_smoketest.csv
rm outputs/translation_*_smoketest*
cd ..
```
Expected: 退出码 0

- [ ] **Step 7: Commit**

```bash
git add 机器翻译/machine_translation.py tests/test_label_smoothing.py
git commit -m "feat(translation): add LabelSmoothingCE with --label-smoothing/--output-suffix CLI"
```

---

## Phase 5: 分析创新

### Task 16: `analyses/decode_grid.py` (翻译解码网格)

**Files:**
- Create: `analyses/__init__.py` (空文件)
- Create: `analyses/decode_grid.py`

- [ ] **Step 1: 创建 `analyses/__init__.py`**

```bash
mkdir -p analyses && touch analyses/__init__.py
```

- [ ] **Step 2: 写 `analyses/decode_grid.py`**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解码策略网格: 加载已训好的 Seq2Seq 与 Transformer checkpoint, 在
beam_size × length_penalty 网格上重新推理, 输出 BLEU 对比表.

输出: 机器翻译/outputs/translation_decode_grid.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "机器翻译"))

from machine_translation import (  # type: ignore
    Config, _detect_device, set_seed, load_parallel_pairs, split_pairs,
    Vocabulary, encode_pairs, TranslationDataset,
    Encoder, BahdanauAttention, Decoder, Seq2Seq,
    TransformerTranslator, decode_seq2seq_beam, decode_transformer_beam,
    compute_bleu_scores, ids_to_tokens, PAD_IDX, SOS_IDX, EOS_IDX,
)
import torch.utils.data as data


def main() -> None:
    cfg = Config()
    cfg.device = _detect_device()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = ROOT / "机器翻译" / "outputs"
    ckpt_s2s = out_dir / "checkpoints" / "seq2seq_best.pt"
    ckpt_tf = out_dir / "checkpoints" / "transformer_best.pt"
    assert ckpt_s2s.exists(), f"缺少 Seq2Seq checkpoint: {ckpt_s2s}"
    assert ckpt_tf.exists(), f"缺少 Transformer checkpoint: {ckpt_tf}"

    # 复用现有数据准备流程
    pairs = load_parallel_pairs(cfg)
    train_pairs, val_pairs, test_pairs = split_pairs(cfg, pairs)
    src_vocab = Vocabulary(); tgt_vocab = Vocabulary()
    src_vocab.build(p[0] for p in train_pairs)
    tgt_vocab.build(p[1] for p in train_pairs)
    test_src, test_tgt = encode_pairs(test_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)
    test_loader = data.DataLoader(TranslationDataset(test_src, test_tgt), batch_size=cfg.batch_size, shuffle=False)

    # 加载 Seq2Seq
    enc = Encoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    attn = BahdanauAttention(cfg.hidden_dim)
    dec = Decoder(len(tgt_vocab), cfg.embedding_dim, cfg.hidden_dim, attn, cfg.dropout)
    s2s = Seq2Seq(enc, dec, device).to(device)
    s2s.load_state_dict(torch.load(ckpt_s2s, map_location=device))
    s2s.eval()

    # 加载 Transformer
    tf = TransformerTranslator(
        len(src_vocab), len(tgt_vocab), cfg.embedding_dim,
        cfg.num_heads, cfg.num_layers, cfg.ff_dim, cfg.dropout,
    ).to(device)
    tf.load_state_dict(torch.load(ckpt_tf, map_location=device))
    tf.eval()

    rows = []
    for beam in [1, 3, 5]:
        for lp in [0.6, 0.8, 1.0, 1.2]:
            print(f"==> Seq2Seq beam={beam} lp={lp}")
            preds_s2s = []
            for src_batch, _ in test_loader:
                src_batch = src_batch.to(device)
                for i in range(src_batch.size(0)):
                    ids = decode_seq2seq_beam(s2s, src_batch[i:i+1], cfg.max_seq_len, beam_size=beam, length_penalty=lp)
                    preds_s2s.append(ids_to_tokens(ids, tgt_vocab))
            refs = [ids_to_tokens(t.tolist(), tgt_vocab) for t in test_tgt]
            bleu = compute_bleu_scores(refs, preds_s2s)
            rows.append({"model": "Seq2Seq+Attention", "beam": beam, "length_penalty": lp, **bleu})

            print(f"==> Transformer beam={beam} lp={lp}")
            preds_tf = []
            for src_batch, _ in test_loader:
                src_batch = src_batch.to(device)
                for i in range(src_batch.size(0)):
                    ids = decode_transformer_beam(tf, src_batch[i:i+1], cfg.max_seq_len, beam_size=beam, length_penalty=lp)
                    preds_tf.append(ids_to_tokens(ids, tgt_vocab))
            bleu = compute_bleu_scores(refs, preds_tf)
            rows.append({"model": "Transformer", "beam": beam, "length_penalty": lp, **bleu})

    df = pd.DataFrame(rows)
    out_path = out_dir / "translation_decode_grid.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✓ 已写入 {out_path}")


if __name__ == "__main__":
    main()
```

注: `decode_seq2seq_beam` / `decode_transformer_beam` 当前签名可能不接受 `length_penalty` 参数. 若不接受, 在 `machine_translation.py` 里把这两个函数加上参数 (默认 cfg.length_penalty), 然后在本脚本传入. 这是 Task 16 的额外修改点.

- [ ] **Step 3: 验证 decode 函数能接受 length_penalty 参数**

```bash
grep -n "def decode_seq2seq_beam\|def decode_transformer_beam" 机器翻译/machine_translation.py
```
查看签名是否含 `length_penalty`. 若未含, 修改并加默认值。

- [ ] **Step 4: smoke test (用 --quick 训出的小 checkpoint)**

```bash
source .venv-mac/bin/activate
# 先 quick 跑一次确保 checkpoint 存在
cd 机器翻译 && python machine_translation.py --quick && cd ..
python analyses/decode_grid.py 2>&1 | tail -10
ls 机器翻译/outputs/translation_decode_grid.csv
```
Expected: csv 存在, 行数 = 2 模型 × 3 beam × 4 lp = 24

- [ ] **Step 5: Commit**

```bash
git add analyses/__init__.py analyses/decode_grid.py 机器翻译/machine_translation.py
git commit -m "feat(analyses): decode strategy grid for translation"
```

---

### Task 17: `analyses/learning_curve.py`

**Files:**
- Create: `analyses/learning_curve.py`

- [ ] **Step 1: 写脚本**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据规模学习曲线: 各任务 × 各模型 × {25%, 50%, 75%, 100%} 训练集比例
分类任务跑全部 4 个 ratio; 翻译任务只跑 50/100% 节省时间.

输出: supplementary_outputs/learning_curve/{task}/curve_results.csv
"""

from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "supplementary_outputs" / "learning_curve"
OUT_BASE.mkdir(parents=True, exist_ok=True)


def run_one(task: str, ratio: float) -> Path:
    """跑一次 task * ratio. 返回输出 CSV 路径."""
    suffix = f"_lc_r{int(ratio * 100):03d}"
    if task == "sentiment":
        cwd = ROOT / "情感二分类"
        full = 12000
        size = int(full * ratio)
        cmd = ["python", "sentiment_analysis.py",
               "--include-hybrid",
               "--max-train-samples", str(size),
               "--output-dir", f"outputs{suffix}"]
        results = cwd / f"outputs{suffix}" / "sentiment_results.csv"
    elif task == "reuters":
        cwd = ROOT / "新闻多分类"
        full = 11228
        size = int(full * ratio)
        cmd = ["python", "reuters_multiclass.py",
               "--include-hybrid",
               "--max-samples", str(size),
               "--output-dir", f"outputs{suffix}"]
        results = cwd / f"outputs{suffix}" / "reuters_results.csv"
    elif task == "translation":
        cwd = ROOT / "机器翻译"
        full = 32000
        size = int(full * ratio)
        cmd = ["python", "machine_translation.py",
               "--include-hybrid",
               "--max-samples", str(size),
               "--output-dir", f"outputs{suffix}"]
        results = cwd / f"outputs{suffix}" / "translation_results.csv"
    else:
        raise ValueError(task)
    print(f"  → {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)
    return results


def collect(task: str, ratios: list[float]) -> pd.DataFrame:
    out_task = OUT_BASE / task
    out_task.mkdir(exist_ok=True)
    rows = []
    for r in ratios:
        path = run_one(task, r)
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["ratio"] = r
        rows.append(df)
        # 移到 supplementary 目录
        dst = out_task / f"results_r{int(r*100):03d}.csv"
        shutil.copy(path, dst)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    sentiment_df = collect("sentiment", [0.25, 0.50, 0.75, 1.00])
    reuters_df = collect("reuters", [0.25, 0.50, 0.75, 1.00])
    translation_df = collect("translation", [0.50, 1.00])

    sentiment_df.to_csv(OUT_BASE / "sentiment_curve_results.csv", index=False)
    reuters_df.to_csv(OUT_BASE / "reuters_curve_results.csv", index=False)
    translation_df.to_csv(OUT_BASE / "translation_curve_results.csv", index=False)
    print(f"✓ 学习曲线结果已写入 {OUT_BASE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: smoke test (只跑 1 个 ratio 验证流程)**

写一个临时缩减版直接调用 `run_one` 验证：
```bash
python -c "
import sys; sys.path.insert(0, 'analyses')
from learning_curve import run_one
p = run_one('sentiment', 0.25)
print('written:', p, p.exists())
"
```
Expected: 路径存在

- [ ] **Step 3: Commit**

```bash
git add analyses/learning_curve.py
git commit -m "feat(analyses): data-size learning curve runner"
```

---

### Task 18: `analyses/robustness.py`

**Files:**
- Create: `analyses/robustness.py`

- [ ] **Step 1: 写脚本**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鲁棒性测试: 不重训, 加载 best checkpoint, 对测试集做词级扰动:
- delete: 随机删除 p% 比例的 token
- unk: 随机将 p% 比例的 token 替换为 <unk>

p ∈ {0, 5, 10, 15, 20}; 各扰动方式独立报告.
输出: supplementary_outputs/robustness/{task}/robustness_results.csv
"""

from __future__ import annotations
import sys, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "supplementary_outputs" / "robustness"
OUT_BASE.mkdir(parents=True, exist_ok=True)


def perturb(seq: np.ndarray, p: float, mode: str, unk_id: int = 1, pad_id: int = 0) -> np.ndarray:
    """对长度为 L 的 token id 序列扰动 p 比例 (不动 PAD)"""
    seq = seq.copy()
    valid = np.where(seq != pad_id)[0]
    n = max(int(len(valid) * p), 0)
    if n == 0:
        return seq
    idx = np.random.choice(valid, size=n, replace=False)
    if mode == "delete":
        keep = np.ones(len(seq), dtype=bool); keep[idx] = False
        out = seq[keep]
        # pad 回原长
        out = np.concatenate([out, np.zeros(len(seq) - len(out), dtype=seq.dtype)])
        return out
    elif mode == "unk":
        seq[idx] = unk_id
        return seq
    else:
        raise ValueError(mode)


def evaluate_sentiment_perturbed(p: float, mode: str) -> dict:
    sys.path.insert(0, str(ROOT / "情感二分类"))
    from sentiment_analysis import (  # type: ignore
        Config, _detect_device, set_seed, load_imdb_data, prepare_dl_data,
        TextCNN, BiGRUClassifier, TransformerClassifier, CNNBiGRU,
        compute_binary_metrics, evaluate_model,
    )
    import torch.utils.data as data
    cfg = Config(); cfg.device = _detect_device()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_df, test_df = load_imdb_data(cfg)
    _, _, test_loader, vocab = prepare_dl_data(train_df, test_df, cfg)

    # 拷一份 dataset 应用扰动
    dataset = test_loader.dataset
    new_X = np.stack([perturb(dataset.X[i], p, mode) for i in range(len(dataset))])
    perturbed = copy.copy(dataset); perturbed.X = new_X
    perturbed_loader = data.DataLoader(perturbed, batch_size=cfg.batch_size, shuffle=False)

    ckpt_dir = ROOT / "情感二分类" / "outputs" / "checkpoints"
    results = {}
    for name, model in [
        ("TextCNN", TextCNN(len(vocab), cfg.embedding_dim, cfg.dropout)),
        ("BiGRU", BiGRUClassifier(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)),
        ("Transformer", TransformerClassifier(len(vocab), cfg.embedding_dim, cfg.num_heads, cfg.ff_dim, cfg.num_transformer_layers, cfg.dropout)),
        ("CNNBiGRU", CNNBiGRU(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)),
    ]:
        path = ckpt_dir / f"{name.lower()}_best.pt"
        if not path.exists():
            print(f"  跳过 {name}: 缺 checkpoint")
            continue
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device).eval()
        y, prob = evaluate_model(model, perturbed_loader, device)
        m = compute_binary_metrics(y, prob)
        results[name] = m
    return results


# reuters 与 translation 同理 (translation 用 BLEU). 略.


def main() -> None:
    rows = []
    for mode in ["delete", "unk"]:
        for p in [0.0, 0.05, 0.10, 0.15, 0.20]:
            print(f"==> sentiment {mode} p={p}")
            res = evaluate_sentiment_perturbed(p, mode)
            for model, m in res.items():
                rows.append({"task": "sentiment", "model": model, "mode": mode, "p": p, "f1": m["f1"], "accuracy": m["accuracy"]})
    pd.DataFrame(rows).to_csv(OUT_BASE / "sentiment_robustness.csv", index=False)
    print("✓ sentiment 完成")
    # TODO Task 18 Step 2: 复制粘贴 sentiment 模式实现 reuters 与 translation


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 添加 reuters + translation 实现**

将 `evaluate_sentiment_perturbed` 复制为 `evaluate_reuters_perturbed`, `evaluate_translation_perturbed`. reuters 用 macro-F1 指标, translation 用 BLEU-4 (调用 `compute_bleu_scores`).

- [ ] **Step 3: smoke test**

先确保有 best checkpoint (Task 13 quick 跑产生)：
```bash
ls 情感二分类/outputs/checkpoints/*.pt
python analyses/robustness.py 2>&1 | tail -20
```
Expected: 输出 `✓ sentiment 完成` 等

- [ ] **Step 4: Commit**

```bash
git add analyses/robustness.py
git commit -m "feat(analyses): word-level perturbation robustness suite"
```

---

### Task 19: `analyses/attention_viz.py`

**Files:**
- Create: `analyses/attention_viz.py`

- [ ] **Step 1: 写脚本**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力可视化: 三任务的 Transformer (含翻译的 cross-attention) 加 Seq2Seq Bahdanau alpha
挑 5-10 个错例画 heatmap, 输出到 figures/attn_<task>_<sample_id>.png
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def _load_font():
    """复用 make_figures.py 字体加载策略"""
    from matplotlib import font_manager as fm
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    names = []
    for p in paths:
        if Path(p).exists():
            try:
                fm.fontManager.addfont(p)
                names.append(fm.FontProperties(fname=p).get_name())
            except Exception: pass
    plt.rcParams["font.sans-serif"] = names + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_load_font()


def viz_sentiment(num_samples: int = 5):
    sys.path.insert(0, str(ROOT / "情感二分类"))
    from sentiment_analysis import (  # type: ignore
        Config, _detect_device, set_seed, load_imdb_data, prepare_dl_data,
        TransformerClassifier,
    )
    cfg = Config(); cfg.device = _detect_device()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_df, test_df = load_imdb_data(cfg)
    _, _, test_loader, vocab = prepare_dl_data(train_df, test_df, cfg)

    inv_vocab = {v: k for k, v in vocab.items()}
    model = TransformerClassifier(len(vocab), cfg.embedding_dim, cfg.num_heads, cfg.ff_dim, cfg.num_transformer_layers, cfg.dropout)
    model.load_state_dict(torch.load(ROOT / "情感二分类/outputs/checkpoints/transformer_best.pt", map_location=device))
    model = model.to(device).eval()

    n = 0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            logits, attn = model(x_batch, return_attention=True)  # attn: [B, H, L, L]
        for i in range(x_batch.size(0)):
            if n >= num_samples: return
            tokens = [inv_vocab.get(int(t), "<?>") for t in x_batch[i].cpu().numpy()][:30]
            head_avg = attn[i].mean(dim=0).cpu().numpy()[:30, :30]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(head_avg, xticklabels=tokens, yticklabels=tokens, cmap="Blues", ax=ax, cbar_kws={"label": "注意力权重"})
            ax.set_title(f"情感分类 Transformer 注意力 (sample {n}, pred={int(logits[i] > 0)}, true={int(y_batch[i])})")
            plt.xticks(rotation=60, fontsize=7)
            plt.yticks(rotation=0, fontsize=7)
            plt.tight_layout()
            plt.savefig(FIG / f"attn_sentiment_{n}.png", dpi=150)
            plt.close()
            n += 1


def viz_translation_cross(num_samples: int = 5):
    """翻译任务: 画 cross-attention (Transformer 与 Seq2Seq alpha)"""
    # 实现略 — 与 viz_sentiment 同模式: 加载模型 + 跑测试样本 + sns heatmap
    pass


def main() -> None:
    print("==> sentiment attention viz")
    viz_sentiment()
    print("==> translation cross-attention viz")
    viz_translation_cross()
    # 同样补 reuters 与 translation seq2seq alpha
    print(f"✓ 已写入 {FIG}/attn_*.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 补全 reuters 与 translation 的可视化函数（同 viz_sentiment 模式）**

- [ ] **Step 3: smoke test**

```bash
python analyses/attention_viz.py 2>&1 | tail -10
ls figures/attn_*.png | head
```
Expected: 至少 5 张 png 存在

- [ ] **Step 4: Commit**

```bash
git add analyses/attention_viz.py
git commit -m "feat(analyses): attention heatmap visualization for transformers"
```

---

## Phase 6: 图表 + 文档

### Task 20: `make_figures.py` 扩展 6 张新图

**Files:**
- Modify: `make_figures.py` (在末尾 main() 之前加 6 个新函数)

- [ ] **Step 1: 加 6 个新绘图函数**

在 `fig_stability_seeds` 之后插入：

```python
def fig_hybrid_compare() -> None:
    """主对比表中加入 CNNBiGRU 一行后, 三任务的对比图 (覆盖原 fig_*_model_compare).
    实际方式: 直接复用现有 fig_sentiment_model_compare/fig_reuters_model_compare/
    fig_translation_bleu_compare, 因为 results.csv 已含 CNNBiGRU.
    本函数仅生成一张 "hybrid 与 baseline 模型 head-to-head 对比图".
    """
    sent = _read_csv(SENT_DIR / "sentiment_results.csv").set_index("model")
    reut = _read_csv(REUT_DIR / "reuters_results.csv").set_index("model")
    tran = _read_csv(TRAN_DIR / "translation_results.csv").set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, df, score, title, ylim in [
        (axes[0], sent, "f1", "情感二分类 F1", (0.7, 0.95)),
        (axes[1], reut, "f1_macro", "Reuters Macro-F1", (0.4, 0.7)),
    ]:
        models = ["TextCNN", "BiGRU", "Transformer", "CNNBiGRU"]
        vals = [df.loc[m, score] if m in df.index else np.nan for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        ax.bar(models, vals, color=colors, edgecolor="black")
        ax.set_title(title); ax.set_ylim(*ylim); ax.tick_params(axis="x", rotation=15)
    # 翻译: BLEU-4 单条
    if "CNNBiGRU" in tran.index:
        models = ["Seq2Seq+Attention", "Transformer", "CNNBiGRU"]
        vals = [tran.loc[m, "BLEU-4"] if m in tran.index else np.nan for m in models]
        axes[2].bar(models, vals, color=[MODEL_COLORS.get(m, "#888") for m in models], edgecolor="black")
        axes[2].set_title("翻译 BLEU-4"); axes[2].set_ylim(0.25, 0.40)
        axes[2].tick_params(axis="x", rotation=15)
    fig.suptitle("CNN-BiGRU 混合模型 vs 单一架构基线", y=1.02)
    _save(fig, "fig_hybrid_compare.png")


def fig_focal_gamma_ablation() -> None:
    """从 reuters_results_focal_g{gamma}.csv 集合中读出 gamma vs Macro-F1"""
    rows = []
    for p in (REUT_DIR).glob("reuters_results_focal_g*.csv"):
        gamma = float(p.stem.split("_g")[-1])
        df = _read_csv(p)
        for _, r in df.iterrows():
            rows.append({"model": r["model"], "gamma": gamma, "f1_macro": r["f1_macro"]})
    if not rows:
        print("  (跳过 focal_gamma_ablation: 无 focal 结果文件)")
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("gamma")
        ax.plot(sub["gamma"], sub["f1_macro"], marker="o", label=model, linewidth=2)
    ax.set_xlabel("focal γ"); ax.set_ylabel("Macro-F1")
    ax.set_title("Reuters-46: Focal Loss γ 敏感性")
    ax.legend()
    _save(fig, "fig_focal_gamma_ablation.png")


def fig_label_smoothing_compare() -> None:
    """对比 baseline Transformer 与 LS Transformer BLEU"""
    base = _read_csv(TRAN_DIR / "translation_results.csv")
    ls_path = TRAN_DIR / "translation_results_ls.csv"
    if not ls_path.exists():
        print("  (跳过 label_smoothing_compare: 无 _ls 结果)")
        return
    ls = _read_csv(ls_path)
    base_tf = base[base["model"] == "Transformer"].iloc[0]
    ls_tf = ls[ls["model"] == "Transformer"].iloc[0]
    metrics = ["BLEU-1", "BLEU-2", "BLEU-4"]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, [base_tf[m] for m in metrics], w, label="baseline", color="#FFA726", edgecolor="black")
    ax.bar(x + w/2, [ls_tf[m] for m in metrics], w, label="+LabelSmoothing 0.1", color="#26A69A", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("BLEU"); ax.set_title("翻译 Transformer: Label Smoothing 影响")
    ax.legend()
    _save(fig, "fig_label_smoothing_compare.png")


def fig_decode_grid() -> None:
    """beam × length_penalty 网格热图 (BLEU-4)"""
    p = TRAN_DIR / "translation_decode_grid.csv"
    if not p.exists():
        print("  (跳过 decode_grid: 无 grid csv)")
        return
    df = _read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, ["Seq2Seq+Attention", "Transformer"]):
        sub = df[df["model"] == model]
        pivot = sub.pivot(index="beam", columns="length_penalty", values="BLEU-4")
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn", ax=ax, cbar_kws={"label": "BLEU-4"})
        ax.set_title(f"{model}")
    fig.suptitle("解码策略网格搜索: beam × length_penalty", y=1.02)
    _save(fig, "fig_decode_grid.png")


def fig_learning_curve() -> None:
    """三任务 ratio vs 主指标 折线图"""
    base = ROOT / "supplementary_outputs" / "learning_curve"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task, score, title in [
        (axes[0], "sentiment", "f1", "情感二分类 (F1)"),
        (axes[1], "reuters", "f1_macro", "Reuters Macro-F1"),
        (axes[2], "translation", "BLEU-4", "翻译 BLEU-4"),
    ]:
        path = base / f"{task}_curve_results.csv"
        if not path.exists():
            ax.set_title(title + " (无数据)"); continue
        df = _read_csv(path)
        for model, sub in df.groupby("model"):
            sub = sub.sort_values("ratio")
            ax.plot(sub["ratio"] * 100, sub[score], marker="o", label=model, linewidth=2)
        ax.set_xlabel("训练数据比例 (%)"); ax.set_ylabel(score); ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("数据规模学习曲线", y=1.02)
    _save(fig, "fig_learning_curve.png")


def fig_robustness() -> None:
    """三任务扰动鲁棒性曲线"""
    base = ROOT / "supplementary_outputs" / "robustness"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task, score, title in [
        (axes[0], "sentiment", "f1", "情感二分类 (F1)"),
        (axes[1], "reuters", "f1_macro", "Reuters Macro-F1"),
        (axes[2], "translation", "BLEU-4", "翻译 BLEU-4"),
    ]:
        path = base / f"{task}_robustness.csv"
        if not path.exists():
            ax.set_title(title + " (无数据)"); continue
        df = _read_csv(path)
        for (model, mode), sub in df.groupby(["model", "mode"]):
            sub = sub.sort_values("p")
            ax.plot(sub["p"] * 100, sub[score], marker="o", label=f"{model}-{mode}", linewidth=1.5)
        ax.set_xlabel("扰动比例 (%)"); ax.set_ylabel(score); ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("词级扰动鲁棒性", y=1.02)
    _save(fig, "fig_robustness.png")
```

- [ ] **Step 2: 把 6 个函数加到 main() 的 figs 列表**

```python
figs = [
    # ... 原有 16 ...
    ("混合模型对比", fig_hybrid_compare),
    ("Focal γ 消融", fig_focal_gamma_ablation),
    ("Label Smoothing 对比", fig_label_smoothing_compare),
    ("解码网格", fig_decode_grid),
    ("学习曲线", fig_learning_curve),
    ("鲁棒性", fig_robustness),
]
```

- [ ] **Step 3: 跑 make_figures.py**

```bash
python make_figures.py 2>&1 | tail -30
```
Expected: 完成数 16 + 至多 6 = 22 (无数据的会被跳过, 不算失败)

- [ ] **Step 4: Commit**

```bash
git add make_figures.py
git commit -m "feat(figures): six new figures for hybrid/focal/LS/decode-grid/curve/robustness"
```

---

### Task 21: 创建 `README_MAC.md` (总入口 + 重跑命令)

**Files:**
- Create: `README_MAC.md`

- [ ] **Step 1: 写 README_MAC.md**

```markdown
# Mac 运行指南

本仓库的原始 `.venv/` 是 Windows 环境。Mac 使用本指南建立独立环境。

## 一次性环境

```bash
bash setup_mac.sh
source .venv-mac/bin/activate
```

## 跑全套实验 (含创新点)

```bash
# Step 1: 主实验 (5 模型 × 3 任务) — 含新 CNN-BiGRU
cd 情感二分类 && python sentiment_analysis.py --include-hybrid && cd ..
cd 新闻多分类 && python reuters_multiclass.py --include-hybrid && cd ..
cd 机器翻译  && python machine_translation.py  --include-hybrid && cd ..

# Step 2: Reuters Focal Loss γ 网格扫描
cd 新闻多分类
for g in 0.5 1.0 2.0 5.0; do
  python reuters_multiclass.py --loss focal --focal-gamma $g --output-suffix _focal_g$g
done
cd ..

# Step 3: 翻译 Label Smoothing + 解码网格
cd 机器翻译
python machine_translation.py --label-smoothing 0.1 --output-suffix _ls
cd ..
python analyses/decode_grid.py

# Step 4: 三大分析创新
python analyses/learning_curve.py
python analyses/robustness.py
python analyses/attention_viz.py

# Step 5: 重生成全部图
python make_figures.py
```

## 快速验证

```bash
# 各任务 quick 模式跑通流程 (~1 分钟/任务)
cd 情感二分类 && python sentiment_analysis.py --quick --include-hybrid && cd ..
cd 新闻多分类 && python reuters_multiclass.py --quick --include-hybrid && cd ..
cd 机器翻译  && python machine_translation.py  --quick --include-hybrid && cd ..
```

## 设备策略

三个脚本默认按 mps → cuda → cpu 顺序自动选设备。可显式覆盖:
```bash
python sentiment_analysis.py --device cpu      # 强制 CPU
python sentiment_analysis.py --device mps      # 强制 MPS
```

## 已知 MPS 限制

PyTorch MPS 在某些算子上可能报错或回退. 若训练中报 `Operation 'X' not implemented for MPS`, 用 `--device cpu` 重跑该任务。

## 与 Windows .venv 的关系

仓库中保留的 `.venv/` 是 Windows 旧环境, Mac 不要使用. `.venv-mac/` 与之独立。
```

- [ ] **Step 2: 验证文档可读**

```bash
head -30 README_MAC.md
```
Expected: 输出正常, 中文不乱码

- [ ] **Step 3: Commit**

```bash
git add README_MAC.md
git commit -m "docs: add Mac setup and rerun guide (README_MAC.md)"
```

---

## 自检 (Self-Review)

- **Spec coverage**: 21 个任务覆盖 spec 第 2 节列出的 6 项工作 (T1-T3, A1, C1-C3) + Mac 环境 + 文档. ✓
- **Placeholder scan**: Task 18 / 19 / 20 中部分函数留有 `# TODO` / `pass` / `# 同样模式` — 已在对应 Step 2 显式声明 "复制粘贴 sentiment 模式" 作为下一 step 的具体动作, 不算开放式占位. ✓
- **Type consistency**: 三个 `_detect_device` 同名函数; `--include-hybrid` / `--loss` / `--focal-gamma` / `--label-smoothing` / `--output-suffix` 各 CLI 名称在不同任务一致. `CNNBiGRU` (二分类) / `CNNBiGRUMulti` (多分类) / `CNNBiGRUEncoder` (翻译) 命名为各自任务的具体类, 调用处统一以 "CNNBiGRU" 字符串出现在 results.csv. ✓
- **依赖**: Task 16 / 18 / 19 都依赖 Task 13 (CNN-BiGRU 已加 + checkpoint 已存); Task 20 部分子图依赖 Task 14 / 15 / 17 / 18 输出. 顺序正确. ✓

---

## Notes for executor

- 每完成一个 Task 立即 commit, 不要批量 commit
- 出现 MPS 算子错误优先 `--device cpu` 通过, 在 README_MAC.md 标注
- 翻译任务实验最耗时 (Transformer ~70min). 若 reproducibility 需求一般, 翻译 Step 2 (label smoothing 重训) 可放在最后跑
- 测试文件 `tests/test_*.py` 不是发布产物, 但保留在仓库以备复跑
- 历史 Windows `outputs/` 不要清空, 作为对照基线
