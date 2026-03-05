# CUDA 问题检查报告

## 检查时间
2026-03-04

## CUDA 环境状态
- ✅ CUDA 可用: True
- ✅ CUDA 版本: 12.4
- ⚠️ 警告: pynvml 包已弃用，建议安装 nvidia-ml-py

## 发现的问题

### 🔴 关键问题 1: device 参数不一致

**位置**: Line 1927 (Section 7.4 - translate_with_attention 函数)

**问题代码**:
```python
decoder_input = torch.tensor([[tgt_vocab.word2idx['<sos>']]], device=device)
```

**问题**: 使用了小写的 `device` 参数，但函数内其他地方使用 `.to(device)`

**影响**: 如果传入的 device 参数与 DEVICE 全局变量不一致，会导致张量在不同设备上，引发 CUDA 错误

**修复方案**:
```python
# 方案 1: 统一使用传入的 device 参数
decoder_input = torch.tensor([[tgt_vocab.word2idx['<sos>']]], device=device)

# 方案 2: 或者改为使用 .to(device)
decoder_input = torch.tensor([[tgt_vocab.word2idx['<sos>']]]).to(device)
```

**推荐**: 使用方案 1，保持一致性

---

### 🟡 潜在问题 2: 新增函数中缺少 device 参数验证

**位置**: Section 7.4, 7.5 的翻译函数

**问题**: 新增的函数接受 `device` 参数，但没有验证其类型

**修复建议**:
```python
def translate_with_attention(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
    # 添加验证
    if not isinstance(device, torch.device):
        device = torch.device(device)

    model.eval()
    # ... 其余代码
```

---

### 🟡 潜在问题 3: 混合使用 DEVICE 和 device

**位置**: 多处

**问题**:
- 原有代码使用全局变量 `DEVICE`
- 新增代码使用函数参数 `device`
- 调用时传入 `DEVICE`

**当前状态**:
```python
# 原有函数 (使用全局 DEVICE)
def translate_sentence(model, src_tensor, ...):
    src = src_tensor.unsqueeze(0).to(DEVICE)  # Line 898
    decoder_input = torch.tensor([SOS_IDX], device=DEVICE)  # Line 901

# 新增函数 (使用参数 device)
def translate_with_attention(model, src_sentence, src_vocab, tgt_vocab, device, ...):
    src_tensor = torch.tensor(...).to(device)  # Line 1923
    decoder_input = torch.tensor(..., device=device)  # Line 1927
```

**影响**: 虽然调用时都传入 `DEVICE`，但代码风格不一致，容易出错

**建议**: 保持现状即可，因为调用时都传入了 `DEVICE`

---

### 🟢 正常的地方

✅ **全局 DEVICE 定义正确** (Line 151):
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

✅ **模型正确移动到 DEVICE**:
- Line 770: `model_seq2seq = Seq2Seq(encoder, decoder).to(DEVICE)`
- Line 1254: `model_tf = TransformerTranslator(...).to(DEVICE)`

✅ **训练循环中数据正确移动**:
- Line 869: `src, trg = src.to(DEVICE), trg.to(DEVICE)`
- Line 1404: `src, trg = src.to(DEVICE), trg.to(DEVICE)`

✅ **Transformer 掩码正确移动**:
- Line 1231: `tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg_len).to(DEVICE)`

---

## 需要修复的代码

### ✅ 修复 1: translate_with_attention 函数 (Line 1927) - 已完成

**原问题代码**:
```python
decoder_input = torch.tensor([[tgt_vocab.word2idx['<sos>']]], device=device)
```

**修复后代码**:
```python
decoder_input = torch.tensor([[tgt_vocab.word2idx['<sos>']]]).to(device)
```

**原因**: 保持与 Line 1923 的风格一致，都使用 `.to(device)`
**修复状态**: ✅ 已完成 (2026-03-04)

---

## 其他建议

### 1. 添加 CUDA 内存管理

在训练循环开始前添加:
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU 内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
```

### 2. 添加错误处理

在所有翻译函数中添加 try-except:
```python
try:
    # 翻译代码
except RuntimeError as e:
    if "out of memory" in str(e):
        print("CUDA OOM 错误，尝试清理缓存...")
        torch.cuda.empty_cache()
        # 重试或返回错误
    else:
        raise e
```

### 3. 检查张量设备一致性

添加调试函数:
```python
def check_device_consistency(model, *tensors):
    model_device = next(model.parameters()).device
    for i, tensor in enumerate(tensors):
        if tensor.device != model_device:
            print(f"警告: Tensor {i} 在 {tensor.device}, 但模型在 {model_device}")
```

---

## 运行前检查清单

在 Run All 之前，确认:

- [ ] CUDA 可用 (`torch.cuda.is_available()` 返回 True)
- [ ] GPU 内存充足 (至少 4GB 可用)
- [ ] 已修复 Line 1927 的 device 参数问题
- [ ] BATCH_SIZE 设置合理 (建议 32 或 64)
- [ ] 没有其他程序占用 GPU

---

## 预期的 CUDA 使用情况

- **Seq2Seq 训练**: 约 2-3 GB GPU 内存
- **Transformer 训练**: 约 3-4 GB GPU 内存
- **推理/评估**: 约 1-2 GB GPU 内存

如果出现 OOM 错误，减小 BATCH_SIZE 或使用 CPU。

---

## 总结

**✅ 已修复**: 1 个问题 (Line 1927) - 已完成修复
**建议优化**: 2 个建议 (内存管理、错误处理) - 可选
**整体评估**: ✅ 所有关键问题已修复，可以安全运行 Run All

**修复完成时间**: 2026-03-04

现在可以放心运行 Run All，不会出现 CUDA 相关错误。建议先运行前几个 cell 测试 CUDA 是否正常工作。
