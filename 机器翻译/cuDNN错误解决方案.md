# cuDNN 错误解决方案

## 错误信息
```
RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
```

## 原因分析

1. **GPU 内存碎片化** - 多次训练后 GPU 显存管理出现问题
2. **cuDNN 版本不兼容** - PyTorch 与 cuDNN 版本不匹配
3. **数值不稳定** - 梯度爆炸或 NaN 值导致 cuDNN 内部错误
4. **批次大小过大** - 超出 GPU 内存限制

## 解决方案（按优先级）

### 方案 1: 重启 Jupyter Kernel（最简单）
1. 点击 Jupyter 菜单：`Kernel` → `Restart Kernel`
2. 重新运行所有单元格（从头开始）

### 方案 2: 配置 cuDNN（已自动添加）
我已经在 notebook 中添加了诊断单元格，运行它会自动配置：
```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()
```

### 方案 3: 减小批次大小
在超参数配置单元格中修改：
```python
BATCH_SIZE = 32  # 改为 16 或 8
```

### 方案 4: 使用 CPU 训练
在超参数配置单元格中修改：
```python
DEVICE = torch.device('cpu')  # 强制使用 CPU
```

### 方案 5: 更新 PyTorch（如果上述方案无效）
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 已添加的修复

1. **GPU 诊断单元格** - 在 5.2 训练单元格之前
2. **NaN 检测** - 在训练循环中自动跳过异常批次
3. **GPU 缓存清理** - 每个 epoch 开始前清理

## 使用建议

1. 先运行新添加的 GPU 诊断单元格
2. 查看 GPU 内存使用情况
3. 如果内存不足，减小 BATCH_SIZE
4. 如果问题持续，使用 CPU 训练（速度较慢但稳定）
