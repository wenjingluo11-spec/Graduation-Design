# Mac 运行指南

本仓库的原始 `.venv/` 是 Windows 环境。Mac 使用本指南建立独立环境。

## 一次性环境初始化

```bash
bash setup_mac.sh
source .venv-mac/bin/activate
```

`setup_mac.sh` 会:
- 创建 `.venv-mac/` 虚拟环境
- 安装 `requirements.txt` 中的所有依赖（torch / pandas / numpy / matplotlib / seaborn / scikit-learn / nltk / jupyter）
- 下载 NLTK punkt

## 跑全套实验（含创新点）

```bash
source .venv-mac/bin/activate

# Step 1: 主实验（5 模型 × 3 任务）— 含 CNN-BiGRU 混合架构
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

# Step 5: 重生成全部图（共 22 张）
python make_figures.py
```

## 快速验证（≈1 分钟/任务）

```bash
source .venv-mac/bin/activate
cd 情感二分类 && python sentiment_analysis.py --quick --include-hybrid --output-dir outputs_smoke && cd ..
cd 新闻多分类 && python reuters_multiclass.py --quick --include-hybrid --output-dir outputs_smoke && cd ..
cd 机器翻译  && python machine_translation.py  --quick --include-hybrid --output-dir outputs_smoke && cd ..
```

`--output-dir outputs_smoke` 把 quick 模式输出隔离到 `outputs_smoke/`，避免覆盖 `outputs/` 里的 baseline。

## 设备策略

三个脚本默认按 `mps → cuda → cpu` 顺序自动选设备。可显式覆盖：

```bash
python sentiment_analysis.py --device cpu      # 强制 CPU
python sentiment_analysis.py --device mps      # 强制 MPS
python sentiment_analysis.py --device cuda     # 强制 CUDA
```

## 已知 MPS 限制

Apple Silicon MPS 后端某些算子（如 `_nested_tensor_from_mask_left_aligned`）尚未实现。
三个脚本顶部已设置：

```python
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

让 PyTorch 在遇到不支持的算子时自动 fallback 到 CPU。**这一行必须在 `import torch` 之前**，否则 `nn.TransformerEncoder` 在 `__init__` 时已锁定 nested-tensor 路径，运行期还是会报错。新增 Transformer 系模型或调整 import 顺序时务必保留。

若仍出错，用 `--device cpu` 重跑该任务即可。

## 测试

```bash
source .venv-mac/bin/activate
pytest tests/                                                  # 全部
pytest tests/test_focal_loss.py -v                             # 单文件
pytest tests/test_focal_loss.py::test_focal_loss_gamma_zero_equals_ce  # 单测
```

`tests/` 通过把任务子目录注入 `sys.path` 直接 import 主脚本中的类（`FocalLoss`、`LabelSmoothingCrossEntropy`、`CNNBiGRU` 等），所以重命名时必须同步更新测试。

## 与 Windows .venv 的关系

仓库中保留的 `.venv/` 是 Windows 旧环境，Mac 不要使用。`.venv-mac/` 与之独立，互不干扰。`.gitignore` 忽略两者。

## 创新点速览

本仓库相对开题报告新增的内容（见 `docs/superpowers/specs/2026-04-29-thesis-innovation-design.md`）：

- **架构创新**：CNN-BiGRU 混合模型 (`--include-hybrid`)，三任务统一对比
- **训练改进**：
  - Reuters 引入 Focal Loss (`--loss focal --focal-gamma γ`)，γ 网格扫描
  - 翻译引入 Label Smoothing (`--label-smoothing 0.1`)
- **解码消融**：`analyses/decode_grid.py`，beam × length_penalty 网格搜索
- **数据效率**：`analyses/learning_curve.py`，data ratio 25/50/75/100% 学习曲线
- **鲁棒性**：`analyses/robustness.py`，词级 delete / unk 扰动 0–20%
- **注意力可视化**：`analyses/attention_viz.py`，Transformer self/cross-attn + Seq2Seq Bahdanau α 热图

## 常见问题

**Q: pytest 报错说找不到模块？**
A: 测试通过 `sys.path.insert` 把任务子目录加进 PYTHONPATH。确保从仓库根目录运行 pytest。

**Q: 中文图表显示方块？**
A: macOS 已通过 `Arial Unicode MS` 等字体路径加载。如仍乱码，检查 `make_figures.py` 顶部的 `_FONT_CANDIDATE_PATHS` 是否在你的系统中存在对应路径。

**Q: 训练很慢？**
A: 默认是 MPS。Apple Silicon 上 nn.GRU 系模型相对较慢（已知 PyTorch MPS 后端的 RNN 实现有 overhead）；如有 CUDA 机器优先用 CUDA。
