#!/usr/bin/env bash
# 一键完整跑全套创新实验 (Mac MPS)
# 估计耗时 6-10 小时, 建议 nohup 后台跑.
# 任意阶段失败会立即停 (set -e), 通过 step 标记可定位.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 激活 venv
source .venv-mac/bin/activate

LOG_DIR="$ROOT/run_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
MAIN="$LOG_DIR/full_run_$TS.log"

stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

stamp "===== 开始全套实验 ====="
stamp "本机: $(uname -srm)  PyTorch: $(python -c 'import torch;print(torch.__version__)')  MPS: $(python -c 'import torch;print(torch.backends.mps.is_available())')"

# -------- Step 1: 主实验 (5 模型 × 3 任务) --------
stamp "===== Step 1.1: sentiment 主实验 (5 模型) ====="
cd "$ROOT/情感二分类"
python sentiment_analysis.py --include-hybrid 2>&1 | tee -a "$LOG_DIR/01_sentiment_$TS.log" >> "$MAIN"
stamp "✓ sentiment 主实验完成"

stamp "===== Step 1.2: reuters 主实验 (5 模型) ====="
cd "$ROOT/新闻多分类"
python reuters_multiclass.py --include-hybrid 2>&1 | tee -a "$LOG_DIR/02_reuters_$TS.log" >> "$MAIN"
stamp "✓ reuters 主实验完成"

stamp "===== Step 1.3: translation 主实验 (Seq2Seq + Transformer + CNNBiGRU) ====="
cd "$ROOT/机器翻译"
python machine_translation.py --include-hybrid 2>&1 | tee -a "$LOG_DIR/03_translation_$TS.log" >> "$MAIN"
stamp "✓ translation 主实验完成"

# -------- Step 2: Reuters Focal Loss γ 网格 --------
stamp "===== Step 2: Reuters Focal Loss γ 网格 ====="
cd "$ROOT/新闻多分类"
for g in 0.5 1.0 2.0 5.0; do
  stamp "  → γ=$g"
  python reuters_multiclass.py --loss focal --focal-gamma "$g" --output-suffix "_focal_g$g" 2>&1 \
    | tee -a "$LOG_DIR/04_focal_g${g}_$TS.log" >> "$MAIN"
done
stamp "✓ Reuters γ 扫描完成 (4 组)"

# -------- Step 3: 翻译 Label Smoothing + 解码网格 --------
stamp "===== Step 3.1: 翻译 Label Smoothing 重训 Transformer ====="
cd "$ROOT/机器翻译"
python machine_translation.py --label-smoothing 0.1 --output-suffix _ls 2>&1 \
  | tee -a "$LOG_DIR/05_ls_$TS.log" >> "$MAIN"
stamp "✓ Label smoothing 重训完成"

stamp "===== Step 3.2: 解码策略网格 (beam × length_penalty) ====="
cd "$ROOT"
python analyses/decode_grid.py 2>&1 | tee -a "$LOG_DIR/06_decode_grid_$TS.log" >> "$MAIN"
stamp "✓ 解码网格完成"

# -------- Step 4: 三大分析创新 --------
stamp "===== Step 4.1: 数据规模学习曲线 ====="
python analyses/learning_curve.py 2>&1 | tee -a "$LOG_DIR/07_learning_curve_$TS.log" >> "$MAIN"
stamp "✓ 学习曲线完成"

stamp "===== Step 4.2: 词级扰动鲁棒性 ====="
python analyses/robustness.py 2>&1 | tee -a "$LOG_DIR/08_robustness_$TS.log" >> "$MAIN"
stamp "✓ 鲁棒性完成"

stamp "===== Step 4.3: 注意力可视化 ====="
python analyses/attention_viz.py 2>&1 | tee -a "$LOG_DIR/09_attention_viz_$TS.log" >> "$MAIN"
stamp "✓ 注意力可视化完成"

# -------- Step 5: 后处理 (汇总 / 审计 / 出图) --------
stamp "===== Step 5.1: summarize_final_results ====="
python summarize_final_results.py 2>&1 | tee -a "$MAIN" || stamp "  (警告: summarize 失败, 不阻塞)"

stamp "===== Step 5.2: audit_outputs ====="
python audit_outputs.py 2>&1 | tee -a "$MAIN" || stamp "  (警告: audit 失败, 不阻塞)"

stamp "===== Step 5.3: make_figures (22 张) ====="
python make_figures.py 2>&1 | tee -a "$LOG_DIR/10_make_figures_$TS.log" >> "$MAIN"
stamp "✓ 图表生成完成"

stamp "===== 🎉 全套实验完成 ====="
stamp "总主日志: $MAIN"
stamp "各阶段子日志: $LOG_DIR/*_$TS.log"
