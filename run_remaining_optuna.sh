#!/usr/bin/env bash
# 等当前 Reuters Optuna 跑完 → 跑 Sentiment Optuna → 跑 Translation Optuna
# 串行执行避免 MPS 设备争抢

set -euo pipefail
cd "$(dirname "$0")"
source .venv-mac/bin/activate

LOG_DIR="run_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
MAIN="$LOG_DIR/optuna_remaining_$TS.log"

stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

# ---- Step 0: 等 Reuters Optuna 跑完 ----
REUTERS_PID=62698
if kill -0 "$REUTERS_PID" 2>/dev/null; then
  stamp "===== 等待 Reuters Optuna (PID=$REUTERS_PID) 跑完 ====="
  while kill -0 "$REUTERS_PID" 2>/dev/null; do
    sleep 30
  done
  stamp "✓ Reuters Optuna 已结束"
else
  stamp "===== Reuters Optuna 已不在跑,直接进入下一步 ====="
fi

# ---- Step 1: Sentiment Optuna (30 trials, ~60 min) ----
stamp "===== Step 1: Sentiment Optuna (30 trials) ====="
python analyses/optuna_search_sentiment.py --n-trials 30 --epochs 12 --patience 4 \
  2>&1 | tee "$LOG_DIR/optuna_sentiment_$TS.log" >> "$MAIN"
stamp "✓ Sentiment Optuna 完成"

# ---- Step 2: Translation Optuna (15 trials × 8 epochs, ~4-5h) ----
stamp "===== Step 2: Translation Optuna (15 trials × 8 epochs) ====="
python analyses/optuna_search_translation.py --n-trials 15 --epochs 8 --patience 3 \
  --val-sample 200 --test-sample 600 \
  2>&1 | tee "$LOG_DIR/optuna_translation_$TS.log" >> "$MAIN"
stamp "✓ Translation Optuna 完成"

stamp "===== 🎉 全部 Optuna 完成 ====="
