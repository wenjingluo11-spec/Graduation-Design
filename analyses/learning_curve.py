#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据规模学习曲线 (T17):

对每个任务,在不同 data ratio 下重训所有模型 (含 CNNBiGRU) 一次,
收集各 ratio 的 *_results.csv 并合并成 curve_results.csv.

- sentiment / reuters: ratios = [0.25, 0.50, 0.75, 1.00]
- translation: ratios = [0.50, 1.00] (训练昂贵,只两个点)

输出:
  supplementary_outputs/learning_curve/sentiment_curve_results.csv
  supplementary_outputs/learning_curve/reuters_curve_results.csv
  supplementary_outputs/learning_curve/translation_curve_results.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "supplementary_outputs" / "learning_curve"

TASK_CONFIG = {
    "sentiment": {
        "cwd": ROOT / "情感二分类",
        "script": "sentiment_analysis.py",
        "flag": "--max-train-samples",
        "full_size": 12000,
        "results_csv": "sentiment_results.csv",
    },
    "reuters": {
        "cwd": ROOT / "新闻多分类",
        "script": "reuters_multiclass.py",
        "flag": "--max-samples",
        "full_size": 11228,
        "results_csv": "reuters_results.csv",
    },
    "translation": {
        "cwd": ROOT / "机器翻译",
        "script": "machine_translation.py",
        "flag": "--max-samples",
        "full_size": 32000,
        "results_csv": "translation_results.csv",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="sentiment,reuters,translation", help="逗号分隔的任务名")
    p.add_argument(
        "--ratios-classification", default="0.25,0.50,0.75,1.00",
        help="分类任务的训练比例 (sentiment / reuters)",
    )
    p.add_argument(
        "--ratios-translation", default="0.50,1.00",
        help="翻译任务的训练比例",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="所有子调用都附加 --quick (用于 smoke,跑得快但指标不可信)",
    )
    p.add_argument(
        "--include-hybrid", action="store_true", default=True,
        help="子调用启用 CNN-BiGRU (默认 True)",
    )
    return p.parse_args()


def run_one(task: str, ratio: float, quick: bool, include_hybrid: bool) -> Path:
    """跑一次 (task, ratio).返回该次输出的 results.csv 路径."""
    cfg = TASK_CONFIG[task]
    size = max(int(cfg["full_size"] * ratio), 1)
    out_dir_name = f"outputs_lc_r{int(ratio * 100):03d}"
    out_dir = cfg["cwd"] / out_dir_name

    cmd: list[str] = [
        sys.executable, cfg["script"],
        cfg["flag"], str(size),
        "--output-dir", out_dir_name,
    ]
    if include_hybrid:
        cmd.append("--include-hybrid")
    if quick:
        cmd.append("--quick")

    print(f"[lc] {task} ratio={ratio} size={size} -> {out_dir_name}")
    subprocess.check_call(cmd, cwd=cfg["cwd"])
    results_path = out_dir / cfg["results_csv"]
    if not results_path.exists():
        raise FileNotFoundError(f"子调用未产出 {results_path}")
    return results_path


def collect(task: str, ratios: list[float], quick: bool, include_hybrid: bool) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for r in ratios:
        path = run_one(task, r, quick, include_hybrid)
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["ratio"] = r
        df["task"] = task
        rows.append(df)
        # 复制到 supplementary 目录归档
        dst = OUT_BASE / task / f"results_r{int(r * 100):03d}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dst)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    cls_ratios = [float(x) for x in args.ratios_classification.split(",")]
    tr_ratios = [float(x) for x in args.ratios_translation.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    for task in tasks:
        ratios = tr_ratios if task == "translation" else cls_ratios
        df = collect(task, ratios, args.quick, args.include_hybrid)
        out_csv = OUT_BASE / f"{task}_curve_results.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✓ {task} 曲线已写入 {out_csv} ({len(df)} 行)")


if __name__ == "__main__":
    main()
