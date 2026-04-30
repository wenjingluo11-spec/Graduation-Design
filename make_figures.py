#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figures.py — 论文图表一键生成脚本

只读取仓库中现有的 CSV 输出，绘制论文/答辩所需的全部对比图。
不需要重跑实验。每张图的依赖都在函数 docstring 中标注。

输出目录: figures/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SENT_DIR = ROOT / "情感二分类" / "outputs"
REUT_DIR = ROOT / "新闻多分类" / "outputs"
TRAN_DIR = ROOT / "机器翻译" / "outputs"
STAB_DIR = ROOT / "supplementary_outputs" / "stability"

from matplotlib import font_manager as _fm

_FONT_CANDIDATE_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_LOADED_NAMES: List[str] = []
for _p in _FONT_CANDIDATE_PATHS:
    if os.path.exists(_p):
        try:
            _fm.fontManager.addfont(_p)
            _LOADED_NAMES.append(_fm.FontProperties(fname=_p).get_name())
        except Exception:
            pass

_FONT_LIST = _LOADED_NAMES + [
    "PingFang HK", "Heiti TC", "STHeiti", "Songti SC",
    "SimHei", "Microsoft YaHei", "DejaVu Sans",
]
print(f"已加载中文字体: {_LOADED_NAMES}")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
sns.set_style("whitegrid")

# sns.set_style 会重置 font.* — 必须放在它之后
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _FONT_LIST
plt.rcParams["axes.unicode_minus"] = False

MODEL_ORDER_CLS = ["NaiveBayes", "TextCNN", "BiGRU", "Transformer"]
MODEL_COLORS = {
    "NaiveBayes": "#7E57C2",
    "TextCNN": "#42A5F5",
    "BiGRU": "#26A69A",
    "Transformer": "#EF5350",
    "Seq2Seq+Attention": "#26A69A",
}


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def _save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {name}")


def _annotate_bars(ax, fmt: str = "{:.3f}", offset: float = 0.005) -> None:
    for bar in ax.patches:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


# ─────────────────────────────────────────────────────────────
# 1. 跨任务总览
# ─────────────────────────────────────────────────────────────
def fig_overview_best_per_task() -> None:
    """final_results_summary.csv → 三任务最优指标柱状图"""
    df = _read_csv(ROOT / "final_results_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#42A5F5", "#26A69A", "#EF5350"]
    bars = ax.bar(df["task"], df["score"], color=colors, edgecolor="black", linewidth=0.8)
    for bar, model, metric in zip(bars, df["best_model"], df["primary_metric"]):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{model}\n{metric}={h:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("最优指标分数")
    ax.set_title("三类任务最优模型与指标总览")
    _save(fig, "fig_overview_best_per_task.png")


# ─────────────────────────────────────────────────────────────
# 2. 情感二分类
# ─────────────────────────────────────────────────────────────
def fig_sentiment_model_compare() -> None:
    """sentiment_results.csv → 4 模型 × 4 指标 分组柱状图"""
    df = _read_csv(SENT_DIR / "sentiment_results.csv")
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER_CLS, ordered=True)
    df = df.sort_values("model")
    metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df[m], width, label=m.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("分数")
    ax.set_title("IMDB 情感二分类: 各模型指标对比")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _save(fig, "fig_sentiment_model_compare.png")


def fig_sentiment_confusion() -> None:
    """sentiment_predictions_best_model.csv → 2×2 混淆矩阵"""
    df = _read_csv(SENT_DIR / "sentiment_predictions_best_model.csv")
    cm = pd.crosstab(df["true_label"], df["pred_label"])
    cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    labels = np.array(
        [[f"TN\n{cm.iat[0,0]}", f"FP\n{cm.iat[0,1]}"],
         [f"FN\n{cm.iat[1,0]}", f"TP\n{cm.iat[1,1]}"]]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        xticklabels=["负面 (0)", "正面 (1)"],
        yticklabels=["负面 (0)", "正面 (1)"],
        cbar_kws={"label": "样本数"}, ax=ax,
    )
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("IMDB 情感二分类 - NaiveBayes 混淆矩阵")
    _save(fig, "fig_sentiment_confusion.png")


def fig_sentiment_error_breakdown() -> None:
    """sentiment_error_summary.csv → correct / FN / FP 饼图 + 柱图"""
    df = _read_csv(SENT_DIR / "sentiment_error_summary.csv")
    df = df.set_index("error_type").reindex(["correct", "false_negative", "false_positive"])
    counts = df["count"].values
    labels_zh = ["正确预测", "假阴性 (FN)", "假阳性 (FP)"]
    colors = ["#66BB6A", "#FFCA28", "#EF5350"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].pie(
        counts, labels=labels_zh, autopct="%1.1f%%", colors=colors,
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[0].set_title("NaiveBayes 预测分布")

    bars = axes[1].bar(labels_zh, counts, color=colors, edgecolor="black")
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 30, f"{int(h)}", ha="center", fontsize=10)
    axes[1].set_ylabel("样本数")
    axes[1].set_title("误差类型计数")
    fig.suptitle("IMDB 情感二分类 - 误差分布", y=1.02)
    _save(fig, "fig_sentiment_error_breakdown.png")


# ─────────────────────────────────────────────────────────────
# 3. Reuters 多分类
# ─────────────────────────────────────────────────────────────
def fig_reuters_model_compare() -> None:
    """reuters_results.csv → 4 模型 × 3 指标 分组柱状图"""
    df = _read_csv(REUT_DIR / "reuters_results.csv")
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER_CLS, ordered=True)
    df = df.sort_values("model")

    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    metric_names = ["Accuracy", "Macro-F1", "Weighted-F1"]
    x = np.arange(len(df))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, n) in enumerate(zip(metrics, metric_names)):
        bars = ax.bar(x + (i - 1) * width, df[m], width, label=n)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("分数")
    ax.set_title("Reuters-46 多分类: 各模型指标对比")
    ax.legend()
    _save(fig, "fig_reuters_model_compare.png")


def fig_reuters_per_class_f1() -> None:
    """reuters_class_report_best_model.csv → 各类别 F1 + 样本数"""
    df = _read_csv(REUT_DIR / "reuters_class_report_best_model.csv")
    df = df.rename(columns={"Unnamed: 0": "label", df.columns[0]: "label"})
    df = df[~df["label"].isin(["accuracy", "macro avg", "weighted avg"])].copy()
    df["label"] = df["label"].astype(int)
    df = df.sort_values("label")

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()
    ax1.bar(df["label"], df["f1-score"], color="#42A5F5",
            edgecolor="black", linewidth=0.4, label="F1")
    ax2.plot(df["label"], df["support"], color="#EF5350",
             marker="o", markersize=4, linewidth=1, label="样本数 (右轴)")

    ax1.set_xlabel("类别 ID")
    ax1.set_ylabel("F1 分数")
    ax2.set_ylabel("测试集样本数")
    ax1.set_xticks(df["label"])
    ax1.tick_params(axis="x", labelsize=7)
    ax1.set_title("Reuters-46 各类别 F1 与样本量分布 (TextCNN)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    _save(fig, "fig_reuters_per_class_f1.png")


def fig_reuters_class_imbalance() -> None:
    """reuters_class_report_best_model.csv → 类别样本数排序长尾分布"""
    df = _read_csv(REUT_DIR / "reuters_class_report_best_model.csv")
    df = df.rename(columns={df.columns[0]: "label"})
    df = df[~df["label"].isin(["accuracy", "macro avg", "weighted avg"])].copy()
    df["label"] = df["label"].astype(int)
    df = df.sort_values("support", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    colors = ["#EF5350" if s >= 100 else ("#42A5F5" if s >= 20 else "#90A4AE")
              for s in df["support"]]
    ax.bar(range(len(df)), df["support"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"].astype(str), fontsize=7)
    ax.set_xlabel("类别 ID (按样本数降序)")
    ax.set_ylabel("测试集样本数")
    ax.set_title("Reuters-46 类别样本量长尾分布")
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#EF5350", label="支持度 ≥ 100"),
        plt.Rectangle((0, 0), 1, 1, color="#42A5F5", label="20 ≤ 支持度 < 100"),
        plt.Rectangle((0, 0), 1, 1, color="#90A4AE", label="支持度 < 20"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    _save(fig, "fig_reuters_class_imbalance.png")


def fig_reuters_top_confusions(model: str = "TextCNN", top_k: int = 15) -> None:
    """reuters_top_confusions.csv → 最频繁混淆 pair 热图"""
    df = _read_csv(REUT_DIR / "reuters_top_confusions.csv")
    df = df[df["model"] == model].nlargest(top_k, "count")

    pivot = df.pivot_table(
        index="true_label", columns="pred_label", values="count", fill_value=0
    ).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds",
                cbar_kws={"label": "误判次数"}, ax=ax)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"Reuters-46 - {model} Top-{top_k} 混淆对")
    _save(fig, f"fig_reuters_top_confusions_{model.lower()}.png")


# ─────────────────────────────────────────────────────────────
# 4. 机器翻译
# ─────────────────────────────────────────────────────────────
def fig_translation_bleu_compare() -> None:
    """translation_results.csv → BLEU-1/2/4 × beam/greedy"""
    df = _read_csv(TRAN_DIR / "translation_results.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append([r["model"], "Beam", r["BLEU-1"], r["BLEU-2"], r["BLEU-4"]])
        rows.append([r["model"], "Greedy", r["BLEU-1-greedy"], r["BLEU-2-greedy"], r["BLEU-4-greedy"]])
    long_df = pd.DataFrame(rows, columns=["model", "decode", "BLEU-1", "BLEU-2", "BLEU-4"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, metric in zip(axes, ["BLEU-1", "BLEU-2", "BLEU-4"]):
        sub = long_df.pivot(index="model", columns="decode", values=metric)
        sub.plot(kind="bar", ax=ax, color={"Beam": "#26A69A", "Greedy": "#FFA726"},
                 edgecolor="black", linewidth=0.5, width=0.7)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("BLEU 分数" if metric == "BLEU-1" else "")
        ax.set_ylim(0, 0.75)
        ax.tick_params(axis="x", rotation=0)
        for p in ax.patches:
            h = p.get_height()
            if not np.isnan(h):
                ax.text(p.get_x() + p.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", fontsize=8)
        ax.legend(title="解码")
    fig.suptitle("西班牙语 → 英语翻译: BLEU 对比", y=1.02)
    _save(fig, "fig_translation_bleu_compare.png")


def fig_translation_decode_ablation() -> None:
    """translation_decode_ablation.csv → BLEU-4 vs 推理速度 散点对比"""
    df = _read_csv(TRAN_DIR / "translation_decode_ablation.csv")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"beam": "o", "greedy": "s"}
    for _, r in df.iterrows():
        c = MODEL_COLORS.get(r["model"], "#888")
        ax.scatter(r["sentences_per_sec"], r["BLEU-4"], s=180, c=c,
                   marker=markers[r["decode_method"]],
                   edgecolor="black", linewidth=0.8,
                   label=f"{r['model']} - {r['decode_method']}")
        ax.annotate(f"BLEU-4={r['BLEU-4']:.4f}\n{r['sentences_per_sec']:.1f} sent/s",
                    xy=(r["sentences_per_sec"], r["BLEU-4"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=8)
    ax.set_xlabel("推理速度 (句/秒)")
    ax.set_ylabel("BLEU-4")
    ax.set_title("解码策略消融: 翻译质量 vs 推理速度")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "fig_translation_decode_ablation.png")


def fig_translation_error_distribution() -> None:
    """translation_error_summary.csv → 各模型/解码 下错误类型堆叠柱"""
    df = _read_csv(TRAN_DIR / "translation_error_summary.csv")
    pivot = df.pivot_table(index="model_decode", columns="error_tag",
                           values="count", fill_value=0)
    pivot = pivot.reindex(["seq2seq_greedy", "seq2seq_beam",
                           "transformer_greedy", "transformer_beam"])
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               colormap="Set2", edgecolor="black", linewidth=0.4)
    for c in ax.containers:
        ax.bar_label(c, label_type="center", fontsize=8,
                     fmt=lambda v: f"{int(v)}" if v > 0 else "")
    ax.set_xlabel("模型 - 解码策略")
    ax.set_ylabel("错误样本数")
    ax.set_title("机器翻译错误类型分布 (按模型/解码)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="错误类型", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, "fig_translation_error_distribution.png")


# ─────────────────────────────────────────────────────────────
# 5. 效率对比 (横跨任务)
# ─────────────────────────────────────────────────────────────
def fig_efficiency_params_vs_score() -> None:
    """三任务 efficiency.csv + results.csv → params vs 主指标"""
    sent = _read_csv(SENT_DIR / "sentiment_results.csv")
    reut = _read_csv(REUT_DIR / "reuters_results.csv")
    tran = _read_csv(TRAN_DIR / "translation_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    def _plot(ax, df, score_col, title, ylim):
        d = df.dropna(subset=["params"]).copy()
        for _, r in d.iterrows():
            c = MODEL_COLORS.get(r["model"], "#888")
            ax.scatter(r["params"] / 1e6, r[score_col], s=200, c=c,
                       edgecolor="black", linewidth=1, label=r["model"])
            ax.annotate(r["model"],
                        xy=(r["params"] / 1e6, r[score_col]),
                        xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set_xlabel("参数量 (M)")
        ax.set_ylabel(score_col)
        ax.set_title(title)
        ax.set_ylim(*ylim)

    _plot(axes[0], sent, "f1", "情感二分类: 参数量 vs F1", (0.75, 0.88))
    _plot(axes[1], reut, "f1_macro", "Reuters-46: 参数量 vs Macro-F1", (0.4, 0.65))
    _plot(axes[2], tran, "BLEU-4", "翻译: 参数量 vs BLEU-4", (0.28, 0.36))
    fig.suptitle("模型参数量与性能权衡", y=1.02)
    _save(fig, "fig_efficiency_params_vs_score.png")


def fig_efficiency_train_time() -> None:
    """三任务 efficiency.csv → 训练耗时柱状对比"""
    sent = _read_csv(SENT_DIR / "sentiment_efficiency.csv").assign(task="情感二分类")
    reut = _read_csv(REUT_DIR / "reuters_efficiency.csv").assign(task="Reuters-46")
    tran = _read_csv(TRAN_DIR / "translation_efficiency.csv").assign(task="机器翻译")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, df, title in zip(axes, [sent, reut, tran],
                              ["情感二分类", "Reuters-46", "机器翻译"]):
        d = df.copy()
        d["model"] = pd.Categorical(d["model"],
                                    categories=MODEL_ORDER_CLS + ["Seq2Seq+Attention"],
                                    ordered=True)
        d = d.sort_values("model")
        colors = [MODEL_COLORS.get(m, "#888") for m in d["model"]]
        ax.bar(d["model"].astype(str), d["train_seconds"], color=colors,
               edgecolor="black", linewidth=0.5)
        for x, v in enumerate(d["train_seconds"]):
            ax.text(x, v + max(d["train_seconds"]) * 0.02,
                    f"{v:.1f}s", ha="center", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("训练耗时 (秒)")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("训练耗时对比", y=1.02)
    _save(fig, "fig_efficiency_train_time.png")


def fig_efficiency_infer_speed() -> None:
    """三任务 efficiency.csv → 推理吞吐对比"""
    sent = _read_csv(SENT_DIR / "sentiment_efficiency.csv")
    reut = _read_csv(REUT_DIR / "reuters_efficiency.csv")
    tran = _read_csv(TRAN_DIR / "translation_efficiency.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, df, title, col in [
        (axes[0], sent, "情感二分类 (样本/秒)", "infer_samples_per_sec"),
        (axes[1], reut, "Reuters-46 (样本/秒)", "infer_samples_per_sec"),
    ]:
        d = df.copy()
        d["model"] = pd.Categorical(d["model"],
                                    categories=MODEL_ORDER_CLS, ordered=True)
        d = d.sort_values("model")
        colors = [MODEL_COLORS.get(m, "#888") for m in d["model"]]
        ax.bar(d["model"].astype(str), d[col], color=colors, edgecolor="black", linewidth=0.5)
        for x, v in enumerate(d[col]):
            ax.text(x, v + max(d[col]) * 0.02, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("吞吐量 (样本/秒)")
        ax.tick_params(axis="x", rotation=15)

    d = tran.copy()
    x = np.arange(len(d))
    w = 0.35
    axes[2].bar(x - w / 2, d["infer_sents_per_sec_beam"], w,
                color="#26A69A", label="Beam", edgecolor="black", linewidth=0.5)
    axes[2].bar(x + w / 2, d["infer_sents_per_sec_greedy"], w,
                color="#FFA726", label="Greedy", edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(d["model"], rotation=15)
    axes[2].set_title("机器翻译 (句/秒)")
    axes[2].set_ylabel("吞吐量 (句/秒)")
    axes[2].legend()
    for i, (b, g) in enumerate(zip(d["infer_sents_per_sec_beam"], d["infer_sents_per_sec_greedy"])):
        axes[2].text(i - w / 2, b + 1, f"{b:.1f}", ha="center", fontsize=8)
        axes[2].text(i + w / 2, g + 1, f"{g:.1f}", ha="center", fontsize=8)

    fig.suptitle("推理吞吐量对比", y=1.02)
    _save(fig, "fig_efficiency_infer_speed.png")


# ─────────────────────────────────────────────────────────────
# 6. 多种子稳定性
# ─────────────────────────────────────────────────────────────
def _collect_stability(task_dir: Path, score_col: str) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(task_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed = seed_dir.name.replace("seed_", "")
        result_csv = next(seed_dir.glob("*_results.csv"), None)
        if not result_csv:
            continue
        df = _read_csv(result_csv)
        for _, r in df.iterrows():
            rows.append({"seed": seed, "model": r["model"], "score": r[score_col]})
    return pd.DataFrame(rows)


def fig_stability_seeds() -> None:
    """supplementary_outputs/stability/* → 三任务 × 各模型 跨种子误差棒"""
    sent = _collect_stability(STAB_DIR / "sentiment", "f1")
    reut = _collect_stability(STAB_DIR / "reuters", "f1_macro")
    tran = _collect_stability(STAB_DIR / "translation", "BLEU-4")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, df, title, ylabel in [
        (axes[0], sent, "情感二分类 (F1)", "F1"),
        (axes[1], reut, "Reuters-46 (Macro-F1)", "Macro-F1"),
        (axes[2], tran, "机器翻译 (BLEU-4)", "BLEU-4"),
    ]:
        if df.empty:
            ax.set_title(title + " (无数据)")
            continue
        agg = df.groupby("model")["score"].agg(["mean", "std", "count"]).reset_index()
        order = [m for m in MODEL_ORDER_CLS + ["Seq2Seq+Attention"]
                 if m in set(agg["model"])]
        agg["model"] = pd.Categorical(agg["model"], categories=order, ordered=True)
        agg = agg.sort_values("model")
        colors = [MODEL_COLORS.get(m, "#888") for m in agg["model"]]
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], color=colors,
               edgecolor="black", linewidth=0.5, capsize=6,
               error_kw={"elinewidth": 1.2, "ecolor": "black"})
        for i, (m, s, n) in enumerate(zip(agg["mean"], agg["std"], agg["count"])):
            ax.text(i, m + (s if not np.isnan(s) else 0) + 0.005,
                    f"{m:.4f}\n±{s:.4f}\n(n={int(n)})",
                    ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(agg["model"].astype(str), rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    fig.suptitle("多随机种子稳定性 (均值 ± 标准差)", y=1.02)
    _save(fig, "fig_stability_seeds.png")


# ─────────────────────────────────────────────────────────────
# 7. 新增创新工作图表
# ─────────────────────────────────────────────────────────────
def fig_hybrid_compare() -> None:
    """主对比表中加入 CNNBiGRU 一行后, 三任务的 head-to-head 对比图.

    依赖: 三任务 outputs/*_results.csv 已含 CNNBiGRU 行 (T10/T11/T12 已实现).
    若任一任务的 results.csv 中缺 CNNBiGRU, 该任务子图静默跳过.
    """
    sent_path = SENT_DIR / "sentiment_results.csv"
    reut_path = REUT_DIR / "reuters_results.csv"
    tran_path = TRAN_DIR / "translation_results.csv"
    if not (sent_path.exists() and reut_path.exists() and tran_path.exists()):
        print("  (跳过 hybrid_compare: 缺一个或多个 results.csv)")
        return
    sent = _read_csv(sent_path).set_index("model")
    reut = _read_csv(reut_path).set_index("model")
    tran = _read_csv(tran_path).set_index("model")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sent_models = ["TextCNN", "BiGRU", "Transformer", "CNNBiGRU"]
    sent_vals = [sent.loc[m, "f1"] if m in sent.index else np.nan for m in sent_models]
    axes[0].bar(sent_models, sent_vals,
                color=[MODEL_COLORS.get(m, "#888") for m in sent_models],
                edgecolor="black")
    axes[0].set_title("情感二分类 F1")
    axes[0].set_ylim(0.7, 0.95)
    axes[0].tick_params(axis="x", rotation=15)

    reut_models = ["TextCNN", "BiGRU", "Transformer", "CNNBiGRU"]
    reut_vals = [reut.loc[m, "f1_macro"] if m in reut.index else np.nan for m in reut_models]
    axes[1].bar(reut_models, reut_vals,
                color=[MODEL_COLORS.get(m, "#888") for m in reut_models],
                edgecolor="black")
    axes[1].set_title("Reuters Macro-F1")
    axes[1].set_ylim(0.4, 0.7)
    axes[1].tick_params(axis="x", rotation=15)

    tran_models = ["Seq2Seq+Attention", "Transformer", "CNNBiGRU"]
    tran_vals = [tran.loc[m, "BLEU-4"] if m in tran.index else np.nan for m in tran_models]
    axes[2].bar(tran_models, tran_vals,
                color=[MODEL_COLORS.get(m, "#888") for m in tran_models],
                edgecolor="black")
    axes[2].set_title("翻译 BLEU-4")
    axes[2].set_ylim(0.25, 0.40)
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle("CNN-BiGRU 混合模型 vs 单一架构基线", y=1.02)
    _save(fig, "fig_hybrid_compare.png")


def fig_focal_gamma_ablation() -> None:
    """γ ∈ {0.5, 1.0, 2.0, 5.0} 的 Macro-F1 折线图 (Reuters).

    数据源: 新闻多分类/outputs/reuters_results_focal_g*.csv (Plan 中要求的 γ-scan 输出)
    """
    rows = []
    for p in REUT_DIR.glob("reuters_results_focal_g*.csv"):
        # 文件名形如 reuters_results_focal_g2.0.csv
        try:
            gamma = float(p.stem.rsplit("_g", 1)[-1])
        except ValueError:
            continue
        df = _read_csv(p)
        for _, r in df.iterrows():
            rows.append({"model": r["model"], "gamma": gamma, "f1_macro": r["f1_macro"]})
    if not rows:
        print("  (跳过 focal_gamma_ablation: 无 reuters_results_focal_g*.csv)")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("gamma")
        ax.plot(sub["gamma"], sub["f1_macro"], marker="o", label=str(model), linewidth=2)
    ax.set_xlabel("focal γ")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Reuters-46: Focal Loss γ 敏感性")
    ax.legend()
    _save(fig, "fig_focal_gamma_ablation.png")


def fig_label_smoothing_compare() -> None:
    """对比 baseline Transformer 与 LS Transformer 的 BLEU-1/2/4."""
    base_path = TRAN_DIR / "translation_results.csv"
    ls_path = TRAN_DIR / "translation_results_ls.csv"
    if not (base_path.exists() and ls_path.exists()):
        print("  (跳过 label_smoothing_compare: 缺 _ls 结果或 baseline)")
        return
    base = _read_csv(base_path)
    ls = _read_csv(ls_path)
    if "Transformer" not in base["model"].values or "Transformer" not in ls["model"].values:
        print("  (跳过 label_smoothing_compare: 无 Transformer 行)")
        return
    base_tf = base[base["model"] == "Transformer"].iloc[0]
    ls_tf = ls[ls["model"] == "Transformer"].iloc[0]
    metrics = ["BLEU-1", "BLEU-2", "BLEU-4"]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, [base_tf[m] for m in metrics], w,
           label="baseline", color="#FFA726", edgecolor="black")
    ax.bar(x + w / 2, [ls_tf[m] for m in metrics], w,
           label="+LabelSmoothing 0.1", color="#26A69A", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("BLEU")
    ax.set_title("翻译 Transformer: Label Smoothing 影响")
    ax.legend()
    _save(fig, "fig_label_smoothing_compare.png")


def fig_decode_grid() -> None:
    """beam × length_penalty 的 BLEU-4 网格热图 (Seq2Seq vs Transformer)."""
    p = TRAN_DIR / "translation_decode_grid.csv"
    if not p.exists():
        print("  (跳过 decode_grid: 无 translation_decode_grid.csv)")
        return
    df = _read_csv(p)
    # decode_grid.py 的 BLEU 列名为 bleu1/bleu2/bleu4 (lowercase, no dash)
    score_col = "bleu4" if "bleu4" in df.columns else "BLEU-4"
    if score_col not in df.columns:
        print(f"  (跳过 decode_grid: 找不到 BLEU-4 列, 仅有 {list(df.columns)})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, ["Seq2Seq+Attention", "Transformer"]):
        sub = df[df["model"] == model]
        if sub.empty:
            ax.set_title(f"{model} (无数据)")
            continue
        pivot = sub.pivot(index="beam", columns="length_penalty", values=score_col)
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn",
                    ax=ax, cbar_kws={"label": "BLEU-4"})
        ax.set_title(model)
    fig.suptitle("解码策略网格搜索: beam × length_penalty", y=1.02)
    _save(fig, "fig_decode_grid.png")


def fig_learning_curve() -> None:
    """三任务 ratio vs 主指标的折线图.

    数据源: supplementary_outputs/learning_curve/{sentiment,reuters,translation}_curve_results.csv
    """
    base = ROOT / "supplementary_outputs" / "learning_curve"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task, score, title in [
        (axes[0], "sentiment", "f1", "情感二分类 (F1)"),
        (axes[1], "reuters", "f1_macro", "Reuters Macro-F1"),
        (axes[2], "translation", "BLEU-4", "翻译 BLEU-4"),
    ]:
        path = base / f"{task}_curve_results.csv"
        if not path.exists():
            ax.set_title(f"{title} (无数据)")
            ax.set_xlabel("训练数据比例 (%)")
            continue
        df = _read_csv(path)
        if score not in df.columns or "ratio" not in df.columns:
            ax.set_title(f"{title} (列缺失)")
            continue
        for model, sub in df.groupby("model"):
            sub = sub.sort_values("ratio")
            ax.plot(sub["ratio"] * 100, sub[score],
                    marker="o", label=str(model), linewidth=2)
        ax.set_xlabel("训练数据比例 (%)")
        ax.set_ylabel(score)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("数据规模学习曲线", y=1.02)
    _save(fig, "fig_learning_curve.png")


def fig_robustness() -> None:
    """三任务扰动鲁棒性曲线.

    数据源: supplementary_outputs/robustness/{sentiment,reuters,translation}_robustness.csv
    """
    base = ROOT / "supplementary_outputs" / "robustness"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task, score, title in [
        (axes[0], "sentiment", "f1", "情感二分类 (F1)"),
        (axes[1], "reuters", "f1_macro", "Reuters Macro-F1"),
        (axes[2], "translation", "bleu4", "翻译 BLEU-4"),
    ]:
        path = base / f"{task}_robustness.csv"
        if not path.exists():
            ax.set_title(f"{title} (无数据)")
            ax.set_xlabel("扰动比例 (%)")
            continue
        df = _read_csv(path)
        if score not in df.columns or "p" not in df.columns or "mode" not in df.columns:
            ax.set_title(f"{title} (列缺失)")
            continue
        for (model, mode), sub in df.groupby(["model", "mode"]):
            sub = sub.sort_values("p")
            ax.plot(sub["p"] * 100, sub[score],
                    marker="o", label=f"{model}-{mode}", linewidth=1.5)
        ax.set_xlabel("扰动比例 (%)")
        ax.set_ylabel(score)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("词级扰动鲁棒性", y=1.02)
    _save(fig, "fig_robustness.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"输出目录: {FIG_DIR}")
    figs = [
        ("跨任务总览", fig_overview_best_per_task),
        ("情感模型对比", fig_sentiment_model_compare),
        ("情感混淆矩阵", fig_sentiment_confusion),
        ("情感误差分布", fig_sentiment_error_breakdown),
        ("Reuters 模型对比", fig_reuters_model_compare),
        ("Reuters 各类 F1", fig_reuters_per_class_f1),
        ("Reuters 长尾分布", fig_reuters_class_imbalance),
        ("Reuters TextCNN 混淆", lambda: fig_reuters_top_confusions("TextCNN")),
        ("Reuters BiGRU 混淆", lambda: fig_reuters_top_confusions("BiGRU")),
        ("翻译 BLEU 对比", fig_translation_bleu_compare),
        ("翻译解码消融", fig_translation_decode_ablation),
        ("翻译错误分布", fig_translation_error_distribution),
        ("效率: 参数量 vs 分数", fig_efficiency_params_vs_score),
        ("效率: 训练耗时", fig_efficiency_train_time),
        ("效率: 推理吞吐", fig_efficiency_infer_speed),
        ("多种子稳定性", fig_stability_seeds),
        ("混合模型对比", fig_hybrid_compare),
        ("Focal γ 消融", fig_focal_gamma_ablation),
        ("Label Smoothing 对比", fig_label_smoothing_compare),
        ("解码网格", fig_decode_grid),
        ("学习曲线", fig_learning_curve),
        ("鲁棒性", fig_robustness),
    ]
    failed = []
    for name, fn in figs:
        try:
            print(f"[{name}]")
            fn()
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            failed.append((name, str(e)))

    print(f"\n完成: 共 {len(figs) - len(failed)} / {len(figs)} 张图")
    if failed:
        print("\n失败列表:")
        for n, e in failed:
            print(f"  - {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
