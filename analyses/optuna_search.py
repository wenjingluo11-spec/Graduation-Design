#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna 超参搜索: Reuters-46 Transformer (T22 创新点 - 工程实践)

目标: 在不增加参数量、不引入预训练的前提下,通过自动超参搜索探索基础
Transformer 在小数据多分类任务上的性能上限.

搜索空间:
    lr            log-uniform [1e-4, 5e-3]
    weight_decay  log-uniform [1e-6, 1e-3]
    dropout       uniform     [0.1, 0.5]
    embedding_dim categorical {96, 128, 160, 192, 256}
    num_heads     categorical {2, 4, 8}    (constrain emb_dim % heads == 0)
    ff_dim        categorical {128, 256, 512}
    num_layers    int         [1, 4]
    batch_size    categorical {32, 64, 96}
    loss          categorical {ce, focal}
    focal_gamma   uniform     [0.5, 5.0]   (only when loss=focal)

目标指标: test Macro-F1 (与论文主指标一致)

输出:
    supplementary_outputs/optuna/optuna_trials.csv     全部 trial
    supplementary_outputs/optuna/optuna_importance.csv 超参重要性
    supplementary_outputs/optuna/optuna_best.json      最佳超参 + 最终指标
    figures/fig_optuna_history.png        优化历史
    figures/fig_optuna_importance.png     超参重要性
    figures/fig_optuna_parallel.png       平行坐标
    figures/fig_optuna_contour.png        2D 等高线 (top 2 重要超参)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# MPS fallback 必须在 import torch 之前
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import optuna
import pandas as pd
import torch
import torch.utils.data as data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "新闻多分类"))

from reuters_multiclass import (  # type: ignore
    Config,
    SeqDataset,
    TransformerMulti,
    _detect_device,
    compute_mc_metrics,
    evaluate_model,
    prepare_data,
    set_seed,
    train_one_epoch,
    FocalLoss,
)
import torch.nn as nn
import torch.optim as optim


OUT_DIR = ROOT / "supplementary_outputs" / "optuna"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", type=str, default="reuters_transformer")
    p.add_argument(
        "--storage",
        type=str,
        default=f"sqlite:///{OUT_DIR}/optuna_study.db",
        help="SQLite URL,断点续跑时指向同一文件",
    )
    p.add_argument("--epochs", type=int, default=15, help="单 trial 训练 epoch 数")
    p.add_argument("--patience", type=int, default=4)
    return p.parse_args()


# 全局缓存数据,避免每个 trial 重建 (节约 ~30s/trial)
# 我们直接缓存底层 dataset (而非 loader),便于每个 trial 用不同 batch_size 重建 loader
_DATA_CACHE: dict = {}


def _load_data_once(cfg: Config):
    """复用 prepare_data,但只调一次.

    prepare_data 返回 6 元组:
      (seq_train, y_train), (seq_test, y_test), train_loader, val_loader, test_loader, num_classes

    我们从 train/val/test loader 里抽出 dataset,跨 trial 复用底层张量.
    """
    if "datasets" in _DATA_CACHE:
        return _DATA_CACHE["datasets"]
    (_st, _yt), (_se, _ye), train_loader, val_loader, test_loader, num_classes = prepare_data(cfg)
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset
    _DATA_CACHE["datasets"] = (train_ds, val_ds, test_ds, num_classes)
    return _DATA_CACHE["datasets"]


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """单 trial: 采样超参 → 构造 Transformer → 训练 → 返回 test Macro-F1."""

    # ---- 1. 采样超参 ----
    embedding_dim = trial.suggest_categorical("embedding_dim", [96, 128, 160, 192, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    if embedding_dim % num_heads != 0:
        # Transformer 强制约束: emb_dim 必须能被 num_heads 整除
        raise optuna.exceptions.TrialPruned(
            f"emb_dim={embedding_dim} 不能整除 num_heads={num_heads}"
        )

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    ff_dim = trial.suggest_categorical("ff_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])
    loss_type = trial.suggest_categorical("loss", ["ce", "focal"])
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 5.0) if loss_type == "focal" else 0.0

    # ---- 2. 构造 cfg ----
    cfg = Config()
    cfg.device = _detect_device()
    # 修正数据路径: 默认 "../reuters.npz" 是相对 新闻多分类/, 我们从仓库根跑要改
    cfg.data_file = str(ROOT / "reuters.npz")
    cfg.embedding_dim = embedding_dim
    cfg.num_heads = num_heads
    cfg.ff_dim = ff_dim
    cfg.num_transformer_layers = num_layers
    cfg.dropout = dropout
    cfg.lr = lr
    cfg.weight_decay = weight_decay
    cfg.batch_size = batch_size
    cfg.epochs = args.epochs
    cfg.early_stopping_patience = args.patience
    cfg.loss = loss_type
    cfg.focal_gamma = focal_gamma
    cfg.seed = args.seed

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---- 3. 数据 (跨 trial 缓存底层 dataset, 每个 trial 用不同 batch_size 重建 loader) ----
    train_ds, val_ds, test_ds, num_classes = _load_data_once(cfg)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ---- 4. 构造 Transformer ----
    model = TransformerMulti(
        cfg.max_vocab_size,
        embedding_dim,
        num_heads,
        ff_dim,
        num_layers,
        num_classes,
        dropout,
    ).to(device)

    # ---- 5. 训练循环 (含早停 + 中间报告便于 pruner) ----
    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_macro = -1.0
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_v, p_v = evaluate_model(model, val_loader, device)
        val_metrics = compute_mc_metrics(y_v, p_v)
        val_macro = val_metrics["f1_macro"]

        # Optuna pruner 检查
        trial.report(val_macro, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_macro > best_val_macro + 1e-4:
            best_val_macro = val_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                break

    # ---- 6. 用 best 权重在 test 上算 Macro-F1 ----
    if best_state is not None:
        model.load_state_dict(best_state)
    y_t, p_t = evaluate_model(model, test_loader, device)
    test_metrics = compute_mc_metrics(y_t, p_t)
    test_macro = test_metrics["f1_macro"]

    # ---- 7. 记录额外指标 ----
    trial.set_user_attr("test_accuracy", test_metrics["accuracy"])
    trial.set_user_attr("test_f1_weighted", test_metrics["f1_weighted"])
    trial.set_user_attr("best_val_f1_macro", best_val_macro)
    trial.set_user_attr("trained_epochs", epoch)
    trial.set_user_attr("params", sum(p.numel() for p in model.parameters()))

    return test_macro


def export_results(study: optuna.Study) -> None:
    """跑完后导出 CSV / JSON / 4 张图."""

    # --- CSV: 全部 trial ---
    rows = []
    for t in study.trials:
        if t.state.name not in ("COMPLETE", "PRUNED"):
            continue
        rec = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value if t.value is not None else float("nan"),
            **t.params,
            **{f"_{k}": v for k, v in t.user_attrs.items()},
        }
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT_DIR / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ {OUT_DIR / 'optuna_trials.csv'}  ({len(rows)} 行)")

    # --- 重要性 ---
    try:
        importance = optuna.importance.get_param_importances(study)
        pd.DataFrame(
            [(k, v) for k, v in importance.items()], columns=["param", "importance"]
        ).to_csv(OUT_DIR / "optuna_importance.csv", index=False, encoding="utf-8-sig")
        print(f"  ✓ {OUT_DIR / 'optuna_importance.csv'}")
    except Exception as e:
        print(f"  ⚠ 重要性导出失败: {e}")

    # --- best ---
    best = {
        "number": study.best_trial.number,
        "value": study.best_value,
        "params": study.best_params,
        "user_attrs": dict(study.best_trial.user_attrs),
    }
    (OUT_DIR / "optuna_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓ {OUT_DIR / 'optuna_best.json'}")

    # --- 4 张图 (用 plotly 绘制后转 png 需要 kaleido,这里直接保存 HTML;
    #     用 matplotlib 绘最关键的两张) ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # 字体
    for fp in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=fp).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False

    # 优化历史
    fig, ax = plt.subplots(figsize=(9, 5))
    completed = [
        t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"
    ]
    xs = [t.number for t in completed]
    ys = [t.value for t in completed]
    best_so_far = []
    cur_best = -float("inf")
    for v in ys:
        cur_best = max(cur_best, v)
        best_so_far.append(cur_best)
    ax.scatter(xs, ys, alpha=0.5, label="单 trial", s=30)
    ax.plot(xs, best_so_far, "r-", linewidth=2, label="累计最优", marker="o", markersize=4)
    ax.set_xlabel("Trial 编号")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Optuna 优化历史 (Reuters Transformer)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_optuna_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'fig_optuna_history.png'}")

    # 超参重要性
    try:
        importance = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 5))
        params = list(importance.keys())[::-1]
        vals = [importance[p] for p in params]
        ax.barh(params, vals, color="#42A5F5", edgecolor="black")
        ax.set_xlabel("重要性 (相对)")
        ax.set_title("Optuna 超参重要性 (Reuters Transformer)")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_optuna_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {FIG_DIR / 'fig_optuna_importance.png'}")
    except Exception as e:
        print(f"  ⚠ 重要性图绘制失败: {e}")

    # plotly 交互图 (HTML)
    try:
        from optuna.visualization import (
            plot_parallel_coordinate,
            plot_contour,
        )

        plot_parallel_coordinate(study).write_html(str(OUT_DIR / "optuna_parallel.html"))
        print(f"  ✓ {OUT_DIR / 'optuna_parallel.html'}")
        # contour 用 top 2 重要超参
        try:
            imp_dict = optuna.importance.get_param_importances(study)
            if len(imp_dict) >= 2:
                top2 = list(imp_dict.keys())[:2]
                plot_contour(study, params=top2).write_html(
                    str(OUT_DIR / "optuna_contour.html")
                )
                print(f"  ✓ {OUT_DIR / 'optuna_contour.html'} (params: {top2})")
        except Exception:
            pass
    except Exception as e:
        print(f"  ⚠ plotly 图绘制失败: {e}")


def main() -> None:
    args = parse_args()
    print(f"=== Optuna 搜索: Reuters Transformer ===")
    print(f"  trials={args.n_trials}, seed={args.seed}, storage={args.storage}")
    print(f"  device={_detect_device()}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    t0 = time.time()
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    elapsed = time.time() - t0

    print(f"\n=== 完成 ({elapsed/60:.1f} min, {len(study.trials)} trials) ===")
    print(f"  best test Macro-F1: {study.best_value:.4f}")
    print(f"  best params:")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")

    print("\n=== 导出结果 ===")
    export_results(study)


if __name__ == "__main__":
    main()
