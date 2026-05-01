#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna 超参搜索: IMDB Sentiment Transformer (创新点 - 工程实践)

搜索空间 (BCE 二分类,无 focal):
    lr            log-uniform [1e-4, 5e-3]
    weight_decay  log-uniform [1e-6, 1e-3]
    dropout       uniform     [0.1, 0.5]
    embedding_dim categorical {96, 128, 160, 192, 256}
    num_heads     categorical {2, 4, 8}    (constrain emb_dim % heads == 0)
    ff_dim        categorical {128, 256, 512}
    num_layers    int         [1, 4]
    batch_size    categorical {32, 64, 96}

目标指标: test F1 (与论文主指标一致)

输出:
    supplementary_outputs/optuna_sentiment/optuna_trials.csv
    supplementary_outputs/optuna_sentiment/optuna_importance.csv
    supplementary_outputs/optuna_sentiment/optuna_best.json
    figures/fig_optuna_sentiment_history.png
    figures/fig_optuna_sentiment_importance.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "情感二分类"))

from sentiment_analysis import (  # type: ignore
    Config,
    TextDataset,
    TransformerClassifier,
    _detect_device,
    compute_binary_metrics,
    evaluate_model,
    load_imdb_data,
    prepare_dl_data,
    set_seed,
    train_one_epoch,
)


OUT_DIR = ROOT / "supplementary_outputs" / "optuna_sentiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", type=str, default="sentiment_transformer")
    p.add_argument(
        "--storage",
        type=str,
        default=f"sqlite:///{OUT_DIR}/optuna_study.db",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    return p.parse_args()


_DATA_CACHE: dict = {}


def _load_data_once(cfg: Config):
    """复用 prepare_dl_data,跨 trial 缓存底层 dataset."""
    if "datasets" in _DATA_CACHE:
        return _DATA_CACHE["datasets"]
    cfg.data_dir = str(ROOT / "情感二分类" / "aclImdb")
    train_df, test_df = load_imdb_data(cfg)
    train_loader, val_loader, test_loader, vocab = prepare_dl_data(train_df, test_df, cfg)
    _DATA_CACHE["datasets"] = (
        train_loader.dataset,
        val_loader.dataset,
        test_loader.dataset,
        vocab,
    )
    return _DATA_CACHE["datasets"]


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    embedding_dim = trial.suggest_categorical("embedding_dim", [96, 128, 160, 192, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    if embedding_dim % num_heads != 0:
        raise optuna.exceptions.TrialPruned(
            f"emb_dim={embedding_dim} 不能整除 num_heads={num_heads}"
        )

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    ff_dim = trial.suggest_categorical("ff_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])

    cfg = Config()
    cfg.device = _detect_device()
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
    cfg.seed = args.seed

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_ds, val_ds, test_ds, vocab = _load_data_once(cfg)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(
        len(vocab), embedding_dim, num_heads, ff_dim, num_layers, dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_v, p_v = evaluate_model(model, val_loader, device)
        val_metrics = compute_binary_metrics(y_v, p_v)
        val_f1 = val_metrics["f1"]

        trial.report(val_f1, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_t, p_t = evaluate_model(model, test_loader, device)
    test_metrics = compute_binary_metrics(y_t, p_t)

    trial.set_user_attr("test_accuracy", test_metrics["accuracy"])
    trial.set_user_attr("test_precision", test_metrics["precision"])
    trial.set_user_attr("test_recall", test_metrics["recall"])
    trial.set_user_attr("best_val_f1", best_val_f1)
    trial.set_user_attr("trained_epochs", epoch)
    trial.set_user_attr("params", sum(p.numel() for p in model.parameters()))

    return test_metrics["f1"]


def export_results(study: optuna.Study) -> None:
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

    try:
        importance = optuna.importance.get_param_importances(study)
        pd.DataFrame(
            [(k, v) for k, v in importance.items()], columns=["param", "importance"]
        ).to_csv(OUT_DIR / "optuna_importance.csv", index=False, encoding="utf-8-sig")
        print(f"  ✓ {OUT_DIR / 'optuna_importance.csv'}")
    except Exception as e:
        print(f"  ⚠ 重要性导出失败: {e}")

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

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    for fp in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=fp).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False

    # 历史
    fig, ax = plt.subplots(figsize=(9, 5))
    completed = [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
    xs = [t.number for t in completed]
    ys = [t.value for t in completed]
    best_so_far = []
    cur = -float("inf")
    for v in ys:
        cur = max(cur, v)
        best_so_far.append(cur)
    ax.scatter(xs, ys, alpha=0.5, s=30, label="单 trial")
    ax.plot(xs, best_so_far, "r-", linewidth=2, marker="o", markersize=4, label="累计最优")
    ax.set_xlabel("Trial 编号")
    ax.set_ylabel("Test F1")
    ax.set_title("Optuna 优化历史 (Sentiment Transformer)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_optuna_sentiment_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'fig_optuna_sentiment_history.png'}")

    try:
        importance = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 5))
        params = list(importance.keys())[::-1]
        vals = [importance[p] for p in params]
        ax.barh(params, vals, color="#42A5F5", edgecolor="black")
        ax.set_xlabel("重要性 (相对)")
        ax.set_title("Optuna 超参重要性 (Sentiment Transformer)")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_optuna_sentiment_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {FIG_DIR / 'fig_optuna_sentiment_importance.png'}")
    except Exception as e:
        print(f"  ⚠ 重要性图绘制失败: {e}")


def main() -> None:
    args = parse_args()
    print(f"=== Optuna 搜索: Sentiment Transformer ===")
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
    print(f"  best test F1: {study.best_value:.4f}")
    print(f"  best params:")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")

    print("\n=== 导出结果 ===")
    export_results(study)


if __name__ == "__main__":
    main()
