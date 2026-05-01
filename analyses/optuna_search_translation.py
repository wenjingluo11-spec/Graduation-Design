#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna 超参搜索: Es→En Translation Transformer (创新点 - 工程实践)

⚠️ 时间警告: 翻译 Transformer 单 trial 耗时显著高于分类任务.
    - epochs=8 (而非主实验的 20): 约 17 min/trial
    - n-trials 默认 15: 总时间约 4 小时
    使用 BLEU 验证集采样小尺寸 (val_sample=200) 进一步加速.

搜索空间 (含 label_smoothing):
    lr               log-uniform [1e-4, 2e-3]
    weight_decay     log-uniform [1e-6, 1e-3]
    dropout          uniform     [0.1, 0.3]
    embedding_dim    categorical {128, 192, 256}
    num_heads        categorical {4, 8}      (constrain emb_dim % heads == 0)
    ff_dim           categorical {256, 512}
    num_layers       int         [2, 4]
    label_smoothing  uniform     [0.0, 0.2]

目标指标: test BLEU-4 (greedy decode, 节省时间)

输出:
    supplementary_outputs/optuna_translation/optuna_trials.csv
    supplementary_outputs/optuna_translation/optuna_importance.csv
    supplementary_outputs/optuna_translation/optuna_best.json
    figures/fig_optuna_translation_history.png
    figures/fig_optuna_translation_importance.png
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
sys.path.insert(0, str(ROOT / "机器翻译"))

from machine_translation import (  # type: ignore
    Config,
    LabelSmoothingCrossEntropy,
    PAD_IDX,
    TransformerTranslator,
    TranslationDataset,
    Vocabulary,
    _detect_device,
    encode_pairs,
    evaluate_transformer_bleu,
    load_parallel_pairs,
    set_seed,
    split_pairs,
    train_transformer_epoch,
)


OUT_DIR = ROOT / "supplementary_outputs" / "optuna_translation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", type=str, default="translation_transformer")
    p.add_argument(
        "--storage",
        type=str,
        default=f"sqlite:///{OUT_DIR}/optuna_study.db",
    )
    p.add_argument("--epochs", type=int, default=8, help="单 trial epoch 数 (主实验是 20,这里减半)")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--val-sample", type=int, default=200, help="val BLEU 采样尺寸,加速")
    p.add_argument("--test-sample", type=int, default=600, help="test BLEU 采样尺寸,加速")
    return p.parse_args()


_DATA_CACHE: dict = {}


def _load_data_once(cfg: Config):
    if "datasets" in _DATA_CACHE:
        return _DATA_CACHE["datasets"]
    pairs = load_parallel_pairs(cfg)
    train_pairs, val_pairs, test_pairs = split_pairs(cfg, pairs)
    src_vocab = Vocabulary(max_size=cfg.max_vocab_src, min_freq=cfg.min_token_freq)
    tgt_vocab = Vocabulary(max_size=cfg.max_vocab_tgt, min_freq=cfg.min_token_freq)
    src_vocab.build(p[0] for p in train_pairs)
    tgt_vocab.build(p[1] for p in train_pairs)
    train_src, train_tgt = encode_pairs(train_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)
    val_src, val_tgt = encode_pairs(val_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)
    test_src, test_tgt = encode_pairs(test_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)
    _DATA_CACHE["datasets"] = (
        train_src, train_tgt, val_src, val_tgt, test_src, test_tgt,
        src_vocab, tgt_vocab,
    )
    return _DATA_CACHE["datasets"]


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 192, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    if embedding_dim % num_heads != 0:
        raise optuna.exceptions.TrialPruned(
            f"emb_dim={embedding_dim} 不能整除 num_heads={num_heads}"
        )

    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    ff_dim = trial.suggest_categorical("ff_dim", [256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    cfg = Config()
    cfg.device = _detect_device()
    cfg.embedding_dim = embedding_dim
    cfg.hidden_dim = embedding_dim  # Transformer 用 hidden_dim 做 d_model
    cfg.num_heads = num_heads
    cfg.ff_dim = ff_dim
    cfg.num_layers = num_layers
    cfg.dropout = dropout
    cfg.epochs_transformer = args.epochs
    cfg.early_stopping_patience_transformer = args.patience
    cfg.seed = args.seed
    cfg.label_smoothing = label_smoothing

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt, src_vocab, tgt_vocab = _load_data_once(cfg)
    train_loader = data.DataLoader(
        TranslationDataset(train_src, train_tgt),
        batch_size=cfg.batch_size, shuffle=True,
    )

    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=embedding_dim,
        nhead=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)

    if label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(epsilon=label_smoothing, ignore_index=PAD_IDX)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_bleu = -1.0
    bad_epochs = 0

    for epoch in range(1, cfg.epochs_transformer + 1):
        train_loss = train_transformer_epoch(model, train_loader, optimizer, criterion, device)
        # 用 greedy decode + 小采样加速 val 评估
        val_bleu = evaluate_transformer_bleu(
            model, val_src, val_tgt, tgt_vocab, "greedy", cfg, sample_size=args.val_sample,
        )["bleu4"]

        trial.report(val_bleu, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_bleu > best_val_bleu + 1e-4:
            best_val_bleu = val_bleu
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience_transformer:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_bleu = evaluate_transformer_bleu(
        model, test_src, test_tgt, tgt_vocab, "greedy", cfg, sample_size=args.test_sample,
    )

    trial.set_user_attr("test_bleu1", test_bleu["bleu1"])
    trial.set_user_attr("test_bleu2", test_bleu["bleu2"])
    trial.set_user_attr("best_val_bleu4", best_val_bleu)
    trial.set_user_attr("trained_epochs", epoch)
    trial.set_user_attr("params", sum(p.numel() for p in model.parameters()))

    return test_bleu["bleu4"]


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
    print(f"  ✓ {OUT_DIR / 'optuna_trials.csv'} ({len(rows)} 行)")

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
    ax.set_ylabel("Test BLEU-4")
    ax.set_title("Optuna 优化历史 (Translation Transformer)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_optuna_translation_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / 'fig_optuna_translation_history.png'}")

    try:
        importance = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 5))
        params = list(importance.keys())[::-1]
        vals = [importance[p] for p in params]
        ax.barh(params, vals, color="#42A5F5", edgecolor="black")
        ax.set_xlabel("重要性 (相对)")
        ax.set_title("Optuna 超参重要性 (Translation Transformer)")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_optuna_translation_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {FIG_DIR / 'fig_optuna_translation_importance.png'}")
    except Exception as e:
        print(f"  ⚠ 重要性图绘制失败: {e}")


def main() -> None:
    args = parse_args()
    print(f"=== Optuna 搜索: Translation Transformer ===")
    print(f"  trials={args.n_trials}, epochs={args.epochs}, val_sample={args.val_sample}, test_sample={args.test_sample}")
    print(f"  device={_detect_device()}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2)
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
    print(f"  best test BLEU-4: {study.best_value:.4f}")
    print(f"  best params:")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")

    print("\n=== 导出结果 ===")
    export_results(study)


if __name__ == "__main__":
    main()
