#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力可视化 (T19):

加载已训好的 Transformer / Seq2Seq 模型, 跑一小批测试样本, 把
注意力权重画成热图. 输出到 figures/attn_<task>_<sample_id>.png.

- 情感二分类:    Transformer 最后一层 self-attention (头平均)
- Reuters-46:    Transformer 最后一层 self-attention (头平均)
- 翻译 Transformer: 最后一层 decoder cross-attention (头平均, 源 vs 目标)
- 翻译 Seq2Seq:    Bahdanau alpha 矩阵 (T_tgt × T_src)
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data as data

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]
_LOADED: List[str] = []
for _p in _FONT_PATHS:
    if Path(_p).exists():
        try:
            fm.fontManager.addfont(_p)
            _LOADED.append(fm.FontProperties(fname=_p).get_name())
        except Exception:
            pass
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _LOADED + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def viz_sentiment(num_samples: int = 5, max_show: int = 30) -> int:
    try:
        sys.path.insert(0, str(ROOT / "情感二分类"))
        from sentiment_analysis import (  # type: ignore
            Config, _detect_device, set_seed, load_imdb_data, prepare_dl_data,
            TransformerClassifier,
        )
        cfg = Config(); cfg.device = _detect_device(); set_seed(cfg.seed)
        device = torch.device(cfg.device)
        train_df, test_df = load_imdb_data(cfg)
        _, _, test_loader, vocab = prepare_dl_data(train_df, test_df, cfg)
    except FileNotFoundError as e:
        print(f"  跳过 sentiment: 数据缺失 ({e})")
        return 0
    inv_vocab = {v: k for k, v in vocab.items()}

    ckpt = ROOT / "情感二分类" / "outputs" / "checkpoints" / "transformer_best.pt"
    if not ckpt.exists():
        print(f"  跳过 sentiment: 缺 checkpoint ({ckpt})")
        return 0

    model = TransformerClassifier(
        len(vocab), cfg.embedding_dim, cfg.num_heads, cfg.ff_dim,
        cfg.num_transformer_layers, cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    n = 0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            logits, attn = model(x_batch, return_attention=True)
        for i in range(x_batch.size(0)):
            if n >= num_samples:
                return n
            tokens_full = [inv_vocab.get(int(t), "<?>") for t in x_batch[i].cpu().numpy()]
            real_len = sum(1 for t in tokens_full if t != "<pad>")
            show = min(max(real_len, 1), max_show)
            tokens = tokens_full[:show]
            head_avg = attn[i].mean(dim=0).cpu().numpy()[:show, :show]

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(head_avg, xticklabels=tokens, yticklabels=tokens,
                        cmap="Blues", ax=ax, cbar_kws={"label": "注意力权重"})
            pred = int(logits[i].sigmoid().item() > 0.5)
            true_label = int(y_batch[i].item())
            ax.set_title(f"情感二分类 Transformer - sample {n} (pred={pred}, true={true_label})")
            plt.xticks(rotation=60, fontsize=7)
            plt.yticks(rotation=0, fontsize=7)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"attn_sentiment_{n}.png", dpi=150, bbox_inches="tight")
            plt.close()
            n += 1
    return n


def viz_reuters(num_samples: int = 5, max_show: int = 30) -> int:
    try:
        sys.path.insert(0, str(ROOT / "新闻多分类"))
        from reuters_multiclass import (  # type: ignore
            Config, _detect_device, set_seed, load_reuters, remap_and_pad,
            SeqDataset, TransformerMulti,
        )
        from sklearn.model_selection import train_test_split
        cfg = Config(); cfg.device = _detect_device(); set_seed(cfg.seed)
        device = torch.device(cfg.device)
        x, y, num_classes = load_reuters(cfg)
        _, x_test, _, y_test = train_test_split(
            x, y, test_size=cfg.test_ratio, random_state=cfg.seed, stratify=y,
        )
    except FileNotFoundError as e:
        print(f"  跳过 reuters: 数据缺失 ({e})")
        return 0

    test_arr = remap_and_pad(x_test, cfg.max_vocab_size, cfg.max_seq_len)
    test_ds = SeqDataset(test_arr, y_test)
    test_loader = data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    ckpt = ROOT / "新闻多分类" / "outputs" / "checkpoints" / "transformer_best.pt"
    if not ckpt.exists():
        print(f"  跳过 reuters: 缺 checkpoint ({ckpt})")
        return 0

    # TransformerMulti constructor: (vocab_size, emb_dim, num_heads, ff_dim, num_layers, num_classes, dropout)
    model = TransformerMulti(
        cfg.max_vocab_size, cfg.embedding_dim, cfg.num_heads,
        cfg.ff_dim, cfg.num_transformer_layers, num_classes, cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    n = 0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            logits, attn = model(x_batch, return_attention=True)
        for i in range(x_batch.size(0)):
            if n >= num_samples:
                return n
            ids = x_batch[i].cpu().numpy()
            real_len = int((ids != 0).sum())
            show = min(max(real_len, 1), max_show)
            tokens = [str(int(t)) for t in ids[:show]]
            head_avg = attn[i].mean(dim=0).cpu().numpy()[:show, :show]
            pred = int(logits[i].argmax().item())
            true_label = int(y_batch[i].item())

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(head_avg, xticklabels=tokens, yticklabels=tokens,
                        cmap="Blues", ax=ax, cbar_kws={"label": "注意力权重"})
            ax.set_title(f"Reuters-46 Transformer - sample {n} (pred={pred}, true={true_label})")
            plt.xticks(rotation=60, fontsize=7)
            plt.yticks(rotation=0, fontsize=7)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"attn_reuters_{n}.png", dpi=150, bbox_inches="tight")
            plt.close()
            n += 1
    return n


def _build_translation_data():
    sys.path.insert(0, str(ROOT / "机器翻译"))
    from machine_translation import (  # type: ignore
        Config, _detect_device, set_seed, load_parallel_pairs, split_pairs,
        Vocabulary, encode_pairs, PAD_IDX, SOS_IDX, EOS_IDX,
    )
    cfg = Config(); cfg.device = _detect_device(); set_seed(cfg.seed)
    device = torch.device(cfg.device)
    pairs = load_parallel_pairs(cfg)
    train_pairs, _, test_pairs = split_pairs(cfg, pairs)
    src_vocab = Vocabulary(max_size=cfg.max_vocab_src, min_freq=cfg.min_token_freq)
    tgt_vocab = Vocabulary(max_size=cfg.max_vocab_tgt, min_freq=cfg.min_token_freq)
    src_vocab.build(p[0] for p in train_pairs)
    tgt_vocab.build(p[1] for p in train_pairs)
    test_src_arr, test_tgt_arr = encode_pairs(test_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)
    return cfg, device, src_vocab, tgt_vocab, test_src_arr, test_tgt_arr, PAD_IDX, SOS_IDX, EOS_IDX


def viz_translation_transformer(num_samples: int = 5) -> int:
    try:
        cfg, device, src_vocab, tgt_vocab, test_src_arr, test_tgt_arr, PAD_IDX, SOS_IDX, EOS_IDX = _build_translation_data()
    except FileNotFoundError as e:
        print(f"  跳过 translation_transformer: 数据缺失 ({e})")
        return 0
    from machine_translation import TransformerTranslator  # type: ignore

    ckpt = ROOT / "机器翻译" / "outputs" / "checkpoints" / "transformer_best.pt"
    if not ckpt.exists():
        print(f"  跳过 translation_transformer: 缺 checkpoint ({ckpt})")
        return 0

    model = TransformerTranslator(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.hidden_dim, nhead=cfg.num_heads,
        num_layers=cfg.num_layers, ff_dim=cfg.ff_dim, dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    n = 0
    for i in range(min(num_samples, len(test_src_arr))):
        src = torch.tensor(test_src_arr[i], dtype=torch.long, device=device).unsqueeze(0)
        tgt_in = torch.tensor(test_tgt_arr[i][:-1], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, cross_attn = model(src, tgt_in, return_attention=True)
        src_tokens = [src_vocab.idx2word.get(int(t), "<?>") for t in test_src_arr[i] if t != PAD_IDX][:25]
        tgt_tokens = [tgt_vocab.idx2word.get(int(t), "<?>") for t in test_tgt_arr[i][:-1] if t != PAD_IDX][:25]
        if not src_tokens or not tgt_tokens:
            continue
        attn = cross_attn[0].mean(dim=0).cpu().numpy()[:len(tgt_tokens), :len(src_tokens)]

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(attn, xticklabels=src_tokens, yticklabels=tgt_tokens,
                    cmap="Blues", ax=ax, cbar_kws={"label": "cross-attn"})
        ax.set_xlabel("源 (西班牙语)")
        ax.set_ylabel("目标 (英语)")
        ax.set_title(f"翻译 Transformer cross-attention - sample {n}")
        plt.xticks(rotation=60, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"attn_translation_transformer_{n}.png", dpi=150, bbox_inches="tight")
        plt.close()
        n += 1
    return n


def viz_translation_seq2seq(num_samples: int = 5) -> int:
    try:
        cfg, device, src_vocab, tgt_vocab, test_src_arr, test_tgt_arr, PAD_IDX, SOS_IDX, EOS_IDX = _build_translation_data()
    except FileNotFoundError as e:
        print(f"  跳过 translation_seq2seq: 数据缺失 ({e})")
        return 0
    from machine_translation import Encoder, Decoder, Seq2Seq  # type: ignore

    ckpt = ROOT / "机器翻译" / "outputs" / "checkpoints" / "seq2seq_best.pt"
    if not ckpt.exists():
        print(f"  跳过 translation_seq2seq: 缺 checkpoint ({ckpt})")
        return 0

    enc = Encoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    dec = Decoder(len(tgt_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    s2s = Seq2Seq(enc, dec, max_len=cfg.max_seq_len).to(device)
    s2s.load_state_dict(torch.load(ckpt, map_location=device))
    s2s.eval()

    n = 0
    for i in range(min(num_samples, len(test_src_arr))):
        src_tensor = torch.tensor(test_src_arr[i], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            encoder_outputs, hidden = s2s.encoder(src_tensor)
        src_mask = (src_tensor != PAD_IDX).long()
        tokens_out: List[int] = [SOS_IDX]
        alphas: List[np.ndarray] = []
        cur = torch.tensor([SOS_IDX], device=device)
        for _ in range(cfg.max_seq_len):
            with torch.no_grad():
                logits, hidden, alpha = s2s.decoder(cur, hidden, encoder_outputs, src_mask)
            alphas.append(alpha[0].cpu().numpy())
            nxt = int(logits.argmax(-1).item())
            tokens_out.append(nxt)
            if nxt == EOS_IDX:
                break
            cur = torch.tensor([nxt], device=device)
        if not alphas:
            continue
        alpha_mat = np.stack(alphas)

        src_tokens = [src_vocab.idx2word.get(int(t), "<?>") for t in test_src_arr[i] if t != PAD_IDX][:25]
        tgt_tokens = [tgt_vocab.idx2word.get(t, "<?>") for t in tokens_out[1:] if t != PAD_IDX][:25]
        if not src_tokens or not tgt_tokens:
            continue
        alpha_show = alpha_mat[:len(tgt_tokens), :len(src_tokens)]

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(alpha_show, xticklabels=src_tokens, yticklabels=tgt_tokens,
                    cmap="Blues", ax=ax, cbar_kws={"label": "Bahdanau α"})
        ax.set_xlabel("源 (西班牙语)")
        ax.set_ylabel("目标 (英语)")
        ax.set_title(f"翻译 Seq2Seq Bahdanau α - sample {n}")
        plt.xticks(rotation=60, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"attn_translation_seq2seq_{n}.png", dpi=150, bbox_inches="tight")
        plt.close()
        n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--tasks", default="sentiment,reuters,translation_transformer,translation_seq2seq")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    counters = {}
    if "sentiment" in tasks:
        counters["sentiment"] = viz_sentiment(args.num_samples)
    if "reuters" in tasks:
        counters["reuters"] = viz_reuters(args.num_samples)
    if "translation_transformer" in tasks:
        counters["translation_transformer"] = viz_translation_transformer(args.num_samples)
    if "translation_seq2seq" in tasks:
        counters["translation_seq2seq"] = viz_translation_seq2seq(args.num_samples)
    total = sum(counters.values())
    print(f"\n已写入 {total} 张热图到 {FIG_DIR}")
    for t, n in counters.items():
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
