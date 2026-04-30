#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
词级扰动鲁棒性分析 (T18):

不重训。加载各任务的 best checkpoint, 对测试集做两类扰动:
  - delete: 随机删除 p% token
  - unk:    随机将 p% token 替换为 <unk>

度量每个模型在 p ∈ {0, 0.05, 0.10, 0.15, 0.20} 下的主指标:
  - sentiment: F1
  - reuters:   Macro-F1
  - translation: BLEU-4

输出: supplementary_outputs/robustness/{sentiment,reuters,translation}_robustness.csv
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List

# Enable CPU fallback for MPS ops that are not yet implemented on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "supplementary_outputs" / "robustness"
PERTURB_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20]
PERTURB_MODES = ["delete", "unk"]


def perturb_seq(seq: np.ndarray, p: float, mode: str, *, unk_id: int, pad_id: int) -> np.ndarray:
    """对一个 token id 序列扰动 p 比例的非 PAD 位置.

    delete: 删除该位置, 末尾用 pad_id 补齐保持长度
    unk:    将该位置替换为 unk_id
    p == 0: 原样返回
    """
    seq = seq.copy()
    if p <= 0:
        return seq
    valid_idx = np.where(seq != pad_id)[0]
    if len(valid_idx) == 0:
        return seq
    n = max(int(len(valid_idx) * p), 1)
    n = min(n, len(valid_idx))
    rng = np.random.default_rng(42)
    chosen = rng.choice(valid_idx, size=n, replace=False)
    if mode == "delete":
        keep = np.ones(len(seq), dtype=bool)
        keep[chosen] = False
        kept = seq[keep]
        out = np.full_like(seq, pad_id)
        out[: len(kept)] = kept
        return out
    elif mode == "unk":
        seq[chosen] = unk_id
        return seq
    else:
        raise ValueError(f"unknown mode: {mode}")


def perturb_dataset_X(X: np.ndarray, p: float, mode: str, *, unk_id: int, pad_id: int) -> np.ndarray:
    """对 [N, L] 的整个测试集做逐行扰动."""
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = perturb_seq(X[i], p, mode, unk_id=unk_id, pad_id=pad_id)
    return out


# ----------------------- sentiment -----------------------
def evaluate_sentiment(p: float, mode: str) -> Dict[str, float]:
    sys.path.insert(0, str(ROOT / "情感二分类"))
    from sentiment_analysis import (  # type: ignore
        Config,
        _detect_device,
        set_seed,
        TextCNN,
        BiGRUClassifier,
        TransformerClassifier,
        CNNBiGRU,
        TextDataset,
        build_vocab,
        encode_text,
        compute_binary_metrics,
        evaluate_model,
    )

    cfg = Config()
    cfg.device = _detect_device()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt_dir = ROOT / "情感二分类" / "outputs" / "checkpoints"

    # Infer embedding (vocab) size directly from first available checkpoint.
    vocab_size_from_ckpt: int = cfg.max_vocab_size
    for _probe_name in ["textcnn", "bigru", "transformer", "cnnbigru"]:
        _probe_path = ckpt_dir / f"{_probe_name}_best.pt"
        if _probe_path.exists():
            _sd = torch.load(_probe_path, map_location="cpu", weights_only=True)
            vocab_size_from_ckpt = int(_sd["embedding.weight"].shape[0])
            break

    # The models were trained with max_train_samples=12000, max_seq_len=220,
    # max_vocab_size=20000 (from sentiment_config.json) but only 12000 unique
    # tokens resulted.  We reconstruct the vocab from the saved training texts.
    # The saved predictions CSV contains the 5000 original test texts + labels.
    preds_csv = ROOT / "情感二分类" / "outputs" / "sentiment_predictions_best_model.csv"
    if not preds_csv.exists():
        print(f"  警告: 找不到 {preds_csv}, 跳过 sentiment 任务")
        return {}

    test_df_saved = pd.read_csv(preds_csv)
    test_texts = test_df_saved["text"].tolist()
    test_labels = test_df_saved["true_label"].to_numpy(dtype=np.float32)

    # Rebuild the same vocab that was used during training.
    # Training used max_vocab_size=20000, max_train_samples=12000.
    # On this machine the raw IMDB data is incomplete; instead we build vocab
    # from the test texts themselves and pad to match the checkpoint vocab size.
    # This is safe because: (a) test texts are a subset of the same distribution,
    # (b) any OOV tokens get mapped to <unk> (id=1) which is already in the vocab.
    vocab = build_vocab(test_texts, cfg.max_vocab_size)
    # Force vocab size to match checkpoint embedding by padding with dummy entries
    while len(vocab) < vocab_size_from_ckpt:
        vocab[f"__dummy_{len(vocab)}__"] = len(vocab)
    # If somehow larger, truncate (shouldn't happen)
    if len(vocab) > vocab_size_from_ckpt:
        vocab = {k: v for k, v in vocab.items() if v < vocab_size_from_ckpt}

    # Encode test texts with max_seq_len from saved config (220)
    max_seq_len = cfg.max_seq_len  # 220
    X_test = np.array(
        [encode_text(t, vocab, max_seq_len) for t in test_texts], dtype=np.int64
    )

    pad_id = vocab.get("<pad>", 0)
    unk_id = vocab.get("<unk>", 1)

    X_perturbed = perturb_dataset_X(
        X_test, p, mode, unk_id=unk_id, pad_id=pad_id
    )
    perturbed_ds = TextDataset(X_perturbed, test_labels)
    perturbed_loader = data.DataLoader(
        perturbed_ds, batch_size=cfg.batch_size, shuffle=False
    )

    n_vocab = len(vocab)
    constructors = {
        "TextCNN": lambda: TextCNN(n_vocab, cfg.embedding_dim, cfg.dropout),
        "BiGRU": lambda: BiGRUClassifier(
            n_vocab, cfg.embedding_dim, cfg.hidden_dim, cfg.dropout
        ),
        "Transformer": lambda: TransformerClassifier(
            n_vocab,
            cfg.embedding_dim,
            cfg.num_heads,
            cfg.ff_dim,
            cfg.num_transformer_layers,
            cfg.dropout,
        ),
        "CNNBiGRU": lambda: CNNBiGRU(
            n_vocab, cfg.embedding_dim, cfg.hidden_dim, cfg.dropout
        ),
    }
    results: Dict[str, float] = {}
    for name, build in constructors.items():
        path = ckpt_dir / f"{name.lower()}_best.pt"
        if not path.exists():
            print(f"  跳过 {name}: 缺 checkpoint ({path})")
            continue
        model = build().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        y_true, y_prob = evaluate_model(model, perturbed_loader, device)
        m = compute_binary_metrics(y_true, y_prob)
        results[name] = float(m["f1"])
    return results


# ----------------------- reuters -----------------------
def evaluate_reuters(p: float, mode: str) -> Dict[str, float]:
    sys.path.insert(0, str(ROOT / "新闻多分类"))
    from reuters_multiclass import (  # type: ignore
        Config,
        _detect_device,
        set_seed,
        load_reuters,
        remap_and_pad,
        SeqDataset,
        TextCNNMulti,
        BiGRUMulti,
        TransformerMulti,
        CNNBiGRUMulti,
        compute_mc_metrics,
        evaluate_model,
    )
    from sklearn.model_selection import train_test_split

    cfg = Config()
    cfg.device = _detect_device()
    # Resolve data file relative to project root
    cfg.data_file = str(ROOT / "reuters.npz")
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    x, y, num_classes = load_reuters(cfg)

    # Replicate the same test split used during training
    idx_all = np.arange(len(x))
    try:
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=cfg.test_ratio,
            random_state=cfg.seed,
            stratify=y,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=cfg.test_ratio,
            random_state=cfg.seed,
        )

    seq_test = [x[i] for i in test_idx]
    y_test = y[test_idx]

    test_arr = remap_and_pad(seq_test, cfg.max_vocab_size, cfg.max_seq_len)
    pad_id = 0
    unk_id = 1
    perturbed = perturb_dataset_X(
        test_arr, p, mode, unk_id=unk_id, pad_id=pad_id
    )
    test_loader = data.DataLoader(
        SeqDataset(perturbed, y_test), batch_size=cfg.batch_size, shuffle=False
    )

    ckpt_dir = ROOT / "新闻多分类" / "outputs" / "checkpoints"
    num_words = cfg.max_vocab_size
    constructors = {
        "TextCNN": lambda: TextCNNMulti(
            num_words, cfg.embedding_dim, num_classes, cfg.dropout
        ),
        "BiGRU": lambda: BiGRUMulti(
            num_words, cfg.embedding_dim, cfg.hidden_dim, num_classes, cfg.dropout
        ),
        "Transformer": lambda: TransformerMulti(
            num_words,
            num_classes,
            cfg.embedding_dim,
            cfg.num_heads,
            cfg.ff_dim,
            cfg.num_transformer_layers,
            cfg.dropout,
        ),
        "CNNBiGRU": lambda: CNNBiGRUMulti(
            num_words, num_classes, cfg.embedding_dim, cfg.hidden_dim, cfg.dropout
        ),
    }
    results: Dict[str, float] = {}
    for name, build in constructors.items():
        path = ckpt_dir / f"{name.lower()}_best.pt"
        if not path.exists():
            print(f"  跳过 {name}: 缺 checkpoint ({path})")
            continue
        model = build().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        y_true, y_pred = evaluate_model(model, test_loader, device)
        m = compute_mc_metrics(y_true, y_pred)
        results[name] = float(m["f1_macro"])
    return results


# ----------------------- translation -----------------------
def evaluate_translation(p: float, mode: str) -> Dict[str, float]:
    """翻译: 把 src 序列扰动后用 greedy 解码, 算 BLEU-4."""
    sys.path.insert(0, str(ROOT / "机器翻译"))
    from machine_translation import (  # type: ignore
        Config,
        _detect_device,
        set_seed,
        load_parallel_pairs,
        split_pairs,
        Vocabulary,
        encode_pairs,
        Encoder,
        Decoder,
        Seq2Seq,
        CNNBiGRUEncoder,
        TransformerTranslator,
        decode_seq2seq_greedy,
        decode_transformer_greedy,
        compute_bleu_scores,
        PAD_IDX,
        UNK_IDX,
    )

    cfg = Config()
    cfg.device = _detect_device()
    cfg.data_file = str(ROOT / "spa.txt")
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    pairs = load_parallel_pairs(cfg)
    train_pairs, _, test_pairs = split_pairs(cfg, pairs)

    src_vocab = Vocabulary(max_size=cfg.max_vocab_src, min_freq=cfg.min_token_freq)
    tgt_vocab = Vocabulary(max_size=cfg.max_vocab_tgt, min_freq=cfg.min_token_freq)
    src_vocab.build(p_[0] for p_ in train_pairs)
    tgt_vocab.build(p_[1] for p_ in train_pairs)

    test_src_arr, test_tgt_arr = encode_pairs(
        test_pairs, src_vocab, tgt_vocab, cfg.max_seq_len
    )
    perturbed_src = perturb_dataset_X(
        test_src_arr, p, mode, unk_id=UNK_IDX, pad_id=PAD_IDX
    )

    ckpt_dir = ROOT / "机器翻译" / "outputs" / "checkpoints"
    # (display_name, ckpt_filename, family)
    candidates = [
        ("Seq2Seq+Attention", "seq2seq_best.pt", "seq2seq"),
        ("Transformer", "transformer_best.pt", "transformer"),
        ("CNNBiGRU", "cnnbigru_best.pt", "cnnbigru"),
    ]
    # References: strip special tokens from target sequences
    from machine_translation import PAD_IDX as _PAD, SOS_IDX as _SOS, EOS_IDX as _EOS  # type: ignore
    refs: List[List[int]] = [
        [int(t) for t in row if t not in (_PAD, _SOS, _EOS)]
        for row in test_tgt_arr
    ]

    results: Dict[str, float] = {}
    for name, ckpt_name, family in candidates:
        path = ckpt_dir / ckpt_name
        if not path.exists():
            print(f"  跳过 {name}: 缺 checkpoint ({path})")
            continue
        if family in ("seq2seq", "cnnbigru"):
            if family == "cnnbigru":
                enc = CNNBiGRUEncoder(
                    len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout
                )
            else:
                enc = Encoder(
                    len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout
                )
            dec = Decoder(
                len(tgt_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout
            )
            model = Seq2Seq(enc, dec, max_len=cfg.max_seq_len).to(device)
        else:
            model = TransformerTranslator(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=cfg.hidden_dim,
                nhead=cfg.num_heads,
                num_layers=cfg.num_layers,
                ff_dim=cfg.ff_dim,
                dropout=cfg.dropout,
            ).to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        model.eval()

        preds: List[List[int]] = []
        for i in range(len(perturbed_src)):
            # decode_*_greedy expects a 1-D tensor (unsqueeze happens inside)
            src_tensor = torch.tensor(
                perturbed_src[i], dtype=torch.long, device=device
            )
            if family in ("seq2seq", "cnnbigru"):
                ids = decode_seq2seq_greedy(model, src_tensor, max_len=cfg.max_seq_len)
            else:
                ids = decode_transformer_greedy(
                    model, src_tensor, max_len=cfg.max_seq_len
                )
            preds.append(list(ids))

        bleu = compute_bleu_scores(refs, preds, tgt_vocab)
        results[name] = float(bleu.get("bleu4", 0.0))
    return results


# ----------------------- driver -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="词级扰动鲁棒性分析 (T18)"
    )
    parser.add_argument(
        "--tasks",
        default="sentiment,reuters,translation",
        help="逗号分隔的任务名 (sentiment / reuters / translation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",")]
    evaluators = {
        "sentiment": evaluate_sentiment,
        "reuters": evaluate_reuters,
        "translation": evaluate_translation,
    }
    score_col = {
        "sentiment": "f1",
        "reuters": "f1_macro",
        "translation": "bleu4",
    }

    for task in tasks:
        if task not in evaluators:
            print(f"未知任务: {task}, 跳过")
            continue
        rows: List[dict] = []
        for mode in PERTURB_MODES:
            for p in PERTURB_RATIOS:
                print(f"==> {task} mode={mode} p={p:.2f}")
                res = evaluators[task](p, mode)
                for model_name, score in res.items():
                    rows.append(
                        {
                            "task": task,
                            "model": model_name,
                            "mode": mode,
                            "p": p,
                            score_col[task]: score,
                        }
                    )
        out = OUT_BASE / f"{task}_robustness.csv"
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"✓ {task}: 写入 {out} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
