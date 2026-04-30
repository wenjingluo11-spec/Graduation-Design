#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解码策略网格搜索 (T16):

- 加载 机器翻译/outputs/checkpoints/seq2seq_best.pt 与 transformer_best.pt
- 对测试集运行 beam_size x length_penalty 网格,计算 BLEU-1/2/4
- 输出: 机器翻译/outputs/translation_decode_grid.csv

用法:
    python analyses/decode_grid.py
    python analyses/decode_grid.py --ckpt-dir 机器翻译/outputs_smoke/checkpoints  # 用 smoke 验证
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# MPS fallback must be set before importing torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "机器翻译"))

# 从 machine_translation 复用一切
from machine_translation import (  # type: ignore
    Config,
    Vocabulary,
    TranslationDataset,
    Encoder,
    Decoder,
    Seq2Seq,
    TransformerTranslator,
    compute_bleu_scores,
    decode_seq2seq_beam,
    decode_transformer_beam,
    encode_pairs,
    ids_to_tokens,
    load_parallel_pairs,
    set_seed,
    split_pairs,
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="解码策略网格搜索: beam_size x length_penalty")
    p.add_argument(
        "--ckpt-dir",
        default=str(ROOT / "机器翻译" / "outputs" / "checkpoints"),
        help="checkpoint 目录,需含 seq2seq_best.pt 与 transformer_best.pt",
    )
    p.add_argument(
        "--output-csv",
        default=str(ROOT / "机器翻译" / "outputs" / "translation_decode_grid.csv"),
    )
    p.add_argument("--beam-sizes", default="1,3,5", help="逗号分隔的 beam size 列表")
    p.add_argument(
        "--length-penalties", default="0.6,0.8,1.0,1.2", help="逗号分隔的 length penalty 列表"
    )
    return p.parse_args()


def _load_cfg_from_ckpt_dir(ckpt_dir: Path) -> Config:
    """Try to load translation_config.json from the outputs directory (parent of checkpoints/).
    Falls back to default Config() if not found."""
    config_path = ckpt_dir.parent / "translation_config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        # Build a Config, overriding fields present in the JSON
        cfg = Config()
        for key, val in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        print(f"已从 {config_path} 读取超参数配置")
    else:
        cfg = Config()
        print(f"未找到 {config_path}，使用默认 Config()")
    return cfg


def main() -> None:
    args = parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_s2s = ckpt_dir / "seq2seq_best.pt"
    ckpt_tf = ckpt_dir / "transformer_best.pt"

    if not ckpt_s2s.exists():
        sys.exit(
            f"缺少 Seq2Seq checkpoint: {ckpt_s2s}\n"
            "请先跑一次完整翻译训练或用 --ckpt-dir 指向有 checkpoint 的目录"
        )
    if not ckpt_tf.exists():
        sys.exit(f"缺少 Transformer checkpoint: {ckpt_tf}")

    # Load config that matches the checkpoint dimensions
    cfg = _load_cfg_from_ckpt_dir(ckpt_dir)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ------------------------------------------------------------------ #
    # 数据准备 (与 run_experiment 一致)                                     #
    # ------------------------------------------------------------------ #
    print("加载平行语料...")
    pairs = load_parallel_pairs(cfg)
    train_pairs, _, test_pairs = split_pairs(cfg, pairs)

    src_vocab = Vocabulary(max_size=cfg.max_vocab_src, min_freq=cfg.min_token_freq)
    tgt_vocab = Vocabulary(max_size=cfg.max_vocab_tgt, min_freq=cfg.min_token_freq)
    src_vocab.build([s for s, _ in train_pairs])
    tgt_vocab.build([t for _, t in train_pairs])
    print(f"src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}, test_pairs={len(test_pairs)}")

    src_test, tgt_test = encode_pairs(test_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)

    # ------------------------------------------------------------------ #
    # 加载 Seq2Seq                                                         #
    # ------------------------------------------------------------------ #
    enc = Encoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    dec = Decoder(len(tgt_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    s2s = Seq2Seq(enc, dec, max_len=cfg.max_seq_len).to(device)
    s2s.load_state_dict(torch.load(ckpt_s2s, map_location=device, weights_only=True))
    s2s.eval()
    print(f"Seq2Seq checkpoint 已加载: {ckpt_s2s}")

    # ------------------------------------------------------------------ #
    # 加载 Transformer                                                     #
    # NOTE: run_experiment uses d_model=cfg.hidden_dim, not embedding_dim #
    # ------------------------------------------------------------------ #
    tf = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.hidden_dim,
        nhead=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    ).to(device)
    tf.load_state_dict(torch.load(ckpt_tf, map_location=device, weights_only=True))
    tf.eval()
    print(f"Transformer checkpoint 已加载: {ckpt_tf}")

    # ------------------------------------------------------------------ #
    # 解码网格                                                             #
    # compute_bleu_scores(refs_int_ids, hyps_int_ids, vocab)              #
    # decode_*_beam(model, src_tensor_1d, beam_size, length_penalty, max_len)
    # ------------------------------------------------------------------ #
    beam_sizes = [int(b) for b in args.beam_sizes.split(",")]
    length_penalties = [float(lp) for lp in args.length_penalties.split(",")]

    # Build refs as list of int-id sequences (strip special tokens from tgt)
    refs_ids = [
        [int(t) for t in tgt_test[i] if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
        for i in range(len(tgt_test))
    ]

    src_tensors = [
        torch.tensor(src_test[i], dtype=torch.long, device=device)
        for i in range(len(src_test))
    ]

    rows = []
    total_combos = len(beam_sizes) * len(length_penalties) * 2
    combo_idx = 0

    for beam in beam_sizes:
        for lp in length_penalties:
            for model_name, model, decode_fn in [
                ("Seq2Seq+Attention", s2s, decode_seq2seq_beam),
                ("Transformer", tf, decode_transformer_beam),
            ]:
                combo_idx += 1
                print(
                    f"[{combo_idx}/{total_combos}] {model_name} beam={beam} lp={lp:.2f} ..."
                )
                hyps_ids = []
                for src_t in src_tensors:
                    # decode_*_beam signature: (model, src_tensor_1d, beam_size, length_penalty, max_len)
                    ids = decode_fn(
                        model,
                        src_t,
                        beam,
                        lp,
                        cfg.max_seq_len,
                    )
                    hyps_ids.append(ids)

                bleu = compute_bleu_scores(refs_ids, hyps_ids, tgt_vocab)
                rows.append(
                    {
                        "model": model_name,
                        "beam": beam,
                        "length_penalty": lp,
                        "bleu1": bleu["bleu1"],
                        "bleu2": bleu["bleu2"],
                        "bleu4": bleu["bleu4"],
                    }
                )
                print(
                    f"    BLEU-1={bleu['bleu1']:.4f}  BLEU-2={bleu['bleu2']:.4f}  BLEU-4={bleu['bleu4']:.4f}"
                )

    df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已写入 {out_path} ({len(df)} 行)")
    print(df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
