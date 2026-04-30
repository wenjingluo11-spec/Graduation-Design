#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器翻译实验（Spanish -> English）

符合开题报告范围：
- 任务：西班牙语到英语机器翻译
- 模型：Seq2Seq + Bahdanau Attention、Transformer
- 指标：BLEU-1 / BLEU-2 / BLEU-4

说明：
- 默认采用中等规模子集，兼顾可复现与资源消耗
- 支持 --quick 快速跑通
"""

from __future__ import annotations

import argparse
import json
import math
import os

# MPS fallback 必须在 import torch 之前设置.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from sklearn.model_selection import train_test_split


def _detect_device() -> str:
    """三级设备 fallback: mps → cuda → cpu"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    data_file: str = "../spa.txt"
    output_dir: str = "outputs"
    seed: int = 42
    max_samples: int = 32000
    max_seq_len: int = 24
    max_vocab_src: int = 18000
    max_vocab_tgt: int = 18000
    min_token_freq: int = 1
    batch_size: int = 64
    embedding_dim: int = 192
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    epochs_seq2seq: int = 20
    epochs_transformer: int = 20
    lr_seq2seq: float = 1e-3
    lr_transformer: float = 8e-4
    early_stopping_patience_seq2seq: int = 5
    early_stopping_patience_transformer: int = 5
    early_stopping_min_delta_bleu: float = 1e-4
    teacher_forcing_ratio: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    bleu_eval_every: int = 1
    bleu_val_sample_size: int = 600
    beam_size: int = 4
    length_penalty: float = 0.7
    quick: bool = False
    device: str = field(default_factory=_detect_device)
    include_hybrid: bool = False
    label_smoothing: float = 0.0
    output_suffix: str = ""


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Spanish->English 机器翻译实验")
    parser.add_argument("--data-file", type=str, default="../spa.txt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-seq2seq", type=int, default=20)
    parser.add_argument("--epochs-transformer", type=int, default=20)
    parser.add_argument("--patience-seq2seq", type=int, default=5)
    parser.add_argument("--patience-transformer", type=int, default=5)
    parser.add_argument("--min-delta-bleu", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-hybrid", action="store_true", help="加入 CNN-BiGRU 混合 encoder")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="0 = 标准 CE; 0.1 是 NMT 经典默认值")
    parser.add_argument("--output-suffix", type=str, default="", help="附加到所有输出文件名后,便于 LS 与 baseline 并存")
    args = parser.parse_args()

    cfg = Config(
        data_file=args.data_file,
        output_dir=args.output_dir,
        seed=args.seed,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs_seq2seq=args.epochs_seq2seq,
        epochs_transformer=args.epochs_transformer,
        early_stopping_patience_seq2seq=args.patience_seq2seq,
        early_stopping_patience_transformer=args.patience_transformer,
        early_stopping_min_delta_bleu=args.min_delta_bleu,
        quick=args.quick,
        include_hybrid=args.include_hybrid,
        label_smoothing=args.label_smoothing,
        output_suffix=args.output_suffix,
    )
    if args.device is not None:
        cfg.device = args.device  # 用户显式指定即生效

    if cfg.quick:
        cfg.max_samples = min(cfg.max_samples, 7000)
        cfg.batch_size = 48
        cfg.max_seq_len = 20
        cfg.embedding_dim = 128
        cfg.hidden_dim = 160
        cfg.ff_dim = 256
        cfg.epochs_seq2seq = 2
        cfg.epochs_transformer = 2
        cfg.early_stopping_patience_seq2seq = 2
        cfg.early_stopping_patience_transformer = 2
        cfg.bleu_val_sample_size = 240

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:
    def __init__(self, max_size: int, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2word: Dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

    def build(self, sentences: Sequence[str]) -> None:
        from collections import Counter

        counter = Counter()
        for sent in sentences:
            counter.update(sent.split())

        words = [w for w, c in counter.items() if c >= self.min_freq]
        words = sorted(words, key=lambda w: (-counter[w], w))
        cap = max(0, self.max_size - len(SPECIAL_TOKENS))
        words = words[:cap]

        for w in words:
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

    def encode(self, sentence: str, max_len: int) -> List[int]:
        ids = [SOS_IDX]
        ids.extend(self.word2idx.get(tok, UNK_IDX) for tok in sentence.split())
        ids.append(EOS_IDX)
        if len(ids) < max_len:
            ids.extend([PAD_IDX] * (max_len - len(ids)))
        else:
            ids = ids[: max_len - 1] + [EOS_IDX]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        words: List[str] = []
        for idx in ids:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            words.append(self.idx2word.get(int(idx), "<unk>"))
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.lower().strip())
    s = re.sub(r"([.!?¿¡])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Záéíóúüñ¿¡.!?\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def resolve_data_file(cfg: Config) -> Path:
    candidates = [
        Path(cfg.data_file),
        Path(__file__).resolve().parent / cfg.data_file,
        Path(__file__).resolve().parent.parent / "spa.txt",
        Path.cwd() / "spa.txt",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"找不到数据文件 spa.txt，尝试过: {candidates}")


def load_parallel_pairs(cfg: Config) -> List[Tuple[str, str]]:
    data_file = resolve_data_file(cfg)
    pairs: List[Tuple[str, str]] = []

    with data_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            # ManyThings: english\tspanish\tmetadata
            eng = normalize_text(parts[0])
            spa = normalize_text(parts[1])
            if not eng or not spa:
                continue
            # 对齐开题报告：Spanish -> English
            src_spa = spa
            tgt_eng = eng
            if len(src_spa.split()) <= (cfg.max_seq_len - 2) and len(tgt_eng.split()) <= (cfg.max_seq_len - 2):
                pairs.append((src_spa, tgt_eng))

    random.shuffle(pairs)
    if cfg.max_samples > 0:
        pairs = pairs[: cfg.max_samples]
    return pairs


def split_pairs(cfg: Config, pairs: Sequence[Tuple[str, str]]):
    idx = np.arange(len(pairs))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )
    val_ratio_adjusted = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_ratio_adjusted,
        random_state=cfg.seed,
        shuffle=True,
    )
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    return train_pairs, val_pairs, test_pairs


class TranslationDataset(data.Dataset):
    def __init__(self, src: np.ndarray, tgt: np.ndarray):
        self.src = torch.tensor(src, dtype=torch.long)
        self.tgt = torch.tensor(tgt, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx]


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(emb)
        return outputs, hidden


class CNNBiGRUEncoder(nn.Module):
    """Stacked CNN-BiGRU encoder for Seq2Seq.

    Conv1d (k=3, pad=1) over the embedding, then bi-GRU.
    Output projected back to ``hidden_dim`` so it is a drop-in replacement
    for ``Encoder`` (BahdanauAttention + Decoder need not change).
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.bigru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # Project bi-GRU outputs and hidden back to hidden_dim
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hid_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(src))                       # [B, L, E]
        conv_out = torch.relu(self.conv(emb.transpose(1, 2)))         # [B, E, L]
        gru_in = conv_out.transpose(1, 2)                             # [B, L, E]
        outputs, hidden = self.bigru(gru_in)                          # outputs: [B, L, 2H]; hidden: [2, B, H]
        outputs = self.out_proj(outputs)                              # [B, L, H]
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)         # [B, 2H]
        new_hidden = torch.tanh(self.hid_proj(hidden_cat)).unsqueeze(0)  # [1, B, H]
        return outputs, new_hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # hidden: [1, B, H] -> [B, 1, H]
        dec = hidden.transpose(0, 1)
        score = self.v(torch.tanh(self.w_enc(encoder_outputs) + self.w_dec(dec))).squeeze(-1)
        score = score.masked_fill(src_mask == 0, -1e9)
        attn = F.softmax(score, dim=1)
        return attn


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.attn = BahdanauAttention(hidden_dim)
        self.gru = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_token: [B]
        emb = self.dropout(self.embedding(input_token)).unsqueeze(1)  # [B,1,E]
        attn = self.attn(hidden, encoder_outputs, src_mask).unsqueeze(1)  # [B,1,S]
        context = torch.bmm(attn, encoder_outputs)  # [B,1,H]
        gru_input = torch.cat([emb, context], dim=2)
        output, hidden = self.gru(gru_input, hidden)
        logits = self.fc(torch.cat([output.squeeze(1), context.squeeze(1), emb.squeeze(1)], dim=1))
        return logits, hidden, attn.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, max_len: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = max_len

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        bsz, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(bsz, tgt_len, vocab_size, device=src.device)

        encoder_outputs, hidden = self.encoder(src)
        src_mask = (src != PAD_IDX)

        dec_input = tgt[:, 0]
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = logits
            teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[:, t] if teacher else logits.argmax(dim=1)
        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        src_mask = None
        tgt_len = tgt_in.size(1)
        tgt_causal_mask = torch.triu(
            torch.ones((tgt_len, tgt_len), device=src.device, dtype=torch.bool),
            diagonal=1,
        )
        src_key_padding_mask = src.eq(PAD_IDX)
        tgt_key_padding_mask = tgt_in.eq(PAD_IDX)

        src_emb = self.pos(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.fc(out)
        if return_attention:
            # Re-run encoder + last decoder cross-attn to extract attention weights
            memory = self.transformer.encoder(
                src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
            )
            last_dec = self.transformer.decoder.layers[-1]
            with torch.no_grad():
                _, cross_attn = last_dec.multihead_attn(
                    tgt_emb, memory, memory,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True, average_attn_weights=False,
                )
            return logits, cross_attn  # cross_attn: [B, num_heads, T_tgt, T_src]
        return logits


def ids_to_tokens(ids: Sequence[int], vocab: Vocabulary) -> List[str]:
    tokens: List[str] = []
    for idx in ids:
        if idx == EOS_IDX:
            break
        if idx in (PAD_IDX, SOS_IDX):
            continue
        tokens.append(vocab.idx2word.get(int(idx), "<unk>"))
    return tokens


def compute_bleu_scores(
    references: Sequence[Sequence[int]],
    hypotheses: Sequence[Sequence[int]],
    vocab: Vocabulary,
) -> Dict[str, float]:
    if len(references) == 0:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

    refs = [[ids_to_tokens(ref, vocab)] for ref in references]
    hyps = [ids_to_tokens(hyp, vocab) for hyp in hypotheses]
    smooth = SmoothingFunction().method1

    b1 = corpus_bleu(refs, hyps, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth)
    b2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth)
    b4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return {"bleu1": float(b1), "bleu2": float(b2), "bleu4": float(b4)}


def train_seq2seq_epoch(
    model: Seq2Seq,
    loader: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> float:
    model.train()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(logits[:, 1:, :].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def decode_seq2seq_greedy(model: Seq2Seq, src_tensor: torch.Tensor, max_len: int) -> List[int]:
    model.eval()
    src = src_tensor.unsqueeze(0)
    encoder_outputs, hidden = model.encoder(src)
    src_mask = src.ne(PAD_IDX)
    token = torch.tensor([SOS_IDX], device=src.device)
    out_ids: List[int] = []
    for _ in range(max_len):
        logits, hidden, _ = model.decoder(token, hidden, encoder_outputs, src_mask)
        next_id = int(logits.argmax(dim=1).item())
        if next_id == EOS_IDX:
            break
        out_ids.append(next_id)
        token = torch.tensor([next_id], device=src.device)
    return out_ids


@torch.no_grad()
def decode_seq2seq_beam(
    model: Seq2Seq,
    src_tensor: torch.Tensor,
    beam_size: int,
    length_penalty: float,
    max_len: int,
) -> List[int]:
    model.eval()
    src = src_tensor.unsqueeze(0)
    encoder_outputs, hidden = model.encoder(src)
    src_mask = src.ne(PAD_IDX)

    beams: List[Tuple[List[int], torch.Tensor, float, bool]] = [([SOS_IDX], hidden, 0.0, False)]

    for _ in range(max_len):
        candidates: List[Tuple[List[int], torch.Tensor, float, bool]] = []
        for seq, hid, score, done in beams:
            if done:
                candidates.append((seq, hid, score, True))
                continue
            token = torch.tensor([seq[-1]], device=src.device)
            logits, next_hidden, _ = model.decoder(token, hid, encoder_outputs, src_mask)
            log_prob = F.log_softmax(logits, dim=1).squeeze(0)
            topv, topi = torch.topk(log_prob, beam_size)
            for lp, idx in zip(topv.tolist(), topi.tolist()):
                new_seq = seq + [int(idx)]
                done_flag = int(idx) == EOS_IDX
                candidates.append((new_seq, next_hidden.clone(), score + float(lp), done_flag))

        def score_fn(item):
            seq, _, s, _ = item
            length = max(1, len(seq) - 1)
            return s / (length ** length_penalty)

        beams = sorted(candidates, key=score_fn, reverse=True)[:beam_size]
        if all(done for _, _, _, done in beams):
            break

    best = beams[0][0]
    decoded: List[int] = []
    for tid in best[1:]:
        if tid == EOS_IDX:
            break
        decoded.append(tid)
    return decoded


@torch.no_grad()
def evaluate_seq2seq_bleu(
    model: Seq2Seq,
    src_data: np.ndarray,
    tgt_data: np.ndarray,
    tgt_vocab: Vocabulary,
    decode_method: str,
    cfg: Config,
    sample_size: int | None = None,
) -> Dict[str, float]:
    idxs = list(range(len(src_data)))
    if sample_size is not None and sample_size < len(idxs):
        idxs = random.sample(idxs, sample_size)

    refs: List[List[int]] = []
    hyps: List[List[int]] = []
    device = torch.device(cfg.device)

    for i in idxs:
        src_tensor = torch.tensor(src_data[i], dtype=torch.long, device=device)
        if decode_method == "beam":
            pred = decode_seq2seq_beam(model, src_tensor, cfg.beam_size, cfg.length_penalty, cfg.max_seq_len)
        else:
            pred = decode_seq2seq_greedy(model, src_tensor, cfg.max_seq_len)
        ref = [int(t) for t in tgt_data[i] if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
        refs.append(ref)
        hyps.append(pred)
    return compute_bleu_scores(refs, hyps, tgt_vocab)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label-smoothed CE with PAD ignore.

    Reference: Inception-V3 / "Rethinking the Inception Architecture for Computer Vision".
    For a target class t and num_classes N: the smoothed distribution puts
    (1 - epsilon) on t and epsilon / (N - 1) on every other class.
    PAD positions (target == ignore_index) contribute 0 to the loss.
    """
    def __init__(self, epsilon: float = 0.1, ignore_index: int = PAD_IDX):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B*T, V] — flattened; target: [B*T]
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            mask = (target != self.ignore_index)
            true_dist = torch.full_like(log_probs, self.epsilon / max(n_classes - 1, 1))
            target_safe = target.clamp(min=0)  # avoid -1 indices for ignored entries
            true_dist.scatter_(1, target_safe.unsqueeze(1), 1 - self.epsilon)
        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = loss * mask.float()
        denom = mask.float().sum().clamp(min=1)
        return loss.sum() / denom


def train_transformer_epoch(
    model: TransformerTranslator,
    loader: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def decode_transformer_greedy(
    model: TransformerTranslator,
    src_tensor: torch.Tensor,
    max_len: int,
) -> List[int]:
    model.eval()
    src = src_tensor.unsqueeze(0)
    gen = [SOS_IDX]
    for _ in range(max_len):
        tgt_in = torch.tensor([gen], dtype=torch.long, device=src.device)
        logits = model(src, tgt_in)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == EOS_IDX:
            break
        gen.append(nxt)
    return gen[1:]


@torch.no_grad()
def decode_transformer_beam(
    model: TransformerTranslator,
    src_tensor: torch.Tensor,
    beam_size: int,
    length_penalty: float,
    max_len: int,
) -> List[int]:
    model.eval()
    src = src_tensor.unsqueeze(0)
    beams: List[Tuple[List[int], float, bool]] = [([SOS_IDX], 0.0, False)]

    for _ in range(max_len):
        cands: List[Tuple[List[int], float, bool]] = []
        for seq, score, done in beams:
            if done:
                cands.append((seq, score, True))
                continue
            tgt_in = torch.tensor([seq], dtype=torch.long, device=src.device)
            logits = model(src, tgt_in)
            log_prob = F.log_softmax(logits[0, -1], dim=0)
            topv, topi = torch.topk(log_prob, beam_size)
            for lp, idx in zip(topv.tolist(), topi.tolist()):
                idx = int(idx)
                ns = seq + [idx]
                cands.append((ns, score + float(lp), idx == EOS_IDX))

        def score_fn(item):
            seq, s, _ = item
            length = max(1, len(seq) - 1)
            return s / (length ** length_penalty)

        beams = sorted(cands, key=score_fn, reverse=True)[:beam_size]
        if all(done for _, _, done in beams):
            break

    best = beams[0][0]
    out: List[int] = []
    for tid in best[1:]:
        if tid == EOS_IDX:
            break
        out.append(tid)
    return out


@torch.no_grad()
def evaluate_transformer_bleu(
    model: TransformerTranslator,
    src_data: np.ndarray,
    tgt_data: np.ndarray,
    tgt_vocab: Vocabulary,
    decode_method: str,
    cfg: Config,
    sample_size: int | None = None,
) -> Dict[str, float]:
    idxs = list(range(len(src_data)))
    if sample_size is not None and sample_size < len(idxs):
        idxs = random.sample(idxs, sample_size)

    refs: List[List[int]] = []
    hyps: List[List[int]] = []
    device = torch.device(cfg.device)
    for i in idxs:
        src_tensor = torch.tensor(src_data[i], dtype=torch.long, device=device)
        if decode_method == "beam":
            pred = decode_transformer_beam(model, src_tensor, cfg.beam_size, cfg.length_penalty, cfg.max_seq_len)
        else:
            pred = decode_transformer_greedy(model, src_tensor, cfg.max_seq_len)
        ref = [int(t) for t in tgt_data[i] if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
        refs.append(ref)
        hyps.append(pred)
    return compute_bleu_scores(refs, hyps, tgt_vocab)


def encode_pairs(
    pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    src_arr = np.array([src_vocab.encode(src, max_len) for src, _ in pairs], dtype=np.int64)
    tgt_arr = np.array([tgt_vocab.encode(tgt, max_len) for _, tgt in pairs], dtype=np.int64)
    return src_arr, tgt_arr


def plot_training_curves(
    history_s2s: Dict[str, List[float]],
    history_tf: Dict[str, List[float]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ep1 = range(1, len(history_s2s["loss"]) + 1)
    ep2 = range(1, len(history_tf["loss"]) + 1)

    axes[0].plot(ep1, history_s2s["loss"], marker="o", label="Seq2Seq")
    axes[0].plot(ep2, history_tf["loss"], marker="s", label="Transformer")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep1, history_s2s["bleu4_val"], marker="o", label="Seq2Seq")
    axes[1].plot(ep2, history_tf["bleu4_val"], marker="s", label="Transformer")
    axes[1].set_title("Validation BLEU-4")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BLEU-4")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def save_translation_examples(
    seq2seq_model: Seq2Seq,
    tf_model: TransformerTranslator,
    test_pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    cfg: Config,
    out_dir: Path,
    n_samples: int = 12,
    suffix: str = "",
) -> pd.DataFrame:
    samples = random.sample(test_pairs, min(n_samples, len(test_pairs)))
    rows: List[Dict[str, str]] = []
    device = torch.device(cfg.device)

    for src_text, tgt_text in samples:
        src_ids = torch.tensor(src_vocab.encode(src_text, cfg.max_seq_len), dtype=torch.long, device=device)
        pred_s2s_ids = decode_seq2seq_beam(seq2seq_model, src_ids, cfg.beam_size, cfg.length_penalty, cfg.max_seq_len)
        pred_s2s_greedy = decode_seq2seq_greedy(seq2seq_model, src_ids, cfg.max_seq_len)
        pred_tf_ids = decode_transformer_beam(tf_model, src_ids, cfg.beam_size, cfg.length_penalty, cfg.max_seq_len)
        pred_tf_greedy = decode_transformer_greedy(tf_model, src_ids, cfg.max_seq_len)

        rows.append(
            {
                "source_spanish": src_text,
                "reference_english": tgt_text,
                "seq2seq_beam_pred": tgt_vocab.decode(pred_s2s_ids),
                "seq2seq_greedy_pred": tgt_vocab.decode(pred_s2s_greedy),
                "transformer_beam_pred": tgt_vocab.decode(pred_tf_ids),
                "transformer_greedy_pred": tgt_vocab.decode(pred_tf_greedy),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"translation_examples{suffix}.csv", index=False, encoding="utf-8-sig")
    return df


def _token_overlap(pred: str, ref: str) -> float:
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())
    if not ref_tokens:
        return 0.0
    return float(len(pred_tokens & ref_tokens) / len(ref_tokens))


def _error_tag(pred: str, ref: str, overlap: float, length_ratio: float) -> str:
    if not pred.strip():
        return "empty_output"
    if overlap < 0.2:
        return "low_keyword_overlap"
    if length_ratio < 0.6:
        return "under_translation"
    if length_ratio > 1.6:
        return "over_translation"
    return "semantic_or_grammar_error"


def save_translation_error_analysis(examples_df: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    rows: List[Dict[str, float | str]] = []
    pred_cols = [
        "seq2seq_beam_pred",
        "seq2seq_greedy_pred",
        "transformer_beam_pred",
        "transformer_greedy_pred",
    ]
    for _, row in examples_df.iterrows():
        ref = str(row["reference_english"])
        ref_len = max(1, len(ref.split()))
        for col in pred_cols:
            pred = str(row[col])
            overlap = _token_overlap(pred, ref)
            length_ratio = float(len(pred.split()) / ref_len)
            rows.append(
                {
                    "source_spanish": row["source_spanish"],
                    "reference_english": ref,
                    "model_decode": col.replace("_pred", ""),
                    "prediction": pred,
                    "token_overlap": overlap,
                    "length_ratio": length_ratio,
                    "error_tag": _error_tag(pred, ref, overlap, length_ratio),
                }
            )
    analysis_df = pd.DataFrame(rows)
    analysis_df.to_csv(out_dir / f"translation_error_analysis{suffix}.csv", index=False, encoding="utf-8-sig")

    summary_df = (
        analysis_df.groupby(["model_decode", "error_tag"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(by=["model_decode", "count"], ascending=[True, False])
    )
    summary_df.to_csv(out_dir / f"translation_error_summary{suffix}.csv", index=False, encoding="utf-8-sig")


def run_experiment(cfg: Config) -> pd.DataFrame:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = cfg.output_suffix

    print("\n配置:")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    print("\n加载平行语料...")
    pairs = load_parallel_pairs(cfg)
    print(f"可用句对: {len(pairs)}")
    if len(pairs) < 1000:
        raise RuntimeError("有效样本过少，请检查数据文件")

    train_pairs, val_pairs, test_pairs = split_pairs(cfg, pairs)
    print(f"train/val/test: {len(train_pairs)} / {len(val_pairs)} / {len(test_pairs)}")

    src_vocab = Vocabulary(max_size=cfg.max_vocab_src, min_freq=cfg.min_token_freq)
    tgt_vocab = Vocabulary(max_size=cfg.max_vocab_tgt, min_freq=cfg.min_token_freq)
    src_vocab.build([s for s, _ in train_pairs])
    tgt_vocab.build([t for _, t in train_pairs])
    print(f"src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}")

    src_train, tgt_train = encode_pairs(train_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)
    src_val, tgt_val = encode_pairs(val_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)
    src_test, tgt_test = encode_pairs(test_pairs, src_vocab, tgt_vocab, cfg.max_seq_len)

    train_loader = data.DataLoader(
        TranslationDataset(src_train, tgt_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available() and cfg.device.startswith("cuda"),
        num_workers=0,
    )

    device = torch.device(cfg.device)

    # Seq2Seq + Attention
    encoder = Encoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    decoder = Decoder(len(tgt_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    seq2seq = Seq2Seq(encoder, decoder, cfg.max_seq_len).to(device)
    opt_s2s = optim.AdamW(seq2seq.parameters(), lr=cfg.lr_seq2seq, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    hist_s2s: Dict[str, List[float]] = {"loss": [], "bleu4_val": []}
    best_bleu_s2s = -1.0
    best_state_s2s = None
    best_epoch_s2s = 0
    bad_epochs = 0
    patience_s2s = cfg.early_stopping_patience_seq2seq
    ckpt_s2s = ckpt_dir / "seq2seq_best.pt"
    train_seconds_s2s = 0.0

    print(f"\n{'=' * 70}\n训练 Seq2Seq + Attention\n{'=' * 70}")
    for ep in range(1, cfg.epochs_seq2seq + 1):
        t0 = time.time()
        loss = train_seq2seq_epoch(
            seq2seq,
            train_loader,
            opt_s2s,
            criterion,
            device,
            teacher_forcing_ratio=cfg.teacher_forcing_ratio,
        )
        val_bleu = evaluate_seq2seq_bleu(
            seq2seq,
            src_val,
            tgt_val,
            tgt_vocab,
            decode_method="beam",
            cfg=cfg,
            sample_size=min(cfg.bleu_val_sample_size, len(src_val)),
        )
        hist_s2s["loss"].append(loss)
        hist_s2s["bleu4_val"].append(val_bleu["bleu4"])
        elapsed = time.time() - t0
        train_seconds_s2s += elapsed
        print(
            f"Epoch {ep:02d}/{cfg.epochs_seq2seq} | "
            f"loss={loss:.4f} | val_bleu4={val_bleu['bleu4']:.4f} | {elapsed:.1f}s"
        )

        if val_bleu["bleu4"] > (best_bleu_s2s + cfg.early_stopping_min_delta_bleu):
            best_bleu_s2s = val_bleu["bleu4"]
            best_epoch_s2s = ep
            best_state_s2s = {k: v.cpu().clone() for k, v in seq2seq.state_dict().items()}
            torch.save(best_state_s2s, ckpt_s2s)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience_s2s:
                print("Seq2Seq 早停触发")
                break

    if ckpt_s2s.exists():
        seq2seq.load_state_dict(torch.load(ckpt_s2s, map_location=device))
    elif best_state_s2s is not None:
        seq2seq.load_state_dict(best_state_s2s)
    print(f"Seq2Seq best epoch={best_epoch_s2s}, best val_bleu4={best_bleu_s2s:.4f}")

    t_s2s_beam = time.time()
    s2s_test = evaluate_seq2seq_bleu(seq2seq, src_test, tgt_test, tgt_vocab, decode_method="beam", cfg=cfg)
    infer_seconds_s2s_beam = time.time() - t_s2s_beam
    t_s2s_greedy = time.time()
    s2s_test_greedy = evaluate_seq2seq_bleu(seq2seq, src_test, tgt_test, tgt_vocab, decode_method="greedy", cfg=cfg)
    infer_seconds_s2s_greedy = time.time() - t_s2s_greedy
    print("\nSeq2Seq 测试 BLEU:")
    print(json.dumps(s2s_test, ensure_ascii=False, indent=2))

    # Transformer
    transformer = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.hidden_dim,
        nhead=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    ).to(device)
    opt_tf = optim.AdamW(transformer.parameters(), lr=cfg.lr_transformer, weight_decay=1e-5)
    if cfg.label_smoothing > 0:
        tf_criterion = LabelSmoothingCrossEntropy(epsilon=cfg.label_smoothing, ignore_index=PAD_IDX)
    else:
        tf_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    hist_tf: Dict[str, List[float]] = {"loss": [], "bleu4_val": []}
    best_bleu_tf = -1.0
    best_state_tf = None
    best_epoch_tf = 0
    bad_epochs = 0
    patience_tf = cfg.early_stopping_patience_transformer
    ckpt_tf = ckpt_dir / "transformer_best.pt"
    train_seconds_tf = 0.0

    print(f"\n{'=' * 70}\n训练 Transformer\n{'=' * 70}")
    for ep in range(1, cfg.epochs_transformer + 1):
        t0 = time.time()
        loss = train_transformer_epoch(transformer, train_loader, opt_tf, tf_criterion, device)
        val_bleu = evaluate_transformer_bleu(
            transformer,
            src_val,
            tgt_val,
            tgt_vocab,
            decode_method="beam",
            cfg=cfg,
            sample_size=min(cfg.bleu_val_sample_size, len(src_val)),
        )
        hist_tf["loss"].append(loss)
        hist_tf["bleu4_val"].append(val_bleu["bleu4"])
        elapsed = time.time() - t0
        train_seconds_tf += elapsed
        print(
            f"Epoch {ep:02d}/{cfg.epochs_transformer} | "
            f"loss={loss:.4f} | val_bleu4={val_bleu['bleu4']:.4f} | {elapsed:.1f}s"
        )

        if val_bleu["bleu4"] > (best_bleu_tf + cfg.early_stopping_min_delta_bleu):
            best_bleu_tf = val_bleu["bleu4"]
            best_epoch_tf = ep
            best_state_tf = {k: v.cpu().clone() for k, v in transformer.state_dict().items()}
            torch.save(best_state_tf, ckpt_tf)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience_tf:
                print("Transformer 早停触发")
                break

    if ckpt_tf.exists():
        transformer.load_state_dict(torch.load(ckpt_tf, map_location=device))
    elif best_state_tf is not None:
        transformer.load_state_dict(best_state_tf)
    print(f"Transformer best epoch={best_epoch_tf}, best val_bleu4={best_bleu_tf:.4f}")

    t_tf_beam = time.time()
    tf_test = evaluate_transformer_bleu(transformer, src_test, tgt_test, tgt_vocab, decode_method="beam", cfg=cfg)
    infer_seconds_tf_beam = time.time() - t_tf_beam
    t_tf_greedy = time.time()
    tf_test_greedy = evaluate_transformer_bleu(
        transformer,
        src_test,
        tgt_test,
        tgt_vocab,
        decode_method="greedy",
        cfg=cfg,
    )
    infer_seconds_tf_greedy = time.time() - t_tf_greedy
    print("\nTransformer 测试 BLEU:")
    print(json.dumps(tf_test, ensure_ascii=False, indent=2))

    # CNN-BiGRU hybrid encoder (Seq2Seq with CNNBiGRUEncoder)
    hyb_s2s_test = None
    hyb_s2s_test_greedy = None
    infer_seconds_hyb_beam = 0.0
    infer_seconds_hyb_greedy = 0.0
    best_bleu_hyb = -1.0
    best_epoch_hyb = 0
    hist_hyb: Dict[str, List[float]] = {"loss": [], "bleu4_val": []}
    train_seconds_hyb = 0.0
    hyb_s2s = None

    if cfg.include_hybrid:
        hyb_encoder = CNNBiGRUEncoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
        hyb_decoder = Decoder(len(tgt_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
        hyb_s2s = Seq2Seq(hyb_encoder, hyb_decoder, cfg.max_seq_len).to(device)
        opt_hyb = optim.AdamW(hyb_s2s.parameters(), lr=cfg.lr_seq2seq, weight_decay=1e-5)

        best_state_hyb = None
        bad_epochs_hyb = 0
        patience_hyb = cfg.early_stopping_patience_seq2seq
        ckpt_hyb = ckpt_dir / "cnnbigru_best.pt"

        print(f"\n{'=' * 70}\n训练 CNN-BiGRU Seq2Seq\n{'=' * 70}")
        for ep in range(1, cfg.epochs_seq2seq + 1):
            t0 = time.time()
            loss = train_seq2seq_epoch(
                hyb_s2s,
                train_loader,
                opt_hyb,
                criterion,
                device,
                teacher_forcing_ratio=cfg.teacher_forcing_ratio,
            )
            val_bleu = evaluate_seq2seq_bleu(
                hyb_s2s,
                src_val,
                tgt_val,
                tgt_vocab,
                decode_method="beam",
                cfg=cfg,
                sample_size=min(cfg.bleu_val_sample_size, len(src_val)),
            )
            hist_hyb["loss"].append(loss)
            hist_hyb["bleu4_val"].append(val_bleu["bleu4"])
            elapsed = time.time() - t0
            train_seconds_hyb += elapsed
            print(
                f"Epoch {ep:02d}/{cfg.epochs_seq2seq} | "
                f"loss={loss:.4f} | val_bleu4={val_bleu['bleu4']:.4f} | {elapsed:.1f}s"
            )

            if val_bleu["bleu4"] > (best_bleu_hyb + cfg.early_stopping_min_delta_bleu):
                best_bleu_hyb = val_bleu["bleu4"]
                best_epoch_hyb = ep
                best_state_hyb = {k: v.cpu().clone() for k, v in hyb_s2s.state_dict().items()}
                torch.save(best_state_hyb, ckpt_hyb)
                bad_epochs_hyb = 0
            else:
                bad_epochs_hyb += 1
                if bad_epochs_hyb >= patience_hyb:
                    print("CNN-BiGRU 早停触发")
                    break

        if ckpt_hyb.exists():
            hyb_s2s.load_state_dict(torch.load(ckpt_hyb, map_location=device))
        elif best_state_hyb is not None:
            hyb_s2s.load_state_dict(best_state_hyb)
        print(f"CNN-BiGRU best epoch={best_epoch_hyb}, best val_bleu4={best_bleu_hyb:.4f}")

        t_hyb_beam = time.time()
        hyb_s2s_test = evaluate_seq2seq_bleu(hyb_s2s, src_test, tgt_test, tgt_vocab, decode_method="beam", cfg=cfg)
        infer_seconds_hyb_beam = time.time() - t_hyb_beam
        t_hyb_greedy = time.time()
        hyb_s2s_test_greedy = evaluate_seq2seq_bleu(hyb_s2s, src_test, tgt_test, tgt_vocab, decode_method="greedy", cfg=cfg)
        infer_seconds_hyb_greedy = time.time() - t_hyb_greedy
        print("\nCNN-BiGRU 测试 BLEU:")
        print(json.dumps(hyb_s2s_test, ensure_ascii=False, indent=2))

    plot_training_curves(hist_s2s, hist_tf, out_dir)
    examples_df = save_translation_examples(
        seq2seq,
        transformer,
        test_pairs,
        src_vocab,
        tgt_vocab,
        cfg,
        out_dir,
        n_samples=80 if not cfg.quick else 12,
        suffix=suffix,
    )
    save_translation_error_analysis(examples_df, out_dir, suffix=suffix)

    n_test_sent = len(src_test)
    ablation_rows = [
        {
            "model": "Seq2Seq+Attention",
            "decode_method": "beam",
            "BLEU-1": s2s_test["bleu1"],
            "BLEU-2": s2s_test["bleu2"],
            "BLEU-4": s2s_test["bleu4"],
            "infer_seconds": infer_seconds_s2s_beam,
            "sentences_per_sec": n_test_sent / max(infer_seconds_s2s_beam, 1e-8),
        },
        {
            "model": "Seq2Seq+Attention",
            "decode_method": "greedy",
            "BLEU-1": s2s_test_greedy["bleu1"],
            "BLEU-2": s2s_test_greedy["bleu2"],
            "BLEU-4": s2s_test_greedy["bleu4"],
            "infer_seconds": infer_seconds_s2s_greedy,
            "sentences_per_sec": n_test_sent / max(infer_seconds_s2s_greedy, 1e-8),
        },
        {
            "model": "Transformer",
            "decode_method": "beam",
            "BLEU-1": tf_test["bleu1"],
            "BLEU-2": tf_test["bleu2"],
            "BLEU-4": tf_test["bleu4"],
            "infer_seconds": infer_seconds_tf_beam,
            "sentences_per_sec": n_test_sent / max(infer_seconds_tf_beam, 1e-8),
        },
        {
            "model": "Transformer",
            "decode_method": "greedy",
            "BLEU-1": tf_test_greedy["bleu1"],
            "BLEU-2": tf_test_greedy["bleu2"],
            "BLEU-4": tf_test_greedy["bleu4"],
            "infer_seconds": infer_seconds_tf_greedy,
            "sentences_per_sec": n_test_sent / max(infer_seconds_tf_greedy, 1e-8),
        },
    ]
    if cfg.include_hybrid and hyb_s2s_test is not None and hyb_s2s_test_greedy is not None:
        ablation_rows.extend(
            [
                {
                    "model": "CNNBiGRU",
                    "decode_method": "beam",
                    "BLEU-1": hyb_s2s_test["bleu1"],
                    "BLEU-2": hyb_s2s_test["bleu2"],
                    "BLEU-4": hyb_s2s_test["bleu4"],
                    "infer_seconds": infer_seconds_hyb_beam,
                    "sentences_per_sec": n_test_sent / max(infer_seconds_hyb_beam, 1e-8),
                },
                {
                    "model": "CNNBiGRU",
                    "decode_method": "greedy",
                    "BLEU-1": hyb_s2s_test_greedy["bleu1"],
                    "BLEU-2": hyb_s2s_test_greedy["bleu2"],
                    "BLEU-4": hyb_s2s_test_greedy["bleu4"],
                    "infer_seconds": infer_seconds_hyb_greedy,
                    "sentences_per_sec": n_test_sent / max(infer_seconds_hyb_greedy, 1e-8),
                },
            ]
        )
    decode_ablation = pd.DataFrame(ablation_rows)
    decode_ablation.to_csv(out_dir / f"translation_decode_ablation{suffix}.csv", index=False, encoding="utf-8-sig")

    result_rows = [
        {
            "model": "Seq2Seq+Attention",
            "BLEU-1": s2s_test["bleu1"],
            "BLEU-2": s2s_test["bleu2"],
            "BLEU-4": s2s_test["bleu4"],
            "BLEU-1-greedy": s2s_test_greedy["bleu1"],
            "BLEU-2-greedy": s2s_test_greedy["bleu2"],
            "BLEU-4-greedy": s2s_test_greedy["bleu4"],
            "best_val_BLEU-4": best_bleu_s2s,
            "best_epoch": best_epoch_s2s,
            "trained_epochs": len(hist_s2s["loss"]),
            "train_seconds": train_seconds_s2s,
            "infer_seconds_beam": infer_seconds_s2s_beam,
            "infer_sents_per_sec_beam": n_test_sent / max(infer_seconds_s2s_beam, 1e-8),
            "infer_seconds_greedy": infer_seconds_s2s_greedy,
            "infer_sents_per_sec_greedy": n_test_sent / max(infer_seconds_s2s_greedy, 1e-8),
            "params": sum(p.numel() for p in seq2seq.parameters()),
        },
        {
            "model": "Transformer",
            "BLEU-1": tf_test["bleu1"],
            "BLEU-2": tf_test["bleu2"],
            "BLEU-4": tf_test["bleu4"],
            "BLEU-1-greedy": tf_test_greedy["bleu1"],
            "BLEU-2-greedy": tf_test_greedy["bleu2"],
            "BLEU-4-greedy": tf_test_greedy["bleu4"],
            "best_val_BLEU-4": best_bleu_tf,
            "best_epoch": best_epoch_tf,
            "trained_epochs": len(hist_tf["loss"]),
            "train_seconds": train_seconds_tf,
            "infer_seconds_beam": infer_seconds_tf_beam,
            "infer_sents_per_sec_beam": n_test_sent / max(infer_seconds_tf_beam, 1e-8),
            "infer_seconds_greedy": infer_seconds_tf_greedy,
            "infer_sents_per_sec_greedy": n_test_sent / max(infer_seconds_tf_greedy, 1e-8),
            "params": sum(p.numel() for p in transformer.parameters()),
        },
    ]

    if cfg.include_hybrid and hyb_s2s is not None and hyb_s2s_test is not None and hyb_s2s_test_greedy is not None:
        result_rows.append(
            {
                "model": "CNNBiGRU",
                "BLEU-1": hyb_s2s_test["bleu1"],
                "BLEU-2": hyb_s2s_test["bleu2"],
                "BLEU-4": hyb_s2s_test["bleu4"],
                "BLEU-1-greedy": hyb_s2s_test_greedy["bleu1"],
                "BLEU-2-greedy": hyb_s2s_test_greedy["bleu2"],
                "BLEU-4-greedy": hyb_s2s_test_greedy["bleu4"],
                "best_val_BLEU-4": best_bleu_hyb,
                "best_epoch": best_epoch_hyb,
                "trained_epochs": len(hist_hyb["loss"]),
                "train_seconds": train_seconds_hyb,
                "infer_seconds_beam": infer_seconds_hyb_beam,
                "infer_sents_per_sec_beam": n_test_sent / max(infer_seconds_hyb_beam, 1e-8),
                "infer_seconds_greedy": infer_seconds_hyb_greedy,
                "infer_sents_per_sec_greedy": n_test_sent / max(infer_seconds_hyb_greedy, 1e-8),
                "params": sum(p.numel() for p in hyb_s2s.parameters()),
            }
        )

    results = pd.DataFrame(result_rows).sort_values(by="BLEU-4", ascending=False)

    results.to_csv(out_dir / f"translation_results{suffix}.csv", index=False, encoding="utf-8-sig")
    eff_cols = [
        "model",
        "params",
        "train_seconds",
        "infer_seconds_beam",
        "infer_sents_per_sec_beam",
        "infer_seconds_greedy",
        "infer_sents_per_sec_greedy",
    ]
    results[eff_cols].to_csv(out_dir / f"translation_efficiency{suffix}.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"translation_config{suffix}.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n最终结果:")
    print(results.to_string(index=False, float_format="%.4f"))
    print("\n示例翻译已保存:")
    print((out_dir / f"translation_examples{suffix}.csv").resolve())
    return results


def main() -> None:
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
