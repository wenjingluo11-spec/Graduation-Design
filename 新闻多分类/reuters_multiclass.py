#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reuters-46 新闻多分类实验

符合开题报告范围：
- 基础模型：RNN 变体（BiGRU）、CNN（TextCNN）、Transformer Encoder
- 任务：Reuters 46 类新闻分类
- 额外传统基线：Multinomial Naive Bayes
"""

from __future__ import annotations

import argparse
import json
import math
import os

# MPS fallback 必须在 import torch 之前设置.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def _detect_device() -> str:
    """三级设备 fallback: mps → cuda → cpu"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    data_file: str = "../reuters.npz"
    output_dir: str = "outputs"
    seed: int = 42
    max_vocab_size: int = 20000
    max_seq_len: int = 240
    embedding_dim: int = 128
    hidden_dim: int = 160
    num_heads: int = 4
    num_transformer_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    max_samples: int = 11228
    device: str = field(default_factory=_detect_device)
    quick: bool = False
    include_hybrid: bool = False
    loss: str = "ce"
    focal_gamma: float = 2.0
    output_suffix: str = ""


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Reuters-46 多分类实验")
    parser.add_argument("--data-file", type=str, default="../reuters.npz")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=11228)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-hybrid", action="store_true", help="加入 CNN-BiGRU 混合模型对比")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--output-suffix", type=str, default="", help="附加到所有输出文件名后,便于 γ-scan 等并存")
    args = parser.parse_args()

    cfg = Config(
        data_file=args.data_file,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        quick=args.quick,
        include_hybrid=args.include_hybrid,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        output_suffix=args.output_suffix,
    )
    if args.device is not None:
        cfg.device = args.device  # 用户显式指定即生效

    if cfg.quick:
        cfg.epochs = 2
        cfg.early_stopping_patience = 2
        cfg.batch_size = 32
        cfg.max_samples = min(cfg.max_samples, 3500)
        cfg.max_vocab_size = 12000
        cfg.max_seq_len = 180

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_reuters(cfg: Config) -> Tuple[List[List[int]], np.ndarray, int]:
    data_file = Path(cfg.data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_file.resolve()}")
    obj = np.load(data_file, allow_pickle=True)
    x = obj["x"].tolist()
    y = obj["y"].astype(np.int64)
    if cfg.max_samples > 0:
        x = x[: cfg.max_samples]
        y = y[: cfg.max_samples]
    num_classes = int(y.max() + 1)
    return x, y, num_classes


def remap_and_pad(seqs: Sequence[Sequence[int]], max_vocab_size: int, max_seq_len: int) -> np.ndarray:
    # 0: pad, 1: unk, 2..: valid tokens
    arr = np.zeros((len(seqs), max_seq_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        remapped = []
        for token in seq[:max_seq_len]:
            if 1 <= token < (max_vocab_size - 1):
                remapped.append(token + 1)  # shift to reserve 0/1
            else:
                remapped.append(1)
        arr[i, : len(remapped)] = remapped
    return arr


def build_bow_matrix(seqs: Sequence[Sequence[int]], max_vocab_size: int) -> csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    data_vals: List[int] = []
    for i, seq in enumerate(seqs):
        local_count: Dict[int, int] = {}
        for token in seq:
            if 1 <= token < (max_vocab_size - 1):
                idx = token
            else:
                idx = 0
            local_count[idx] = local_count.get(idx, 0) + 1
        for idx, cnt in local_count.items():
            rows.append(i)
            cols.append(idx)
            data_vals.append(cnt)
    return csr_matrix((data_vals, (rows, cols)), shape=(len(seqs), max_vocab_size), dtype=np.float32)


class SeqDataset(data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TextCNNMulti(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1),
                nn.Conv1d(emb_dim, 128, kernel_size=4, padding=2),
                nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        feats = [torch.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class BiGRUMulti(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class CNNBiGRUMulti(nn.Module):
    """Stacked CNN-BiGRU 用于 46 类新闻分类.

    Conv1d (k=3, pad=1) → BiGRU → max+mean pool → 46-class softmax 头.
    """
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int, hidden_dim: int, dropout: float, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.bigru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2 * 2, num_classes)  # bi*hidden, max+mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        conv_out = torch.relu(self.conv(emb.transpose(1, 2)))
        gru_in = conv_out.transpose(1, 2)
        gru_out, _ = self.bigru(gru_in)
        max_pool = gru_out.max(dim=1)[0]
        mean_pool = gru_out.mean(dim=1)
        pooled = torch.cat([max_pool, mean_pool], dim=1)
        return self.classifier(self.dropout(pooled))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerMulti(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        pad_mask = x.eq(0)
        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        emb = self.pos(emb)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        pooled = out.masked_fill(pad_mask.unsqueeze(-1), 0.0).sum(dim=1)
        valid_len = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(1)
        pooled = pooled / valid_len
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        if return_attention:
            last_layer = self.encoder.layers[-1]
            with torch.no_grad():
                _, attn = last_layer.self_attn(
                    out, out, out,
                    key_padding_mask=pad_mask,
                    need_weights=True, average_attn_weights=False,
                )
            return logits, attn  # attn: [B, num_heads, L, L]
        return logits


def compute_mc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


class FocalLoss(nn.Module):
    """多分类 focal loss: FL = -(1 - p_t)^γ * log(p_t).

    γ=0 退化为标准交叉熵; γ>0 对易分类样本下放权重, 关注困难/少数类.
    """
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)            # [B]
        pt = torch.exp(-ce_loss)                     # softmax-prob of target class
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: data.DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        preds_all.append(preds)
        labels_all.append(y_batch.numpy())
    return np.concatenate(labels_all), np.concatenate(preds_all)


def run_dl_model(
    model_name: str,
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    test_loader: data.DataLoader,
    checkpoint_dir: Path,
    cfg: Config,
    num_classes: int,
) -> Tuple[Dict[str, float], Dict[str, List[float]], np.ndarray, np.ndarray]:
    device = torch.device(cfg.device)
    model = model.to(device)
    if cfg.loss == "focal":
        criterion = FocalLoss(gamma=cfg.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_acc": [], "val_f1_macro": []}
    best_state = None
    best_macro = -1.0
    best_epoch = 0
    patience = cfg.early_stopping_patience
    bad_epochs = 0
    checkpoint_path = checkpoint_dir / f"{model_name.lower()}_best.pt"
    train_seconds = 0.0

    print(f"\n{'=' * 70}")
    print(f"训练模型: {model_name}")
    print(f"{'=' * 70}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_y, val_pred = evaluate_model(model, val_loader, device)
        val_metrics = compute_mc_metrics(val_y, val_pred)
        history["train_loss"].append(loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        elapsed = time.time() - t0
        train_seconds += elapsed
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | loss={loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f} | {elapsed:.1f}s"
        )

        if val_metrics["f1_macro"] > (best_macro + cfg.early_stopping_min_delta):
            best_macro = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"早停触发: {model_name}")
                break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif best_state is not None:
        model.load_state_dict(best_state)
    print(f"{model_name} best epoch={best_epoch}, best val_f1_macro={best_macro:.4f}")

    t_inf = time.time()
    test_y, test_pred = evaluate_model(model, test_loader, device)
    infer_seconds = time.time() - t_inf
    test_metrics = compute_mc_metrics(test_y, test_pred)
    test_metrics["best_val_f1_macro"] = float(best_macro)
    test_metrics["best_epoch"] = float(best_epoch)
    test_metrics["trained_epochs"] = float(len(history["train_loss"]))
    test_metrics["params"] = float(sum(p.numel() for p in model.parameters()))
    test_metrics["train_seconds"] = float(train_seconds)
    test_metrics["infer_seconds"] = float(infer_seconds)
    test_metrics["infer_samples_per_sec"] = float(len(test_y) / max(infer_seconds, 1e-8))
    print(f"\n{model_name} 测试集指标:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(classification_report(test_y, test_pred, digits=4, zero_division=0))
    return test_metrics, history, test_y, test_pred


def run_nb_baseline(
    train_seq: Sequence[Sequence[int]],
    train_y: np.ndarray,
    test_seq: Sequence[Sequence[int]],
    test_y: np.ndarray,
    cfg: Config,
) -> Tuple[Dict[str, float], np.ndarray]:
    print(f"\n{'=' * 70}")
    print("训练模型: Naive Bayes (BoW)")
    print(f"{'=' * 70}")
    t0 = time.time()
    x_train = build_bow_matrix(train_seq, cfg.max_vocab_size)
    model = MultinomialNB(alpha=0.2)
    model.fit(x_train, train_y)
    train_seconds = time.time() - t0

    t_inf = time.time()
    x_test = build_bow_matrix(test_seq, cfg.max_vocab_size)
    pred = model.predict(x_test)
    infer_seconds = time.time() - t_inf
    metrics = compute_mc_metrics(test_y, pred)
    metrics["params"] = float("nan")
    metrics["train_seconds"] = float(train_seconds)
    metrics["infer_seconds"] = float(infer_seconds)
    metrics["infer_samples_per_sec"] = float(len(test_y) / max(infer_seconds, 1e-8))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(classification_report(test_y, pred, digits=4, zero_division=0))
    return metrics, pred


def save_reuters_error_analysis(
    y_true: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    out_dir: Path,
    top_k_per_model: int = 25,
    suffix: str = "",
) -> None:
    rows: List[Dict[str, float]] = []
    for model_name, y_pred in pred_by_model.items():
        cm = confusion_matrix(y_true, y_pred)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cnt = int(cm[i, j])
                if i == j or cnt <= 0:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "true_label": int(i),
                        "pred_label": int(j),
                        "count": cnt,
                    }
                )
    if not rows:
        return

    conf_df = pd.DataFrame(rows).sort_values(by=["model", "count"], ascending=[True, False])
    top_df = conf_df.groupby("model", as_index=False).head(top_k_per_model).reset_index(drop=True)
    top_df.to_csv(out_dir / f"reuters_top_confusions{suffix}.csv", index=False, encoding="utf-8-sig")


def plot_history(history: Dict[str, List[float]], model_name: str, out_dir: Path) -> None:
    if not history["train_loss"]:
        return
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ep, history["train_loss"], marker="o")
    axes[0].set_title(f"{model_name} Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["val_acc"], marker="o", label="val_acc")
    axes[1].plot(ep, history["val_f1_macro"], marker="s", label="val_f1_macro")
    axes[1].set_title(f"{model_name} Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name.lower()}_history.png", dpi=150)
    plt.close(fig)


def plot_conf_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def prepare_data(cfg: Config):
    seqs, labels, num_classes = load_reuters(cfg)
    idx_all = np.arange(len(seqs))
    try:
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=cfg.test_ratio,
            random_state=cfg.seed,
            stratify=labels,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=cfg.test_ratio,
            random_state=cfg.seed,
            stratify=None,
        )

    try:
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=cfg.val_ratio,
            random_state=cfg.seed,
            stratify=labels[train_idx],
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=cfg.val_ratio,
            random_state=cfg.seed,
            stratify=None,
        )

    seq_train = [seqs[i] for i in train_idx]
    seq_val = [seqs[i] for i in val_idx]
    seq_test = [seqs[i] for i in test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    x_train = remap_and_pad(seq_train, cfg.max_vocab_size, cfg.max_seq_len)
    x_val = remap_and_pad(seq_val, cfg.max_vocab_size, cfg.max_seq_len)
    x_test = remap_and_pad(seq_test, cfg.max_vocab_size, cfg.max_seq_len)

    train_loader = data.DataLoader(SeqDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = data.DataLoader(SeqDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)
    test_loader = data.DataLoader(SeqDataset(x_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    return (
        (seq_train, y_train),
        (seq_test, y_test),
        train_loader,
        val_loader,
        test_loader,
        num_classes,
    )


def run_experiment(cfg: Config) -> pd.DataFrame:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = cfg.output_suffix

    print("\n配置:")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    (seq_train, y_train), (seq_test, y_test), train_loader, val_loader, test_loader, num_classes = prepare_data(cfg)
    print(f"训练样本: {len(seq_train)}, 测试样本: {len(seq_test)}, 类别数: {num_classes}")

    results: List[Dict[str, float]] = []
    pred_by_model: Dict[str, np.ndarray] = {}

    nb_metrics, nb_pred = run_nb_baseline(seq_train, y_train, seq_test, y_test, cfg)
    plot_conf_matrix(y_test, nb_pred, "Naive Bayes Confusion Matrix", out_dir / "naive_bayes_confusion_matrix.png")
    results.append({"model": "NaiveBayes", **nb_metrics})
    pred_by_model["NaiveBayes"] = nb_pred

    cnn = TextCNNMulti(cfg.max_vocab_size, cfg.embedding_dim, num_classes, cfg.dropout)
    cnn_metrics, cnn_hist, cnn_y, cnn_pred = run_dl_model(
        "TextCNN",
        cnn,
        train_loader,
        val_loader,
        test_loader,
        ckpt_dir,
        cfg,
        num_classes,
    )
    plot_history(cnn_hist, "TextCNN", out_dir)
    plot_conf_matrix(cnn_y, cnn_pred, "TextCNN Confusion Matrix", out_dir / "textcnn_confusion_matrix.png")
    results.append({"model": "TextCNN", **cnn_metrics})
    pred_by_model["TextCNN"] = cnn_pred

    rnn = BiGRUMulti(cfg.max_vocab_size, cfg.embedding_dim, cfg.hidden_dim, num_classes, cfg.dropout)
    rnn_metrics, rnn_hist, rnn_y, rnn_pred = run_dl_model(
        "BiGRU",
        rnn,
        train_loader,
        val_loader,
        test_loader,
        ckpt_dir,
        cfg,
        num_classes,
    )
    plot_history(rnn_hist, "BiGRU", out_dir)
    plot_conf_matrix(rnn_y, rnn_pred, "BiGRU Confusion Matrix", out_dir / "bigru_confusion_matrix.png")
    results.append({"model": "BiGRU", **rnn_metrics})
    pred_by_model["BiGRU"] = rnn_pred

    tfm = TransformerMulti(
        vocab_size=cfg.max_vocab_size,
        emb_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_layers=cfg.num_transformer_layers,
        num_classes=num_classes,
        dropout=cfg.dropout,
    )
    tfm_metrics, tfm_hist, tfm_y, tfm_pred = run_dl_model(
        "Transformer",
        tfm,
        train_loader,
        val_loader,
        test_loader,
        ckpt_dir,
        cfg,
        num_classes,
    )
    plot_history(tfm_hist, "Transformer", out_dir)
    plot_conf_matrix(tfm_y, tfm_pred, "Transformer Confusion Matrix", out_dir / "transformer_confusion_matrix.png")
    results.append({"model": "Transformer", **tfm_metrics})
    pred_by_model["Transformer"] = tfm_pred

    if cfg.include_hybrid:
        hybrid = CNNBiGRUMulti(cfg.max_vocab_size, num_classes, cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
        h_metrics, h_hist, h_y, h_pred = run_dl_model(
            "CNNBiGRU",
            hybrid,
            train_loader,
            val_loader,
            test_loader,
            ckpt_dir,
            cfg,
            num_classes,
        )
        plot_history(h_hist, "CNNBiGRU", out_dir)
        plot_conf_matrix(h_y, h_pred, "CNNBiGRU Confusion Matrix", out_dir / "cnnbigru_confusion_matrix.png")
        results.append({"model": "CNNBiGRU", **h_metrics})
        pred_by_model["CNNBiGRU"] = h_pred

    results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    print("\n最终结果:")
    print(results_df.to_string(index=False, float_format="%.4f"))

    results_df.to_csv(out_dir / f"reuters_results{suffix}.csv", index=False, encoding="utf-8-sig")
    eff_cols = ["model", "params", "train_seconds", "infer_seconds", "infer_samples_per_sec"]
    eff_cols = [c for c in eff_cols if c in results_df.columns]
    if eff_cols:
        results_df[eff_cols].to_csv(out_dir / f"reuters_efficiency{suffix}.csv", index=False, encoding="utf-8-sig")

    save_reuters_error_analysis(
        y_true=y_test,
        pred_by_model=pred_by_model,
        out_dir=out_dir,
        top_k_per_model=30 if not cfg.quick else 12,
        suffix=suffix,
    )

    best_model = str(results_df.iloc[0]["model"])
    if best_model in pred_by_model:
        best_report = classification_report(y_test, pred_by_model[best_model], output_dict=True, zero_division=0)
        pd.DataFrame(best_report).T.to_csv(
            out_dir / f"reuters_class_report_best_model{suffix}.csv",
            encoding="utf-8-sig",
        )

    (out_dir / f"reuters_config{suffix}.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results_df


def main() -> None:
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
