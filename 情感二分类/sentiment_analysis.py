#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感二分类实验（IMDB）

符合开题报告范围：
- 基础模型：RNN 变体（BiGRU）、CNN（TextCNN）、Transformer Encoder
- 任务：文本情感二分类
- 额外提供传统基线：Naive Bayes + TF-IDF

说明：
- 默认使用中等规模子集，兼顾可复现与运行成本
- 支持 --quick 快速验证整条流程
"""

from __future__ import annotations

import argparse
import json
import math
import os

# MPS fallback 必须在 import torch 之前设置,否则 nn.TransformerEncoder 在
# __init__ 阶段已锁定 use_nested_tensor 路径,后续 forward 仍会触发 MPS 不实现的算子.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
import re
import time
from collections import Counter
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    data_dir: str = "aclImdb"
    output_dir: str = "outputs"
    seed: int = 42
    max_vocab_size: int = 20000
    max_seq_len: int = 220
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    num_transformer_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    val_ratio: float = 0.1
    max_train_samples: int = 12000
    max_test_samples: int = 5000
    min_df_tfidf: int = 3
    device: str = field(default_factory=_detect_device)
    quick: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="IMDB 情感二分类实验")
    parser.add_argument("--data-dir", type=str, default="aclImdb")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true", help="快速跑通模式")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        quick=args.quick,
    )
    if args.device is not None:
        cfg.device = args.device  # 用户显式指定即生效

    if cfg.quick:
        cfg.epochs = 2
        cfg.early_stopping_patience = 2
        cfg.batch_size = 32
        cfg.max_train_samples = min(cfg.max_train_samples, 3000)
        cfg.max_test_samples = min(cfg.max_test_samples, 1500)
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


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


def read_imdb_split(split_dir: Path, label: int) -> pd.DataFrame:
    texts: List[str] = []
    labels: List[int] = []
    for file_path in sorted(split_dir.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        texts.append(clean_text(content))
        labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})


def load_imdb_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(cfg.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root.resolve()}")

    train_pos = read_imdb_split(root / "train" / "pos", 1)
    train_neg = read_imdb_split(root / "train" / "neg", 0)
    test_pos = read_imdb_split(root / "test" / "pos", 1)
    test_neg = read_imdb_split(root / "test" / "neg", 0)

    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    if cfg.max_train_samples > 0:
        train_df = train_df.iloc[: cfg.max_train_samples].copy()
    if cfg.max_test_samples > 0:
        test_df = test_df.iloc[: cfg.max_test_samples].copy()

    return train_df, test_df


def build_vocab(texts: Sequence[str], max_vocab_size: int) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max(0, max_vocab_size - 2)):
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


class TextDataset(data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, dropout: float, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1),
                nn.Conv1d(emb_dim, 128, kernel_size=4, padding=2),
                nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128 * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        feats = [torch.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.classifier(out).squeeze(1)


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(emb_dim)
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
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        pad_mask = x.eq(self.pad_idx)
        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        emb = self.pos_encoding(emb)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        pooled = out.masked_fill(pad_mask.unsqueeze(-1), 0.0).sum(dim=1)
        valid_len = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(1)
        pooled = pooled / valid_len
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(1)
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


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


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
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(y_batch.numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def run_dl_model(
    model_name: str,
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    test_loader: data.DataLoader,
    checkpoint_dir: Path,
    cfg: Config,
) -> Tuple[Dict[str, float], Dict[str, List[float]], np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(cfg.device)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience = cfg.early_stopping_patience
    patience_count = 0
    checkpoint_path = checkpoint_dir / f"{model_name.lower()}_best.pt"
    train_seconds = 0.0

    print(f"\n{'=' * 70}")
    print(f"训练模型: {model_name}")
    print(f"{'=' * 70}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_labels, val_probs = evaluate_model(model, val_loader, device)
        val_metrics = compute_binary_metrics(val_labels, val_probs)

        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        elapsed = time.time() - t0
        train_seconds += elapsed
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | {elapsed:.1f}s"
        )

        if val_metrics["f1"] > (best_val_f1 + cfg.early_stopping_min_delta):
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"早停触发: {model_name}")
                break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif best_state is not None:
        model.load_state_dict(best_state)
    print(f"{model_name} best epoch={best_epoch}, best val_f1={best_val_f1:.4f}")

    t_inf = time.time()
    test_labels, test_probs = evaluate_model(model, test_loader, device)
    infer_seconds = time.time() - t_inf
    test_metrics = compute_binary_metrics(test_labels, test_probs)
    test_metrics["best_val_f1"] = float(best_val_f1)
    test_metrics["best_epoch"] = float(best_epoch)
    test_metrics["trained_epochs"] = float(len(history["train_loss"]))
    test_metrics["params"] = float(sum(p.numel() for p in model.parameters()))
    test_metrics["train_seconds"] = float(train_seconds)
    test_metrics["infer_seconds"] = float(infer_seconds)
    test_metrics["infer_samples_per_sec"] = float(len(test_labels) / max(infer_seconds, 1e-8))

    y_pred = (test_probs >= 0.5).astype(int)
    print(f"\n{model_name} 测试集指标:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(classification_report(test_labels, y_pred, target_names=["negative", "positive"], digits=4))
    return test_metrics, history, test_labels, y_pred, test_probs


def run_nb_baseline(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    cfg: Config,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    print(f"\n{'=' * 70}")
    print("训练模型: Naive Bayes + TF-IDF")
    print(f"{'=' * 70}")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        max_features=cfg.max_vocab_size,
        ngram_range=(1, 2),
        min_df=cfg.min_df_tfidf,
        max_df=0.9,
        sublinear_tf=True,
    )
    x_train = tfidf.fit_transform(train_texts)
    model = MultinomialNB(alpha=0.1)
    model.fit(x_train, train_labels)
    train_seconds = time.time() - t0

    t_inf = time.time()
    x_test = tfidf.transform(test_texts)
    y_prob = model.predict_proba(x_test)[:, 1]
    infer_seconds = time.time() - t_inf
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_binary_metrics(test_labels, y_prob)
    metrics["params"] = float("nan")
    metrics["train_seconds"] = float(train_seconds)
    metrics["infer_seconds"] = float(infer_seconds)
    metrics["infer_samples_per_sec"] = float(len(test_labels) / max(infer_seconds, 1e-8))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(classification_report(test_labels, y_pred, target_names=["negative", "positive"], digits=4))
    return metrics, y_pred, y_prob


def save_sentiment_error_analysis(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    best_model: str,
    out_dir: Path,
    top_k: int = 300,
) -> None:
    y_true = test_df["label"].to_numpy(dtype=int)
    data = pd.DataFrame(
        {
            "text": test_df["text"].tolist(),
            "true_label": y_true,
            "pred_label": y_pred.astype(int),
            "pred_prob_pos": y_prob.astype(float),
        }
    )
    data["correct"] = (data["true_label"] == data["pred_label"]).astype(int)
    data["confidence"] = (data["pred_prob_pos"] - 0.5).abs() * 2.0
    data["error_type"] = np.where(
        data["correct"] == 1,
        "correct",
        np.where(data["pred_label"] == 1, "false_positive", "false_negative"),
    )

    data.to_csv(out_dir / "sentiment_predictions_best_model.csv", index=False, encoding="utf-8-sig")

    errors = data[data["correct"] == 0].sort_values(by="confidence", ascending=False).head(top_k)
    errors.insert(0, "best_model", best_model)
    errors.to_csv(out_dir / "sentiment_error_analysis.csv", index=False, encoding="utf-8-sig")

    summary = (
        data.groupby("error_type", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(by="count", ascending=False)
    )
    summary.insert(0, "best_model", best_model)
    summary.to_csv(out_dir / "sentiment_error_summary.csv", index=False, encoding="utf-8-sig")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, save_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_history(history: Dict[str, List[float]], model_name: str, output_dir: Path) -> None:
    if not history["train_loss"]:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set_title(f"{model_name} Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="val_acc")
    axes[1].plot(epochs, history["val_f1"], marker="s", label="val_f1")
    axes[1].set_title(f"{model_name} Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_history.png", dpi=150)
    plt.close(fig)


def prepare_dl_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, Dict[str, int]]:
    train_part, val_part = train_test_split(
        train_df,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_df["label"],
    )

    vocab = build_vocab(train_part["text"].tolist(), cfg.max_vocab_size)

    x_train = np.array([encode_text(t, vocab, cfg.max_seq_len) for t in train_part["text"]], dtype=np.int64)
    y_train = train_part["label"].to_numpy(dtype=np.float32)
    x_val = np.array([encode_text(t, vocab, cfg.max_seq_len) for t in val_part["text"]], dtype=np.int64)
    y_val = val_part["label"].to_numpy(dtype=np.float32)
    x_test = np.array([encode_text(t, vocab, cfg.max_seq_len) for t in test_df["text"]], dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.float32)

    train_loader = data.DataLoader(TextDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = data.DataLoader(TextDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)
    test_loader = data.DataLoader(TextDataset(x_test, y_test), batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, vocab


def run_experiment(cfg: Config) -> pd.DataFrame:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n配置:")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    print("\n加载 IMDB 数据...")
    train_df, test_df = load_imdb_data(cfg)
    print(f"训练样本: {len(train_df)}, 测试样本: {len(test_df)}")

    train_loader, val_loader, test_loader, vocab = prepare_dl_data(train_df, test_df, cfg)
    print(f"词表大小: {len(vocab)}")

    train_part, _ = train_test_split(
        train_df,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_df["label"],
    )

    results: List[Dict[str, float]] = []
    pred_by_model: Dict[str, np.ndarray] = {}
    prob_by_model: Dict[str, np.ndarray] = {}

    # Baseline
    nb_metrics, nb_pred, nb_prob = run_nb_baseline(
        train_part["text"].tolist(),
        train_part["label"].to_numpy(),
        test_df["text"].tolist(),
        test_df["label"].to_numpy(),
        cfg,
    )
    plot_confusion_matrix(
        confusion_matrix(test_df["label"].to_numpy(), nb_pred),
        ["negative", "positive"],
        "Naive Bayes Confusion Matrix",
        out_dir / "naive_bayes_confusion_matrix.png",
    )
    results.append({"model": "NaiveBayes", **nb_metrics})
    pred_by_model["NaiveBayes"] = nb_pred
    prob_by_model["NaiveBayes"] = nb_prob

    # TextCNN
    cnn = TextCNN(len(vocab), cfg.embedding_dim, cfg.dropout)
    cnn_metrics, cnn_hist, cnn_y, cnn_pred, cnn_prob = run_dl_model(
        "TextCNN",
        cnn,
        train_loader,
        val_loader,
        test_loader,
        ckpt_dir,
        cfg,
    )
    plot_history(cnn_hist, "TextCNN", out_dir)
    plot_confusion_matrix(
        confusion_matrix(cnn_y, cnn_pred),
        ["negative", "positive"],
        "TextCNN Confusion Matrix",
        out_dir / "textcnn_confusion_matrix.png",
    )
    results.append({"model": "TextCNN", **cnn_metrics})
    pred_by_model["TextCNN"] = cnn_pred
    prob_by_model["TextCNN"] = cnn_prob

    # BiGRU
    bigru = BiGRUClassifier(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.dropout)
    rnn_metrics, rnn_hist, rnn_y, rnn_pred, rnn_prob = run_dl_model(
        "BiGRU",
        bigru,
        train_loader,
        val_loader,
        test_loader,
        ckpt_dir,
        cfg,
    )
    plot_history(rnn_hist, "BiGRU", out_dir)
    plot_confusion_matrix(
        confusion_matrix(rnn_y, rnn_pred),
        ["negative", "positive"],
        "BiGRU Confusion Matrix",
        out_dir / "bigru_confusion_matrix.png",
    )
    results.append({"model": "BiGRU", **rnn_metrics})
    pred_by_model["BiGRU"] = rnn_pred
    prob_by_model["BiGRU"] = rnn_prob

    # Transformer
    transformer = TransformerClassifier(
        vocab_size=len(vocab),
        emb_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_layers=cfg.num_transformer_layers,
        dropout=cfg.dropout,
    )
    tf_metrics, tf_hist, tf_y, tf_pred, tf_prob = run_dl_model(
        "Transformer",
        transformer,
        train_loader,
        val_loader,
        test_loader,
        ckpt_dir,
        cfg,
    )
    plot_history(tf_hist, "Transformer", out_dir)
    plot_confusion_matrix(
        confusion_matrix(tf_y, tf_pred),
        ["negative", "positive"],
        "Transformer Confusion Matrix",
        out_dir / "transformer_confusion_matrix.png",
    )
    results.append({"model": "Transformer", **tf_metrics})
    pred_by_model["Transformer"] = tf_pred
    prob_by_model["Transformer"] = tf_prob

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False).reset_index(drop=True)
    print("\n最终结果:")
    print(results_df.to_string(index=False, float_format="%.4f"))
    results_df.to_csv(out_dir / "sentiment_results.csv", index=False, encoding="utf-8-sig")

    eff_cols = ["model", "params", "train_seconds", "infer_seconds", "infer_samples_per_sec"]
    eff_cols = [c for c in eff_cols if c in results_df.columns]
    if eff_cols:
        results_df[eff_cols].to_csv(out_dir / "sentiment_efficiency.csv", index=False, encoding="utf-8-sig")

    best_model = str(results_df.iloc[0]["model"])
    if best_model in pred_by_model and best_model in prob_by_model:
        save_sentiment_error_analysis(
            test_df=test_df,
            y_pred=pred_by_model[best_model],
            y_prob=prob_by_model[best_model],
            best_model=best_model,
            out_dir=out_dir,
            top_k=400 if not cfg.quick else 120,
        )

    (out_dir / "sentiment_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results_df


def main() -> None:
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
