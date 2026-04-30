import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "情感二分类"))

import torch


def test_cnnbigru_sentiment_forward():
    from sentiment_analysis import CNNBiGRU
    m = CNNBiGRU(vocab_size=200, emb_dim=32, hidden_dim=24, dropout=0.1)
    x = torch.randint(1, 200, (4, 50))
    out = m(x)
    assert out.shape == (4,)


def test_cnnbigru_sentiment_param_count():
    """参数量量级与 TextCNN/BiGRU 同档 (~2-3M with default config 128/128)"""
    from sentiment_analysis import CNNBiGRU
    m = CNNBiGRU(vocab_size=20000, emb_dim=128, hidden_dim=128, dropout=0.1)
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 < n < 5_000_000, f"unexpected param count: {n}"


sys.path.insert(0, str(ROOT / "新闻多分类"))


def test_cnnbigru_reuters_forward():
    from reuters_multiclass import CNNBiGRUMulti
    m = CNNBiGRUMulti(vocab_size=200, num_classes=46, emb_dim=32, hidden_dim=24, dropout=0.1)
    x = torch.randint(1, 200, (4, 50))
    out = m(x)
    assert out.shape == (4, 46)


def test_cnnbigru_reuters_param_count():
    """与 TextCNNMulti / BiGRUMulti 同档"""
    from reuters_multiclass import CNNBiGRUMulti
    m = CNNBiGRUMulti(vocab_size=20000, num_classes=46, emb_dim=128, hidden_dim=160, dropout=0.1)
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 < n < 5_000_000, f"unexpected param count: {n}"
