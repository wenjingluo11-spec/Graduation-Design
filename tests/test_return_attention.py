import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "情感二分类"))
sys.path.insert(0, str(ROOT / "新闻多分类"))

import torch


def test_sentiment_return_attention():
    from sentiment_analysis import TransformerClassifier
    m = TransformerClassifier(
        vocab_size=100, emb_dim=32, num_heads=2, ff_dim=64,
        num_layers=2, dropout=0.1,
    )
    x = torch.randint(1, 100, (3, 20))
    logits = m(x)
    assert logits.shape == (3,)
    logits2, attn = m(x, return_attention=True)
    assert logits2.shape == (3,)
    assert attn.shape == (3, 2, 20, 20)


def test_reuters_return_attention():
    from reuters_multiclass import TransformerMulti
    m = TransformerMulti(
        vocab_size=100, num_classes=46, emb_dim=32, num_heads=2,
        ff_dim=64, num_layers=2, dropout=0.1,
    )
    x = torch.randint(1, 100, (3, 20))
    logits = m(x)
    assert logits.shape == (3, 46)
    logits2, attn = m(x, return_attention=True)
    assert logits2.shape == (3, 46)
    assert attn.shape == (3, 2, 20, 20)
