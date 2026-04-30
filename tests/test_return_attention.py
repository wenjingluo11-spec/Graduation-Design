import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "情感二分类"))
sys.path.insert(0, str(ROOT / "新闻多分类"))
sys.path.insert(0, str(ROOT / "机器翻译"))

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


def test_translation_transformer_attention():
    from machine_translation import TransformerTranslator
    m = TransformerTranslator(
        src_vocab_size=100, tgt_vocab_size=120, d_model=32,
        nhead=2, num_layers=2, ff_dim=64, dropout=0.1,
    )
    src = torch.randint(1, 100, (2, 8))
    tgt = torch.randint(1, 120, (2, 6))
    logits = m(src, tgt)
    assert logits.shape == (2, 6, 120)
    logits2, attn = m(src, tgt, return_attention=True)
    assert logits2.shape == (2, 6, 120)
    assert attn.shape == (2, 2, 6, 8)


def test_seq2seq_forward_unchanged():
    from machine_translation import Encoder, Decoder, Seq2Seq
    enc = Encoder(vocab_size=100, emb_dim=16, hidden_dim=24, dropout=0.1)
    dec = Decoder(vocab_size=120, emb_dim=16, hidden_dim=24, dropout=0.1)
    s2s = Seq2Seq(enc, dec, max_len=20)
    src = torch.randint(1, 100, (2, 8))
    tgt = torch.randint(1, 120, (2, 6))
    out = s2s(src, tgt, teacher_forcing_ratio=1.0)
    assert out.shape == (2, 6, 120)
