import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "机器翻译"))

import torch


def test_ls_zero_close_to_ce():
    from machine_translation import LabelSmoothingCrossEntropy, PAD_IDX
    ls = LabelSmoothingCrossEntropy(epsilon=0.0, ignore_index=PAD_IDX)
    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    torch.manual_seed(0)
    logits = torch.randn(20, 50)
    target = torch.randint(1, 50, (20,))  # avoid PAD
    assert torch.allclose(ls(logits, target), ce(logits, target), atol=1e-4)


def test_ls_ignores_pad():
    from machine_translation import LabelSmoothingCrossEntropy, PAD_IDX
    ls = LabelSmoothingCrossEntropy(epsilon=0.1, ignore_index=PAD_IDX)
    logits = torch.randn(4, 50)
    target = torch.tensor([PAD_IDX, PAD_IDX, 1, 2])
    loss = ls(logits, target)
    assert torch.isfinite(loss)


def test_ls_returns_finite_when_all_pad():
    """All-PAD batch should not produce NaN/Inf"""
    from machine_translation import LabelSmoothingCrossEntropy, PAD_IDX
    ls = LabelSmoothingCrossEntropy(epsilon=0.1, ignore_index=PAD_IDX)
    logits = torch.randn(4, 50)
    target = torch.tensor([PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX])
    loss = ls(logits, target)
    assert torch.isfinite(loss)
