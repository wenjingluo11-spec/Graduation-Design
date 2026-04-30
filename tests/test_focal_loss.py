import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "新闻多分类"))

import torch


def test_focal_loss_gamma_zero_equals_ce():
    from reuters_multiclass import FocalLoss
    fl = FocalLoss(gamma=0.0)
    ce = torch.nn.CrossEntropyLoss()
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    assert torch.allclose(fl(logits, target), ce(logits, target), atol=1e-5)


def test_focal_loss_gamma_positive_smaller_than_ce_for_confident_correct():
    from reuters_multiclass import FocalLoss
    fl = FocalLoss(gamma=2.0)
    ce_loss = torch.nn.CrossEntropyLoss()
    # 制造 4 个高置信度正确预测 (logit 集中在正确类)
    logits = torch.zeros(4, 3)
    logits[range(4), [0, 1, 2, 0]] = 10.0
    target = torch.tensor([0, 1, 2, 0])
    fl_val = fl(logits, target)
    ce_val = ce_loss(logits, target)
    # focal 应严格小于 ce (因为对易样本下放权重)
    assert fl_val < ce_val


def test_focal_loss_returns_finite_for_pad_targets():
    from reuters_multiclass import FocalLoss
    fl = FocalLoss(gamma=2.0, ignore_index=-100)
    logits = torch.randn(4, 5)
    # 混入 ignore_index
    target = torch.tensor([-100, -100, 1, 2])
    loss = fl(logits, target)
    assert torch.isfinite(loss)
