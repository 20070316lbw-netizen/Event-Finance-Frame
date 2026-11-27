"""横截面排序损失函数"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PairwiseRankLoss(nn.Module):
    """成对排序损失 (RankNet / Logistic Pairwise Loss)

    对同一横截面内满足 ``y_i > y_j`` 的股票对, 惩罚 ``score_i <= score_j`` 的情况.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        mask = target_diff > 0
        if not mask.any():
            return scores.sum() * 0.0
        return F.softplus(self.margin - score_diff[mask]).mean()


class ICLoss(nn.Module):
    """负 Pearson IC 损失, 直接最大化预测分数与目标收益的横截面线性相关性"""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if scores.numel() < 2:
            return scores.sum() * 0.0
        s = scores - scores.mean()
        t = targets - targets.mean()
        denom = (s.pow(2).sum().sqrt() * t.pow(2).sum().sqrt()).clamp_min(self.eps)
        return -(s * t).sum() / denom
