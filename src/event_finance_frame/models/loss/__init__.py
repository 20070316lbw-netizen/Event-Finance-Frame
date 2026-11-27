"""损失函数子包"""

from event_finance_frame.models.loss.rankloss import ICLoss, PairwiseRankLoss

__all__ = ["ICLoss", "PairwiseRankLoss"]
