"""通用指标: 收益率等"""

from event_finance_frame.metrics.returns import (
    future_returns,
    historical_return,
    log_returns,
)

__all__ = ["future_returns", "historical_return", "log_returns"]
