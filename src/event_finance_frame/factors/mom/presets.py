"""预设经典动量因子 (基于 base_momfactor 构建)"""

from __future__ import annotations

import pandas as pd

from event_finance_frame.factors import factor
from event_finance_frame.factors.mom.base import base_momfactor

__all__ = [
    "mom_1m",
    "mom_3_1",
    "mom_6_1",
    "mom_12_1",
]


@factor(
    "mom_12_1",
    source="log_returns",
    lookback=252,
    description="经典 12-1 月动量 (window=231, skip=21)",
)
def mom_12_1(log_returns: pd.DataFrame | pd.Series, *, col: str = "log_return") -> pd.Series:
    """过去 12 个月剔除最近 1 个月的动量因子 (231 + 21 个交易日)"""
    out = base_momfactor(log_returns, window=231, skip=21, col=col)
    out.name = "mom_12_1"
    return out


@factor(
    "mom_6_1",
    source="log_returns",
    lookback=126,
    description="6-1 月中期动量 (window=105, skip=21)",
)
def mom_6_1(log_returns: pd.DataFrame | pd.Series, *, col: str = "log_return") -> pd.Series:
    """过去 6 个月剔除最近 1 个月的动量因子 (105 + 21 个交易日)"""
    out = base_momfactor(log_returns, window=105, skip=21, col=col)
    out.name = "mom_6_1"
    return out


@factor(
    "mom_3_1",
    source="log_returns",
    lookback=63,
    description="3-1 月动量 (window=42, skip=21)",
)
def mom_3_1(log_returns: pd.DataFrame | pd.Series, *, col: str = "log_return") -> pd.Series:
    """过去 3 个月剔除最近 1 个月的动量因子 (42 + 21 个交易日)"""
    out = base_momfactor(log_returns, window=42, skip=21, col=col)
    out.name = "mom_3_1"
    return out


@factor(
    "mom_1m",
    source="log_returns",
    lookback=21,
    description="1 月短期动量 (window=21, skip=0)",
)
def mom_1m(log_returns: pd.DataFrame | pd.Series, *, col: str = "log_return") -> pd.Series:
    """过去 21 个交易日的动量因子"""
    out = base_momfactor(log_returns, window=21, skip=0, col=col)
    out.name = "mom_1m"
    return out
