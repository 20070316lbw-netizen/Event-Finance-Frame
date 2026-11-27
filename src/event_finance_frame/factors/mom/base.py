"""通用动量因子算子: 基于 SQL 对数收益率面板的滚动求和与跳期后移"""

from __future__ import annotations

import pandas as pd

from event_finance_frame.error import EventFinanceError, FactorErrorCode
from event_finance_frame.factors import factor

__all__ = ["base_momfactor"]


@factor(
    "mom_base",
    source="log_returns",
    lookback=lambda window, skip=0, **_: window + skip,
    description="对数收益率滚动求和动量因子 (按 skip 期后移)",
)
def base_momfactor(
    log_returns: pd.DataFrame | pd.Series,
    window: int,
    *,
    skip: int = 0,
    col: str = "log_return",
) -> pd.Series:
    """计算滚动窗口对数动量, 按 skip 期整体后移

    Args:
        log_returns: [date, ticker] MultiIndex DataFrame 或 Series (不修改输入)
        window: 正整数, 窗口内单期对数收益率数量 (按每只证券自身交易记录计数)
        skip: 默认 0, 非负整数, 剔除最近 skip 期收益 (skip 不包含在 window 内)
        col: 当输入为 DataFrame 时提取的对数收益率列名, 默认 'log_return'
    """
    cap = "mom_base"
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise EventFinanceError(
            FactorErrorCode.INVALID_ARGUMENT,
            f"window 必须是 >= 1 的整数, 实际为 {window!r}",
            capability=cap,
        )
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
        raise EventFinanceError(
            FactorErrorCode.INVALID_ARGUMENT,
            f"skip 必须是非负整数, 实际为 {skip!r}",
            capability=cap,
        )

    if isinstance(log_returns, pd.DataFrame):
        if col not in log_returns.columns:
            raise EventFinanceError(
                FactorErrorCode.MISSING_COLUMN,
                f"输入 DataFrame 缺少对数收益率列 {col!r}",
                capability=cap,
            )
        series = log_returns[col]
    else:
        series = log_returns

    out_name = f"mom_{window}_{skip}"
    if series.empty:
        return pd.Series(dtype="float64", index=series.index, name=out_name)

    values = series.astype("float64")
    rolled = (
        values.groupby(level="ticker", sort=False)
        .rolling(window=window, min_periods=window)
        .sum()
        .droplevel(0)
    )
    if skip > 0:
        rolled = rolled.groupby(level="ticker", sort=False).shift(skip)

    result = rolled.reindex(series.index)
    result.name = out_name
    return result
