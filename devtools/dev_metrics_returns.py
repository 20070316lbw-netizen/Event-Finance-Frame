"""用 test.duckdb 的真实 AAPL 行情目测 metrics.returns 三个函数的输出

单元测试见 test/test_metrics_returns.py; 本脚本只用于人工核对真实数据
"""

from __future__ import annotations

import pandas as pd

from event_finance_frame.config import TEST_DB
from event_finance_frame.db import get_duckdb
from event_finance_frame.metrics.returns import (
    future_returns,
    historical_return,
    log_returns,
)

with get_duckdb(path=TEST_DB, read_only=True) as con:
    prices = con.execute(
        """
        SELECT ticker, date, close
        FROM prices
        WHERE ticker = 'AAPL'
          AND date BETWEEN DATE '2026-01-01' AND DATE '2026-01-31'
        ORDER BY date
        """
    ).df()

prices = prices.set_index(["ticker", "date"]).sort_index()

fr = future_returns(df=prices, price_col="close")
hr = historical_return(df=prices, price_col="close")
lr = log_returns(df=prices, price_col="close")

result = pd.concat(
    [
        fr.rename("future_return"),
        hr.rename("historical_return"),
        lr.rename("log_return"),
    ],
    axis=1,
)

print(result)
