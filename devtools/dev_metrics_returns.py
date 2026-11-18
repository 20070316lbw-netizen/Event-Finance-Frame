"""event_finance_frame.metrics.returns 下的三个函数已经全部实际测试全部跑通完毕"""

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
