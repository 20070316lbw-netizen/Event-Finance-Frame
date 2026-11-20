"""metrics.returns 的离线单元测试: 数值正确性, 按 ticker 隔离与输入校验"""

import math

import numpy as np
import pandas as pd
import pytest

from event_finance_frame.metrics import future_returns, historical_return, log_returns


def _panel(prices: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "date": pd.date_range(start, periods=len(values)),
                "ticker": ticker,
                "adj_close": values,
            }
        )
        for ticker, values in prices.items()
    ]
    return pd.concat(frames).set_index(["date", "ticker"])


def test_future_returns_values_and_name():
    df = _panel({"A": [1.0, 2.0, 4.0, 8.0]})
    out = future_returns(df=df, n_periods=1, gap=1)
    assert out.name == "forward_return_1_gap_1"
    # t=0: price[2]/price[1]-1 = 1; t=1: price[3]/price[2]-1 = 1; 末尾不足为 NaN
    assert out.tolist()[:2] == [1.0, 1.0]
    assert out.iloc[2:].isna().all()


def test_historical_return_values():
    df = _panel({"A": [1.0, 2.0, 4.0]})
    out = historical_return(df=df, n_periods=2)
    assert out.name == "historical_return_2"
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == 3.0


def test_log_returns_values():
    df = _panel({"A": [1.0, np.e, np.e**3]})
    out = log_returns(df=df)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1:].round(9).tolist() == [1.0, 2.0]


def test_returns_do_not_leak_across_tickers():
    df = _panel({"A": [1.0, 2.0], "B": [10.0, 30.0]})
    hist = historical_return(df=df, n_periods=1)
    assert hist.loc[(pd.Timestamp("2024-01-02"), "A")] == 1.0
    assert hist.loc[(pd.Timestamp("2024-01-02"), "B")] == 2.0
    assert pd.isna(hist.loc[(pd.Timestamp("2024-01-01"), "B")])


def test_index_level_order_does_not_matter():
    df = _panel({"A": [1.0, 2.0, 4.0]})
    swapped = df.swaplevel().sort_index()
    out = historical_return(df=swapped, n_periods=1)
    assert out.dropna().tolist() == [1.0, 1.0]


def test_empty_input_returns_empty_series():
    df = _panel({"A": [1.0]}).iloc[:0]
    assert future_returns(df=df).empty
    assert historical_return(df=df).empty
    assert log_returns(df=df).empty


def test_missing_price_column_raises_key_error():
    with pytest.raises(KeyError):
        historical_return(df=_panel({"A": [1.0, 2.0]}), price_col="close")


def test_missing_index_level_raises_key_error():
    df = _panel({"A": [1.0, 2.0]}).reset_index("ticker")
    with pytest.raises(KeyError):
        log_returns(df=df)


def test_duplicate_keys_raise_value_error():
    df = _panel({"A": [1.0, 2.0]})
    with pytest.raises(ValueError, match="重复"):
        log_returns(df=pd.concat([df, df]))


def test_non_positive_price_raises_value_error():
    with pytest.raises(ValueError, match="非正数"):
        log_returns(df=_panel({"A": [1.0, 0.0]}))


@pytest.mark.parametrize(("n_periods", "gap"), [(0, 1), (1, -1), (1.5, 1), (True, 1)])
def test_invalid_periods_raise_value_error(n_periods, gap):
    with pytest.raises(ValueError):
        future_returns(df=_panel({"A": [1.0, 2.0]}), n_periods=n_periods, gap=gap)
