"""收益率相关指标

输入统一为 [date, ticker] 两层 MultiIndex 的行情面板 (层顺序不限), 例如::

    prices = con.execute("SELECT date, ticker, adj_close FROM prices").df()
    panel = prices.set_index(["date", "ticker"])

核心函数:
    future_returns():    未来持有期收益率 (使用未来数据, 只能做标签)
    historical_return(): 截至当天的历史收益率 (可做特征)
    log_returns():       相邻两条记录的对数收益率
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["future_returns", "historical_return", "log_returns"]

_INDEX_LEVELS = ("date", "ticker")


def _price_series(df: pd.DataFrame, price_col: str) -> pd.Series:
    """校验行情面板并取出按索引排序后的价格列

    Raises:
        KeyError: 缺少 price_col 列或 date / ticker 索引层
        ValueError: 索引键缺失或重复, 或非缺失价格不为正数
    """
    if price_col not in df.columns:
        raise KeyError(f"行情面板缺少价格列: {price_col}")
    missing = [name for name in _INDEX_LEVELS if name not in (df.index.names or [])]
    if missing:
        raise KeyError(
            f"行情面板缺少索引层: {missing}, 当前索引层为 {list(df.index.names)}"
        )

    for name in _INDEX_LEVELS:
        if df.index.get_level_values(name).isna().any():
            raise ValueError(f"行情面板索引层 {name} 存在缺失值")
    if df.index.has_duplicates:
        raise ValueError("行情面板存在重复的 [date, ticker] 键")

    prices = pd.to_numeric(df[price_col], errors="raise").astype("float64")
    if (prices.dropna() <= 0).any():
        raise ValueError(f"价格列 {price_col} 存在非正数")
    return prices.sort_index()


def _check_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数, 实际为 {value!r}")


def future_returns(
    *, df: pd.DataFrame, n_periods: int = 5, gap: int = 1, price_col: str = "adj_close"
) -> pd.Series:
    """生成未来持有期的简单收益率标签

    警告:
        使用未来数据, 只能作为监督模型的标签, 不能当作基准日的特征,
        也不能出现在回测引擎的 PIT 数据视图里

    Args:
        df: [date, ticker] 索引的行情面板, 含 price_col 列, 键非空且唯一
        n_periods: 持有期数, 默认 5, 正整数, 按每只证券的交易记录计数
        gap: 当天到入场的记录数, 默认 1, 非负整数
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数

    Returns:
        按索引排序的 Series, name 为 forward_return_{n_periods}_gap_{gap}
        值为 price[t+gap+n_periods] / price[t+gap] - 1, 单位为小数收益率
        未来记录不足或端点缺失时为 NaN, 不填补价格; 空输入返回空 Series
        与 split_dates 合用时, embargo 取 gap + n_periods,
        因为标签用到的最后价格落在 t + gap + n_periods

    Raises:
        ValueError: 参数不合法, 索引键缺失或重复, 或价格非正
        KeyError: price_col 指定的列或 date / ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1.0, 2.0, 4.0]}, index=index)
        >>> future_returns(df=prices, n_periods=1, gap=0).iloc[:2].tolist()
        [1.0, 1.0]
    """
    _check_int("n_periods", n_periods, minimum=1)
    _check_int("gap", gap, minimum=0)
    prices = _price_series(df, price_col)
    grouped = prices.groupby(level="ticker", sort=False)
    entry = grouped.shift(-gap)
    exit_price = grouped.shift(-(gap + n_periods))
    return (exit_price / entry - 1).rename(f"forward_return_{n_periods}_gap_{gap}")


def historical_return(
    *,
    df: pd.DataFrame,
    n_periods: int = 5,
    price_col: str = "adj_close",
) -> pd.Series:
    """计算截至基准日期的历史简单收益率

    Args:
        df: [date, ticker] 索引的行情面板, 含 price_col 列, 键非空且唯一
        n_periods: 回看期数, 默认 5, 正整数
            按每只证券自己的交易记录计数, 不补齐交易日
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数

    Returns:
        按索引排序的 Series, name 为 historical_return_{n_periods}
        值为 price[t] / price[t-n_periods] - 1, 单位为小数收益率
        包含当天价格; 历史记录不足或端点缺失时为 NaN, 不填补价格
        空输入返回空 Series, 可用于历史特征

    Raises:
        ValueError: 参数不合法, 索引键缺失或重复, 或价格非正
        KeyError: price_col 指定的列或 date / ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1.0, 2.0, 4.0]}, index=index)
        >>> historical_return(df=prices, n_periods=1).iloc[1:].tolist()
        [1.0, 1.0]
    """
    _check_int("n_periods", n_periods, minimum=1)
    prices = _price_series(df, price_col)
    grouped = prices.groupby(level="ticker", sort=False)
    previous = grouped.shift(n_periods)
    return (prices / previous - 1).rename(f"historical_return_{n_periods}")


def log_returns(
    *,
    df: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.Series:
    """在每只证券内计算相邻两条价格的对数收益率

    Args:
        df: [date, ticker] 索引的行情面板, 含 price_col 列, 键非空且唯一
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数

    Returns:
        按索引排序的 Series, name 为 log_return, 单位为对数收益率
        值为 log(price[t]) - log(price[t-1]); 每只证券首行或相邻价格缺失时
        为 NaN, 不填补交易记录或价格; 空输入返回空 Series

    Raises:
        ValueError: 索引键缺失或重复, 或价格非正
        KeyError: price_col 指定的列或 date / ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=2), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1.0, np.e]}, index=index)
        >>> round(float(log_returns(df=prices).iloc[1]), 6)
        1.0
    """
    prices = _price_series(df, price_col)
    return (
        np.log(prices).groupby(level="ticker", sort=False).diff().rename("log_return")
    )
