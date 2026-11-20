"""市场日历生成与入库: 日期、星期几、是否开盘

核心函数:
    build_calendar(): 由开盘日集合生成连续自然日 DataFrame (date, weekday, is_open)
    load_calendar():  从 prices 推导开盘日, 写入 market_calendar 表并刷新 prices_calendar 视图
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import duckdb
import pandas as pd
from loguru import logger

from event_finance_frame.config import DDL, INSERT, VIEW
from event_finance_frame.db.operation import operation_sql, read_sql

__all__ = ["CALENDAR_COLUMNS", "build_calendar", "load_calendar"]
# 调用方法: from calendar import *

CALENDAR_COLUMNS = ["date", "weekday", "is_open"]
_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def build_calendar(
    open_dates: Iterable[date | str | pd.Timestamp],
    *,
    start: date | str | None = None,
    end: date | str | None = None,
) -> pd.DataFrame:
    """生成覆盖 [start, end] 全部自然日的市场日历

    Args:
        open_dates: 开盘日期集合 (通常取自 prices 的去重日期)
        start: 起始日期, 默认取 open_dates 的最小值
        end: 结束日期, 默认取 open_dates 的最大值

    Returns:
        pd.DataFrame: 列为 date (datetime64), weekday (Mon..Sun), is_open (bool)
        按 date 升序排列; open_dates 为空且未指定起止日期时返回空表

    Raises:
        ValueError: start > end
    """

    # 遍历传入的每个日期 d , 用 pd.Timestamp(d) 统一转成 pandas 的时间格式
    # .normalize() 把时间里的“时分秒”全部清零
    normalized = {pd.Timestamp(d).normalize() for d in open_dates}
    if not normalized and (start is None or end is None):
        """专门指定 datetime64[ns]、string、bool），是为了防止后面把空表塞进 DuckDB 时，数据库猜不出列的类型而报错"""
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "weekday": pd.Series(dtype="string"),
                "is_open": pd.Series(dtype="bool"),
            }
        )

    """
    如果你传了 start，就用你的；
    没传（None），就自动取开盘日里的最早一天 min(normalized)（end 同理）
    接着检查如果起始日期晚于结束日期，直接抛 ValueError 报错拦截
    """
    start_ts = pd.Timestamp(start).normalize() if start is not None else min(normalized)
    end_ts = pd.Timestamp(end).normalize() if end is not None else max(normalized)
    if start_ts > end_ts:
        raise ValueError(
            f"日历起始日期晚于结束日期: {start_ts.date()} > {end_ts.date()}"
        )

    # 按天（D）生成从头到尾连续不断的自然日（包含周末和节假日）
    all_days = pd.date_range(start_ts, end_ts, freq="D")
    return pd.DataFrame(
        {
            "date": all_days,
            "weekday": [
                _WEEKDAY_NAMES[d.weekday()] for d in all_days
            ],  # d.weekday() 会返回 0~6（0 代表周一），拿它当索引去查文件开头定义的 ("Mon", "Tue", ...)，就换成了英文星期几
            "is_open": all_days.isin(
                normalized
            ),  # all_days.isin(normalized)：判断每一天在不在开盘日集合里，直接生成一列 True/False。前面把开盘日转成 set，就是为了让这里的 isin 查得飞快
        }
    )


def load_calendar(
    connection: duckdb.DuckDBPyConnection,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
) -> int:
    """从 prices 提取开盘日, 写入 market_calendar 表并更新 prices_calendar 视图

    Args:
        connection: 可写的 lab 库连接, 调用前不应有未结束的事务
        start: 起始日期, 默认取 prices 最小日期
        end: 结束日期, 默认取 prices 最大日期

    Returns:
        int: 写入 market_calendar 的自然日行数
    """

    # fetchall() 返回的是元组列表，长得像 [(date1,), (date2,), ...]，所以外面套列表推导式 [row[0] for row in ...]，把日期从元组里剥出来
    open_dates = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT date FROM prices ORDER BY date"
        ).fetchall()
    ]
    df = build_calendar(open_dates, start=start, end=end)
    written = operation_sql(
        read_sql(INSERT / "market_calendar.sql"),
        read_sql(DDL / "market_calendar.sql"),
        connection,
        df,
    )
    connection.execute(read_sql(VIEW / "prices_calendar.sql"))
    n_open = int(df["is_open"].sum()) if not df.empty else 0  # 算开盘天数
    logger.info(
        f"market_calendar 写入 {written} 天 (其中开盘 {n_open} 天, 休市 {written - n_open} 天)"
    )
    return written
