"""市场日历生成、入库与交易日历序列查询

核心函数与类:
    build_calendar(): 由开盘日集合生成连续自然日 DataFrame (date, weekday, is_open)
    load_calendar():  从 prices 推导开盘日, 写入 market_calendar 表并刷新 prices_calendar 视图
    normalize_date(): 将常见日期表示统一规范为 datetime.date
    TradingCalendar:  不可变的交易日有序序列, 支持二分定位、前后交易日查询、偏移与切片
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from typing import overload

import duckdb
import pandas as pd
from loguru import logger

from event_finance_frame.config import DDL, INSERT, VIEW
from event_finance_frame.db.operation import operation_sql, read_sql
from event_finance_frame.error import ClockErrorCode, EventFinanceError

__all__ = [
    "CALENDAR_COLUMNS",
    "DateLike",
    "TradingCalendar",
    "build_calendar",
    "load_calendar",
    "normalize_date",
]
# 调用方法: from calendar import *

DateLike = date | str | pd.Timestamp

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


def normalize_date(value: DateLike) -> date:
    """将常见日期表示 (date / datetime / pd.Timestamp / str) 规范为 datetime.date"""
    if value is pd.NaT:
        raise ValueError("日期不能为 NaT")
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError("日期不能为 NaT")
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("日期字符串不能为空")
        try:
            return date.fromisoformat(text)
        except ValueError:
            ts = pd.Timestamp(text)
            if pd.isna(ts):
                raise ValueError(f"无效的日期字符串: {value!r}") from None
            return ts.date()
    raise TypeError(f"不支持的日期类型: {type(value).__name__}")


class TradingCalendar:
    """不可变的有序交易日历序列

    提供基于二分查找的交易日判断、前后交易日定位、交易日偏移、回看窗口与区间切片.
    """

    def __init__(self, trading_dates: Iterable[DateLike]) -> None:
        unique_dates = sorted({normalize_date(d) for d in trading_dates})
        if not unique_dates:
            raise EventFinanceError(
                ClockErrorCode.EMPTY_CALENDAR,
                "交易日历不能为空",
                capability="TradingCalendar",
            )
        self._dates: tuple[date, ...] = tuple(unique_dates)

    @classmethod
    def from_duckdb(
        cls,
        connection: duckdb.DuckDBPyConnection,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> TradingCalendar:
        """从 DuckDB 加载交易日历 (优先读取 market_calendar, 否则回退到 prices)"""
        start_dt = normalize_date(start) if start is not None else None
        end_dt = normalize_date(end) if end is not None else None
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise EventFinanceError(
                ClockErrorCode.INVALID_DATE_RANGE,
                f"起始日期晚于结束日期: {start_dt} > {end_dt}",
                capability="TradingCalendar.from_duckdb",
            )

        has_market_cal = bool(
            connection.execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = 'market_calendar'
                """
            ).fetchone()[0]
        )

        conditions: list[str] = []
        params: list[date] = []
        if has_market_cal:
            base_sql = "SELECT date FROM market_calendar"
            conditions.append("is_open = TRUE")
        else:
            base_sql = "SELECT DISTINCT date FROM prices"

        if start_dt is not None:
            conditions.append("date >= CAST(? AS DATE)")
            params.append(start_dt)
        if end_dt is not None:
            conditions.append("date <= CAST(? AS DATE)")
            params.append(end_dt)

        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"{base_sql}{where_sql} ORDER BY date"
        rows = connection.execute(sql, params).fetchall()
        return cls(row[0] for row in rows)

    @property
    def dates(self) -> tuple[date, ...]:
        return self._dates

    @property
    def start_date(self) -> date:
        return self._dates[0]

    @property
    def end_date(self) -> date:
        return self._dates[-1]

    def __len__(self) -> int:
        return len(self._dates)

    @overload
    def __getitem__(self, idx: int) -> date: ...

    @overload
    def __getitem__(self, idx: slice) -> tuple[date, ...]: ...

    def __getitem__(self, idx: int | slice) -> date | tuple[date, ...]:
        return self._dates[idx]

    def __iter__(self) -> Iterator[date]:
        return iter(self._dates)

    def __contains__(self, item: object) -> bool:
        try:
            dt = normalize_date(item)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        pos = bisect_left(self._dates, dt)
        return pos < len(self._dates) and self._dates[pos] == dt

    def __repr__(self) -> str:
        return (
            f"TradingCalendar(n={len(self._dates)}, "
            f"start={self.start_date.isoformat()}, end={self.end_date.isoformat()})"
        )

    def is_trading_day(self, d: DateLike) -> bool:
        dt = normalize_date(d)
        pos = bisect_left(self._dates, dt)
        return pos < len(self._dates) and self._dates[pos] == dt

    def index_of(self, d: DateLike) -> int:
        dt = normalize_date(d)
        pos = bisect_left(self._dates, dt)
        if pos >= len(self._dates) or self._dates[pos] != dt:
            raise EventFinanceError(
                ClockErrorCode.NON_TRADING_DAY,
                f"非交易日: {dt}",
                capability="TradingCalendar.index_of",
            )
        return pos

    def next_trading_day(self, d: DateLike, *, inclusive: bool = False) -> date | None:
        dt = normalize_date(d)
        pos = bisect_left(self._dates, dt) if inclusive else bisect_right(self._dates, dt)
        return self._dates[pos] if pos < len(self._dates) else None

    def prev_trading_day(self, d: DateLike, *, inclusive: bool = False) -> date | None:
        dt = normalize_date(d)
        pos = bisect_right(self._dates, dt) - 1 if inclusive else bisect_left(self._dates, dt) - 1
        return self._dates[pos] if pos >= 0 else None

    def shift(self, d: DateLike, n: int) -> date | None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise EventFinanceError(
                ClockErrorCode.INVALID_STEP,
                f"偏移量 n 必须是整数, 实际为 {n!r}",
                capability="TradingCalendar.shift",
            )
        target = self.index_of(d) + n
        return self._dates[target] if 0 <= target < len(self._dates) else None

    def window(self, end: DateLike, lookback: int) -> tuple[date, ...]:
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise EventFinanceError(
                ClockErrorCode.INVALID_STEP,
                f"lookback 必须是 >= 1 的整数, 实际为 {lookback!r}",
                capability="TradingCalendar.window",
            )
        end_dt = normalize_date(end)
        right = bisect_right(self._dates, end_dt)
        left = max(0, right - lookback)
        return self._dates[left:right]

    def slice(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> TradingCalendar:
        start_dt = normalize_date(start) if start is not None else None
        end_dt = normalize_date(end) if end is not None else None
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise EventFinanceError(
                ClockErrorCode.INVALID_DATE_RANGE,
                f"起始日期晚于结束日期: {start_dt} > {end_dt}",
                capability="TradingCalendar.slice",
            )
        left = bisect_left(self._dates, start_dt) if start_dt is not None else 0
        right = bisect_right(self._dates, end_dt) if end_dt is not None else len(self._dates)
        return TradingCalendar(self._dates[left:right])
