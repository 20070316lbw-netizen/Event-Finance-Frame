"""交易日时钟: 沿 TradingCalendar 推进或跳转模拟当前日期 now"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Literal

import duckdb

from event_finance_frame.calendar import DateLike, TradingCalendar, normalize_date
from event_finance_frame.error import ClockErrorCode, EventFinanceError

__all__ = ["Clock"]

_SEEK_METHODS = frozenset({"exact", "next", "prev"})


class Clock:
    """围绕交易日历运行的单调时钟

    Args:
        calendar: 底层交易日历
        start: 可选起始日期截断
        end: 可选结束日期截断
    """

    def __init__(
        self,
        calendar: TradingCalendar,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> None:
        self._calendar = (
            calendar.slice(start=start, end=end)
            if (start is not None or end is not None)
            else calendar
        )
        self._idx = 0

    @classmethod
    def from_duckdb(
        cls,
        connection: duckdb.DuckDBPyConnection,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> Clock:
        """从 DuckDB 数据库构建交易日历并初始化时钟"""
        return cls(TradingCalendar.from_duckdb(connection, start=start, end=end))

    @property
    def calendar(self) -> TradingCalendar:
        return self._calendar

    @property
    def now(self) -> date:
        return self._calendar[self._idx]

    @property
    def index(self) -> int:
        return self._idx

    @property
    def start_date(self) -> date:
        return self._calendar.start_date

    @property
    def end_date(self) -> date:
        return self._calendar.end_date

    @property
    def remaining(self) -> int:
        return len(self._calendar) - 1 - self._idx

    @property
    def is_end(self) -> bool:
        return self._idx == len(self._calendar) - 1

    @property
    def prev_date(self) -> date | None:
        return self._calendar[self._idx - 1] if self._idx > 0 else None

    @property
    def next_date(self) -> date | None:
        return self._calendar[self._idx + 1] if self._idx + 1 < len(self._calendar) else None

    def __len__(self) -> int:
        return len(self._calendar)

    def __repr__(self) -> str:
        return (
            f"Clock(now={self.now.isoformat()}, "
            f"index={self._idx}/{len(self._calendar) - 1}, "
            f"range=[{self.start_date.isoformat()}, {self.end_date.isoformat()}])"
        )

    def step(self, n: int = 1) -> date | None:
        """向前推进 n 个交易日; 超出结束日期时保持原位并返回 None"""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise EventFinanceError(
                ClockErrorCode.INVALID_STEP,
                f"步长 n 必须是 >= 1 的整数, 实际为 {n!r}",
                capability="Clock.step",
            )
        target = self._idx + n
        if target >= len(self._calendar):
            return None
        self._idx = target
        return self.now

    def seek(
        self,
        target: DateLike,
        *,
        method: Literal["exact", "next", "prev"] = "exact",
        allow_backward: bool = False,
    ) -> date:
        """跳转到指定交易日

        Args:
            target: 目标日期
            method: 'exact' 要求精确匹配交易日; 'next' 取 >= target 的首个交易日;
                    'prev' 取 <= target 的最后一个交易日
            allow_backward: 默认 False, 禁止向过去回退
        """
        if method not in _SEEK_METHODS:
            raise ValueError(
                f"不支持的 seek method: {method!r}, 仅支持 'exact', 'next', 'prev'"
            )

        target_dt = normalize_date(target)
        if method == "exact":
            if target_dt < self.start_date or target_dt > self.end_date:
                raise EventFinanceError(
                    ClockErrorCode.OUT_OF_RANGE,
                    f"目标日期 {target_dt} 超出时钟范围 [{self.start_date}, {self.end_date}]",
                    capability="Clock.seek",
                )
            target_idx = self._calendar.index_of(target_dt)
        elif method == "next":
            next_dt = self._calendar.next_trading_day(target_dt, inclusive=True)
            if next_dt is None:
                raise EventFinanceError(
                    ClockErrorCode.OUT_OF_RANGE,
                    f"目标日期 {target_dt} 之后不存在有效交易日 (end={self.end_date})",
                    capability="Clock.seek",
                )
            target_idx = self._calendar.index_of(next_dt)
        else:
            prev_dt = self._calendar.prev_trading_day(target_dt, inclusive=True)
            if prev_dt is None:
                raise EventFinanceError(
                    ClockErrorCode.OUT_OF_RANGE,
                    f"目标日期 {target_dt} 之前不存在有效交易日 (start={self.start_date})",
                    capability="Clock.seek",
                )
            target_idx = self._calendar.index_of(prev_dt)

        if target_idx < self._idx and not allow_backward:
            raise EventFinanceError(
                ClockErrorCode.BACKWARD_SEEK,
                f"禁止时钟向过去回退: {self.now} -> {self._calendar[target_idx]}",
                capability="Clock.seek",
            )

        self._idx = target_idx
        return self.now

    def reset(self) -> date:
        """将时钟重置回起始交易日"""
        self._idx = 0
        return self.now

    def iter_dates(self) -> Iterator[date]:
        """从当前 now 逐日推进直到 end_date (包含当前与末尾交易日)"""
        while True:
            yield self.now
            if self.step() is None:
                break

    def __iter__(self) -> Iterator[date]:
        return self.iter_dates()
