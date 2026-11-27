"""测试 TradingCalendar 与 Clock"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd
import pytest

from event_finance_frame import (
    Clock,
    ClockErrorCode,
    EventFinanceError,
    TradingCalendar,
)
from event_finance_frame.calendar import load_calendar, normalize_date
from event_finance_frame.config import DDL
from event_finance_frame.db.operation import read_sql


def test_normalize_date():
    expected = date(2024, 1, 2)
    assert normalize_date(date(2024, 1, 2)) == expected
    assert normalize_date(datetime(2024, 1, 2, 15, 30)) == expected
    assert normalize_date(pd.Timestamp("2024-01-02 16:00:00")) == expected
    assert normalize_date("2024-01-02") == expected
    assert normalize_date("  2024-01-02  ") == expected
    assert normalize_date("2024/01/02") == expected

    with pytest.raises(ValueError):
        normalize_date(pd.NaT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        normalize_date("")
    with pytest.raises(ValueError):
        normalize_date("NaT")
    with pytest.raises(TypeError):
        normalize_date(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        normalize_date(None)  # type: ignore[arg-type]


def test_trading_calendar_basics():
    cal = TradingCalendar(["2024-01-05", date(2024, 1, 2), pd.Timestamp("2024-01-03"), "2024-01-02"])
    assert cal.dates == (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5))
    assert cal.start_date == date(2024, 1, 2)
    assert cal.end_date == date(2024, 1, 5)
    assert len(cal) == 3
    assert cal[0] == date(2024, 1, 2)
    assert cal[1:] == (date(2024, 1, 3), date(2024, 1, 5))
    assert list(cal) == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]
    assert "2024-01-03" in cal
    assert "2024-01-04" not in cal
    assert 12345 not in cal

    with pytest.raises(EventFinanceError) as exc:
        TradingCalendar([])
    assert exc.value.code is ClockErrorCode.EMPTY_CALENDAR


def test_trading_calendar_queries():
    cal = TradingCalendar(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])

    assert cal.is_trading_day("2024-01-03")
    assert not cal.is_trading_day("2024-01-04")
    assert cal.index_of("2024-01-05") == 2

    with pytest.raises(EventFinanceError) as exc:
        cal.index_of("2024-01-04")
    assert exc.value.code is ClockErrorCode.NON_TRADING_DAY

    assert cal.next_trading_day("2024-01-03", inclusive=False) == date(2024, 1, 5)
    assert cal.next_trading_day("2024-01-03", inclusive=True) == date(2024, 1, 3)
    assert cal.next_trading_day("2024-01-04", inclusive=False) == date(2024, 1, 5)
    assert cal.next_trading_day("2024-01-08", inclusive=False) is None

    assert cal.prev_trading_day("2024-01-05", inclusive=False) == date(2024, 1, 3)
    assert cal.prev_trading_day("2024-01-05", inclusive=True) == date(2024, 1, 5)
    assert cal.prev_trading_day("2024-01-04", inclusive=False) == date(2024, 1, 3)
    assert cal.prev_trading_day("2024-01-02", inclusive=False) is None

    assert cal.shift("2024-01-03", 0) == date(2024, 1, 3)
    assert cal.shift("2024-01-03", 2) == date(2024, 1, 8)
    assert cal.shift("2024-01-05", -2) == date(2024, 1, 2)
    assert cal.shift("2024-01-08", 1) is None
    assert cal.shift("2024-01-02", -1) is None
    with pytest.raises(EventFinanceError) as exc_shift:
        cal.shift("2024-01-03", True)  # type: ignore[arg-type]
    assert exc_shift.value.code is ClockErrorCode.INVALID_STEP

    assert cal.window("2024-01-05", 2) == (date(2024, 1, 3), date(2024, 1, 5))
    assert cal.window("2024-01-04", 10) == (date(2024, 1, 2), date(2024, 1, 3))
    with pytest.raises(EventFinanceError) as exc_win:
        cal.window("2024-01-05", 0)
    assert exc_win.value.code is ClockErrorCode.INVALID_STEP
    with pytest.raises(EventFinanceError) as exc_win_bool:
        cal.window("2024-01-05", True)  # type: ignore[arg-type]
    assert exc_win_bool.value.code is ClockErrorCode.INVALID_STEP

    sliced = cal.slice(start="2024-01-03", end="2024-01-06")
    assert sliced.dates == (date(2024, 1, 3), date(2024, 1, 5))
    with pytest.raises(EventFinanceError) as exc_rng:
        cal.slice(start="2024-01-08", end="2024-01-02")
    assert exc_rng.value.code is ClockErrorCode.INVALID_DATE_RANGE
    with pytest.raises(EventFinanceError) as exc_empty:
        cal.slice(start="2024-01-06", end="2024-01-07")
    assert exc_empty.value.code is ClockErrorCode.EMPTY_CALENDAR


def test_trading_calendar_from_duckdb_with_and_without_market_calendar():
    con = duckdb.connect(":memory:")
    con.execute(read_sql(DDL / "prices.sql"))
    con.executemany(
        "INSERT INTO prices (date, ticker, close, adj_close, volume) VALUES (?, 'AAA', 10.0, 10.0, 100)",
        [("2024-01-02",), ("2024-01-03",), ("2024-01-05",)],
    )

    # 无 market_calendar 时回退读取 prices
    cal_fallback = TradingCalendar.from_duckdb(con)
    assert cal_fallback.dates == (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5))

    # 建好 market_calendar 后优先读取 is_open = TRUE
    load_calendar(con)
    cal_mc = TradingCalendar.from_duckdb(con, start="2024-01-03", end="2024-01-05")
    assert cal_mc.dates == (date(2024, 1, 3), date(2024, 1, 5))

    with pytest.raises(EventFinanceError) as exc_rng:
        TradingCalendar.from_duckdb(con, start="2024-01-05", end="2024-01-02")
    assert exc_rng.value.code is ClockErrorCode.INVALID_DATE_RANGE

    with pytest.raises(EventFinanceError) as exc_empty:
        TradingCalendar.from_duckdb(con, start="2024-01-10", end="2024-01-12")
    assert exc_empty.value.code is ClockErrorCode.EMPTY_CALENDAR
    con.close()


def test_clock_stepping_and_properties():
    cal = TradingCalendar(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    clk = Clock(cal)

    assert len(clk) == 4
    assert clk.start_date == date(2024, 1, 2)
    assert clk.end_date == date(2024, 1, 5)
    assert clk.now == date(2024, 1, 2)
    assert clk.index == 0
    assert clk.remaining == 3
    assert not clk.is_end
    assert clk.prev_date is None
    assert clk.next_date == date(2024, 1, 3)

    assert clk.step() == date(2024, 1, 3)
    assert clk.prev_date == date(2024, 1, 2)
    assert clk.next_date == date(2024, 1, 4)
    assert clk.remaining == 2

    assert clk.step(2) == date(2024, 1, 5)
    assert clk.is_end
    assert clk.remaining == 0
    assert clk.next_date is None

    # 超出范围不移动当前位置
    assert clk.step() is None
    assert clk.now == date(2024, 1, 5)

    for bad in (0, -1, True):
        with pytest.raises(EventFinanceError) as exc:
            clk.step(bad)
        assert exc.value.code is ClockErrorCode.INVALID_STEP


def test_clock_seek_and_reset():
    cal = TradingCalendar(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    clk = Clock(cal)

    assert clk.seek("2024-01-03") == date(2024, 1, 3)
    assert clk.seek("2024-01-04", method="next") == date(2024, 1, 5)

    with pytest.raises(EventFinanceError) as exc_back:
        clk.seek("2024-01-03")
    assert exc_back.value.code is ClockErrorCode.BACKWARD_SEEK
    assert clk.now == date(2024, 1, 5)

    assert clk.seek("2024-01-04", method="prev", allow_backward=True) == date(2024, 1, 3)

    with pytest.raises(EventFinanceError) as exc_nontrading:
        clk.seek("2024-01-04", method="exact")
    assert exc_nontrading.value.code is ClockErrorCode.NON_TRADING_DAY

    for method, target in [
        ("exact", "2024-01-01"),
        ("exact", "2024-01-09"),
        ("next", "2024-01-09"),
        ("prev", "2024-01-01"),
    ]:
        with pytest.raises(EventFinanceError) as exc_oor:
            clk.seek(target, method=method, allow_backward=True)  # type: ignore[arg-type]
        assert exc_oor.value.code is ClockErrorCode.OUT_OF_RANGE

    with pytest.raises(ValueError):
        clk.seek("2024-01-05", method="invalid")  # type: ignore[arg-type]

    assert clk.reset() == date(2024, 1, 2)
    assert clk.index == 0


def test_clock_iteration_and_seek_in_loop():
    cal = TradingCalendar(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    clk = Clock(cal, start="2024-01-03", end="2024-01-05")

    visited: list[tuple[date, date]] = []
    for d in clk:
        visited.append((d, clk.now))
    assert visited == [
        (date(2024, 1, 3), date(2024, 1, 3)),
        (date(2024, 1, 4), date(2024, 1, 4)),
        (date(2024, 1, 5), date(2024, 1, 5)),
    ]

    # 循环内手动 seek 跳日: 从 01-03 跳至 01-04 后, 下一轮步进到 01-05
    cal_long = TradingCalendar(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    clk_long = Clock(cal_long, start="2024-01-03", end="2024-01-05")
    seen: list[date] = []
    for d in clk_long:
        seen.append(d)
        if d == date(2024, 1, 3):
            clk_long.seek("2024-01-04")
    assert seen == [date(2024, 1, 3), date(2024, 1, 5)]


def test_clock_from_duckdb():
    con = duckdb.connect(":memory:")
    con.execute(read_sql(DDL / "prices.sql"))
    con.executemany(
        "INSERT INTO prices (date, ticker, close, adj_close, volume) VALUES (?, 'AAA', 10.0, 10.0, 100)",
        [("2024-01-02",), ("2024-01-03",), ("2024-01-04",)],
    )
    clk = Clock.from_duckdb(con, start="2024-01-03")
    assert clk.start_date == date(2024, 1, 3)
    assert clk.end_date == date(2024, 1, 4)
    assert clk.now == date(2024, 1, 3)
    con.close()
