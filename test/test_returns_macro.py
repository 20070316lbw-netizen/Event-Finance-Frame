"""测试收益率表宏: pit_historical_return / pit_log_return / label_forward_return"""

from __future__ import annotations

import math

import duckdb
import pytest

from event_finance_frame.config import DDL, MACRO
from event_finance_frame.db.operation import read_sql


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute(read_sql(DDL / "prices.sql"))
    c.execute(read_sql(MACRO / "returns.sql"))

    # AAA: adj_close 每天翻倍; close 固定为 adj_close 的 10 倍 (用来区分 price_col)
    # BBB: 第 3 天 adj_close 缺失, 第 4 天 adj_close 为 0 (脏数据), 用来检查不填补价格
    rows = [
        ("2024-01-01", "AAA", 10.0, 1.0),
        ("2024-01-02", "AAA", 20.0, 2.0),
        ("2024-01-03", "AAA", 40.0, 4.0),
        ("2024-01-04", "AAA", 80.0, 8.0),
        ("2024-01-05", "AAA", 160.0, 16.0),
        ("2024-01-01", "BBB", 100.0, 100.0),
        ("2024-01-02", "BBB", 110.0, 110.0),
        ("2024-01-03", "BBB", 121.0, None),
        ("2024-01-04", "BBB", 133.1, 0.0),
        ("2024-01-05", "BBB", 146.41, 146.41),
    ]
    c.executemany(
        "INSERT INTO prices (date, ticker, close, adj_close, volume) VALUES (?, ?, ?, ?, 0)",
        rows,
    )
    yield c
    c.close()


def _fetch(conn, sql: str) -> list[tuple]:
    return conn.execute(sql).fetchall()


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


# ---------- pit_historical_return ----------


def test_historical_return_values(conn):
    rows = _fetch(
        conn,
        """
        SELECT date::VARCHAR, historical_return
        FROM pit_historical_return(DATE '2024-01-05', n_periods := 2)
        WHERE ticker = 'AAA'
        """,
    )
    assert [r[0] for r in rows] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    assert [r[1] for r in rows] == [None, None, 3.0, 3.0, 3.0]


def test_historical_return_does_not_fill_missing_or_bad_prices(conn):
    rows = _fetch(
        conn,
        """
        SELECT date::VARCHAR, historical_return
        FROM pit_historical_return(DATE '2024-01-05', n_periods := 1)
        WHERE ticker = 'BBB'
        """,
    )
    values = dict(rows)
    assert _close(values["2024-01-02"], 0.1)
    # 缺失和非正价格既不能当分子也不能当分母, 也不会被前值填补
    assert values["2024-01-03"] is None
    assert values["2024-01-04"] is None
    assert values["2024-01-05"] is None


def test_historical_return_price_col_close(conn):
    rows = _fetch(
        conn,
        """
        SELECT historical_return
        FROM pit_historical_return(DATE '2024-01-05', n_periods := 1, price_col := 'close')
        WHERE ticker = 'BBB' AND date >= DATE '2024-01-02'
        """,
    )
    assert all(_close(r[0], 0.1) for r in rows)


def test_historical_return_is_point_in_time(conn):
    rows = _fetch(
        conn,
        "SELECT max(date)::VARCHAR, count(*) FROM pit_historical_return(DATE '2024-01-03', n_periods := 1)",
    )
    assert rows == [("2024-01-03", 6)]


# ---------- pit_log_return ----------


def test_log_return_values(conn):
    rows = _fetch(
        conn,
        """
        SELECT log_return
        FROM pit_log_return(DATE '2024-01-05')
        WHERE ticker = 'AAA'
        """,
    )
    assert rows[0][0] is None
    assert all(_close(r[0], math.log(2)) for r in rows[1:])


def test_log_return_handles_bad_prices_as_null(conn):
    rows = _fetch(
        conn,
        """
        SELECT date::VARCHAR, log_return
        FROM pit_log_return(DATE '2024-01-05')
        WHERE ticker = 'BBB'
        """,
    )
    values = dict(rows)
    assert _close(values["2024-01-02"], math.log(1.1))
    assert values["2024-01-03"] is None
    assert values["2024-01-04"] is None
    assert values["2024-01-05"] is None


def test_log_return_is_point_in_time(conn):
    rows = _fetch(
        conn, "SELECT max(date)::VARCHAR FROM pit_log_return(DATE '2024-01-02')"
    )
    assert rows == [("2024-01-02",)]


# ---------- label_forward_return ----------


def test_forward_return_values_and_dates(conn):
    rows = _fetch(
        conn,
        """
        SELECT date::VARCHAR, entry_date::VARCHAR, exit_date::VARCHAR, forward_return
        FROM label_forward_return(n_periods := 2, gap := 1)
        WHERE ticker = 'AAA'
        """,
    )
    assert rows == [
        ("2024-01-01", "2024-01-02", "2024-01-04", 3.0),
        ("2024-01-02", "2024-01-03", "2024-01-05", 3.0),
        ("2024-01-03", "2024-01-04", None, None),
        ("2024-01-04", "2024-01-05", None, None),
        ("2024-01-05", None, None, None),
    ]


def test_forward_return_gap_zero(conn):
    rows = _fetch(
        conn,
        """
        SELECT entry_date = date, forward_return
        FROM label_forward_return(n_periods := 1, gap := 0)
        WHERE ticker = 'AAA' AND date = DATE '2024-01-01'
        """,
    )
    assert rows == [(True, 1.0)]


def test_forward_return_does_not_leak_across_tickers(conn):
    rows = _fetch(
        conn,
        """
        SELECT ticker, forward_return
        FROM label_forward_return(n_periods := 1, gap := 0)
        WHERE date = DATE '2024-01-05'
        """,
    )
    assert rows == [("AAA", None), ("BBB", None)]


# ---------- 参数校验 ----------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pit_historical_return(DATE '2024-01-05', n_periods := 0)",
        "SELECT * FROM pit_historical_return(DATE '2024-01-05', n_periods := 1.5)",
        "SELECT * FROM pit_historical_return(DATE '2024-01-05', price_col := 'open')",
        "SELECT * FROM pit_log_return(DATE '2024-01-05', price_col := 'volume')",
        "SELECT * FROM label_forward_return(n_periods := 0)",
        "SELECT * FROM label_forward_return(gap := -1)",
        "SELECT * FROM label_forward_return(price_col := 'high')",
    ],
)
def test_invalid_arguments_raise(conn, sql):
    with pytest.raises(duckdb.Error):
        conn.execute(sql).fetchall()


def test_invalid_arguments_raise_on_empty_prices():
    c = duckdb.connect(":memory:")
    c.execute(read_sql(DDL / "prices.sql"))
    c.execute(read_sql(MACRO / "returns.sql"))
    with pytest.raises(duckdb.Error):
        c.execute("SELECT * FROM label_forward_return(n_periods := 0)").fetchall()
    assert c.execute(
        "SELECT count(*) FROM pit_log_return(DATE '2024-01-01')"
    ).fetchone() == (0,)
    c.close()
