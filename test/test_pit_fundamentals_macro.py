"""测试 PIT 基本面表宏: pit_fundamentals 与 pit_fundamentals_by_ticker"""

from __future__ import annotations

import duckdb
import pytest

from event_finance_frame.config import DDL, MACRO
from event_finance_frame.db.operation import read_sql


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute(read_sql(DDL / "universe.sql"))
    c.execute(read_sql(DDL / "fundamental_instant.sql"))
    c.execute(read_sql(DDL / "fundamental_period.sql"))
    c.execute(read_sql(MACRO / "pit_fundamentals.sql"))

    # 2 期 universe 快照:
    #   2024-01-01: 只有 AAA
    #   2024-06-01: AAA + BBB
    c.executemany(
        "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("2024-01-01", "AAA", "0000000001", "Alpha", "NASDAQ", None, "Tech"),
            ("2024-06-01", "AAA", "0000000001", "Alpha", "NASDAQ", None, "Tech"),
            ("2024-06-01", "BBB", "0000000002", "Beta", "NYSE", None, "Finance"),
        ],
    )

    # AAA 的 instant 事实: 7 个季度的 Assets + 1 条修订 + 1 条 8-K
    instant_rows = [
        ("0000000001", "us-gaap", "Assets", "USD", "2022-09-30", 10.0, "a1", "2022-11-01", "10-Q", 2022, "Q3", None),
        ("0000000001", "us-gaap", "Assets", "USD", "2022-12-31", 20.0, "a2", "2023-02-01", "10-K", 2022, "FY", None),
        ("0000000001", "us-gaap", "Assets", "USD", "2023-03-31", 30.0, "a3", "2023-05-01", "10-Q", 2023, "Q1", None),
        ("0000000001", "us-gaap", "Assets", "USD", "2023-06-30", 40.0, "a4", "2023-08-01", "10-Q", 2023, "Q2", None),
        ("0000000001", "us-gaap", "Assets", "USD", "2023-09-30", 50.0, "a5", "2023-11-01", "10-Q", 2023, "Q3", None),
        ("0000000001", "us-gaap", "Assets", "USD", "2023-12-31", 60.0, "a6", "2024-02-01", "10-K", 2023, "FY", None),
        # 同一报告期 2023-12-31 在 2024-05-01 被 10-K/A 修订为 66.0
        ("0000000001", "us-gaap", "Assets", "USD", "2023-12-31", 66.0, "a6-rev", "2024-05-01", "10-K/A", 2023, "FY", None),
        # 2024-03-31 季报于 2024-05-10 申报
        ("0000000001", "us-gaap", "Assets", "USD", "2024-03-31", 70.0, "a7", "2024-05-10", "10-Q", 2024, "Q1", None),
        # 8-K 应被过滤
        ("0000000001", "us-gaap", "Assets", "USD", "2024-04-30", 999.0, "a8k", "2024-05-02", "8-K", 2024, "Q2", None),
        # BBB 的 instant 事实
        ("0000000002", "us-gaap", "Assets", "USD", "2023-12-31", 500.0, "b1", "2024-02-15", "10-K", 2023, "FY", None),
    ]
    c.executemany(
        "INSERT INTO fundamental_instant VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        instant_rows,
    )

    # AAA 的 period 事实
    period_rows = [
        ("0000000001", "us-gaap", "Revenues", "USD", "2023-01-01", "2023-12-31", 100.0, "a6", "2024-02-01", "10-K", 2023, "FY", None),
        ("0000000001", "us-gaap", "Revenues", "USD", "2024-01-01", "2024-03-31", 30.0, "a7", "2024-05-10", "10-Q", 2024, "Q1", None),
    ]
    c.executemany(
        "INSERT INTO fundamental_period VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        period_rows,
    )

    yield c
    c.close()


def test_pit_filed_boundary_and_restatement(conn):
    # 2024-02-01 当天 filed == as_of_date, 2023-12-31 的 10-K 尚不可见
    rows_on_filed = conn.execute(
        """
        SELECT end_date::VARCHAR, val
        FROM pit_fundamentals_by_ticker('AAA', DATE '2024-02-01', 2)
        WHERE tag = 'Assets'
        ORDER BY report_rank
        """
    ).fetchall()
    assert rows_on_filed == [("2023-09-30", 50.0), ("2023-06-30", 40.0)]

    # 2024-02-02 次日可见初版 60.0 (看不到 2024-05-01 的修订版 66.0)
    rows_next_day = conn.execute(
        """
        SELECT end_date::VARCHAR, val
        FROM pit_fundamentals_by_ticker('AAA', DATE '2024-02-02', 2)
        WHERE tag = 'Assets'
        ORDER BY report_rank
        """
    ).fetchall()
    assert rows_next_day == [("2023-12-31", 60.0), ("2023-09-30", 50.0)]

    # 2024-05-02 修订版生效, 2023-12-31 更新为 66.0 且去重为 1 条; 8-K 被过滤
    rows_after_rev = conn.execute(
        """
        SELECT end_date::VARCHAR, val
        FROM pit_fundamentals_by_ticker('AAA', DATE '2024-05-02', 2)
        WHERE tag = 'Assets'
        ORDER BY report_rank
        """
    ).fetchall()
    assert rows_after_rev == [("2023-12-31", 66.0), ("2023-09-30", 50.0)]


def test_default_five_reports_and_custom_limit(conn):
    # 默认 n_reports = 5, 2024-05-11 可看到全部 7 期中的最新 5 期
    rows_default = conn.execute(
        """
        SELECT report_rank, end_date::VARCHAR, val
        FROM pit_fundamentals_by_ticker('AAA', DATE '2024-05-11')
        WHERE tag = 'Assets'
        ORDER BY report_rank
        """
    ).fetchall()
    assert rows_default == [
        (1, "2024-03-31", 70.0),
        (2, "2023-12-31", 66.0),
        (3, "2023-09-30", 50.0),
        (4, "2023-06-30", 40.0),
        (5, "2023-03-31", 30.0),
    ]

    # 自定义取 1 条 (只拿最新一期)
    rows_one = conn.execute(
        """
        SELECT fact_kind, tag, start_date::VARCHAR, end_date::VARCHAR, val
        FROM pit_fundamentals_by_ticker('AAA', DATE '2024-05-11', 1)
        ORDER BY fact_kind, tag
        """
    ).fetchall()
    assert rows_one == [
        ("instant", "Assets", None, "2024-03-31", 70.0),
        ("period", "Revenues", "2024-01-01", "2024-03-31", 30.0),
    ]


def test_pit_universe_membership_and_all_vs_single_parity(conn):
    # 2024-03-01 时 BBB 尚未进入 universe
    tickers_mar = conn.execute(
        "SELECT DISTINCT ticker FROM pit_fundamentals(DATE '2024-03-01') ORDER BY ticker"
    ).fetchall()
    assert tickers_mar == [("AAA",)]
    assert (
        conn.execute(
            "SELECT count(*) FROM pit_fundamentals_by_ticker('BBB', DATE '2024-03-01')"
        ).fetchone()[0]
        == 0
    )

    # 2024-07-01 时 BBB 已进入 universe, 全体宏和单标的宏对 AAA/BBB 结果完全一致
    tickers_jul = conn.execute(
        "SELECT DISTINCT ticker FROM pit_fundamentals(DATE '2024-07-01') ORDER BY ticker"
    ).fetchall()
    assert tickers_jul == [("AAA",), ("BBB",)]

    for ticker in ("AAA", "BBB"):
        from_all = conn.execute(
            """
            SELECT * FROM pit_fundamentals(DATE '2024-07-01', 3)
            WHERE ticker = ?
            ORDER BY fact_kind, tag, report_rank
            """,
            [ticker],
        ).fetchall()
        from_single = conn.execute(
            """
            SELECT * FROM pit_fundamentals_by_ticker(?, DATE '2024-07-01', 3)
            ORDER BY fact_kind, tag, report_rank
            """,
            [ticker],
        ).fetchall()
        assert from_single == from_all
