"""测试 M3 PITDataView 时点隔离、宏一致性与面板契约"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest

from event_finance_frame import (
    Clock,
    EventFinanceError,
    PITDataView,
    PITErrorCode,
    TradingCalendar,
    get_duckdb,
    validate_panel_index,
)
from event_finance_frame.calendar import load_calendar
from event_finance_frame.config import DB_STORAGE, DDL, MACRO
from event_finance_frame.db.operation import read_sql


def _create_base_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(read_sql(DDL / "prices.sql"))
    con.execute(read_sql(DDL / "universe.sql"))
    con.execute(read_sql(DDL / "fundamental_instant.sql"))
    con.execute(read_sql(DDL / "fundamental_period.sql"))
    con.execute(read_sql(MACRO / "pit_fundamentals.sql"))
    con.execute(read_sql(MACRO / "returns.sql"))

    # 连续 5 个交易日: 2024-01-02 .. 2024-01-05, 2024-01-08 (跳过周末 01-06/01-07)
    price_rows = [
        ("2024-01-02", "AAA", 10.0, 11.0, 9.0, 10.0, 1.0, 100),
        ("2024-01-03", "AAA", 20.0, 21.0, 19.0, 20.0, 2.0, 200),
        ("2024-01-04", "AAA", 40.0, 41.0, 39.0, 40.0, 4.0, 300),
        ("2024-01-05", "AAA", 80.0, 81.0, 79.0, 80.0, 8.0, 400),
        ("2024-01-08", "AAA", 160.0, 161.0, 159.0, 160.0, 16.0, 500),
        ("2024-01-02", "BBB", 100.0, 101.0, 99.0, 100.0, 100.0, 10),
        ("2024-01-03", "BBB", 110.0, 111.0, 109.0, 110.0, 110.0, 20),
        ("2024-01-04", "BBB", 121.0, 122.0, 120.0, 121.0, None, 30),
        ("2024-01-05", "BBB", 133.1, 134.0, 132.0, 133.1, 0.0, 40),
        ("2024-01-08", "BBB", 146.41, 147.0, 145.0, 146.41, 146.41, 50),
    ]
    con.executemany(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        price_rows,
    )
    load_calendar(con)

    # 2024-01-02: 仅 AAA; 2024-01-05: AAA + BBB
    con.executemany(
        "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("2024-01-02", "AAA", "0000000001", "Alpha", "NASDAQ", None, "Tech"),
            ("2024-01-05", "AAA", "0000000001", "Alpha", "NASDAQ", None, "Tech"),
            ("2024-01-05", "BBB", "0000000002", "Beta", "NYSE", None, "Finance"),
        ],
    )

    # 基本面:
    #   2024-01-03 filed: 2023-09-30 Assets=50.0 (2024-01-04 起可见)
    #   2024-01-04 filed: 2023-12-31 Assets=60.0 (2024-01-05 起可见)
    #   2024-01-05 filed: 2023-12-31 Assets=66.0 10-K/A 修订 (2024-01-08 起可见)
    #   8-K 应被过滤
    con.executemany(
        "INSERT INTO fundamental_instant VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("0000000001", "us-gaap", "Assets", "USD", "2023-09-30", 50.0, "a1", "2024-01-03", "10-Q", 2023, "Q3", None),
            ("0000000001", "us-gaap", "Assets", "USD", "2023-12-31", 60.0, "a2", "2024-01-04", "10-K", 2023, "FY", None),
            ("0000000001", "us-gaap", "Assets", "USD", "2023-12-31", 66.0, "a2-rev", "2024-01-05", "10-K/A", 2023, "FY", None),
            ("0000000001", "us-gaap", "Assets", "USD", "2024-01-02", 999.0, "a8k", "2024-01-02", "8-K", 2024, "Q1", None),
            ("0000000002", "us-gaap", "Assets", "USD", "2023-12-31", 500.0, "b1", "2024-01-03", "10-K", 2023, "FY", None),
        ],
    )
    con.executemany(
        "INSERT INTO fundamental_period VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("0000000001", "us-gaap", "Revenues", "USD", "2023-01-01", "2023-12-31", 100.0, "a2", "2024-01-04", "10-K", 2023, "FY", None),
        ],
    )
    return con


def test_future_data_injection_invariance():
    """核心证明: 向数据库注入 T 之后的未来行情、未来成分与未来修订财报, 在 T 时刻查询结果完全不变"""
    con = _create_base_db()
    cal = TradingCalendar(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    clk = Clock(cal)
    clk.seek("2024-01-04")
    pit = PITDataView(con, clk)

    # 注入前在 T = 2024-01-04 快照全部查询结果
    univ_before = pit.universe()
    tickers_before = pit.universe_tickers()
    px_all_before = pit.prices()
    px_bbb_before = pit.prices("BBB")
    latest_before = pit.latest_prices()
    fund_before = pit.fundamentals()
    hret_before = pit.historical_returns(1, universe_only=False)
    lret_before = pit.log_returns(universe_only=False)

    # 注入大量 T = 2024-01-04 之后的未来数据 + 对历史报告期的未来修订版
    con.executemany(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("2024-01-09", "AAA", 320.0, 321.0, 319.0, 320.0, 32.0, 600),
            ("2024-01-09", "CCC", 50.0, 51.0, 49.0, 50.0, 50.0, 999),
        ],
    )
    con.execute(
        "INSERT INTO universe VALUES ('2024-01-09', 'CCC', '0000000003', 'Gamma', 'NYSE', NULL, 'Energy')"
    )
    con.executemany(
        "INSERT INTO fundamental_instant VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # filed == T (2024-01-04) 当天不可见, 仅从次日起可见
            ("0000000001", "us-gaap", "Assets", "USD", "2023-09-30", 55.5, "a1-rev-t", "2024-01-04", "10-Q/A", 2023, "Q3", None),
            # filed > T 的未来修订
            ("0000000001", "us-gaap", "Assets", "USD", "2023-09-30", 88.8, "a1-rev-fut", "2024-01-10", "10-Q/A", 2023, "Q3", None),
        ],
    )
    con.execute(
        "INSERT INTO fundamental_period VALUES ('0000000001', 'us-gaap', 'Revenues', 'USD', '2024-01-01', '2024-03-31', 42.0, 'a3', '2024-01-04', '10-Q', 2024, 'Q1', NULL)"
    )

    assert pit.universe_tickers() == tickers_before
    pd.testing.assert_frame_equal(pit.universe(), univ_before)
    pd.testing.assert_frame_equal(pit.prices(), px_all_before)
    pd.testing.assert_frame_equal(pit.prices("BBB"), px_bbb_before)
    pd.testing.assert_frame_equal(pit.latest_prices(), latest_before)
    pd.testing.assert_frame_equal(pit.fundamentals(), fund_before)
    pd.testing.assert_frame_equal(pit.historical_returns(1, universe_only=False), hret_before)
    pd.testing.assert_frame_equal(pit.log_returns(universe_only=False), lret_before)
    con.close()


def test_pit_evolution_along_clock():
    con = _create_base_db()
    clk = Clock.from_duckdb(con)
    pit = PITDataView(con, clk)

    # 2024-01-02: universe 只有 AAA; filed=2024-01-03 尚不可见, 8-K 被过滤
    assert pit.universe_tickers() == ("AAA",)
    assert pit.fundamentals().empty

    # 2024-01-03: 仍只有 AAA, filed == 2024-01-03 当天不可见
    clk.step()
    assert clk.now == date(2024, 1, 3)
    assert pit.fundamentals().empty

    # 2024-01-04: 2023-09-30 Assets=50.0 可见; filed == 2024-01-04 的 10-K 尚不可见
    clk.step()
    assert clk.now == date(2024, 1, 4)
    f04 = pit.fundamentals("AAA", tags="Assets")
    assert list(zip(f04["end_date"].dt.strftime("%Y-%m-%d"), f04["val"], strict=True)) == [
        ("2023-09-30", 50.0),
    ]

    # 2024-01-05: BBB 进入 universe; AAA 2023-12-31 初版 60.0 可见 (修订版 filed=01-05 尚不可见)
    clk.step()
    assert clk.now == date(2024, 1, 5)
    assert pit.universe_tickers() == ("AAA", "BBB")
    f05 = pit.fundamentals("AAA", tags="Assets")
    assert list(zip(f05["end_date"].dt.strftime("%Y-%m-%d"), f05["val"], strict=True)) == [
        ("2023-12-31", 60.0),
        ("2023-09-30", 50.0),
    ]

    # 2024-01-08: AAA 2023-12-31 修订版 66.0 生效且去重为 1 条
    clk.step()
    assert clk.now == date(2024, 1, 8)
    f08 = pit.fundamentals("AAA", tags="Assets")
    assert list(zip(f08["end_date"].dt.strftime("%Y-%m-%d"), f08["val"], strict=True)) == [
        ("2023-12-31", 66.0),
        ("2023-09-30", 50.0),
    ]
    con.close()


def test_future_access_denied_and_argument_validation():
    con = _create_base_db()
    clk = Clock.from_duckdb(con)
    clk.seek("2024-01-04")
    pit = PITDataView(con, clk)

    with pytest.raises(EventFinanceError) as exc_end:
        pit.prices(end="2024-01-05")
    assert exc_end.value.code is PITErrorCode.FUTURE_ACCESS_DENIED

    with pytest.raises(EventFinanceError) as exc_start:
        pit.prices(start="2024-01-05")
    assert exc_start.value.code is PITErrorCode.FUTURE_ACCESS_DENIED

    invalid_calls = [
        lambda: pit.prices(lookback=2, start="2024-01-02"),
        lambda: pit.prices(start="2024-01-04", end="2024-01-02"),
        lambda: pit.prices(columns=["bad_col"]),
        lambda: pit.prices(columns="close"),  # type: ignore[arg-type]
        lambda: pit.prices(lookback=True),  # type: ignore[arg-type]
        lambda: pit.prices(lookback=0),
        lambda: pit.prices(universe_only="yes"),  # type: ignore[arg-type]
        lambda: pit.fundamentals(n_reports=True),  # type: ignore[arg-type]
        lambda: pit.fundamentals(fact_kind="other"),  # type: ignore[arg-type]
        lambda: pit.historical_returns(0),
        lambda: pit.historical_returns(True),  # type: ignore[arg-type]
        lambda: pit.historical_returns(1, price_col="open"),  # type: ignore[arg-type]
        lambda: pit.log_returns(price_col="volume"),  # type: ignore[arg-type]
    ]
    for fn in invalid_calls:
        with pytest.raises(EventFinanceError) as exc:
            fn()
        assert exc.value.code is PITErrorCode.INVALID_ARGUMENT
    con.close()


def test_lookback_before_clock_start():
    con = _create_base_db()
    # 时钟从 2024-01-05 开始, 但数据库有 01-02..01-08 全部历史
    clk = Clock.from_duckdb(con, start="2024-01-05", end="2024-01-08")
    pit = PITDataView(con, clk)
    assert clk.now == date(2024, 1, 5)

    px = pit.prices("AAA", lookback=3, columns=["close"])
    dates = [d.date() for d in px.index.get_level_values("date")]
    assert dates == [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]

    ret = pit.historical_returns(1, tickers="AAA", lookback=2)
    ret_dates = [d.date() for d in ret.index.get_level_values("date")]
    assert ret_dates == [date(2024, 1, 4), date(2024, 1, 5)]
    assert list(ret["historical_return"]) == pytest.approx([1.0, 1.0])
    con.close()


def test_parity_with_sql_macros():
    con = _create_base_db()
    clk = Clock.from_duckdb(con)
    clk.seek("2024-01-08")
    pit = PITDataView(con, clk)

    # 基本面一致性
    pit_f = pit.fundamentals(n_reports=3)
    macro_f = con.execute(
        """
        SELECT ticker, cik, fact_kind, taxonomy, tag, unit,
               start_date, end_date, val, accn, filed, form, fy, fp, frame, report_rank
        FROM pit_fundamentals(DATE '2024-01-08', 3)
        ORDER BY ticker, fact_kind, tag, unit, report_rank
        """
    ).df()
    pd.testing.assert_frame_equal(pit_f.reset_index(drop=True), macro_f.reset_index(drop=True))

    # 简单收益率一致性 (含 NULL / 0 价与 close/adj_close 切换)
    for price_col in ("adj_close", "close"):
        pit_h = pit.historical_returns(2, price_col=price_col, universe_only=False)
        macro_h = con.execute(
            f"""
            SELECT date, ticker, historical_return
            FROM pit_historical_return(DATE '2024-01-08', 2, '{price_col}')
            ORDER BY date, ticker
            """
        ).df()
        macro_h["date"] = pd.to_datetime(macro_h["date"]).astype("datetime64[ns]")
        macro_h = macro_h.set_index(["date", "ticker"]).sort_index()
        pd.testing.assert_frame_equal(pit_h, macro_h)

        pit_l = pit.log_returns(price_col=price_col, universe_only=False)
        macro_l = con.execute(
            f"""
            SELECT date, ticker, log_return
            FROM pit_log_return(DATE '2024-01-08', '{price_col}')
            ORDER BY date, ticker
            """
        ).df()
        macro_l["date"] = pd.to_datetime(macro_l["date"]).astype("datetime64[ns]")
        macro_l = macro_l.set_index(["date", "ticker"]).sort_index()
        pd.testing.assert_frame_equal(pit_l, macro_l)

    con.close()


def test_panel_index_contract_and_empty_results():
    con = _create_base_db()
    clk = Clock.from_duckdb(con)
    clk.seek("2024-01-05")
    pit = PITDataView(con, clk)

    for df in (
        pit.prices(),
        pit.prices([]),
        pit.prices("NONEXISTENT"),
        pit.historical_returns(2),
        pit.historical_returns(2, tickers=[]),
        pit.log_returns(),
        pit.log_returns(tickers="NONEXISTENT"),
    ):
        validate_panel_index(df)

    assert pit.latest_prices([]).empty
    assert pit.fundamentals([]).empty
    assert pit.fundamentals(tags=[]).empty
    con.close()


def test_no_label_leakage_on_pit_view():
    attrs = [name for name in dir(PITDataView) if not name.startswith("__")]
    assert all("forward" not in name and "label" not in name for name in attrs)


def test_real_lab_duckdb_readonly_smoke():
    if not DB_STORAGE.exists():
        pytest.skip(f"{DB_STORAGE} 不存在, 跳过真实库只读冒烟测试")

    with get_duckdb(path=DB_STORAGE, read_only=True) as con:
        clk = Clock.from_duckdb(con, start="2024-06-03", end="2024-06-05")
        pit = PITDataView(con, clk)

        univ = pit.universe()
        assert not univ.empty
        assert "AAPL" in pit.universe_tickers()

        px = pit.prices(["AAPL", "MSFT"], lookback=5)
        validate_panel_index(px)
        assert not px.empty

        latest = pit.latest_prices(["AAPL", "MSFT"])
        assert not latest.empty

        fund = pit.fundamentals(["AAPL"], tags=["Assets"], n_reports=2)
        assert not fund.empty
        assert (fund["filed"].dt.date < clk.now).all()

        ret = pit.historical_returns(5, tickers=["AAPL", "MSFT"], lookback=3)
        validate_panel_index(ret)
        assert not ret.empty
