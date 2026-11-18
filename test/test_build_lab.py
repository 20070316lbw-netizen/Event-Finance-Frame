"""离线测试 raw_data → lab.duckdb 建库流水线 (切段、复权、universe 映射、幂等性)。"""

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from event_finance_frame.config import DDL
from event_finance_frame.db.operation import read_sql
from event_finance_frame.pipeline import build_lab
from event_finance_frame.sources.sp500_history import (
    CIK_PLACEHOLDER,
    HISTORY_FILE,
    SEC_FILE,
    WIKI_FILE,
)


def _create_raw_db(path: Path) -> None:
    """构造一个带拆股、分红、代码复用、坏值的迷你 raw_data.duckdb"""
    with duckdb.connect(str(path)) as conn:
        for name in ("raw_ohlcv", "raw_symbol", "raw_dividend", "raw_split"):
            conn.execute(read_sql(DDL / f"{name}.sql"))

        # 全市场交易日历 (10 天)
        mkt_dates = [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
            "2020-01-09",
            "2020-01-10",
            "2020-01-13",
            "2020-01-14",
        ]
        ohlcv_rows = [
            (d, "MKT", 10.0, 10.0, 10.0, 10.0, 1000) for d in mkt_dates
        ]
        # AAA: 01-03 拆股 2:1 (100->50), 01-05(非交易日) 分红 2.0 顺延至 01-06 (50->48)
        ohlcv_rows += [
            ("2020-01-01", "AAA", 100.0, 100.0, 100.0, 100.0, 100),
            ("2020-01-02", "AAA", 100.0, 100.0, 100.0, 100.0, 100),
            ("2020-01-03", "AAA", 50.0, 50.0, 50.0, 50.0, 200),
            ("2020-01-06", "AAA", 48.0, 48.0, 48.0, 48.0, 200),
            ("2020-01-07", "AAA", 48.0, 48.0, 48.0, 48.0, 200),
        ]
        # BBB: 旧段 01-01~01-03 (01-02 拆股 2:1), 空档 4 天后 01-10~01-14 新段 (最新活跃)
        ohlcv_rows += [
            ("2020-01-01", "BBB", 20.0, 20.0, 20.0, 20.0, 100),
            ("2020-01-02", "BBB", 10.0, 10.0, 10.0, 10.0, 100),
            ("2020-01-03", "BBB", 10.0, 10.0, 10.0, 10.0, 100),
            ("2020-01-10", "BBB", 50.0, 50.0, 50.0, 50.0, 300),
            ("2020-01-13", "BBB", 52.0, 52.0, 52.0, 52.0, 300),
            ("2020-01-14", "BBB", 52.0, 52.0, 52.0, 52.0, 300),
        ]
        # OLD: 已退市单段代码 (最后交易日 01-02)
        ohlcv_rows += [
            ("2020-01-01", "OLD", 30.0, 30.0, 30.0, 30.0, 50),
            ("2020-01-02", "OLD", 31.0, 31.0, 31.0, 31.0, 50),
        ]
        # BAD: close=0 应丢弃, volume=NULL 应保留
        ohlcv_rows += [
            ("2020-01-01", "BAD", 0.0, 0.0, 0.0, 0.0, 10),
            ("2020-01-02", "BAD", 10.0, 10.0, 10.0, 10.0, None),
        ]
        conn.executemany(
            "INSERT INTO raw_ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", ohlcv_rows
        )

        conn.executemany(
            "INSERT INTO raw_split VALUES (?, ?, ?, ?)",
            [
                ("AAA", "2020-01-03", 2.0, 1.0),
                ("BBB", "2020-01-02", 2.0, 1.0),
                ("BBB", "2020-01-07", 5.0, 1.0),  # 空档期事件, 应丢弃
                ("BAD", "2020-01-02", 0.0, 0.0),  # 无效拆股, 应丢弃
            ],
        )
        conn.executemany(
            "INSERT INTO raw_dividend VALUES (?, ?, ?)",
            [
                ("AAA", "2020-01-05", 2.0),  # 周末除息, 顺延到 01-06
                ("BAD", "2020-01-02", 0.0),  # 金额 0, 应丢弃
            ],
        )
        conn.executemany(
            "INSERT INTO raw_symbol VALUES (?, ?, ?, NULL, 0, 100, 0, NULL, NULL, ?, 0, ?)",
            [
                ("AAA", "Alpha Corp", "NASDAQ", "AAA", "2020-01-07"),
                ("BBB", "Beta New Inc", "NYSE", "BBB", "2020-01-14"),
                ("OLD", "Old Co", "NYSE", "OLD", "2020-01-02"),
            ],
        )


def _create_sp500_dir(sp500_dir: Path) -> None:
    sp500_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2019-12-01", "tickers": "BBB,OLD"},
            {"date": "2020-01-10", "tickers": "AAA,BBB"},
        ]
    ).to_csv(sp500_dir / HISTORY_FILE, index=False)

    pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "cik": "0000002222",
                "name": "Beta New Wiki",
                "sub_industry": "Tech",
            }
        ]
    ).to_csv(sp500_dir / WIKI_FILE, index=False)

    sec = {
        "0": {"cik_str": 1111, "ticker": "AAA", "title": "Alpha SEC"},
        "1": {"cik_str": 9999, "ticker": "OLD", "title": "Reused Elsewhere"},
    }
    (sp500_dir / SEC_FILE).write_text(json.dumps(sec), encoding="utf-8")


def test_build_lab_segments_adjustments_and_universe(tmp_path: Path):
    raw_db = tmp_path / "raw_data.duckdb"
    lab_db = tmp_path / "lab.duckdb"
    sp500_dir = tmp_path / "sp500"
    reused_csv = tmp_path / "reused_tickers.csv"

    _create_raw_db(raw_db)
    _create_sp500_dir(sp500_dir)

    counts = build_lab(
        db_path=lab_db,
        raw_db=raw_db,
        sp500_dir=sp500_dir,
        gap_days=3,
        reused_path=reused_csv,
    )
    # 10 (MKT) + 5 (AAA) + 6 (BBB) + 2 (OLD) + 1 (BAD, close=0 被剔除) = 24
    assert counts["prices"] == 24
    assert counts["universe"] == 4

    with duckdb.connect(str(lab_db), read_only=True) as conn:
        # 1. AAA 全收益复权连续等于 48.0
        aaa_adj = conn.execute(
            "SELECT adj_close FROM prices WHERE ticker = 'AAA' ORDER BY date"
        ).fetchall()
        assert [row[0] for row in aaa_adj] == pytest.approx([48.0] * 5)

        # 2. BBB 切成 BBB~20200103 与 BBB 两段, 空档期拆股不串段
        bbb_old = conn.execute(
            "SELECT close, adj_close FROM prices WHERE ticker = 'BBB~20200103' ORDER BY date"
        ).fetchall()
        assert [row[0] for row in bbb_old] == [20.0, 10.0, 10.0]
        assert [row[1] for row in bbb_old] == pytest.approx([10.0, 10.0, 10.0])

        bbb_new = conn.execute(
            "SELECT close, adj_close FROM prices WHERE ticker = 'BBB' ORDER BY date"
        ).fetchall()
        assert bbb_new == [(50.0, 50.0), (52.0, 52.0), (52.0, 52.0)]

        # 3. BAD close=0 被删, volume=NULL 保留
        bad = conn.execute(
            "SELECT date::VARCHAR, close, volume FROM prices WHERE ticker = 'BAD'"
        ).fetchall()
        assert bad == [("2020-01-02", 10.0, None)]

        # 4. calendar 视图有 10 个交易日
        assert conn.execute("SELECT count(*) FROM calendar").fetchone() == (10,)

        # 5. universe 映射:
        #    2019-12-01 的 BBB 落在旧段 BBB~20200103, cik 填占位
        #    2019-12-01 的 OLD 已退市 (不在最后 10 天内), cik 填占位但保留 raw_symbol 名称
        #    2020-01-10 的 BBB 映射到最新活跃段 (用 wiki 补齐)
        #    2020-01-10 的 AAA 活跃 (01-07 在最后 10 天内), 用 SEC 补 cik
        univ = conn.execute(
            """
            SELECT snapshot_date::VARCHAR, ticker, cik, name, exchange, sub_industry
            FROM universe
            ORDER BY snapshot_date, ticker
            """
        ).fetchall()
        assert univ == [
            ("2019-12-01", "BBB~20200103", CIK_PLACEHOLDER, "BBB~20200103", None, None),
            ("2019-12-01", "OLD", CIK_PLACEHOLDER, "Old Co", "NYSE", None),
            ("2020-01-10", "AAA", "0000001111", "Alpha SEC", "NASDAQ", None),
            ("2020-01-10", "BBB", "0000002222", "Beta New Wiki", "NYSE", "Tech"),
        ]

    # 6. 复用名单 CSV 内容检查
    reused = pd.read_csv(reused_csv)
    assert reused["ticker"].tolist() == ["BBB~20200103", "BBB"]

    # 7. 幂等性: 再跑一次结果完全一致
    counts_again = build_lab(
        db_path=lab_db,
        raw_db=raw_db,
        sp500_dir=sp500_dir,
        gap_days=3,
        reused_path=reused_csv,
    )
    assert counts_again == counts
