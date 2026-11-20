"""离线测试 Dolt CSV → lab.duckdb 一步建库流水线 (切段、复权、幽灵拆股、休市日、universe、幂等性)。"""

import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd
import pytest

from event_finance_frame.pipeline import build_lab
from event_finance_frame.sources.sp500_history import (
    CIK_PLACEHOLDER,
    HISTORY_FILE,
    SEC_FILE,
    WIKI_FILE,
)


def _create_stocks_csv_dir(csv_dir: Path) -> None:
    """构造带拆股、幽灵重复拆股、分红、代码复用、坏值与休市日的迷你 Dolt CSV 目录"""
    csv_dir.mkdir(parents=True, exist_ok=True)

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
    ohlcv_rows = [(d, "MKT", 10.0, 10.0, 10.0, 10.0, 1000) for d in mkt_dates]
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
    # 休市日假数据 (2020-01-04 周六): 全市场 volume=0, 应被整体过滤, 不进入 prices 和 calendar
    ohlcv_rows += [
        ("2020-01-04", "MKT", 10.0, 10.0, 10.0, 10.0, 0),
        ("2020-01-04", "AAA", 50.0, 50.0, 50.0, 50.0, 0),
    ]
    ohlcv_df = pd.DataFrame(
        ohlcv_rows,
        columns=["date", "act_symbol", "open", "high", "low", "close", "volume"],
    )
    ohlcv_df["volume"] = ohlcv_df["volume"].astype("Int64")
    ohlcv_df.to_csv(csv_dir / "ohlcv.csv", index=False)

    pd.DataFrame(
        [
            ("AAA", "2020-01-02", 2.0, 1.0),  # 幽灵重复拆股 (01-02 价格未跳变), 应过滤
            ("AAA", "2020-01-03", 2.0, 1.0),  # 真实拆股 (100 -> 50)
            ("BBB", "2020-01-02", 2.0, 1.0),
            ("BBB", "2020-01-07", 5.0, 1.0),  # 空档期事件, 应丢弃
            ("BAD", "2020-01-02", 0.0, 0.0),  # 无效拆股, 应丢弃
        ],
        columns=["act_symbol", "ex_date", "to_factor", "for_factor"],
    ).to_csv(csv_dir / "split.csv", index=False)

    pd.DataFrame(
        [
            ("AAA", "2020-01-05", 2.0),  # 周末除息, 顺延到 01-06
            ("BAD", "2020-01-02", 0.0),  # 金额 0, 应丢弃
        ],
        columns=["act_symbol", "ex_date", "amount"],
    ).to_csv(csv_dir / "dividend.csv", index=False)

    pd.DataFrame(
        [
            ("AAA", "Alpha Corp", "NASDAQ", None, 0, 100, 0, None, None, "AAA", 0, "2020-01-07"),
            ("BBB", "Beta New Inc", "NYSE", None, 0, 100, 0, None, None, "BBB", 0, "2020-01-14"),
            ("OLD", "Old Co", "NYSE", None, 0, 100, 0, None, None, "OLD", 0, "2020-01-02"),
        ],
        columns=[
            "act_symbol",
            "security_name",
            "listing_exchange",
            "market_category",
            "is_etf",
            "round_lot_size",
            "is_test_issue",
            "financial_status",
            "cqs_symbol",
            "nasdaq_symbol",
            "is_next_shares",
            "last_seen",
        ],
    ).to_csv(csv_dir / "symbol.csv", index=False)


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


def _create_facts_zip(zip_path: Path) -> None:
    """构造迷你 companyfacts.zip: 仅含 AAA (0000001111), 故意缺少 BBB (0000002222) 测试容错"""
    payload = {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2019-12-31",
                                "val": 1000.0,
                                "accn": "0001-20-000001",
                                "filed": "2020-02-01",
                                "form": "10-K",
                                "fy": 2019,
                                "fp": "FY",
                                "frame": "CY2019Q4I",
                            },
                            # 主键重复行 (无 frame), 应去重保留带 frame 的记录
                            {
                                "end": "2019-12-31",
                                "val": 1000.0,
                                "accn": "0001-20-000001",
                                "filed": "2020-02-01",
                                "form": "10-K",
                                "fy": 2019,
                                "fp": "FY",
                            },
                            # 非法日期字符串, 应被过滤而不崩溃
                            {
                                "end": "not-a-date",
                                "val": 999.0,
                                "accn": "0001-20-000002",
                                "filed": "2020-02-01",
                                "form": "10-K",
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2019-01-01",
                                "end": "2019-12-31",
                                "val": 500.0,
                                "accn": "0001-20-000001",
                                "filed": "2020-02-01",
                                "form": "10-K",
                                "fy": 2019,
                                "fp": "FY",
                            },
                            # start_date > end_date 脏行, 应被过滤
                            {
                                "start": "2020-01-01",
                                "end": "2019-12-31",
                                "val": 10.0,
                                "accn": "0001-20-000003",
                                "filed": "2020-02-01",
                                "form": "10-K",
                            },
                        ]
                    }
                },
            }
        }
    }
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("CIK0000001111.json", json.dumps(payload))


def test_build_lab_segments_adjustments_and_universe(tmp_path: Path):
    csv_dir = tmp_path / "stocks_csv"
    lab_db = tmp_path / "lab.duckdb"
    sp500_dir = tmp_path / "sp500"
    facts_zip = tmp_path / "companyfacts.zip"
    reused_csv = tmp_path / "reused_tickers.csv"

    _create_stocks_csv_dir(csv_dir)
    _create_sp500_dir(sp500_dir)
    _create_facts_zip(facts_zip)

    counts = build_lab(
        db_path=lab_db,
        csv_dir=csv_dir,
        sp500_dir=sp500_dir,
        facts_zip=facts_zip,
        gap_days=3,
        reused_path=reused_csv,
    )
    # 10 (MKT) + 5 (AAA) + 6 (BBB) + 2 (OLD) + 1 (BAD, close=0 被剔除) = 24
    # 2020-01-04 休市日假数据 (2 行) 被整体剔除
    assert counts["prices"] == 24
    assert counts["market_calendar"] == 14
    assert counts["universe"] == 4
    assert counts["fundamental_instant"] == 1
    assert counts["fundamental_period"] == 1

    with duckdb.connect(str(lab_db), read_only=True) as conn:
        # 1. AAA 全收益复权连续等于 48.0 (01-02 幽灵重复拆股被过滤)
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

        # 4. calendar 视图有 10 个交易日 (不含 2020-01-04);
        #    market_calendar 含 2020-01-01 ~ 2020-01-14 共 14 个自然日 (10 开盘, 4 休市)
        assert conn.execute("SELECT count(*) FROM calendar").fetchone() == (10,)
        cal_summary = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE is_open), count(*) FILTER (WHERE NOT is_open) FROM market_calendar"
        ).fetchone()
        assert cal_summary == (14, 10, 4)
        assert conn.execute(
            "SELECT weekday, is_open FROM market_calendar WHERE date = DATE '2020-01-04'"
        ).fetchone() == ("Sat", False)

        # prices_calendar 视图把 market_calendar 和 prices 贴在一起
        pc_aaa = conn.execute(
            "SELECT date::VARCHAR, weekday, is_open, close FROM prices_calendar WHERE ticker = 'AAA' ORDER BY date LIMIT 1"
        ).fetchone()
        assert pc_aaa == ("2020-01-01", "Wed", True, 100.0)
        pc_closed = conn.execute(
            "SELECT date::VARCHAR, weekday, is_open, ticker FROM prices_calendar WHERE date = DATE '2020-01-04'"
        ).fetchall()
        assert pc_closed == [("2020-01-04", "Sat", False, None)]

        # 5. universe 映射
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
        csv_dir=csv_dir,
        sp500_dir=sp500_dir,
        facts_zip=facts_zip,
        gap_days=3,
        reused_path=reused_csv,
    )
    assert counts_again == counts
