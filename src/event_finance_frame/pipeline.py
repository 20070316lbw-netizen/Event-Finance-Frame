"""建库流水线: Dolt CSV + 标普500 历史成分 + SEC Company Facts → lab.duckdb

核心函数: build_lab()

步骤:
    1. prices + ticker_segment + calendar: 从 data/stocks_csv/*.csv 一步清洗、切段、复权入库
    2. universe: 标普500 历史成分快照, 代码映射到对应的段 (sources/sp500_history.py)
    3. fundamental_instant + fundamental_period: 从 data/companyfacts.zip 按 universe CIK 分批入库

重复执行结果不变: 每张表都先清空再写入
数据源只读本地文件; 刷新成分调用 sources.sp500_history.download_sp500_sources(),
刷新基本面调用 sources.fundamental.download_company_facts()
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from event_finance_frame.calendar import load_calendar
from event_finance_frame.config import (
    COMPANY_FACTS_ZIP,
    DB_STORAGE,
    DDL,
    INSERT,
    MACRO,
    REUSED_TICKERS,
    SP500_DIR,
    STOCKS_CSV_DIR,
)
from event_finance_frame.db.operation import get_duckdb, operation_sql, read_sql
from event_finance_frame.sources.fundamental import load_fundamentals_from_zip
from event_finance_frame.sources.raw_prices import (
    DEFAULT_GAP_DAYS,
    load_prices_from_csv,
)
from event_finance_frame.sources.sp500_history import build_universe_frame

__all__ = ["build_lab"]


def build_lab(
    *,
    db_path: Path = DB_STORAGE,
    csv_dir: Path = STOCKS_CSV_DIR,
    sp500_dir: Path = SP500_DIR,
    facts_zip: Path | None = COMPANY_FACTS_ZIP,
    gap_days: int = DEFAULT_GAP_DAYS,
    reused_path: Path | None = REUSED_TICKERS,
) -> dict[str, int]:
    """一次调用建好 lab.duckdb 的全部核心表 (prices, ticker_segment, universe, fundamentals, calendar)

    Args:
        db_path: 目标库, 默认 config.DB_STORAGE (data/lab.duckdb)
        csv_dir: Dolt 导出的 CSV 目录, 默认 config.STOCKS_CSV_DIR (data/stocks_csv)
        sp500_dir: 标普500 历史成分本地缓存目录
        facts_zip: SEC companyfacts.zip 路径, None 表示跳过基本面写入
        gap_days: 代码切段阈值 (交易日)
        reused_path: 代码复用名单输出路径, None 表示不导出

    Returns:
        dict[str, int]: 各表写入行数
    """
    with get_duckdb(path=db_path, read_only=False) as connection:
        n_prices = load_prices_from_csv(
            connection, csv_dir=csv_dir, gap_days=gap_days, reused_path=reused_path
        )
        n_calendar = load_calendar(connection)
        connection.execute(read_sql(MACRO / "returns.sql"))

        start = connection.execute("SELECT min(date) FROM prices").fetchone()[0]
        df = build_universe_frame(sp500_dir, start=start)
        n_universe = operation_sql(
            read_sql(INSERT / "universe.sql"),
            read_sql(DDL / "universe.sql"),
            connection,
            df,
        )
        logger.info(f"universe 写入 {n_universe} 行")

        counts: dict[str, int] = {
            "prices": n_prices,
            "market_calendar": n_calendar,
            "universe": n_universe,
        }
        if facts_zip is not None:
            counts.update(load_fundamentals_from_zip(connection, zip_path=facts_zip))
            connection.execute(read_sql(MACRO / "pit_fundamentals.sql"))

        connection.execute("CHECKPOINT")

    return counts


if __name__ == "__main__":
    print(build_lab())
