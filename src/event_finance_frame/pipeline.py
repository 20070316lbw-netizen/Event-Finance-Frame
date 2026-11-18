"""建库流水线: raw_data.duckdb + 标普500 历史成分 → lab.duckdb

核心函数: build_lab()

步骤:
    1. prices: 全量行情切段、复权后写入 (sources/raw_prices.py)
    2. universe: 标普500 历史成分快照, 代码映射到对应的段 (sources/sp500_history.py)
    3. calendar: 交易日视图

重复执行结果不变: 每张表都先清空再写入
数据源只读本地缓存; 需要刷新历史成分时先调用 sources.sp500_history.download_sp500_sources()
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from event_finance_frame.config import (
    DB_STORAGE,
    DDL,
    INSERT,
    RAW_DB,
    REUSED_TICKERS,
    SP500_DIR,
    VIEW,
)
from event_finance_frame.db.operation import get_duckdb, operation_sql, read_sql
from event_finance_frame.sources.raw_prices import (
    DEFAULT_GAP_DAYS,
    load_prices_from_raw,
)
from event_finance_frame.sources.sp500_history import build_universe_frame

__all__ = ["build_lab"]


def build_lab(
    *,
    db_path: Path = DB_STORAGE,
    raw_db: Path = RAW_DB,
    sp500_dir: Path = SP500_DIR,
    gap_days: int = DEFAULT_GAP_DAYS,
    reused_path: Path | None = REUSED_TICKERS,
) -> dict[str, int]:
    """一次调用建好 lab.duckdb 的 prices, universe, calendar

    Args:
        db_path: 目标库, 默认 config.DB_STORAGE (data/lab.duckdb)
        raw_db: 原始行情库, 只读挂载
        sp500_dir: 标普500 历史成分本地缓存目录
        gap_days: 代码切段阈值 (交易日)
        reused_path: 代码复用名单输出路径, None 表示不导出

    Returns:
        dict[str, int]: 各表写入行数, 键为 prices, universe
    """
    with get_duckdb(path=db_path, read_only=False) as connection:
        n_prices = load_prices_from_raw(
            connection, raw_db=raw_db, gap_days=gap_days, reused_path=reused_path
        )

        start = connection.execute("SELECT min(date) FROM prices").fetchone()[0]
        df = build_universe_frame(sp500_dir, start=start)
        n_universe = operation_sql(
            read_sql(INSERT / "universe.sql"),
            read_sql(DDL / "universe.sql"),
            connection,
            df,
        )
        logger.info(f"universe 写入 {n_universe} 行")

        connection.execute(read_sql(VIEW / "calendar.sql"))
        connection.execute("CHECKPOINT")

    return {"prices": n_prices, "universe": n_universe}


if __name__ == "__main__":
    print(build_lab())
