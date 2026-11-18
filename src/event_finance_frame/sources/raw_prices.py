"""把 raw_data.duckdb 的全量行情装入 lab.duckdb 的 prices 表

核心函数: load_prices_from_raw()

处理步骤 (全部在 DuckDB 内用 SQL 完成, 不经过 pandas):
    1. 只读 ATTACH raw_data.duckdb 为 raw
    2. 清洗: 剔除 close <= 0 的行 (免费数据, 其余毛刺原样保留)
    3. 切段: 同一代码前后两条行情间隔超过 gap_days 个交易日时视为不同证券
       (代码被退市公司让出后又给了新公司, 看起来像 "诈尸")
       最新一段沿用原代码, 更早的段改名为 {symbol}~{YYYYMMDD}, 结果写入 ticker_segment
    4. 按段计算全收益复权价 adj_close, 写入 prices
    5. 把被切段的代码名单导出到 REUSED_TICKERS, 方便人工核对
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger

from event_finance_frame.config import DDL, INSERT, RAW_DB, REUSED_TICKERS
from event_finance_frame.db.operation import operation_sql, read_sql

__all__ = [
    "DEFAULT_GAP_DAYS",
    "RAW_ALIAS",
    "attach_raw",
    "export_reused_tickers",
    "load_prices_from_raw",
]

DEFAULT_GAP_DAYS = 120
RAW_ALIAS = "raw"


def attach_raw(connection: duckdb.DuckDBPyConnection, raw_db: Path = RAW_DB) -> None:
    """只读挂载 raw 库为 raw, 并建立清洗后的临时视图 clean_ohlcv

    已经挂载时跳过, 可重复调用

    Args:
        connection: 可写的 lab 库连接
        raw_db: raw_data.duckdb 路径, 默认 config.RAW_DB

    Raises:
        FileNotFoundError: raw 库不存在
        duckdb.Error: 挂载失败, 例如被其他进程加了写锁
    """
    raw_db = Path(raw_db)
    if not raw_db.exists():
        raise FileNotFoundError(f"raw 数据库不存在: {raw_db}")

    attached = {
        row[0] for row in connection.execute("SELECT database_name FROM duckdb_databases()").fetchall()
    }
    if RAW_ALIAS not in attached:
        connection.execute(f"ATTACH '{raw_db.as_posix()}' AS {RAW_ALIAS} (READ_ONLY)")
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW clean_ohlcv AS
        SELECT * FROM {RAW_ALIAS}.raw_ohlcv
        WHERE close > 0
        """
    )


def load_prices_from_raw(
    connection: duckdb.DuckDBPyConnection,
    *,
    raw_db: Path = RAW_DB,
    gap_days: int = DEFAULT_GAP_DAYS,
    reused_path: Path | None = REUSED_TICKERS,
) -> int:
    """切段并写入全量 prices, 重复执行结果不变 (先清空再写入)

    Args:
        connection: 可写的 lab 库连接, 调用前不应有未结束的事务
        raw_db: raw_data.duckdb 路径
        gap_days: 切段阈值, 前后两条行情的交易日序号之差大于该值时切开, 默认 120
        reused_path: 被切段的代码名单输出路径, None 表示不导出

    Returns:
        int: 写入 prices 的行数

    Raises:
        ValueError: gap_days 不是正整数
        duckdb.Error: 挂载或写入失败, 失败时当前步骤回滚
    """
    if int(gap_days) != gap_days or gap_days <= 0:
        raise ValueError(f"gap_days 必须是正整数: {gap_days}")

    attach_raw(connection, raw_db)
    connection.execute(f"SET VARIABLE gap_days = {int(gap_days)}")

    n_segments = operation_sql(
        read_sql(INSERT / "ticker_segment.sql"),
        read_sql(DDL / "ticker_segment.sql"),
        connection,
    )
    n_reused = connection.execute(
        "SELECT count(*) FROM ticker_segment WHERE ticker <> symbol"
    ).fetchone()[0]
    logger.info(
        f"切段完成: {n_segments} 段, 其中 {n_reused} 段为旧段已改名 (阈值 {gap_days} 个交易日)"
    )

    logger.info("开始写入 prices (约 3,000 万行, 需要几十秒)")
    written = operation_sql(
        read_sql(INSERT / "prices.sql"),
        read_sql(DDL / "prices.sql"),
        connection,
    )
    logger.info(f"prices 写入 {written} 行")

    if reused_path is not None:
        export_reused_tickers(connection, Path(reused_path))
    return written


def export_reused_tickers(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    """导出被切成多段的代码名单 (CSV), 每段一行

    列: symbol, ticker, seg_no, start_date, end_date, n_days, gap_before
    按 symbol, seg_no 排序

    Returns:
        int: 涉及的原始代码数量
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    query = """
        SELECT symbol, ticker, seg_no, start_date, end_date, n_days, gap_before
        FROM ticker_segment
        WHERE symbol IN (SELECT symbol FROM ticker_segment WHERE seg_no > 0)
        ORDER BY symbol, seg_no
    """
    connection.execute(f"COPY ({query}) TO '{path.as_posix()}' (HEADER, DELIMITER ',')")
    n_symbols = connection.execute(
        "SELECT count(DISTINCT symbol) FROM ticker_segment WHERE seg_no > 0"
    ).fetchone()[0]
    logger.info(f"代码复用名单已导出: {path} ({n_symbols} 个代码)")
    return n_symbols
