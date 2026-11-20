"""直接把 Dolt 导出的 CSV (ohlcv/dividend/split/symbol) 一步写入 lab.duckdb

核心函数: load_prices_from_csv()

全部清洗、切段、拆股校验与复权、日历更新逻辑集中在单个 SQL 文件:
    db/insert/dolt_prices_pipeline.sql

无需 raw_data.duckdb 中间库:
    1. 校验 CSV 目录存在所需的 4 份文件
    2. 设置 DuckDB 会话变量 stocks_csv_dir 与 gap_days
    3. 事务内执行 db/insert/dolt_prices_pipeline.sql:
       - 一次性扫描 ohlcv.csv 清洗 (剔除 close <= 0 与全市场零成交休市日)
       - 按 gap_days 切段写入 ticker_segment
       - 读 split.csv / dividend.csv 校验拆股跳变后按段计算全收益复权价写入 prices
       - 更新 calendar 视图
    4. 导出被切段的代码名单到 REUSED_TICKERS, 方便人工核对
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger

from event_finance_frame.config import INSERT, REUSED_TICKERS, STOCKS_CSV_DIR
from event_finance_frame.db.operation import read_sql

__all__ = [
    "DEFAULT_GAP_DAYS",
    "PIPELINE_SQL",
    "REQUIRED_CSV_FILES",
    "export_reused_tickers",
    "load_prices_from_csv",
    "validate_csv_dir",
]

DEFAULT_GAP_DAYS = 120
PIPELINE_SQL = INSERT / "dolt_prices_pipeline.sql"
REQUIRED_CSV_FILES = ("ohlcv.csv", "dividend.csv", "split.csv", "symbol.csv")


def validate_csv_dir(csv_dir: Path = STOCKS_CSV_DIR) -> Path:
    """校验 Dolt 导出目录及 4 份必需 CSV 文件存在, 返回解析后的目录 Path"""
    csv_dir = Path(csv_dir).resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"Dolt CSV 目录不存在: {csv_dir}")
    missing = [name for name in REQUIRED_CSV_FILES if not (csv_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dolt CSV 目录 {csv_dir} 缺少文件: {', '.join(missing)}")
    return csv_dir


def load_prices_from_csv(
    connection: duckdb.DuckDBPyConnection,
    *,
    csv_dir: Path = STOCKS_CSV_DIR,
    gap_days: int = DEFAULT_GAP_DAYS,
    reused_path: Path | None = REUSED_TICKERS,
) -> int:
    """从 Dolt CSV 一步写入 ticker_segment, prices, calendar (先清空再写入)

    同时在当前连接上设置会话变量 stocks_csv_dir, 供后续 universe.sql 读取 symbol.csv

    Args:
        connection: 可写的 lab 库连接, 调用前不应有未结束的事务
        csv_dir: Dolt 导出的 CSV 目录, 默认 config.STOCKS_CSV_DIR (data/stocks_csv)
        gap_days: 切段阈值, 前后两条行情的交易日序号之差大于该值时切开, 默认 120
        reused_path: 被切段的代码名单输出路径, None 表示不导出

    Returns:
        int: 写入 prices 的行数

    Raises:
        FileNotFoundError: CSV 目录或必需文件缺失
        ValueError: gap_days 不是正整数
        duckdb.Error: 读取或写入失败, 失败时当前事务回滚
    """
    if int(gap_days) != gap_days or gap_days <= 0:
        raise ValueError(f"gap_days 必须是正整数: {gap_days}")

    resolved_dir = validate_csv_dir(csv_dir)
    escaped_dir = resolved_dir.as_posix().replace("'", "''")
    connection.execute(f"SET VARIABLE stocks_csv_dir = '{escaped_dir}'")
    connection.execute(f"SET VARIABLE gap_days = {int(gap_days)}")

    logger.info(f"开始执行 Dolt CSV 一体流水线: {resolved_dir}")
    pipeline_sql = read_sql(PIPELINE_SQL)
    connection.execute("BEGIN TRANSACTION")
    try:
        row = connection.execute(pipeline_sql).fetchone()
        if row is None:
            raise ValueError("dolt_prices_pipeline.sql 未返回写入行数")
        written = int(row[0])
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise

    n_segments = connection.execute("SELECT count(*) FROM ticker_segment").fetchone()[0]
    n_reused = connection.execute(
        "SELECT count(*) FROM ticker_segment WHERE ticker <> symbol"
    ).fetchone()[0]
    logger.info(
        f"流水线完成: {n_segments} 段 (旧段改名 {n_reused} 段, 阈值 {gap_days} 交易日), "
        f"prices 写入 {written} 行"
    )

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
