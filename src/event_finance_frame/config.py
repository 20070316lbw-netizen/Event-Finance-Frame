from __future__ import annotations

from pathlib import Path

# /Users/liu/Event-Finance-Frame
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "event_finance_frame"
DB = ROOT / "src" / "event_finance_frame" / "db"
DDL = DB / "ddl"
INSERT = DB / "insert"
MACRO = DB / "macro"
VIEW = DB / "view"
SOURCES = SRC / "sources"
DATA = ROOT / "data"
DB_STORAGE = DATA / "lab.duckdb"
TEST_DB = DATA / "test.duckdb"
RAW_DB = DATA / "raw_data.duckdb"  # 已废弃, 保留兼容仅作提示
STOCKS_CSV_DIR = DATA / "stocks_csv"  # Dolt 导出的原始 CSV 目录 (ohlcv/dividend/split/symbol)
SP500_DIR = DATA / "sp500"  # 标普500 历史成分与 CIK 映射的本地缓存
COMPANY_FACTS_ZIP = DATA / "companyfacts.zip"  # SEC Company Facts 归档
REUSED_TICKERS = DATA / "reused_tickers.csv"  # 代码复用 (诈尸) 切段名单


if __name__ == "__main__":
    print(DB)
    print(DB_STORAGE)
