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
RAW_DB = DATA / "raw_data.duckdb"  # 供应商原始落地库, 只读
SP500_DIR = DATA / "sp500"  # 标普500 历史成分与 CIK 映射的本地缓存
REUSED_TICKERS = DATA / "reused_tickers.csv"  # 代码复用 (诈尸) 切段名单


if __name__ == "__main__":
    print(DB)
    print(DB_STORAGE)
