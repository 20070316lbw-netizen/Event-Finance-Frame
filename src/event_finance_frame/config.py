from __future__ import annotations

from pathlib import Path

# /Users/liu/Event-Finance-Frame
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
SRC = ROOT / "src" / "event_finance_frame"
FACTOR_CONFIGS = SRC / "factors" / "configs"
DB = ROOT / "src" / "event_finance_frame" / "db"
DDL = DB / "ddl"
INSERT = DB / "insert"
MACRO = DB / "macro"
VIEW = DB / "view"
SOURCES = SRC / "sources"
DATA = ROOT / "data"
DB_STORAGE = DATA / "lab.duckdb"
TEST_DB = DATA / "test.duckdb"
STOCKS_CSV_DIR = DATA / "stocks_csv"  # Dolt 导出的原始 CSV 目录 (ohlcv/dividend/split/symbol)
SP500_DIR = DATA / "sp500"  # 标普500 历史成分与 CIK 映射的本地缓存
COMPANY_FACTS_ZIP = DATA / "companyfacts.zip"  # SEC Company Facts 归档
REUSED_TICKERS = DATA / "reused_tickers.csv"  # 代码复用 (诈尸) 切段名单
