"""检查数据源输出能按列名写入 db/ddl 中的正式表结构"""

import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd
import pytest

from event_finance_frame.db.operation import operation_sql, read_sql
from event_finance_frame.sources.fundamental import read_company_facts
from event_finance_frame.sources.universe import UNIVERSE_COLUMNS

DDL = Path(__file__).resolve().parents[1] / "src" / "event_finance_frame" / "db" / "ddl"


@pytest.fixture
def connection():
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


def _insert(connection, table: str, df: pd.DataFrame) -> int:
    columns = ", ".join(df.columns)
    return operation_sql(
        sql=f"INSERT INTO {table} ({columns}) SELECT {columns} FROM df;",
        ddl=read_sql(DDL / f"{table}.sql"),
        connection=connection,
        df=df,
    )


def test_prices_ddl_is_valid(connection):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["AAA"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "adj_close": [1.0],
            "volume": [100],
        }
    )
    assert _insert(connection, "prices", df) == 1


def test_fundamental_frames_match_ddl(tmp_path: Path, connection):
    fact = {
        "end": "2023-12-31",
        "val": 1.0,
        "accn": "a",
        "form": "10-K",
        "filed": "2024-02-01",
        "fy": 2023,
        "fp": "FY",
    }
    payload = {
        "facts": {
            "us-gaap": {
                "Assets": {"units": {"USD": [fact]}},
                "Revenues": {"units": {"USD": [{**fact, "start": "2023-01-01"}]}},
            }
        }
    }
    path = tmp_path / "facts.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("CIK0000000001.json", json.dumps(payload))

    period, instant = read_company_facts(path, ["0000000001"])

    assert _insert(connection, "fundamental_period", period) == 1
    assert _insert(connection, "fundamental_instant", instant) == 1


def test_universe_frame_matches_ddl(connection):
    df = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2026-11-17"]),
            "ticker": ["AAPL"],
            "cik": ["0000320193"],
            "name": ["Apple"],
            "exchange": pd.Series([pd.NA], dtype="string"),
            "sic": pd.Series([pd.NA], dtype="string"),
            "sub_industry": ["Hardware"],
        }
    )[UNIVERSE_COLUMNS]
    assert _insert(connection, "universe", df) == 1
