from pathlib import Path

import duckdb
import pandas as pd
import pytest

from event_finance_frame.db.operation import (
    get_duckdb,
    operation_sql,
    read_sql,
)

# 测试专用表，不会创建到正式数据库中。
DDL = """
CREATE TABLE IF NOT EXISTS test_prices (
    ticker VARCHAR NOT NULL,
    close DOUBLE NOT NULL,
    PRIMARY KEY (ticker)
);
"""


@pytest.fixture
def connection():
    """每个使用这个 fixture 的测试都有独立的内存数据库。"""
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


def test_read_sql_preserves_content(tmp_path: Path):
    """SQL 文件中的中文、换行应当完整读出。"""
    path = tmp_path / "example.sql"
    content = "-- 测试中文\nSELECT 1;\n"
    path.write_text(content, encoding="utf-8")

    assert read_sql(path) == content


def test_read_sql_missing_file(tmp_path: Path):
    """读取不存在的 SQL 文件应报错。"""
    with pytest.raises(FileNotFoundError):
        read_sql(tmp_path / "missing.sql")


def test_writable_connection_creates_database(tmp_path: Path):
    """可写模式应自动创建父目录，并将数据保存到磁盘。"""
    path = tmp_path / "nested" / "data" / "test.duckdb"

    with get_duckdb(path=path, read_only=False) as conn:
        conn.execute("CREATE TABLE sample (value INTEGER)")
        conn.execute("INSERT INTO sample VALUES (42)")

    assert path.is_file()

    # 重新打开，确认数据确实保存了。
    with get_duckdb(path=path) as conn:
        assert conn.execute("SELECT value FROM sample").fetchall() == [(42,)]


def test_read_only_missing_database(tmp_path: Path):
    """默认只读模式不能悄悄创建不存在的数据库或目录。"""
    path = tmp_path / "missing" / "test.duckdb"

    with pytest.raises(FileNotFoundError):
        get_duckdb(path=path)

    assert not path.parent.exists()


def test_read_only_connection_rejects_writes(tmp_path: Path):
    """默认连接应允许读取，但拒绝写入。"""
    path = tmp_path / "test.duckdb"

    with get_duckdb(path=path, read_only=False) as conn:
        conn.execute("CREATE TABLE sample (value INTEGER)")
        conn.execute("INSERT INTO sample VALUES (1)")

    with get_duckdb(path=path) as conn:
        assert conn.execute("SELECT * FROM sample").fetchall() == [(1,)]

        with pytest.raises(duckdb.Error):
            conn.execute("INSERT INTO sample VALUES (2)")

    # 确认被拒绝的写入没有改变数据。
    with get_duckdb(path=path) as conn:
        assert conn.execute("SELECT * FROM sample").fetchall() == [(1,)]


def test_operation_sql_inserts_dataframe(connection):
    """应能从传入的 df 写入数据，并返回实际写入行数。"""
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "close": [10.5, 20.0],
        }
    )

    written = operation_sql(
        sql="INSERT INTO test_prices SELECT ticker, close FROM df;",
        ddl=DDL,
        connection=connection,
        df=df,
    )

    assert written == 2
    assert connection.execute(
        "SELECT ticker, close FROM test_prices ORDER BY ticker"
    ).fetchall() == [
        ("AAA", 10.5),
        ("BBB", 20.0),
    ]


def test_operation_sql_without_dataframe(connection):
    """不传 DataFrame 时，也应能执行普通 INSERT。"""
    written = operation_sql(
        sql="INSERT INTO test_prices VALUES ('AAA', 10.5);",
        ddl=DDL,
        connection=connection,
    )

    assert written == 1
    assert connection.execute("SELECT * FROM test_prices").fetchall() == [("AAA", 10.5)]


def test_failed_write_rolls_back_new_table(connection):
    """首次建表后写入失败，建表本身也应回滚。"""
    with pytest.raises(duckdb.Error):
        operation_sql(
            sql="INSERT INTO test_prices VALUES ('AAA', NULL);",
            ddl=DDL,
            connection=connection,
        )

    assert connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = 'test_prices'
        """
    ).fetchone() == (0,)

    # 回滚后，连接应当还能正常使用。
    written = operation_sql(
        sql="INSERT INTO test_prices VALUES ('BBB', 20.0);",
        ddl=DDL,
        connection=connection,
    )

    assert written == 1


def test_failed_batch_preserves_previously_committed_data(connection):
    """批量写入中途失败，应撤销本批数据，保留之前已提交的数据。"""
    operation_sql(
        sql="INSERT INTO test_prices VALUES ('OLD', 5.0);",
        ddl=DDL,
        connection=connection,
    )

    with pytest.raises(duckdb.Error):
        operation_sql(
            sql="""
                INSERT INTO test_prices VALUES ('NEW', 10.0);
                INSERT INTO test_prices VALUES ('OLD', 99.0);
            """,
            ddl=DDL,
            connection=connection,
        )

    # NEW 应被回滚，OLD 的原始数据应保留。
    assert connection.execute(
        "SELECT ticker, close FROM test_prices ORDER BY ticker"
    ).fetchall() == [("OLD", 5.0)]

    # 再次写入，确认函数没有关闭连接或留下未结束的事务。
    written = operation_sql(
        sql="INSERT INTO test_prices VALUES ('AFTER', 30.0);",
        ddl=DDL,
        connection=connection,
    )

    assert written == 1
