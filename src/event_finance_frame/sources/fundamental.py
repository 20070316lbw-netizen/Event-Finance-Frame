"""下载 SEC Company Facts 归档, 并将指定公司的事实拆成两张表"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

import duckdb
import pandas as pd
from loguru import logger

from event_finance_frame.config import COMPANY_FACTS_ZIP, DDL, INSERT
from event_finance_frame.db.operation import read_sql
from event_finance_frame.sources.sp500_history import CIK_PLACEHOLDER

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "download_company_facts",
    "load_fundamentals_from_zip",
    "read_company_facts",
]

DEFAULT_BATCH_SIZE = 50

_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def download_company_facts(destination: str | Path, *, user_agent: str) -> Path:
    """下载 SEC 归档, 检查 ZIP 后再替换目标文件

    Args:
        destination: 本地 ZIP 路径, 父目录须存在
        user_agent: 包含调用者名称和联系邮箱的 SEC User-Agent

    Returns:
        Path: 下载完成的本地归档路径

    Raises:
        OSError: 请求失败或文件无法写入
        ValueError: User-Agent 或归档为空
        BadZipFile: 下载内容无法作为 ZIP 打开

    Example:
        >>> archive = download_company_facts(  # doctest: +SKIP
        ...     "data/companyfacts.zip", user_agent="Your Name you@example.com"
        ... )
        >>> archive.name  # doctest: +SKIP
        'companyfacts.zip'
    """
    if not user_agent.strip():
        raise ValueError("SEC User-Agent 须包含调用者名称和联系邮箱")
    destination = Path(destination)
    request = Request(_ARCHIVE_URL, headers={"User-Agent": user_agent})
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / "companyfacts.zip"
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            copyfileobj(response, output, length=1024 * 1024)
        with ZipFile(temporary) as archive:
            if not archive.namelist():
                raise ValueError("SEC Company Facts 归档为空")
        temporary.replace(destination)
    return destination


# 列名和列顺序与 db/ddl/fundamental_period.sql, fundamental_instant.sql 一致
PERIOD_COLUMNS = [
    "cik",
    "taxonomy",
    "tag",
    "unit",
    "start_date",
    "end_date",
    "val",
    "accn",
    "filed",
    "form",
    "fy",
    "fp",
    "frame",
]

INSTANT_COLUMNS = [
    "cik",
    "taxonomy",
    "tag",
    "unit",
    "end_date",
    "val",
    "accn",
    "filed",
    "form",
    "fy",
    "fp",
    "frame",
]


def _columns(names: list[str]) -> dict[str, list[object]]:
    return {name: [] for name in names}


def _append(
    bucket: dict[str, list[object]],
    cik: str,
    taxonomy: str,
    tag: str,
    unit: str,
    fact: dict[str, object],
) -> None:
    bucket["cik"].append(cik)
    bucket["taxonomy"].append(taxonomy)
    bucket["tag"].append(tag)
    bucket["unit"].append(unit)
    if "start_date" in bucket:
        bucket["start_date"].append(fact.get("start"))
    bucket["end_date"].append(fact.get("end"))
    bucket["val"].append(fact.get("val"))
    bucket["accn"].append(fact.get("accn"))
    bucket["filed"].append(fact.get("filed"))
    bucket["form"].append(fact.get("form"))
    bucket["fy"].append(fact.get("fy"))
    bucket["fp"].append(fact.get("fp"))
    bucket["frame"].append(fact.get("frame"))


def _as_table(
    data: dict[str, list[object]],
    date_columns: tuple[str, ...],
    *,
    coerce_errors: bool = False,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: pd.Series(values, dtype=object) for column, values in data.items()}
    )
    dt_errors = "coerce" if coerce_errors else "raise"
    num_errors = "coerce" if coerce_errors else "raise"
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors=dt_errors)
    for column, dtype in (("val", "float64"), ("fy", "Int64")):
        numeric = pd.to_numeric(frame[column], errors=num_errors)
        if not isinstance(numeric, pd.Series):
            raise TypeError(f"{column} 必须是一维 Series")
        frame[column] = numeric.astype(dtype)
    return frame


def _read_batch(
    archive: ZipFile,
    ciks: Iterable[str],
    *,
    coerce_errors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period = _columns(PERIOD_COLUMNS)
    instant = _columns(INSTANT_COLUMNS)
    for cik in dict.fromkeys(ciks):
        payload = json.loads(archive.read(f"CIK{cik}.json"))
        for taxonomy, tags in (payload.get("facts") or {}).items():
            for tag, node in tags.items():
                for unit, facts in (node.get("units") or {}).items():
                    for fact in facts:
                        bucket = period if fact.get("start") is not None else instant
                        _append(bucket, cik, taxonomy, tag, unit, fact)
    return (
        _as_table(period, ("start_date", "end_date", "filed"), coerce_errors=coerce_errors),
        _as_table(instant, ("end_date", "filed"), coerce_errors=coerce_errors),
    )


def read_company_facts(
    zip_path: str | Path, ciks: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取指定 CIK 的公司事实, 按期间数和时点数拆开

    Args:
        zip_path: companyfacts.zip 的本地路径, 支持 str 或 Path
        ciks: 10位 CIK, 对应压缩包内的 CIK##########.json

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 依次为 period_facts, instant_facts 的列
        带 start 的事实进入第一张表, 其余进入第二张表
        日期列为 datetime64, fy 为可空整数
        缺失的 frame, fy, fp 为空值
        没有事实时返回带列名的空表

    Raises:
        FileNotFoundError: zip 不存在
        KeyError: 某个 CIK 在压缩包中没有对应文件
        json.JSONDecodeError: 公司文件不是有效 JSON

    Example:
        >>> from event_finance_frame.sources.fundamental import read_company_facts
        >>> period, instant = read_company_facts(  # doctest: +SKIP
        ...     "data/companyfacts.zip", ["0000320193"]
        ... )
        >>> period.columns[-1]  # doctest: +SKIP
        'frame'
        >>> "start_date" in instant.columns  # doctest: +SKIP
        False
    """
    with ZipFile(zip_path) as archive:
        return _read_batch(archive, ciks, coerce_errors=False)


def load_fundamentals_from_zip(
    connection: duckdb.DuckDBPyConnection,
    *,
    zip_path: Path = COMPANY_FACTS_ZIP,
    ciks: Iterable[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    """分批从 companyfacts.zip 解析并写入 fundamental_instant 与 fundamental_period

    重复执行结果不变 (先清空再写入), 自动跳过压缩包中缺失的 CIK、非法日期/数值与主键重复行

    Args:
        connection: 可写的 lab 库连接, 调用前不应有未结束的事务
        zip_path: companyfacts.zip 路径, 默认 config.COMPANY_FACTS_ZIP
        ciks: 要入库的 10 位 CIK 列表; None 时自动从 universe 表取全部非占位 CIK
        batch_size: 每批解析并写入的 CIK 数量, 默认 50, 控制内存峰值

    Returns:
        dict[str, int]: 两张表的最终行数, 键为 fundamental_instant, fundamental_period

    Raises:
        FileNotFoundError: companyfacts.zip 不存在
        ValueError: batch_size 不是正整数
    """
    if int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError(f"batch_size 必须是正整数: {batch_size}")

    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"SEC Company Facts 归档不存在: {zip_path}")

    if ciks is None:
        rows = connection.execute(
            "SELECT DISTINCT cik FROM universe WHERE cik <> ? ORDER BY cik",
            [CIK_PLACEHOLDER],
        ).fetchall()
        target_ciks = [row[0] for row in rows]
    else:
        target_ciks = list(dict.fromkeys(ciks))

    instant_ddl = read_sql(DDL / "fundamental_instant.sql")
    period_ddl = read_sql(DDL / "fundamental_period.sql")
    instant_insert = read_sql(INSERT / "fundamental_instant.sql")
    period_insert = read_sql(INSERT / "fundamental_period.sql")

    with ZipFile(zip_path) as archive:
        available = set(archive.namelist())
        present = [cik for cik in target_ciks if f"CIK{cik}.json" in available]
        missing = [cik for cik in target_ciks if f"CIK{cik}.json" not in available]
        if missing:
            sample = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            logger.warning(
                f"companyfacts.zip 中缺少 {len(missing)} 个 CIK, 已跳过: {sample}{suffix}"
            )

        logger.info(
            f"开始分批写入基本面: {len(present)} 个 CIK (batch_size={batch_size}, 来源 {zip_path.name})"
        )
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(instant_ddl)
            connection.execute(period_ddl)
            connection.execute("DELETE FROM fundamental_instant")
            connection.execute("DELETE FROM fundamental_period")

            for start in range(0, len(present), batch_size):
                batch = present[start : start + batch_size]
                period_df, instant_df = _read_batch(archive, batch, coerce_errors=True)
                if not instant_df.empty:
                    connection.register("df", instant_df)
                    connection.execute(instant_insert)
                    connection.unregister("df")
                if not period_df.empty:
                    connection.register("df", period_df)
                    connection.execute(period_insert)
                    connection.unregister("df")

            n_instant = int(
                connection.execute("SELECT count(*) FROM fundamental_instant").fetchone()[0]
            )
            n_period = int(
                connection.execute("SELECT count(*) FROM fundamental_period").fetchone()[0]
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    logger.info(
        f"基本面写入完成: fundamental_instant {n_instant:,} 行, fundamental_period {n_period:,} 行"
    )
    return {
        "fundamental_instant": n_instant,
        "fundamental_period": n_period,
    }
