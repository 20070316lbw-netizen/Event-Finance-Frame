"""从 Wikipedia 抓取当前标普500成员, 整理成 universe 表需要的列

已知限制: Wikipedia 只提供当前成员, 以抓取当天作为 snapshot_date,
不包含历史上被剔除的成员, 直接用于回测会带来幸存者偏差
"""

from __future__ import annotations

from datetime import date
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd

__all__ = ["UNIVERSE_COLUMNS", "fetch_sp500_universe"]

# 列名和列顺序与 db/ddl/universe.sql 一致
UNIVERSE_COLUMNS = [
    "snapshot_date",
    "ticker",
    "cik",
    "name",
    "exchange",
    "sic",
    "sub_industry",
]


def fetch_sp500_universe(snapshot_date: date | None = None) -> pd.DataFrame:
    """抓取最新成员名单, 返回符合 universe 表 DDL 的数据

    请求最多等待 15 秒

    Args:
        snapshot_date: 快照日期, 默认 None 表示抓取当天 (本地日期)
            Wikipedia 只有当前名单, 传入过去日期不会得到当时的成员

    Returns:
        pd.DataFrame: RangeIndex, 按 ticker 升序排列, 列为 UNIVERSE_COLUMNS
        snapshot_date 为 datetime64, 整列相同
        ticker 将点号替换为连字符, 如 BRK.B 转为 BRK-B
        cik 补零到 10 位字符串, 与 SEC companyfacts 文件名一致
        exchange, sic 页面不提供, 为空值, 留给后续 SEC 数据补齐

    Raises:
        urllib.error.URLError: 页面请求失败
        TimeoutError: 请求超过 15 秒
        ValueError: 页面没有当前成员表, HTML 解析失败, 或 CIK 缺失
        KeyError: 当前成员表缺少使用的列

    Example:
        >>> from event_finance_frame.sources.universe import fetch_sp500_universe
        >>> universe = fetch_sp500_universe()  # doctest: +SKIP
        >>> universe.columns.tolist()  # doctest: +SKIP
        ['snapshot_date', 'ticker', 'cik', 'name', 'exchange', 'sic', 'sub_industry']
    """
    request = Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "EventFinanceFrame/0.1 (S&P 500 research)"},
    )
    with urlopen(request, timeout=15) as response:
        html = response.read().decode("utf-8")

    universe = pd.read_html(
        StringIO(html),
        attrs={"id": "constituents"},
        flavor="lxml",
        keep_default_na=False,
        converters={"CIK": str},
    )[0].rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sub-Industry": "sub_industry",
            "CIK": "cik",
        }
    )
    universe = universe.loc[:, ["ticker", "cik", "name", "sub_industry"]].copy()

    tickers = pd.Series(universe["ticker"], dtype="str")
    universe["ticker"] = tickers.str.replace(".", "-", regex=False)

    ciks = pd.Series(universe["cik"], dtype="str").str.strip()
    if bool((ciks == "").any()):
        missing = universe.loc[ciks == "", "ticker"].tolist()
        raise ValueError(f"以下成员缺少 CIK: {', '.join(missing)}")
    universe["cik"] = ciks.str.zfill(10)

    universe["snapshot_date"] = pd.Timestamp(snapshot_date or date.today())
    universe["exchange"] = pd.Series(pd.NA, index=universe.index, dtype="string")
    universe["sic"] = pd.Series(pd.NA, index=universe.index, dtype="string")

    universe = universe.loc[:, UNIVERSE_COLUMNS]
    return universe.sort_values("ticker").reset_index(drop=True)


if __name__ == "__main__":
    universe = fetch_sp500_universe()
    print(universe)
