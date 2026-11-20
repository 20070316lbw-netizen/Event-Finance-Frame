"""标普500 历史成分: 下载本地缓存, 整理成 universe 写入所需的 DataFrame

数据源 (下载后缓存在 config.SP500_DIR, 建库时只读本地文件):
    sp500_history.csv     fja05680/sp500 的 "S&P 500 Historical Components & Changes (Updated).csv"
                          每行一个调整日期, tickers 为当日全部成员, 逗号分隔
    company_tickers.json  SEC 当前 ticker → CIK 映射, 用来补调出成员的 CIK
    wiki_current.csv      Wikipedia 当前成员 (cik, name, sub_industry)

代码统一用点号, 例如 BRK.B, 与 Dolt 行情 (prices 表) 一致
fja 列表本身就是点号; SEC 和 yfinance 用减号, 读入时转换

已知限制:
    fja 只给 ticker, 没有证券 ID; 已改名的公司 (如 FB → META) 改名前的历史挂在旧代码上
    早年调出、查不到 CIK 的成员, cik 填 CIK_PLACEHOLDER
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from loguru import logger

from event_finance_frame.config import SP500_DIR

__all__ = [
    "CIK_PLACEHOLDER",
    "HISTORY_FILE",
    "SEC_FILE",
    "WIKI_FILE",
    "build_universe_frame",
    "download_sp500_sources",
    "read_sp500_membership",
]

CIK_PLACEHOLDER = "0000000000"  # SEC 不存在 CIK 0, 用作查不到时的占位

HISTORY_FILE = "sp500_history.csv"
SEC_FILE = "company_tickers.json"
WIKI_FILE = "wiki_current.csv"

_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
_SEC_URL = "https://www.sec.gov/files/company_tickers.json"
_USER_AGENT = "EventFinanceFrame/0.1 (S&P 500 research)"
_TIMEOUT = 60


def download_sp500_sources(dest: Path = SP500_DIR) -> None:
    """下载三份数据源到本地缓存目录, 覆盖旧文件 (需要联网)

    Raises:
        urllib.error.URLError: 请求失败
    """
    from event_finance_frame.sources.universe import fetch_sp500_universe

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for url, name in ((_HISTORY_URL, HISTORY_FILE), (_SEC_URL, SEC_FILE)):
        request = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(request, timeout=_TIMEOUT) as response:
            (dest / name).write_bytes(response.read())
        logger.info(f"已下载 {name}")

    wiki = fetch_sp500_universe()
    wiki["ticker"] = wiki["ticker"].str.replace("-", ".", regex=False)
    wiki.loc[:, ["ticker", "cik", "name", "sub_industry"]].to_csv(dest / WIKI_FILE, index=False)
    logger.info(f"已保存 {WIKI_FILE} ({len(wiki)} 只)")


def read_sp500_membership(path: Path, start: date | str | None = None) -> pd.DataFrame:
    """读取 fja05680 历史成分 CSV, 展开成长表

    Args:
        path: sp500_history.csv 路径
        start: 起始日期; 保留 start 当天仍然有效的那个快照 (即 <= start 的最后一个)
            以及之后的所有快照. None 表示全部保留

    Returns:
        pd.DataFrame: 列 snapshot_date (datetime64), symbol (str)
        按 snapshot_date, symbol 排序, 同一快照内 symbol 去重
    """
    raw = pd.read_csv(path, dtype={"tickers": "string"})
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    raw = raw.sort_values("date")

    if start is not None:
        start_ts = pd.Timestamp(start)
        before = raw.loc[raw["date"] <= start_ts, "date"]
        cutoff = before.max() if not before.empty else raw["date"].min()
        raw = raw.loc[raw["date"] >= cutoff]

    members = (
        raw.assign(symbol=raw["tickers"].str.split(","))
        .explode("symbol")
        .rename(columns={"date": "snapshot_date"})
    )
    members["symbol"] = members["symbol"].str.strip()
    members = members.loc[members["symbol"].notna() & (members["symbol"] != "")]
    return (
        members.loc[:, ["snapshot_date", "symbol"]]
        .drop_duplicates()
        .sort_values(["snapshot_date", "symbol"])
        .reset_index(drop=True)
    )


def _read_sec(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sec = pd.DataFrame(list(data.values()))
    sec = pd.DataFrame(
        {
            "symbol": sec["ticker"].astype(str).str.upper().str.replace("-", ".", regex=False),
            "sec_cik": sec["cik_str"].astype(str).str.zfill(10),
            "sec_name": sec["title"].astype(str),
        }
    )
    # 同一 ticker 只保留第一条 (SEC 文件按市值排序)
    return sec.drop_duplicates("symbol", keep="first")


def _read_wiki(path: Path) -> pd.DataFrame:
    wiki = pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(
        {
            "symbol": wiki["ticker"],
            "wiki_cik": wiki["cik"].str.zfill(10),
            "wiki_name": wiki["name"],
            "sub_industry": wiki["sub_industry"].replace("", pd.NA),
        }
    ).drop_duplicates("symbol")


def build_universe_frame(
    sp500_dir: Path = SP500_DIR, start: date | str | None = None
) -> pd.DataFrame:
    """组装 universe 写入用的 DataFrame (尚未映射代码切段)

    代码切段映射、CIK 是否可信、exchange 补齐, 交给 db/insert/universe.sql 处理

    Returns:
        pd.DataFrame: 列 snapshot_date, symbol, wiki_cik, wiki_name, sub_industry, sec_cik, sec_name
        后五列按 symbol 关联, 查不到为空
    """
    sp500_dir = Path(sp500_dir)
    members = read_sp500_membership(sp500_dir / HISTORY_FILE, start=start)
    frame = members.merge(_read_wiki(sp500_dir / WIKI_FILE), on="symbol", how="left").merge(
        _read_sec(sp500_dir / SEC_FILE), on="symbol", how="left"
    )
    logger.info(
        f"历史成分: {members['snapshot_date'].nunique()} 个快照, "
        f"{members['symbol'].nunique()} 个代码, {len(members)} 行"
    )
    return frame.loc[
        :,
        ["snapshot_date", "symbol", "wiki_cik", "wiki_name", "sub_industry", "sec_cik", "sec_name"],
    ]
