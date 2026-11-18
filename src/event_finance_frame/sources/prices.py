"""从 Yahoo 下载行情, 清洗后返回 [date, ticker] 面板

核心函数: get_prices()

"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import date, datetime

import pandas as pd
from loguru import logger

DateLike = str | date | datetime

# 一次并发下载几百只股票时, Yahoo 可能返回空行情, yfinance 会把它记成
# possibly delisted, 先缩小批次和线程数, 再重试整段缺失的证券
_CHUNK_SIZE = 40
_DOWNLOAD_THREADS = 4
_MAX_ATTEMPTS = 3
_CHUNK_PAUSE_SECONDS = 1.0
_RETRY_PAUSE_SECONDS = (2.0, 5.0)
_DOWNLOAD_TIMEOUT = 30

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]
_NUMERIC_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def to_date_ticker_frame(*, df: pd.DataFrame) -> pd.DataFrame:
    """把 date 和 ticker 列设为索引, 得到 [date, ticker] 面板

    Args:
        df: 普通索引 DataFrame, 必需列为 date 和 ticker, 无默认值
            date 可转换为日期, ticker 为证券代码, 其余列沿用输入单位
            本函数会将输入 df 的 date 列转为日期类型

    Returns:
        按 [date, ticker] 升序排序的 MultiIndex DataFrame
        保留其余数据列及缺失值, 不补齐日期或删除记录
        空输入返回保留列结构的空 DataFrame

    Raises:
        KeyError: date 或 ticker 列不存在
        ValueError: date 无法转换为日期

    Example:
        >>> raw = pd.DataFrame({
        ...     "date": ["2024-01-03", "2024-01-02"],
        ...     "ticker": ["A", "A"], "close": [110.0, 100.0],
        ... })
        >>> prices = to_date_ticker_frame(df=raw)
        >>> list(prices.index.names)
        ['date', 'ticker']
        >>> prices["close"].tolist()
        [100.0, 110.0]
    """

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    return df.set_index(["date", "ticker"]).sort_index()


def get_prices(
    tickers: str | Sequence[str],
    start: DateLike,
    end: DateLike | None = None,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """抓取一批股票的历史行情, 返回标准 [date, ticker] 行情面板

    Args:
        tickers: 单个 ticker 或 ticker 列表, 如 "AAPL" 或 ["AAPL", "MSFT"]
        start: 起始日期(含), 如 "2020-01-01"
        end: 结束日期, 不含当天, 默认 None, 结束范围由 yfinance 决定
        interval: 行情周期, 默认 "1d", 也可传 yfinance 支持的 "1wk" 或 "1mo"
        auto_adjust: 原样传给 yfinance; True 时 yfinance 直接返回复权价并可能
            丢弃 Adj Close 列, False(默认)保留原始 close 与 adj_close 两列
        progress: 是否显示 yfinance 下载进度条, 默认 True, 仅限关键词传入
            证券较多时分批下载, 每批各有一条进度条
            进度按该批已处理的 ticker 数量推进, 包含失败任务, 不包括清洗和入库

    Returns:
        [date, ticker] MultiIndex DataFrame, 按索引升序排序
        列为 open, high, low, close, adj_close, volume, 单位沿用 Yahoo 行情
        close 缺失的记录删除, 其他缺失字段保留缺失值
        整段 close 都缺失时会降低并发再试, 仍没有则跳过该证券
        找不到数据时返回保留索引和列结构的空 DataFrame

    Raises:
        ValueError: 多证券响应缺少 ticker 列层级

    Example:
        下载示例需联网
        >>> prices = get_prices("AAPL", start="2024-01-01", end="2024-02-01", progress=False)
        >>> list(prices.index.names)
        ['date', 'ticker']
    """
    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)

    logger.info(
        f"Fetching prices for {len(ticker_list)} ticker(s) "
        f"from {start} to {end or 'today'} (interval={interval})"
    )

    import yfinance as yf

    frames: list[pd.DataFrame] = []
    pending = ticker_list
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if not pending:
            break
        threads: bool | int = _DOWNLOAD_THREADS if attempt == 1 else False
        if attempt > 1:
            pause = _RETRY_PAUSE_SECONDS[attempt - 2]
            logger.warning(
                f"{len(pending)} 个 ticker 没有可用 close, {pause:g}s 后进行第 "
                f"{attempt} 次下载. Yahoo 限流时, yfinance 会把这些证券标成 "
                "possibly delisted"
            )
            _pause(pause)
        pending = _download_pending(
            yf.download,
            pending,
            frames,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=progress,
            threads=threads,
        )

    if pending:
        logger.warning(
            f"以下 {len(pending)} 个 ticker 多次下载后仍没有 close, 已跳过: "
            + ", ".join(pending)
        )
    if not frames:
        return _empty_frame()
    return to_date_ticker_frame(df=pd.concat(frames, ignore_index=True))


def _download_pending(
    download: Callable[..., pd.DataFrame | None],
    tickers: list[str],
    frames: list[pd.DataFrame],
    *,
    start: DateLike,
    end: DateLike | None,
    interval: str,
    auto_adjust: bool,
    progress: bool,
    threads: bool | int,
) -> list[str]:
    """分批下载, 返回这一轮结束后仍然没有可用 close 的 ticker"""
    failed: list[str] = []
    for index, chunk in enumerate(_chunks(tickers, _CHUNK_SIZE)):
        if index > 0:
            _pause(_CHUNK_PAUSE_SECONDS)
        raw = download(
            chunk,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by="ticker",
            progress=progress,
            threads=threads,
            timeout=_DOWNLOAD_TIMEOUT,
        )
        ok, bad = _split_prices(raw, chunk, auto_adjust)
        frames.extend(ok)
        failed.extend(bad)
    return failed


def _chunks(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _pause(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _split_prices(
    raw: pd.DataFrame | None,
    tickers: Sequence[str],
    auto_adjust: bool,
) -> tuple[list[pd.DataFrame], list[str]]:
    """拆出有 close 的证券, 整段缺失的留给重试"""
    if raw is None or raw.empty:
        return [], list(tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        columns = {
            str(name).upper(): name
            for name in dict.fromkeys(raw.columns.get_level_values(0))
        }
        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        for ticker in tickers:
            column = columns.get(ticker.upper())
            if column is None:
                failed.append(ticker)
                continue
            single = raw.loc[:, column].copy()
            if not isinstance(single, pd.DataFrame):
                raise TypeError(f"{ticker} 的行情必须是 DataFrame")
            cleaned = _clean_single(single, ticker, auto_adjust)
            if cleaned.empty:
                failed.append(ticker)
            else:
                frames.append(cleaned)
        return frames, failed

    if len(tickers) != 1:
        raise ValueError("多证券行情缺少 ticker 列层级, 无法确认证券身份")
    cleaned = _clean_single(raw.copy(), tickers[0], auto_adjust)
    if cleaned.empty:
        return [], list(tickers)
    return [cleaned], []


def _clean_single(
    df: pd.DataFrame, ticker: str, auto_adjust: bool = False
) -> pd.DataFrame:
    """把 yfinance 返回的单只证券行情整理成标准长表"""
    df = df.rename(columns=_COLUMN_MAP)
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["ticker"] = ticker

    # 只有数据源明确已复权时, close 才能当作 adj_close; 缺少复权价时保留空值
    if "adj_close" not in df.columns and auto_adjust:
        df["adj_close"] = df.get("close")

    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.loc[:, _OUTPUT_COLUMNS].copy()

    df[_NUMERIC_COLUMNS] = df[_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    before = len(df)
    df = df.dropna(subset=["close"])
    dropped = before - len(df)
    if dropped and not df.empty:
        logger.debug(f"{ticker}: 丢弃 {dropped} 行 close 缺失的记录")

    return df


def _empty_frame() -> pd.DataFrame:
    return to_date_ticker_frame(df=pd.DataFrame(columns=_OUTPUT_COLUMNS))


if __name__ == "__main__":
    prices = get_prices(tickers="AAPL", start="2026-11-01", end="2026-11-05")
    print(prices)
