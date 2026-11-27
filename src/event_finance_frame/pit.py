"""Point-in-Time (时点隔离) 数据视图

严格遵守 Design.md 时点规则:
    - prices: date <= now
    - universe: snapshot_date <= now 的最新一期快照
    - fundamental_*: filed < now (次日可见), 常规报表 (10-K/10-Q/10-K/A/10-Q/A), 修订取最新 filed/accn
    - 严禁调用或暴露 label_forward_return 等未来标签宏
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Literal

import duckdb
import pandas as pd

from event_finance_frame.calendar import DateLike, normalize_date
from event_finance_frame.clock import Clock
from event_finance_frame.error import EventFinanceError, PITErrorCode

__all__ = [
    "FUNDAMENTAL_COLUMNS",
    "PRICE_COLUMNS",
    "UNIVERSE_COLUMNS",
    "PITDataView",
]

PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)
_VALID_PRICE_COLS = frozenset(PRICE_COLUMNS)
_VALID_RETURN_PRICE_COLS = frozenset({"adj_close", "close"})
_VALID_FACT_KINDS = frozenset({"instant", "period"})

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "ticker",
    "cik",
    "name",
    "exchange",
    "sic",
    "sub_industry",
)

FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "ticker",
    "cik",
    "fact_kind",
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
    "report_rank",
)


def _validate_positive_int(value: Any, *, name: str, capability: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EventFinanceError(
            PITErrorCode.INVALID_ARGUMENT,
            f"{name} 必须是 >= 1 的整数, 实际为 {value!r}",
            capability=capability,
        )
    return value


def _normalize_str_filter(
    values: str | Sequence[str] | None,
    *,
    name: str,
    capability: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        if not values:
            raise EventFinanceError(
                PITErrorCode.INVALID_ARGUMENT,
                f"{name} 不能为空字符串",
                capability=capability,
            )
        return (values,)
    if isinstance(values, Sequence):
        seen: dict[str, None] = {}
        for item in values:
            if not isinstance(item, str) or not item:
                raise EventFinanceError(
                    PITErrorCode.INVALID_ARGUMENT,
                    f"{name} 中的元素必须是非空字符串, 实际为 {item!r}",
                    capability=capability,
                )
            seen[item] = None
        return tuple(seen.keys())
    raise EventFinanceError(
        PITErrorCode.INVALID_ARGUMENT,
        f"{name} 必须是字符串或字符串序列, 实际为 {type(values).__name__}",
        capability=capability,
    )


def _validate_price_columns(
    columns: Sequence[str] | None,
    *,
    capability: str,
) -> tuple[str, ...]:
    if columns is None:
        return PRICE_COLUMNS
    if isinstance(columns, str) or not isinstance(columns, Sequence) or len(columns) == 0:
        raise EventFinanceError(
            PITErrorCode.INVALID_ARGUMENT,
            "columns 必须是非空列名序列",
            capability=capability,
        )
    seen: dict[str, None] = {}
    for col in columns:
        if not isinstance(col, str) or col not in _VALID_PRICE_COLS:
            raise EventFinanceError(
                PITErrorCode.INVALID_ARGUMENT,
                f"非法行情列 {col!r}, 允许的列: {PRICE_COLUMNS}",
                capability=capability,
            )
        seen[col] = None
    return tuple(seen.keys())


def _resolve_universe_only(
    tickers: tuple[str, ...] | None,
    universe_only: bool | None,
    *,
    capability: str,
) -> bool:
    if universe_only is not None:
        if not isinstance(universe_only, bool):
            raise EventFinanceError(
                PITErrorCode.INVALID_ARGUMENT,
                f"universe_only 必须是 bool 或 None, 实际为 {type(universe_only).__name__}",
                capability=capability,
            )
        return universe_only
    return tickers is None


def _empty_panel(value_cols: Sequence[str]) -> pd.DataFrame:
    empty_index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([], dtype="datetime64[ns]", name="date"),
            pd.Index([], dtype="object", name="ticker"),
        ],
        names=["date", "ticker"],
    )
    return pd.DataFrame(columns=list(value_cols), index=empty_index)


def _to_panel(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return _empty_panel(value_cols)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out = out.set_index(["date", "ticker"]).sort_index()
    return out.loc[:, list(value_cols)]


class PITDataView:
    """绑定到 Clock 的只读 Point-in-Time 数据视图

    每次查询实时读取 clock.now, 仅暴露在 now 收盘时已经公开可见的历史数据.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, clock: Clock) -> None:
        self._conn = connection
        self._clock = clock
        self._has_market_calendar = self._table_exists("market_calendar")

    @property
    def now(self) -> date:
        return self._clock.now

    @property
    def clock(self) -> Clock:
        return self._clock

    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = ?
            """,
            [table_name],
        ).fetchone()
        return bool(row and row[0] > 0)

    def _check_not_future(self, dt: date, *, arg_name: str, capability: str) -> date:
        if dt > self.now:
            raise EventFinanceError(
                PITErrorCode.FUTURE_ACCESS_DENIED,
                f"{arg_name}={dt} 晚于当前模拟时点 now={self.now}, 禁止访问未来数据",
                capability=capability,
            )
        return dt

    def _resolve_lookback_start(self, end_dt: date, lookback: int) -> date | None:
        if self._has_market_calendar:
            sql = """
            SELECT min(date)
            FROM (
                SELECT date
                FROM market_calendar
                WHERE is_open = TRUE
                  AND date <= CAST(? AS DATE)
                ORDER BY date DESC
                LIMIT ?
            )
            """
        else:
            sql = """
            SELECT min(date)
            FROM (
                SELECT DISTINCT date
                FROM prices
                WHERE date <= CAST(? AS DATE)
                ORDER BY date DESC
                LIMIT ?
            )
            """
        row = self._conn.execute(sql, [end_dt, lookback]).fetchone()
        if row is None or row[0] is None:
            return None
        return normalize_date(row[0])

    def _build_ticker_scope_cte(
        self,
        as_of: date,
        tickers: tuple[str, ...] | None,
        use_universe: bool,
    ) -> tuple[str, list[Any]]:
        if use_universe:
            if tickers is None:
                cte = """
                target_tickers AS (
                    SELECT DISTINCT ticker
                    FROM universe
                    WHERE snapshot_date = (
                        SELECT max(snapshot_date)
                        FROM universe
                        WHERE snapshot_date <= CAST(? AS DATE)
                    )
                )
                """
                return cte, [as_of]
            placeholders = ", ".join(["?"] * len(tickers))
            cte = f"""
            target_tickers AS (
                SELECT DISTINCT ticker
                FROM universe
                WHERE snapshot_date = (
                    SELECT max(snapshot_date)
                    FROM universe
                    WHERE snapshot_date <= CAST(? AS DATE)
                )
                  AND ticker IN ({placeholders})
            )
            """
            return cte, [as_of, *tickers]

        if tickers is not None:
            values_sql = ", ".join(["(?)"] * len(tickers))
            cte = f"""
            target_tickers(ticker) AS (
                VALUES {values_sql}
            )
            """
            return cte, [*tickers]

        return "", []

    def universe(self) -> pd.DataFrame:
        """返回截至 now 最新一期的成分股快照"""
        sql = """
        SELECT snapshot_date, ticker, cik, name, exchange, sic, sub_industry
        FROM universe
        WHERE snapshot_date = (
            SELECT max(snapshot_date)
            FROM universe
            WHERE snapshot_date <= CAST(? AS DATE)
        )
        ORDER BY ticker
        """
        df = self._conn.execute(sql, [self.now]).df()
        return df.loc[:, list(UNIVERSE_COLUMNS)]

    def universe_tickers(self) -> tuple[str, ...]:
        """返回截至 now 最新一期成分股代码元组 (升序)"""
        sql = """
        SELECT ticker
        FROM universe
        WHERE snapshot_date = (
            SELECT max(snapshot_date)
            FROM universe
            WHERE snapshot_date <= CAST(? AS DATE)
        )
        ORDER BY ticker
        """
        rows = self._conn.execute(sql, [self.now]).fetchall()
        return tuple(row[0] for row in rows)

    def prices(
        self,
        tickers: str | Sequence[str] | None = None,
        *,
        lookback: int | None = None,
        start: DateLike | None = None,
        end: DateLike | None = None,
        columns: Sequence[str] | None = None,
        universe_only: bool | None = None,
    ) -> pd.DataFrame:
        """查询截至 now 可见的日频行情面板, 返回 (date, ticker) MultiIndex DataFrame"""
        cap = "PITDataView.prices"
        cols = _validate_price_columns(columns, capability=cap)
        ticker_list = _normalize_str_filter(tickers, name="tickers", capability=cap)
        use_universe = _resolve_universe_only(ticker_list, universe_only, capability=cap)

        if lookback is not None and start is not None:
            raise EventFinanceError(
                PITErrorCode.INVALID_ARGUMENT,
                "lookback 与 start 不能同时指定",
                capability=cap,
            )
        if lookback is not None:
            _validate_positive_int(lookback, name="lookback", capability=cap)

        end_dt = (
            self._check_not_future(normalize_date(end), arg_name="end", capability=cap)
            if end is not None
            else self.now
        )

        start_dt: date | None = None
        if start is not None:
            start_dt = self._check_not_future(
                normalize_date(start), arg_name="start", capability=cap
            )
            if start_dt > end_dt:
                raise EventFinanceError(
                    PITErrorCode.INVALID_ARGUMENT,
                    f"start ({start_dt}) 不能晚于 end ({end_dt})",
                    capability=cap,
                )
        elif lookback is not None:
            start_dt = self._resolve_lookback_start(end_dt, lookback)
            if start_dt is None:
                return _empty_panel(cols)

        if ticker_list == ():
            return _empty_panel(cols)

        scope_cte, scope_params = self._build_ticker_scope_cte(
            self.now, ticker_list, use_universe
        )
        select_cols = ", ".join(f"p.{c}" for c in cols)
        with_sql = f"WITH {scope_cte}" if scope_cte else ""
        join_sql = "JOIN target_tickers AS t USING (ticker)" if scope_cte else ""

        where_clauses = ["p.date <= CAST(? AS DATE)"]
        params: list[Any] = [*scope_params, end_dt]
        if start_dt is not None:
            where_clauses.append("p.date >= CAST(? AS DATE)")
            params.append(start_dt)

        sql = f"""
        {with_sql}
        SELECT p.date, p.ticker, {select_cols}
        FROM prices AS p
        {join_sql}
        WHERE {" AND ".join(where_clauses)}
        ORDER BY p.date, p.ticker
        """
        df = self._conn.execute(sql, params).df()
        return _to_panel(df, cols)

    def latest_prices(
        self,
        tickers: str | Sequence[str] | None = None,
        *,
        columns: Sequence[str] | None = None,
        universe_only: bool | None = None,
    ) -> pd.DataFrame:
        """查询 now 当天收盘行情截面, 返回以 ticker 为索引的 DataFrame"""
        cap = "PITDataView.latest_prices"
        cols = _validate_price_columns(columns, capability=cap)
        ticker_list = _normalize_str_filter(tickers, name="tickers", capability=cap)
        use_universe = _resolve_universe_only(ticker_list, universe_only, capability=cap)

        if ticker_list == ():
            return pd.DataFrame(columns=list(cols), index=pd.Index([], name="ticker"))

        scope_cte, scope_params = self._build_ticker_scope_cte(
            self.now, ticker_list, use_universe
        )
        select_cols = ", ".join(f"p.{c}" for c in cols)
        with_sql = f"WITH {scope_cte}" if scope_cte else ""
        join_sql = "JOIN target_tickers AS t USING (ticker)" if scope_cte else ""

        sql = f"""
        {with_sql}
        SELECT p.ticker, {select_cols}
        FROM prices AS p
        {join_sql}
        WHERE p.date = CAST(? AS DATE)
        ORDER BY p.ticker
        """
        df = self._conn.execute(sql, [*scope_params, self.now]).df()
        if df.empty:
            return pd.DataFrame(columns=list(cols), index=pd.Index([], name="ticker"))
        return df.set_index("ticker").loc[:, list(cols)]

    def fundamentals(
        self,
        tickers: str | Sequence[str] | None = None,
        *,
        tags: str | Sequence[str] | None = None,
        fact_kind: Literal["instant", "period"] | None = None,
        n_reports: int = 5,
    ) -> pd.DataFrame:
        """查询截至 now 可见的 PIT 基本面数据 (filed < now, 修订取最新版本)"""
        cap = "PITDataView.fundamentals"
        ticker_list = _normalize_str_filter(tickers, name="tickers", capability=cap)
        tag_list = _normalize_str_filter(tags, name="tags", capability=cap)
        _validate_positive_int(n_reports, name="n_reports", capability=cap)
        if fact_kind is not None and fact_kind not in _VALID_FACT_KINDS:
            raise EventFinanceError(
                PITErrorCode.INVALID_ARGUMENT,
                f"fact_kind 必须是 'instant', 'period' 或 None, 实际为 {fact_kind!r}",
                capability=cap,
            )

        if ticker_list == () or tag_list == ():
            return pd.DataFrame(columns=list(FUNDAMENTAL_COLUMNS))

        univ_filters = [
            """
            snapshot_date = (
                SELECT max(snapshot_date)
                FROM universe
                WHERE snapshot_date <= CAST(? AS DATE)
            )
            """,
            "cik <> '0000000000'",
        ]
        params: list[Any] = [self.now]
        if ticker_list is not None:
            placeholders = ", ".join(["?"] * len(ticker_list))
            univ_filters.append(f"ticker IN ({placeholders})")
            params.extend(ticker_list)

        tag_clause = ""
        if tag_list is not None:
            placeholders = ", ".join(["?"] * len(tag_list))
            tag_clause = f"AND f.tag IN ({placeholders})"

        branches: list[str] = []
        if fact_kind in (None, "instant"):
            branches.append(
                f"""
                SELECT
                    'instant' AS fact_kind,
                    f.cik,
                    f.taxonomy,
                    f.tag,
                    f.unit,
                    CAST(NULL AS DATE) AS start_date,
                    f.end_date,
                    f.val,
                    f.accn,
                    f.filed,
                    f.form,
                    f.fy,
                    f.fp,
                    f.frame
                FROM fundamental_instant AS f
                SEMI JOIN target_ciks AS c USING (cik)
                WHERE f.filed < CAST(? AS DATE)
                  AND f.form IN ('10-K', '10-Q', '10-K/A', '10-Q/A')
                  {tag_clause}
                """
            )
            params.append(self.now)
            if tag_list is not None:
                params.extend(tag_list)

        if fact_kind in (None, "period"):
            branches.append(
                f"""
                SELECT
                    'period' AS fact_kind,
                    f.cik,
                    f.taxonomy,
                    f.tag,
                    f.unit,
                    f.start_date,
                    f.end_date,
                    f.val,
                    f.accn,
                    f.filed,
                    f.form,
                    f.fy,
                    f.fp,
                    f.frame
                FROM fundamental_period AS f
                SEMI JOIN target_ciks AS c USING (cik)
                WHERE f.filed < CAST(? AS DATE)
                  AND f.form IN ('10-K', '10-Q', '10-K/A', '10-Q/A')
                  {tag_clause}
                """
            )
            params.append(self.now)
            if tag_list is not None:
                params.extend(tag_list)

        visible_sql = "\nUNION ALL\n".join(branches)
        params.append(n_reports)

        sql = f"""
        WITH pit_univ AS (
            SELECT ticker, cik
            FROM universe
            WHERE {" AND ".join(univ_filters)}
        ),
        target_ciks AS (
            SELECT DISTINCT cik FROM pit_univ
        ),
        visible_facts AS (
            {visible_sql}
        ),
        deduped AS (
            SELECT *
            FROM visible_facts
            QUALIFY row_number() OVER (
                PARTITION BY fact_kind, cik, taxonomy, tag, unit, start_date, end_date
                ORDER BY filed DESC, accn DESC
            ) = 1
        ),
        ranked AS (
            SELECT
                u.ticker,
                d.cik,
                d.fact_kind,
                d.taxonomy,
                d.tag,
                d.unit,
                d.start_date,
                d.end_date,
                d.val,
                d.accn,
                d.filed,
                d.form,
                d.fy,
                d.fp,
                d.frame,
                row_number() OVER (
                    PARTITION BY u.ticker, d.fact_kind, d.cik, d.taxonomy, d.tag, d.unit
                    ORDER BY d.end_date DESC, d.start_date DESC NULLS LAST, d.filed DESC, d.accn DESC
                ) AS report_rank
            FROM deduped AS d
            JOIN pit_univ AS u USING (cik)
        )
        SELECT
            ticker, cik, fact_kind, taxonomy, tag, unit,
            start_date, end_date, val, accn, filed, form, fy, fp, frame, report_rank
        FROM ranked
        WHERE report_rank <= ?
        ORDER BY ticker, fact_kind, tag, unit, report_rank
        """
        df = self._conn.execute(sql, params).df()
        return df.loc[:, list(FUNDAMENTAL_COLUMNS)]

    def _query_returns(
        self,
        *,
        ret_expr: str,
        out_col: str,
        extra_params: Sequence[Any],
        tickers: str | Sequence[str] | None,
        lookback: int | None,
        price_col: Literal["adj_close", "close"],
        universe_only: bool | None,
        capability: str,
    ) -> pd.DataFrame:
        if price_col not in _VALID_RETURN_PRICE_COLS:
            raise EventFinanceError(
                PITErrorCode.INVALID_ARGUMENT,
                f"price_col 只能是 'adj_close' 或 'close', 实际为 {price_col!r}",
                capability=capability,
            )
        if lookback is not None:
            _validate_positive_int(lookback, name="lookback", capability=capability)

        ticker_list = _normalize_str_filter(tickers, name="tickers", capability=capability)
        use_universe = _resolve_universe_only(ticker_list, universe_only, capability=capability)
        if ticker_list == ():
            return _empty_panel([out_col])

        out_start: date | None = None
        if lookback is not None:
            out_start = self._resolve_lookback_start(self.now, lookback)
            if out_start is None:
                return _empty_panel([out_col])

        scope_cte, scope_params = self._build_ticker_scope_cte(
            self.now, ticker_list, use_universe
        )
        price_expr = "p.close" if price_col == "close" else "p.adj_close"
        with_prefix = f"{scope_cte}," if scope_cte else ""
        join_sql = "JOIN target_tickers AS t USING (ticker)" if scope_cte else ""
        out_filter = "WHERE date >= CAST(? AS DATE)" if out_start is not None else ""

        params: list[Any] = [*scope_params, self.now, *extra_params]
        if out_start is not None:
            params.append(out_start)

        sql = f"""
        WITH {with_prefix}
        px AS (
            SELECT
                p.date,
                p.ticker,
                CASE WHEN {price_expr} > 0 THEN {price_expr} END AS price
            FROM prices AS p
            {join_sql}
            WHERE p.date <= CAST(? AS DATE)
        ),
        ret AS (
            SELECT
                date,
                ticker,
                {ret_expr} AS {out_col}
            FROM px
        )
        SELECT date, ticker, {out_col}
        FROM ret
        {out_filter}
        ORDER BY date, ticker
        """
        df = self._conn.execute(sql, params).df()
        return _to_panel(df, [out_col])

    def historical_returns(
        self,
        n_periods: int = 5,
        *,
        tickers: str | Sequence[str] | None = None,
        lookback: int | None = None,
        price_col: Literal["adj_close", "close"] = "adj_close",
        universe_only: bool | None = None,
    ) -> pd.DataFrame:
        """查询截至 now 的 PIT 历史简单收益率面板"""
        cap = "PITDataView.historical_returns"
        _validate_positive_int(n_periods, name="n_periods", capability=cap)
        return self._query_returns(
            ret_expr="price / lag(price, CAST(? AS INTEGER)) OVER (PARTITION BY ticker ORDER BY date) - 1",
            out_col="historical_return",
            extra_params=[n_periods],
            tickers=tickers,
            lookback=lookback,
            price_col=price_col,
            universe_only=universe_only,
            capability=cap,
        )

    def log_returns(
        self,
        *,
        tickers: str | Sequence[str] | None = None,
        lookback: int | None = None,
        price_col: Literal["adj_close", "close"] = "adj_close",
        universe_only: bool | None = None,
    ) -> pd.DataFrame:
        """查询截至 now 的 PIT 相邻交易日对数收益率面板"""
        return self._query_returns(
            ret_expr="ln(price) - ln(lag(price) OVER (PARTITION BY ticker ORDER BY date))",
            out_col="log_return",
            extra_params=(),
            tickers=tickers,
            lookback=lookback,
            price_col=price_col,
            universe_only=universe_only,
            capability="PITDataView.log_returns",
        )
