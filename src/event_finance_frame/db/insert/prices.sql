-- 全量行情写入 prices: 按 ticker_segment 改名, 并计算全收益复权价 adj_close
-- 前置条件 (由 sources/raw_prices.py 准备):
--   1. raw_data.duckdb 已只读 ATTACH 为 raw
--   2. 已建临时视图 clean_ohlcv (剔除 close <= 0 的行)
--   3. ticker_segment 已生成
--
-- 复权方法 (向后复权, 最新价格 = 原始价格):
--   拆股: ex 日之前的价格乘以 for_factor / to_factor
--   分红: ex 日之前的价格乘以 1 - amount / 前一交易日原始收盘价
--   adj_close(t) = close(t) * 所有晚于 t 的事件因子之积
-- 事件日期不是交易日时, 挪到该段内 >= ex_date 的第一个交易日; 落在段外 (空档期) 的事件丢弃
-- 丢弃无效事件: 拆股比例 <= 0, 分红金额 <= 0, 分红金额 >= 前收盘价
DELETE FROM prices;

INSERT INTO prices (date, ticker, open, high, low, close, adj_close, volume)
WITH px AS (
    SELECT
        o.date,
        s.ticker,
        s.symbol,
        s.start_date,
        s.end_date,
        o.open,
        o.high,
        o.low,
        o.close,
        o.volume,
        lag(o.close) OVER (PARTITION BY s.ticker ORDER BY o.date) AS prev_close
    FROM clean_ohlcv AS o
    JOIN ticker_segment AS s
        ON o.act_symbol = s.symbol
        AND o.date BETWEEN s.start_date AND s.end_date
),
raw_events AS (
    SELECT act_symbol, ex_date, 'split' AS kind, for_factor / to_factor AS value
    FROM raw.raw_split
    WHERE to_factor > 0 AND for_factor > 0
    UNION ALL
    SELECT act_symbol, ex_date, 'dividend' AS kind, amount AS value
    FROM raw.raw_dividend
    WHERE amount > 0
),
mapped AS (
    -- ASOF: 找同一代码 >= ex_date 的第一条行情
    SELECT e.kind, e.value, e.ex_date, p.ticker, p.date, p.prev_close, p.start_date
    FROM raw_events AS e
    ASOF JOIN px AS p
        ON e.act_symbol = p.symbol
        AND p.date >= e.ex_date
),
factors AS (
    SELECT
        ticker,
        date,
        CASE kind
            WHEN 'split' THEN value
            ELSE 1 - value / prev_close
        END AS f
    FROM mapped
    WHERE ex_date >= start_date                          -- 事件必须落在该段内
        AND prev_close IS NOT NULL                       -- 段首日没有可复权的历史
        AND (kind = 'split' OR value < prev_close)
),
ev AS (
    SELECT ticker, date, sum(ln(f)) AS log_f
    FROM factors
    GROUP BY ticker, date
)
SELECT
    p.date,
    p.ticker,
    p.open,
    p.high,
    p.low,
    p.close,
    p.close * exp(coalesce(sum(e.log_f) OVER (
        PARTITION BY p.ticker ORDER BY p.date DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0)) AS adj_close,
    p.volume
FROM px AS p
LEFT JOIN ev AS e USING (ticker, date);
