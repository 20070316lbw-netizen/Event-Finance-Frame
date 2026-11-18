-- 依据行情空档切段, 生成 ticker_segment
-- 前置条件 (由 sources/raw_prices.py 准备):
--   1. raw_data.duckdb 已只读 ATTACH 为 raw
--   2. 已建临时视图 clean_ohlcv (剔除 close <= 0 的行)
--   3. 已 SET VARIABLE gap_days = <阈值>
-- 交易日序号用全市场日历计算, 间隔 = 前后两条行情的交易日序号之差
DELETE FROM ticker_segment;

INSERT INTO ticker_segment (symbol, ticker, seg_no, start_date, end_date, n_days, gap_before)
WITH cal AS (
    SELECT date, row_number() OVER (ORDER BY date) AS di
    FROM (SELECT DISTINCT date FROM clean_ohlcv)
),
gaps AS (
    SELECT
        o.act_symbol,
        o.date,
        c.di - lag(c.di) OVER (PARTITION BY o.act_symbol ORDER BY o.date) AS gap
    FROM clean_ohlcv AS o
    JOIN cal AS c USING (date)
),
flagged AS (
    SELECT
        act_symbol,
        date,
        CASE WHEN gap > getvariable('gap_days') THEN gap END AS new_seg_gap,
        sum(CASE WHEN gap > getvariable('gap_days') THEN 1 ELSE 0 END) OVER (
            PARTITION BY act_symbol ORDER BY date ROWS UNBOUNDED PRECEDING
        ) AS seg_no
    FROM gaps
),
seg AS (
    SELECT
        act_symbol AS symbol,
        seg_no,
        min(date) AS start_date,
        max(date) AS end_date,
        count(*) AS n_days,
        max(new_seg_gap) AS gap_before
    FROM flagged
    GROUP BY act_symbol, seg_no
)
SELECT
    symbol,
    CASE
        WHEN seg_no = max(seg_no) OVER (PARTITION BY symbol) THEN symbol
        ELSE symbol || '~' || strftime(end_date, '%Y%m%d')
    END AS ticker,
    seg_no,
    start_date,
    end_date,
    n_days,
    gap_before
FROM seg;
