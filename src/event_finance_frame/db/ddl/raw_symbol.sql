-- 来源: data/stocks_csv/symbol.csv, 导入 data/raw_data.duckdb
-- 原始落地层: 不加主键与检查约束, 保证供应商文件可以原样重灌
CREATE TABLE IF NOT EXISTS raw_symbol (
    act_symbol VARCHAR,             -- 标的代码
    security_name VARCHAR,          -- 证券全称
    listing_exchange VARCHAR,       -- 上市交易所
    market_category VARCHAR,        -- 市场分类
    is_etf BIGINT,                  -- 是否 ETF, 0/1
    round_lot_size BIGINT,          -- 整手股数
    is_test_issue BIGINT,           -- 是否测试代码, 0/1
    financial_status VARCHAR,       -- 财务状态
    cqs_symbol VARCHAR,             -- CQS 代码
    nasdaq_symbol VARCHAR,          -- NASDAQ 代码
    is_next_shares BIGINT,          -- 是否 NextShares, 0/1
    last_seen DATE                  -- 最后出现日期
);
