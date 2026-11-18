-- 来源: data/stocks_csv/ohlcv.csv, 导入 data/raw_data.duckdb
-- 原始落地层: 不加主键与检查约束, 保证供应商文件可以原样重灌
-- 空值情况: 仅 volume 有 5 行为空, 其余列无空值
CREATE TABLE IF NOT EXISTS raw_ohlcv (
    date DATE,                      -- 交易日
    act_symbol VARCHAR,             -- 标的代码, 对应 raw_symbol.act_symbol
    open DOUBLE,                    -- 开盘价
    high DOUBLE,                    -- 最高价
    low DOUBLE,                     -- 最低价
    close DOUBLE,                   -- 收盘价
    volume BIGINT                   -- 成交量
);
