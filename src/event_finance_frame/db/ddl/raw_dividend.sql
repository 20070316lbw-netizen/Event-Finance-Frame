-- 来源: data/stocks_csv/dividend.csv, 导入 data/raw_data.duckdb
-- 原始落地层: 不加主键与检查约束, 保证供应商文件可以原样重灌
CREATE TABLE IF NOT EXISTS raw_dividend (
    act_symbol VARCHAR,             -- 标的代码
    ex_date DATE,                   -- 除息日
    amount DOUBLE                   -- 每股分红金额
);
