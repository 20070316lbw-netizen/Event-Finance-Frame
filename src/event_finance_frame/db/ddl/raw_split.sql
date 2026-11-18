-- 来源: data/stocks_csv/split.csv, 导入 data/raw_data.duckdb
-- 原始落地层: 不加主键与检查约束, 保证供应商文件可以原样重灌
CREATE TABLE IF NOT EXISTS raw_split (
    act_symbol VARCHAR,             -- 标的代码
    ex_date DATE,                   -- 拆股生效日
    to_factor DOUBLE,               -- 拆股后比例
    for_factor DOUBLE               -- 拆股前比例
);
