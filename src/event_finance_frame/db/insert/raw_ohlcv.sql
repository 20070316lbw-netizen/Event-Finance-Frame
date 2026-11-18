-- 用 DuckDB 原生 read_csv 从源 CSV 批量写入 raw_ohlcv, 配合 operation_sql 使用 (无需 df)
-- 列名与列顺序和 ddl/raw_ohlcv.sql 一致
-- CSV 路径相对于仓库根目录, 从其他工作目录执行时改这里的前缀即可
INSERT INTO raw_ohlcv (date, act_symbol, open, high, low, close, volume)
SELECT date, act_symbol, open, high, low, close, volume
FROM read_csv(
    'data/stocks_csv/ohlcv.csv',
    header = true,
    columns = {
        'date': 'DATE',
        'act_symbol': 'VARCHAR',
        'open': 'DOUBLE',
        'high': 'DOUBLE',
        'low': 'DOUBLE',
        'close': 'DOUBLE',
        'volume': 'BIGINT'
    }
);
