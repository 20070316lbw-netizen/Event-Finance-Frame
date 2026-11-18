-- 用 DuckDB 原生 read_csv 从源 CSV 批量写入 raw_dividend, 配合 operation_sql 使用 (无需 df)
-- 列名与列顺序和 ddl/raw_dividend.sql 一致
-- CSV 路径相对于仓库根目录, 从其他工作目录执行时改这里的前缀即可
INSERT INTO raw_dividend (act_symbol, ex_date, amount)
SELECT act_symbol, ex_date, amount
FROM read_csv(
    'data/stocks_csv/dividend.csv',
    header = true,
    columns = {
        'act_symbol': 'VARCHAR',
        'ex_date': 'DATE',
        'amount': 'DOUBLE'
    }
);
