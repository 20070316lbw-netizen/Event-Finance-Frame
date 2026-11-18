-- 用 DuckDB 原生 read_csv 从源 CSV 批量写入 raw_symbol, 配合 operation_sql 使用 (无需 df)
-- 列名与列顺序和 ddl/raw_symbol.sql 一致
-- CSV 路径相对于仓库根目录, 从其他工作目录执行时改这里的前缀即可
INSERT INTO raw_symbol (
    act_symbol, security_name, listing_exchange, market_category,
    is_etf, round_lot_size, is_test_issue, financial_status,
    cqs_symbol, nasdaq_symbol, is_next_shares, last_seen
)
SELECT
    act_symbol, security_name, listing_exchange, market_category,
    is_etf, round_lot_size, is_test_issue, financial_status,
    cqs_symbol, nasdaq_symbol, is_next_shares, last_seen
FROM read_csv(
    'data/stocks_csv/symbol.csv',
    header = true,
    columns = {
        'act_symbol': 'VARCHAR',
        'security_name': 'VARCHAR',
        'listing_exchange': 'VARCHAR',
        'market_category': 'VARCHAR',
        'is_etf': 'BIGINT',
        'round_lot_size': 'BIGINT',
        'is_test_issue': 'BIGINT',
        'financial_status': 'VARCHAR',
        'cqs_symbol': 'VARCHAR',
        'nasdaq_symbol': 'VARCHAR',
        'is_next_shares': 'BIGINT',
        'last_seen': 'DATE'
    }
);
