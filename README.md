# Event-Finance-Frame

[![CI](https://github.com/20070316lbw-netizen/Event-Finance-Frame/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/Event-Finance-Frame/actions/workflows/ci.yml)
![Last Commit](https://img.shields.io/github/last-commit/20070316lbw-netizen/Event-Finance-Frame)

面向美股标普 500 历史成分的 **Point-in-Time（时点隔离）日频数据与回测引擎框架**，基于 DuckDB 构建本地分析库，围绕“时间推进”组织行情、成分快照、SEC 财报与交易日历。

完整架构与设计约定见 [Design.md](Design.md)。

## 核心特性

- **严格时点（PIT）隔离**：
  - 行情在交易日 $T$ 收盘时可见 `date <= T`；
  - SEC 财报按申报日 `filed < T` 过滤（次日生效），同一报告期被修订时自动取最新可见版本；
  - 成分股采用 `snapshot_date <= T` 的最新标普 500 历史快照，尽量避免幸存者偏差。
- **单 SQL 一体化 Dolt CSV 管道**：直接读取 Dolt 导出的 `data/stocks_csv/*.csv` 写入 `lab.duckdb`（无需中间库），一次完成坏价过滤、全市场零成交休市日剔除、代码复用切段（`{symbol}~{YYYYMMDD}`）、幽灵重复拆股校验与全收益复权。
- **DuckDB PIT 表宏**：内置 `pit_fundamentals(as_of_date, n_reports := 5)` 与 `pit_fundamentals_by_ticker(target_ticker, as_of_date, n_reports := 5)`，毫秒级获取任意截止日可见的最近 $N$ 期常规财报（`10-K`/`10-Q` 及其修订版）。
- **市场日历与联结视图**：生成连续自然日表 `market_calendar`（`date`, `weekday`, `is_open`）并与 `prices` 联结为 `prices_calendar` 视图。
- **交易日历与单调时钟（M2）**：`TradingCalendar` 与 `Clock` 支持从 DuckDB 构建交易日序列、按交易日步进 `step()`、按策略跳转 `seek()`（默认禁止时间倒退）以及区间迭代。
- **PIT 时点隔离视图（M3）**：`PITDataView` 绑定到 `Clock`，统一暴露截至 `clock.now` 的成分快照、行情面板、最新截面行情、PIT 基本面（含修订去重）与历史收益率，杜绝未来数据与标签宏泄漏。

## 数据库核心对象（`data/lab.duckdb`）

| 类别 | 名称 | 说明 |
|---|---|---|
| 表 | `prices` | 切段并全收益复权后的日频 OHLCV 行情 |
| 表 | `ticker_segment` | 代码复用切段映射（超过 120 交易日空档自动切段） |
| 表 | `market_calendar` | 连续自然日历（日期、星期几、是否开盘） |
| 表 | `universe` | 标普 500 历史成分快照与 CIK/行业元数据 |
| 表 | `fundamental_instant` | SEC Company Facts 时点类财务事实（资产负债表等） |
| 表 | `fundamental_period` | SEC Company Facts 期间类财务事实（利润表/现金流量表等） |
| 视图 | `calendar` | 全部有效开盘交易日 |
| 视图 | `prices_calendar` | `market_calendar` 与 `prices` 联结视图 |
| 表宏 | `pit_fundamentals` / `pit_fundamentals_by_ticker` | 全体 / 单标的 PIT 财报查询宏 |
| 表宏 | `pit_historical_return` / `pit_log_return` | 截至 `as_of_date` 的历史简单收益率 / 对数收益率（可作特征） |
| 表宏 | `label_forward_return` | 未来持有期收益率，**仅用作机器学习标签，回测引擎禁止调用** |

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 构建本地数据库

将 Dolt 导出的 4 份 CSV 放入 `data/stocks_csv/`（`ohlcv.csv`, `dividend.csv`, `split.csv`, `symbol.csv`），SEC 归档放入 `data/companyfacts.zip`，然后运行：

```bash
uv run python -m event_finance_frame.pipeline
```

### 3. 运行检查与测试

```bash
uv run ruff check .
uv run pytest -v
```

## 里程碑进度

- [x] **M0**：DDL 规范、数据源字段与项目基础框架
- [x] **M1**：全量数据管道（Dolt CSV 行情 → 市场日历 → 标普 500 历史成分 → SEC 基本面与 PIT 表宏）
- [x] **M2**：Calendar 与 Clock 时间推进器
- [x] **M3**：PIT DataView（防未来数据隔离层）
- [ ] **M4**：Broker 与 Portfolio（收盘价撮合、手续费、滑点、持仓记账）
- [ ] **M5**：示例策略、绩效报告与银行业扩展数据源
