"""回测调仓事件日志采集与纯文本 (.txt) 报告生成器"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from event_finance_frame.broker import Fill, OrderSide
from event_finance_frame.pit import PITDataView
from event_finance_frame.portfolio import PortfolioSnapshot

if TYPE_CHECKING:
    from event_finance_frame.engine import BacktestResult
    from event_finance_frame.metrics import PerformanceMetrics

__all__ = [
    "BacktestJournal",
    "PendingDecision",
    "RebalanceEvent",
    "TickerContext",
    "format_backtest_log",
    "write_backtest_log",
]


@dataclass(frozen=True)
class TickerContext:
    """调仓时刻单只标的的 PIT 信息快照 (元数据 + 当日行情 + 因子值/排名)"""

    ticker: str
    name: str | None = None
    sub_industry: str | None = None
    exchange: str | None = None
    cik: str | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = None
    volume: int | None = None
    factor_values: dict[str, float] = field(default_factory=dict)
    composite_score: float | None = None
    rank: int | None = None


@dataclass(frozen=True)
class PendingDecision:
    """策略在调仓日暂存给日志器的决策上下文"""

    date: date
    target_weights: dict[str, float] = field(default_factory=dict)
    selected_tickers: tuple[str, ...] = ()
    factor_values: dict[str, dict[str, float]] = field(default_factory=dict)
    composite_scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    reason: str = "rebalance"


@dataclass(frozen=True)
class RebalanceEvent:
    """单次调仓事件全貌"""

    date: date
    reason: str
    cash_before: float
    equity_before: float
    positions_before: dict[str, float]
    positions_after: dict[str, float]
    target_weights: dict[str, float]
    selected_tickers: tuple[str, ...]
    fills: tuple[Fill, ...]
    ticker_contexts: dict[str, TickerContext]
    snapshot_after: PortfolioSnapshot


def _opt_float(val: Any) -> float | None:
    return float(val) if pd.notna(val) else None


def _opt_int(val: Any) -> int | None:
    return int(val) if pd.notna(val) else None


def _opt_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return None
    text = str(val).strip()
    return text or None


class BacktestJournal:
    """记录回测全过程每次调仓时间、买卖明细及操作时标的 PIT 信息"""

    def __init__(self) -> None:
        self._events: list[RebalanceEvent] = []
        self._pending: PendingDecision | None = None

    @property
    def events(self) -> tuple[RebalanceEvent, ...]:
        return tuple(self._events)

    def set_decision(self, decision: PendingDecision) -> None:
        """由策略在调仓日调用, 上报当期目标权重、选中标的与因子快照"""
        self._pending = decision

    def record_bar(
        self,
        *,
        pit: PITDataView,
        fills: Sequence[Fill],
        cash_before: float,
        equity_before: float,
        positions_before: Mapping[str, float],
        positions_after: Mapping[str, float],
        snapshot_after: PortfolioSnapshot,
    ) -> RebalanceEvent | None:
        """在每个交易日收盘完成撮合与盯市后记录调仓事件 (无成交且无决策时跳过)"""
        pending = self._pending
        self._pending = None
        has_pending = pending is not None and pending.date == pit.now

        if not fills and not has_pending:
            return None

        target_weights = dict(pending.target_weights) if has_pending and pending else {}
        selected = tuple(pending.selected_tickers) if has_pending and pending else ()
        reason = pending.reason if has_pending and pending else "trade"

        involved = sorted(
            {f.ticker for f in fills}
            | set(target_weights.keys())
            | set(positions_before.keys())
            | set(positions_after.keys())
        )
        contexts: dict[str, TickerContext] = {}
        if involved:
            latest_px = pit.latest_prices(tickers=involved, universe_only=False)
            univ_df = pit.universe()
            univ_map = (
                univ_df.set_index("ticker").to_dict("index")
                if not univ_df.empty
                else {}
            )

            for ticker in involved:
                px_row = latest_px.loc[ticker] if ticker in latest_px.index else None
                u_row = univ_map.get(ticker)
                f_vals = (
                    dict(pending.factor_values.get(ticker, {}))
                    if has_pending and pending
                    else {}
                )
                comp_score = (
                    pending.composite_scores.get(ticker)
                    if has_pending and pending
                    else None
                )
                rank = pending.ranks.get(ticker) if has_pending and pending else None

                contexts[ticker] = TickerContext(
                    ticker=ticker,
                    name=_opt_str(u_row.get("name")) if u_row else None,
                    sub_industry=_opt_str(u_row.get("sub_industry")) if u_row else None,
                    exchange=_opt_str(u_row.get("exchange")) if u_row else None,
                    cik=_opt_str(u_row.get("cik")) if u_row else None,
                    open=_opt_float(px_row["open"]) if px_row is not None else None,
                    high=_opt_float(px_row["high"]) if px_row is not None else None,
                    low=_opt_float(px_row["low"]) if px_row is not None else None,
                    close=_opt_float(px_row["close"]) if px_row is not None else None,
                    adj_close=_opt_float(px_row["adj_close"]) if px_row is not None else None,
                    volume=_opt_int(px_row["volume"]) if px_row is not None else None,
                    factor_values=f_vals,
                    composite_score=comp_score,
                    rank=rank,
                )

        event = RebalanceEvent(
            date=pit.now,
            reason=reason,
            cash_before=float(cash_before),
            equity_before=float(equity_before),
            positions_before=dict(positions_before),
            positions_after=dict(positions_after),
            target_weights=target_weights,
            selected_tickers=selected,
            fills=tuple(fills),
            ticker_contexts=contexts,
            snapshot_after=snapshot_after,
        )
        self._events.append(event)
        return event


def _fmt_price(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "N/A"


def _fmt_vol(val: int | None) -> str:
    return f"{val:,}" if val is not None else "N/A"


def _format_config_section(config: Any) -> list[str]:
    if config is None:
        return []
    lines = ["[回测配置]"]
    data = getattr(config, "data", None)
    strat = getattr(config, "strategy", None)
    broker = getattr(config, "broker", None)
    port = getattr(config, "portfolio", None)

    if data is not None:
        lines.append(
            f"  数据源: {data.db_path} | 区间: {data.start_date or '最早'} -> {data.end_date or '最新'} | 价格口径: {data.price_col}"
        )
    if strat is not None:
        lines.append(
            f"  策略: {strat.name} | 调仓频率: {strat.rebalance_freq} | top_k={strat.top_k} | ascending={strat.ascending} | min_trade_value={strat.min_trade_value:.2f}"
        )
        bundle = getattr(strat, "factor_bundle", None)
        if bundle is not None:
            factor_cfg_ref = getattr(strat, "factor_config", None) or "(inline)"
            lines.append(
                f"  因子组合: {bundle.name} (配置: {factor_cfg_ref}, 合成方式: {bundle.combine})"
            )
            for fc in bundle.factors:
                lines.append(
                    f"    - {fc.output_name} (name={fc.name}, weight={fc.weight}, params={fc.params})"
                )
    if broker is not None:
        lines.append(
            f"  成本参数: commission_rate={broker.commission_rate}, min_commission={broker.min_commission}, per_share={broker.per_share_commission}, slippage_bps={broker.slippage_bps}"
        )
    if port is not None:
        lines.append(f"  初始资金: {port.initial_cash:,.2f}")
    lines.append("")
    return lines


def format_backtest_log(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    *,
    config: Any = None,
) -> str:
    """将完整回测过程、绩效指标与调仓明细格式化为可读纯文本"""
    sep = "=" * 88
    sub_sep = "-" * 88
    lines: list[str] = [
        sep,
        "Event-Finance-Frame 回测调仓日志报告",
        sep,
        "",
        *_format_config_section(config),
        "[绩效摘要]",
        f"  回测区间: {metrics.start_date} -> {metrics.end_date} ({metrics.n_days} 个交易日)",
        f"  初始资金: {metrics.initial_cash:,.2f} -> 期末权益: {metrics.final_equity:,.2f}",
        f"  累计收益: {metrics.total_return:+.2%} | 年化收益 (CAGR): {metrics.cagr:+.2%}",
        f"  年化波动: {metrics.annualized_volatility:.2%} | Sharpe: {metrics.sharpe_ratio:.2f} | 最大回撤: {metrics.max_drawdown:.2%}",
        f"  调仓次数: {len(result.events)} | 成交笔数: {metrics.n_trades} | 年化换手率: {metrics.annualized_turnover:.2%}",
        f"  交易成本: 手续费 {metrics.total_commission:,.2f} + 滑点 {metrics.total_slippage:,.2f}",
        "",
        sep,
        f"[调仓明细] (共 {len(result.events)} 次调仓事件)",
        sep,
    ]

    if not result.events:
        lines.append("  (回测期间无调仓成交记录)")
        return "\n".join(lines) + "\n"

    for idx, ev in enumerate(result.events, start=1):
        buys = [f for f in ev.fills if f.side is OrderSide.BUY]
        sells = [f for f in ev.fills if f.side is OrderSide.SELL]
        snap = ev.snapshot_after
        lines.extend(
            [
                "",
                f"--- #{idx:03d} 调仓日: {ev.date} (reason={ev.reason}) ---",
                f"  调仓前: 现金 {ev.cash_before:,.2f} | 估算权益 {ev.equity_before:,.2f} | 持仓数 {len(ev.positions_before)}",
                f"  调仓后: 现金 {snap.cash:,.2f} | 持仓市值 {snap.market_value:,.2f} | 权益 {snap.equity:,.2f} | 持仓数 {snap.n_positions}",
                f"  成交通计: 卖出 {len(sells)} 笔, 买入 {len(buys)} 笔",
            ]
        )
        if ev.selected_tickers:
            preview = ", ".join(ev.selected_tickers[:15])
            suffix = f" ... (共 {len(ev.selected_tickers)} 只)" if len(ev.selected_tickers) > 15 else ""
            lines.append(f"  当期入选标的: {preview}{suffix}")

        if not ev.fills:
            lines.append("  (当日无实际成交订单)")
            lines.append(sub_sep)
            continue

        for fill in ev.fills:
            ctx = ev.ticker_contexts.get(fill.ticker)
            qty_before = ev.positions_before.get(fill.ticker, 0.0)
            qty_after = ev.positions_after.get(fill.ticker, 0.0)
            target_w = ev.target_weights.get(fill.ticker, 0.0)

            meta_parts: list[str] = []
            if ctx is not None:
                if ctx.name:
                    meta_parts.append(ctx.name)
                if ctx.sub_industry:
                    meta_parts.append(ctx.sub_industry)
                if ctx.exchange:
                    meta_parts.append(ctx.exchange)
                if ctx.cik:
                    meta_parts.append(f"CIK:{ctx.cik}")
            meta_str = " | ".join(meta_parts) if meta_parts else "无成分元数据(已调出或库中缺失)"

            lines.append(
                f"  [{fill.side:<4}] {fill.ticker:<8} 股数={fill.quantity:>10.4f} "
                f"(持仓 {qty_before:.4f} -> {qty_after:.4f}) | 目标权重={target_w:.2%}"
            )
            lines.append(
                f"         参考价={fill.market_price:.4f} -> 成交价={fill.fill_price:.4f} | "
                f"金额={fill.gross_amount:,.2f} | 手续费={fill.commission:.2f} | 滑点={fill.slippage_cost:.2f} | 现金流={fill.net_cash_flow:+,.2f}"
            )
            lines.append(f"         标的信息: {meta_str}")

            if ctx is not None:
                lines.append(
                    f"         当日行情: O={_fmt_price(ctx.open)} H={_fmt_price(ctx.high)} "
                    f"L={_fmt_price(ctx.low)} C={_fmt_price(ctx.close)} AdjC={_fmt_price(ctx.adj_close)} Vol={_fmt_vol(ctx.volume)}"
                )
                if ctx.factor_values or ctx.composite_score is not None or ctx.rank is not None:
                    factor_parts = [f"{k}={v:+.6f}" for k, v in ctx.factor_values.items()]
                    if ctx.composite_score is not None:
                        factor_parts.append(f"综合分={ctx.composite_score:+.6f}")
                    if ctx.rank is not None:
                        factor_parts.append(f"横截面排名=#{ctx.rank}")
                    lines.append(f"         因子状态: {' | '.join(factor_parts)}")

        lines.append(sub_sep)

    return "\n".join(lines) + "\n"


def write_backtest_log(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    path: str | Path,
    *,
    config: Any = None,
) -> Path:
    """生成回测调仓日志并写入 UTF-8 .txt 文件"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = format_backtest_log(result, metrics, config=config)
    out.write_text(text, encoding="utf-8")
    return out
