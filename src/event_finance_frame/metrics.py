"""回测绩效统计与评估指标"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from event_finance_frame.engine import BacktestResult

__all__ = [
    "PerformanceMetrics",
    "evaluate_backtest",
]


@dataclass(frozen=True)
class PerformanceMetrics:
    """组合回测核心绩效汇总"""

    start_date: date
    end_date: date
    n_days: int
    initial_cash: float
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    annualized_turnover: float
    total_commission: float
    total_slippage: float


def evaluate_backtest(
    result: BacktestResult,
    *,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """根据 BacktestResult 计算年化收益、波动率、夏普比率、最大回撤与换手率"""
    if isinstance(trading_days_per_year, bool) or not isinstance(trading_days_per_year, int) or trading_days_per_year < 1:
        raise ValueError(f"trading_days_per_year 必须是 >= 1 的整数, 实际为 {trading_days_per_year!r}")
    if isinstance(risk_free_rate, bool) or not isinstance(risk_free_rate, (int, float)) or not math.isfinite(risk_free_rate):
        raise ValueError(f"risk_free_rate 必须是有限实数, 实际为 {risk_free_rate!r}")

    curve = result.equity_curve
    fills = result.fills
    initial_cash = result.portfolio.initial_cash
    final_equity = result.final_equity
    total_return = result.total_return
    n_trades = len(fills)

    if curve.empty:
        return PerformanceMetrics(
            start_date=result.start_date,
            end_date=result.end_date,
            n_days=0,
            initial_cash=initial_cash,
            final_equity=final_equity,
            total_return=total_return,
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            n_trades=n_trades,
            annualized_turnover=0.0,
            total_commission=result.portfolio.total_commission,
            total_slippage=result.portfolio.total_slippage,
        )

    equity = curve["equity"].astype("float64")
    n_days = len(equity)
    daily_ret = equity.pct_change().dropna()

    if n_days > 1 and initial_cash > 0:
        if final_equity > 0:
            cagr = (final_equity / initial_cash) ** (trading_days_per_year / (n_days - 1)) - 1.0
        else:
            cagr = -1.0
    else:
        cagr = 0.0

    if len(daily_ret) >= 2:
        std_daily = float(daily_ret.std(ddof=1))
        ann_vol = std_daily * math.sqrt(trading_days_per_year)
        mean_excess_daily = float(daily_ret.mean()) - float(risk_free_rate) / trading_days_per_year
        sharpe = (mean_excess_daily * trading_days_per_year) / ann_vol if ann_vol > 1e-12 else 0.0
    else:
        ann_vol = 0.0
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    mean_equity = float(equity.mean())
    gross_traded = float(fills["gross_amount"].sum()) if not fills.empty else 0.0
    ann_turnover = (
        (gross_traded / mean_equity) * (trading_days_per_year / n_days) / 2.0
        if mean_equity > 0 and n_days > 0
        else 0.0
    )

    return PerformanceMetrics(
        start_date=result.start_date,
        end_date=result.end_date,
        n_days=n_days,
        initial_cash=initial_cash,
        final_equity=final_equity,
        total_return=total_return,
        cagr=float(cagr),
        annualized_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(max_drawdown),
        n_trades=n_trades,
        annualized_turnover=float(ann_turnover),
        total_commission=float(curve["total_commission"].iloc[-1]),
        total_slippage=float(curve["total_slippage"].iloc[-1]),
    )
