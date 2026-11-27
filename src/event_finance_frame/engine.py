"""日频事件驱动回测执行引擎: Clock -> PITDataView -> Strategy -> Broker -> Portfolio"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from event_finance_frame.broker import Broker, Fill, Order
from event_finance_frame.clock import Clock
from event_finance_frame.journal import BacktestJournal, RebalanceEvent
from event_finance_frame.pit import PITDataView
from event_finance_frame.portfolio import Portfolio

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Strategy",
]


class Strategy(Protocol):
    """回测策略协议: 在交易日收盘基于 PITDataView 与当前 Portfolio 生成委托单"""

    def on_bar(self, pit: PITDataView, portfolio: Portfolio) -> Sequence[Order]: ...


@dataclass(frozen=True)
class BacktestResult:
    """回测运行结果汇总"""

    start_date: date
    end_date: date
    n_days: int
    portfolio: Portfolio
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    final_equity: float
    total_return: float
    events: tuple[RebalanceEvent, ...] = ()


class BacktestEngine:
    """串行日频回测主循环引擎"""

    def __init__(
        self,
        *,
        clock: Clock,
        pit: PITDataView,
        broker: Broker,
        portfolio: Portfolio,
    ) -> None:
        if pit.clock is not clock:
            raise ValueError("pit.clock 必须与传入的 clock 为同一实例")
        self._clock = clock
        self._pit = pit
        self._broker = broker
        self._portfolio = portfolio

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def pit(self) -> PITDataView:
        return self._pit

    @property
    def broker(self) -> Broker:
        return self._broker

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    def run(
        self,
        strategy: Strategy | Callable[[PITDataView, Portfolio], Sequence[Order]],
        *,
        reset_clock: bool = False,
        journal: BacktestJournal | None = None,
    ) -> BacktestResult:
        """沿 Clock 逐日推进运行策略并记录每日净值 (可选采集调仓日志)"""
        invoke = getattr(strategy, "on_bar", strategy)
        if not callable(invoke):
            raise TypeError("strategy 必须实现 on_bar(pit, portfolio) 方法或本身为可调用对象")

        if journal is not None:
            attach = getattr(strategy, "attach_journal", None)
            if callable(attach):
                attach(journal)

        if reset_clock:
            self._clock.reset()

        start_dt = self._clock.now
        last_dt = start_dt
        n_days = 0

        for d in self._clock:
            last_dt = d
            n_days += 1

            if journal is not None:
                cash_before = self._portfolio.cash
                positions_before = {
                    t: pos.quantity for t, pos in self._portfolio.positions.items()
                }
                equity_before = (
                    self._portfolio.snapshots[-1].equity
                    if self._portfolio.snapshots
                    else self._portfolio.initial_cash
                )

            orders = invoke(self._pit, self._portfolio)
            day_fills: list[Fill] = []
            if orders:
                day_fills = self._broker.execute(
                    orders, pit=self._pit, portfolio=self._portfolio
                )
            snap = self._portfolio.mark_to_market_pit(
                self._pit,
                price_col=self._broker.price_col,
            )

            if journal is not None:
                positions_after = {
                    t: pos.quantity for t, pos in self._portfolio.positions.items()
                }
                journal.record_bar(
                    pit=self._pit,
                    fills=day_fills,
                    cash_before=cash_before,
                    equity_before=equity_before,
                    positions_before=positions_before,
                    positions_after=positions_after,
                    snapshot_after=snap,
                )

        curve = self._portfolio.equity_curve()
        final_equity = (
            float(curve["equity"].iloc[-1]) if not curve.empty else self._portfolio.cash
        )
        total_return = (
            final_equity / self._portfolio.initial_cash - 1.0
            if self._portfolio.initial_cash > 0
            else 0.0
        )
        return BacktestResult(
            start_date=start_dt,
            end_date=last_dt,
            n_days=n_days,
            portfolio=self._portfolio,
            equity_curve=curve,
            fills=self._portfolio.fills_frame(),
            final_equity=final_equity,
            total_return=total_return,
            events=journal.events if journal is not None else (),
        )
