"""组合权重构建与目标权重调仓器"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from event_finance_frame.broker import Broker, Fill, Order
from event_finance_frame.error import BrokerErrorCode, EventFinanceError
from event_finance_frame.pit import PITDataView
from event_finance_frame.portfolio import Portfolio

__all__ = [
    "equal_weight_targets",
    "rebalance_to_weights",
    "select_top_k",
]

_EPS = 1e-9


def _is_finite_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def equal_weight_targets(
    tickers: Sequence[str],
    *,
    max_weight: float = 1.0,
) -> dict[str, float]:
    """为给定标的代码列表生成等权目标权重字典"""
    cap = "equal_weight_targets"
    if not _is_finite_real(max_weight) or max_weight <= 0 or max_weight > 1.0:
        raise EventFinanceError(
            BrokerErrorCode.INVALID_CONFIG,
            f"max_weight 必须在 (0, 1] 之间, 实际为 {max_weight!r}",
            capability=cap,
        )

    unique = list(dict.fromkeys(str(t) for t in tickers if isinstance(t, str) and t.strip()))
    if not unique:
        return {}
    weight = min(float(max_weight), 1.0 / len(unique))
    return {ticker: weight for ticker in unique}


def select_top_k(
    scores: pd.Series,
    k: int,
    *,
    ascending: bool = False,
) -> list[str]:
    """从单日横截面因子分数 Series 中选取前 k 只有效标的 (按 ticker 升序打破平局)"""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise EventFinanceError(
            BrokerErrorCode.INVALID_CONFIG,
            f"k 必须是 >= 1 的整数, 实际为 {k!r}",
            capability="select_top_k",
        )
    if scores.empty:
        return []

    clean = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []

    if isinstance(clean.index, pd.MultiIndex) and "ticker" in clean.index.names:
        tickers = clean.index.get_level_values("ticker")
    else:
        tickers = clean.index

    pairs = [(float(val), str(t)) for t, val in zip(tickers, clean.to_numpy(), strict=True)]
    if ascending:
        pairs.sort(key=lambda item: (item[0], item[1]))
    else:
        pairs.sort(key=lambda item: (-item[0], item[1]))
    return [ticker for _, ticker in pairs[:k]]


def _build_buy_orders(
    buy_candidates: list[tuple[str, float, float]],
    *,
    projected_cash: float,
    broker: Broker,
    portfolio: Portfolio,
    execution_date: Any,
    allow_fractional: bool,
) -> list[Order]:
    if not buy_candidates or projected_cash <= _EPS:
        return []

    can_margin = broker.allow_margin and portfolio.allow_margin
    quantities = [qty for _, qty, _ in buy_candidates]

    def _estimate(qtys: list[float]) -> tuple[list[Order], list[Fill]]:
        orders: list[Order] = []
        fills: list[Fill] = []
        for (ticker, _, px), q in zip(buy_candidates, qtys, strict=True):
            if q <= _EPS:
                continue
            order = Order.buy(ticker, q)
            orders.append(order)
            fills.append(
                broker.estimate_fill(
                    order,
                    execution_date=execution_date,
                    market_price=px,
                )
            )
        return orders, fills

    orders, fills = _estimate(quantities)
    if can_margin:
        return orders

    for _ in range(8):
        if not fills:
            return []
        total_outflow = sum(-f.net_cash_flow for f in fills)
        if total_outflow <= projected_cash + _EPS:
            return orders

        total_comm = sum(f.commission for f in fills)
        available_gross = projected_cash - total_comm
        if available_gross <= _EPS:
            return []
        total_gross = sum(f.gross_amount for f in fills)
        if total_gross <= _EPS:
            return []

        ratio = min(1.0, (available_gross / total_gross) * (1.0 - 1e-6))
        if allow_fractional:
            quantities = [q * ratio for q in quantities]
        else:
            quantities = [float(math.floor(q * ratio)) for q in quantities]
        orders, fills = _estimate(quantities)

    # 整数股离散舍入可能仍有微小超支时, 从名义金额最大的买单逐股扣减
    if not allow_fractional:
        while fills and sum(-f.net_cash_flow for f in fills) > projected_cash + _EPS:
            max_idx = max(range(len(quantities)), key=lambda i: quantities[i] * buy_candidates[i][2])
            if quantities[max_idx] < 1.0:
                break
            quantities[max_idx] -= 1.0
            orders, fills = _estimate(quantities)

    return orders if sum(-f.net_cash_flow for f in fills) <= projected_cash + _EPS else []


def rebalance_to_weights(
    target_weights: Mapping[str, float],
    *,
    pit: PITDataView,
    portfolio: Portfolio,
    broker: Broker,
    min_trade_value: float = 0.0,
    allow_fractional: bool = True,
    cash_buffer: float = 0.0,
) -> list[Order]:
    """将当前组合调仓至目标权重, 自动生成先卖后买的 Order 列表 (含费用与现金保护)"""
    cap = "rebalance_to_weights"
    if not _is_finite_real(min_trade_value) or min_trade_value < 0:
        raise EventFinanceError(
            BrokerErrorCode.INVALID_CONFIG,
            f"min_trade_value 必须是有限非负数, 实际为 {min_trade_value!r}",
            capability=cap,
        )
    if not _is_finite_real(cash_buffer) or cash_buffer < 0 or cash_buffer >= 1.0:
        raise EventFinanceError(
            BrokerErrorCode.INVALID_CONFIG,
            f"cash_buffer 必须在 [0, 1) 之间, 实际为 {cash_buffer!r}",
            capability=cap,
        )

    clean_targets: dict[str, float] = {}
    for ticker, w in target_weights.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise EventFinanceError(
                BrokerErrorCode.INVALID_CONFIG,
                f"target_weights 包含非法 ticker: {ticker!r}",
                capability=cap,
            )
        if not _is_finite_real(w) or w < 0:
            raise EventFinanceError(
                BrokerErrorCode.INVALID_CONFIG,
                f"标的 {ticker} 的目标权重必须为有限非负数, 实际为 {w!r}",
                capability=cap,
            )
        if w > 0:
            clean_targets[ticker] = float(w)

    total_weight = sum(clean_targets.values())
    if total_weight > 1.0 + _EPS:
        raise EventFinanceError(
            BrokerErrorCode.INVALID_CONFIG,
            f"目标权重之和不能超过 1.0, 实际为 {total_weight:.6f}",
            capability=cap,
        )

    all_tickers = sorted(set(portfolio.positions.keys()) | set(clean_targets.keys()))
    if not all_tickers:
        return []

    latest = pit.latest_prices(
        tickers=all_tickers,
        columns=[broker.price_col],
        universe_only=False,
    )
    tradable_prices: dict[str, float] = {}
    for ticker, val in latest[broker.price_col].items():
        if pd.notna(val) and _is_finite_real(float(val)) and float(val) > 0:
            tradable_prices[str(ticker)] = float(val)

    equity = portfolio.cash
    for ticker, pos in portfolio.positions.items():
        mark_px = tradable_prices.get(ticker, portfolio.last_prices.get(ticker, pos.avg_cost))
        equity += pos.quantity * mark_px

    if equity <= _EPS:
        return []

    investable = equity * (1.0 - float(cash_buffer))
    sell_orders: list[Order] = []
    buy_candidates: list[tuple[str, float, float]] = []

    for ticker in all_tickers:
        px = tradable_prices.get(ticker)
        if px is None:
            continue

        curr_qty = portfolio.quantity_of(ticker)
        target_w = clean_targets.get(ticker, 0.0)
        if target_w <= 0.0:
            if curr_qty > _EPS:
                sell_orders.append(Order.sell(ticker, curr_qty))
            continue

        target_val = investable * target_w
        delta_val = target_val - curr_qty * px
        if abs(delta_val) < min_trade_value:
            continue

        raw_qty = abs(delta_val) / px
        qty = raw_qty if allow_fractional else float(math.floor(raw_qty))
        if qty <= _EPS:
            continue

        if delta_val < 0:
            sell_qty = min(qty, curr_qty) if not broker.allow_short else qty
            if sell_qty > _EPS:
                sell_orders.append(Order.sell(ticker, sell_qty))
        else:
            buy_candidates.append((ticker, qty, px))

    projected_cash = portfolio.cash
    for order in sell_orders:
        fill = broker.estimate_fill(
            order,
            execution_date=pit.now,
            market_price=tradable_prices[order.ticker],
        )
        projected_cash += fill.net_cash_flow

    buy_orders = _build_buy_orders(
        buy_candidates,
        projected_cash=projected_cash,
        broker=broker,
        portfolio=portfolio,
        execution_date=pit.now,
        allow_fractional=allow_fractional,
    )
    return [*sell_orders, *buy_orders]
