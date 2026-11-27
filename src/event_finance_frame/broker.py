"""经纪商撮合与交易成本模型: 收盘价撮合、比例/每股/最低手续费与滑点"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from event_finance_frame.calendar import DateLike, normalize_date
from event_finance_frame.error import BrokerErrorCode, EventFinanceError
from event_finance_frame.pit import PITDataView

if TYPE_CHECKING:
    from event_finance_frame.portfolio import Portfolio

__all__ = [
    "Broker",
    "Fill",
    "Order",
    "OrderSide",
]

_VALID_PRICE_COLS = frozenset({"adj_close", "close"})
_EPS = 1e-9


def _is_finite_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    """日频市价/收盘价委托单"""

    ticker: str
    side: OrderSide
    quantity: float
    order_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise EventFinanceError(
                BrokerErrorCode.INVALID_ORDER,
                f"订单 ticker 必须是非空字符串, 实际为 {self.ticker!r}",
                capability="Order",
            )
        if isinstance(self.side, str) and not isinstance(self.side, OrderSide):
            try:
                object.__setattr__(self, "side", OrderSide(self.side))
            except ValueError:
                raise EventFinanceError(
                    BrokerErrorCode.INVALID_ORDER,
                    f"非法订单方向 {self.side!r}, 仅支持 'BUY' 或 'SELL'",
                    capability="Order",
                ) from None
        elif not isinstance(self.side, OrderSide):
            raise EventFinanceError(
                BrokerErrorCode.INVALID_ORDER,
                f"非法订单方向 {self.side!r}",
                capability="Order",
            )

        if not _is_finite_real(self.quantity) or self.quantity <= 0:
            raise EventFinanceError(
                BrokerErrorCode.INVALID_ORDER,
                f"订单数量必须是有限正数, 实际为 {self.quantity!r}",
                capability="Order",
            )
        object.__setattr__(self, "quantity", float(self.quantity))

    @classmethod
    def buy(cls, ticker: str, quantity: float, order_id: str | None = None) -> Order:
        return cls(ticker=ticker, side=OrderSide.BUY, quantity=quantity, order_id=order_id)

    @classmethod
    def sell(cls, ticker: str, quantity: float, order_id: str | None = None) -> Order:
        return cls(ticker=ticker, side=OrderSide.SELL, quantity=quantity, order_id=order_id)


@dataclass(frozen=True)
class Fill:
    """成交回报记录"""

    date: date
    ticker: str
    side: OrderSide
    quantity: float
    market_price: float
    fill_price: float
    gross_amount: float
    commission: float
    slippage_cost: float
    net_cash_flow: float
    order_id: str | None = None


def _normalize_orders(orders: Order | Sequence[Order], *, capability: str) -> list[Order]:
    if isinstance(orders, Order):
        return [orders]
    if isinstance(orders, (str, bytes)) or not isinstance(orders, Sequence):
        raise EventFinanceError(
            BrokerErrorCode.INVALID_ORDER,
            f"orders 必须是 Order 或 Order 序列, 实际为 {type(orders).__name__}",
            capability=capability,
        )
    result: list[Order] = []
    for item in orders:
        if not isinstance(item, Order):
            raise EventFinanceError(
                BrokerErrorCode.INVALID_ORDER,
                f"orders 元素必须是 Order 实例, 实际为 {type(item).__name__}",
                capability=capability,
            )
        result.append(item)
    return result


class Broker:
    """日频收盘价撮合经纪商

    默认按当前交易日收盘价撮合:
        - 买入成交价 = market_price * (1 + slippage_bps / 10_000)
        - 卖出成交价 = market_price * (1 - slippage_bps / 10_000)
        - 手续费 = max(min_commission, gross_amount * commission_rate + quantity * per_share_commission)
    """

    def __init__(
        self,
        *,
        commission_rate: float = 0.0,
        min_commission: float = 0.0,
        per_share_commission: float = 0.0,
        slippage_bps: float = 0.0,
        price_col: Literal["adj_close", "close"] = "adj_close",
        allow_short: bool = False,
        allow_margin: bool = False,
    ) -> None:
        cap = "Broker"
        for name, val in (
            ("commission_rate", commission_rate),
            ("min_commission", min_commission),
            ("per_share_commission", per_share_commission),
        ):
            if not _is_finite_real(val) or val < 0:
                raise EventFinanceError(
                    BrokerErrorCode.INVALID_CONFIG,
                    f"{name} 必须是有限非负数, 实际为 {val!r}",
                    capability=cap,
                )
        if not _is_finite_real(slippage_bps) or slippage_bps < 0 or slippage_bps >= 10_000:
            raise EventFinanceError(
                BrokerErrorCode.INVALID_CONFIG,
                f"slippage_bps 必须在 [0, 10000) 之间, 实际为 {slippage_bps!r}",
                capability=cap,
            )
        if price_col not in _VALID_PRICE_COLS:
            raise EventFinanceError(
                BrokerErrorCode.INVALID_CONFIG,
                f"price_col 只能是 'adj_close' 或 'close', 实际为 {price_col!r}",
                capability=cap,
            )
        if not isinstance(allow_short, bool) or not isinstance(allow_margin, bool):
            raise EventFinanceError(
                BrokerErrorCode.INVALID_CONFIG,
                "allow_short 和 allow_margin 必须是 bool",
                capability=cap,
            )

        self._commission_rate = float(commission_rate)
        self._min_commission = float(min_commission)
        self._per_share_commission = float(per_share_commission)
        self._slippage_bps = float(slippage_bps)
        self._price_col: Literal["adj_close", "close"] = price_col
        self._allow_short = allow_short
        self._allow_margin = allow_margin

    @property
    def commission_rate(self) -> float:
        return self._commission_rate

    @property
    def min_commission(self) -> float:
        return self._min_commission

    @property
    def per_share_commission(self) -> float:
        return self._per_share_commission

    @property
    def slippage_bps(self) -> float:
        return self._slippage_bps

    @property
    def price_col(self) -> Literal["adj_close", "close"]:
        return self._price_col

    @property
    def allow_short(self) -> bool:
        return self._allow_short

    @property
    def allow_margin(self) -> bool:
        return self._allow_margin

    def estimate_fill(
        self,
        order: Order,
        *,
        execution_date: DateLike,
        market_price: float,
    ) -> Fill:
        """按给定市场价和当前成本参数计算单笔成交回报 (不修改组合状态)"""
        cap = "Broker.estimate_fill"
        if not isinstance(order, Order):
            raise EventFinanceError(
                BrokerErrorCode.INVALID_ORDER,
                f"期望 Order 实例, 实际为 {type(order).__name__}",
                capability=cap,
            )
        if not _is_finite_real(market_price) or market_price <= 0:
            raise EventFinanceError(
                BrokerErrorCode.MISSING_PRICE,
                f"{order.ticker} 成交价格无效或缺失: {market_price!r}",
                capability=cap,
            )

        exec_dt = normalize_date(execution_date)
        mkt_px = float(market_price)
        slip_rate = self._slippage_bps / 10_000.0
        fill_price = (
            mkt_px * (1.0 + slip_rate)
            if order.side is OrderSide.BUY
            else mkt_px * (1.0 - slip_rate)
        )
        gross_amount = order.quantity * fill_price
        slippage_cost = order.quantity * abs(fill_price - mkt_px)
        raw_comm = (
            gross_amount * self._commission_rate
            + order.quantity * self._per_share_commission
        )
        commission = max(self._min_commission, raw_comm)
        net_cash_flow = (
            -(gross_amount + commission)
            if order.side is OrderSide.BUY
            else gross_amount - commission
        )
        return Fill(
            date=exec_dt,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            market_price=mkt_px,
            fill_price=fill_price,
            gross_amount=gross_amount,
            commission=commission,
            slippage_cost=slippage_cost,
            net_cash_flow=net_cash_flow,
            order_id=order.order_id,
        )

    def execute_with_prices(
        self,
        orders: Order | Sequence[Order],
        *,
        execution_date: DateLike,
        prices: Mapping[str, float],
        portfolio: Portfolio,
    ) -> list[Fill]:
        """在指定行情字典上原子撮合并记账 (先卖后买, 任一笔校验失败则整批回滚)"""
        cap = "Broker.execute_with_prices"
        order_list = _normalize_orders(orders, capability=cap)
        if not order_list:
            return []

        exec_dt = normalize_date(execution_date)
        sorted_orders = sorted(
            order_list,
            key=lambda o: 0 if o.side is OrderSide.SELL else 1,
        )

        can_short = self._allow_short and portfolio.allow_short
        can_margin = self._allow_margin and portfolio.allow_margin
        sim_cash = portfolio.cash
        sim_pos: dict[str, float] = {
            ticker: pos.quantity for ticker, pos in portfolio.positions.items()
        }
        planned_fills: list[Fill] = []

        for order in sorted_orders:
            raw_px = prices.get(order.ticker)
            if raw_px is None or not _is_finite_real(raw_px) or raw_px <= 0:
                raise EventFinanceError(
                    BrokerErrorCode.MISSING_PRICE,
                    f"{exec_dt} 缺少 {order.ticker} 有效成交价: {raw_px!r}",
                    capability=cap,
                )

            fill = self.estimate_fill(
                order,
                execution_date=exec_dt,
                market_price=float(raw_px),
            )
            sim_cash += fill.net_cash_flow
            if not can_margin and sim_cash < -_EPS:
                raise EventFinanceError(
                    BrokerErrorCode.INSUFFICIENT_CASH,
                    f"{exec_dt} 买入 {order.ticker} 资金不足: 预计剩余现金 {sim_cash:.4f}",
                    capability=cap,
                )

            old_qty = sim_pos.get(order.ticker, 0.0)
            signed_delta = fill.quantity if fill.side is OrderSide.BUY else -fill.quantity
            new_qty = old_qty + signed_delta
            if not can_short and new_qty < -_EPS:
                raise EventFinanceError(
                    BrokerErrorCode.INSUFFICIENT_POSITION,
                    f"{exec_dt} 卖出 {order.ticker} 持仓不足: 持有 {old_qty}, 卖出 {fill.quantity}",
                    capability=cap,
                )

            sim_pos[order.ticker] = 0.0 if abs(new_qty) <= _EPS else new_qty
            planned_fills.append(fill)

        for fill in planned_fills:
            portfolio.apply_fill(fill)

        return planned_fills

    def execute(
        self,
        orders: Order | Sequence[Order],
        *,
        pit: PITDataView,
        portfolio: Portfolio,
    ) -> list[Fill]:
        """从 PITDataView 读取 pit.now 收盘价并撮合记账"""
        cap = "Broker.execute"
        order_list = _normalize_orders(orders, capability=cap)
        if not order_list:
            return []

        unique_tickers = list(dict.fromkeys(o.ticker for o in order_list))
        latest = pit.latest_prices(
            tickers=unique_tickers,
            columns=[self._price_col],
            universe_only=False,
        )
        prices: dict[str, float] = {}
        for ticker, val in latest[self._price_col].items():
            if pd.notna(val):
                prices[str(ticker)] = float(val)

        return self.execute_with_prices(
            order_list,
            execution_date=pit.now,
            prices=prices,
            portfolio=portfolio,
        )
