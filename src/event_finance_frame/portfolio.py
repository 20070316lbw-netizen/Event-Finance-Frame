"""投资组合持仓、现金流水与每日盯市记账"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from event_finance_frame.broker import Fill, OrderSide
from event_finance_frame.calendar import DateLike, normalize_date
from event_finance_frame.error import EventFinanceError, PortfolioErrorCode
from event_finance_frame.pit import PITDataView

__all__ = [
    "Portfolio",
    "PortfolioSnapshot",
    "Position",
]

_EPS = 1e-9
_EQUITY_COLUMNS = [
    "cash",
    "market_value",
    "equity",
    "realized_pnl",
    "unrealized_pnl",
    "total_commission",
    "total_slippage",
    "n_positions",
]
_FILL_COLUMNS = [
    "date",
    "ticker",
    "side",
    "quantity",
    "market_price",
    "fill_price",
    "gross_amount",
    "commission",
    "slippage_cost",
    "net_cash_flow",
    "order_id",
]


def _is_finite_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


@dataclass(frozen=True)
class Position:
    """单一标的持仓信息 (avg_cost 为加权平均成交价, 不含手续费)"""

    ticker: str
    quantity: float
    avg_cost: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    """每日收盘盯市后的组合净值快照"""

    date: date
    cash: float
    market_value: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_commission: float
    total_slippage: float
    n_positions: int


class Portfolio:
    """投资组合账本

    会计恒等式 (任意时刻均成立):
        ``realized_pnl + unrealized_pnl - total_commission == equity - initial_cash``
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        *,
        allow_short: bool = False,
        allow_margin: bool = False,
    ) -> None:
        if not _is_finite_real(initial_cash) or initial_cash < 0:
            raise EventFinanceError(
                PortfolioErrorCode.INVALID_CASH,
                f"initial_cash 必须是有限非负数, 实际为 {initial_cash!r}",
                capability="Portfolio",
            )
        if not isinstance(allow_short, bool) or not isinstance(allow_margin, bool):
            raise EventFinanceError(
                PortfolioErrorCode.INVALID_CASH,
                "allow_short 和 allow_margin 必须是 bool",
                capability="Portfolio",
            )

        self._initial_cash = float(initial_cash)
        self._cash = float(initial_cash)
        self._allow_short = allow_short
        self._allow_margin = allow_margin
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._snapshots: list[PortfolioSnapshot] = []
        self._realized_pnl = 0.0
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._last_prices: dict[str, float] = {}

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def allow_short(self) -> bool:
        return self._allow_short

    @property
    def allow_margin(self) -> bool:
        return self._allow_margin

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    @property
    def snapshots(self) -> tuple[PortfolioSnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def total_commission(self) -> float:
        return self._total_commission

    @property
    def total_slippage(self) -> float:
        return self._total_slippage

    @property
    def last_prices(self) -> dict[str, float]:
        return dict(self._last_prices)

    def quantity_of(self, ticker: str) -> float:
        pos = self._positions.get(ticker)
        return pos.quantity if pos is not None else 0.0

    def apply_fill(self, fill: Fill) -> None:
        """将一笔成交回报过账到现金与持仓 (校验失败时保持原状态不变)"""
        cap = "Portfolio.apply_fill"
        if not isinstance(fill, Fill):
            raise EventFinanceError(
                PortfolioErrorCode.INVALID_FILL,
                f"期望 Fill 实例, 实际为 {type(fill).__name__}",
                capability=cap,
            )
        if (
            not _is_finite_real(fill.quantity)
            or fill.quantity <= 0
            or not _is_finite_real(fill.fill_price)
            or fill.fill_price <= 0
            or not _is_finite_real(fill.market_price)
            or fill.market_price <= 0
            or not _is_finite_real(fill.commission)
            or fill.commission < 0
            or not _is_finite_real(fill.slippage_cost)
            or fill.slippage_cost < 0
            or not _is_finite_real(fill.net_cash_flow)
        ):
            raise EventFinanceError(
                PortfolioErrorCode.INVALID_FILL,
                f"成交回报字段包含非法数值: {fill}",
                capability=cap,
            )

        new_cash = self._cash + fill.net_cash_flow
        if not self._allow_margin and new_cash < -_EPS:
            raise EventFinanceError(
                PortfolioErrorCode.INSUFFICIENT_CASH,
                f"成交后现金不足: {new_cash:.4f}",
                capability=cap,
            )
        if not self._allow_margin and new_cash < 0:
            new_cash = 0.0

        old_pos = self._positions.get(fill.ticker)
        old_qty = old_pos.quantity if old_pos is not None else 0.0
        old_avg = old_pos.avg_cost if old_pos is not None else 0.0
        signed_delta = fill.quantity if fill.side is OrderSide.BUY else -fill.quantity
        new_qty = old_qty + signed_delta

        if not self._allow_short and new_qty < -_EPS:
            raise EventFinanceError(
                PortfolioErrorCode.INSUFFICIENT_POSITION,
                f"{fill.ticker} 卖出超过当前持仓: 持有 {old_qty}, 卖出 {fill.quantity}",
                capability=cap,
            )

        realized_delta = 0.0
        if abs(old_qty) <= _EPS or old_qty * signed_delta > 0:
            # 新开仓或同向加仓: 更新加权平均成交价
            new_avg = (abs(old_qty) * old_avg + abs(signed_delta) * fill.fill_price) / abs(new_qty)
            updated_pos: Position | None = Position(
                ticker=fill.ticker,
                quantity=new_qty,
                avg_cost=new_avg,
            )
        else:
            # 减仓、清仓或反向开仓
            closed_abs = min(abs(old_qty), abs(signed_delta))
            realized_delta = math.copysign(closed_abs, old_qty) * (fill.fill_price - old_avg)
            if abs(new_qty) <= _EPS:
                updated_pos = None
            elif abs(signed_delta) > abs(old_qty):
                # 反手开新仓
                updated_pos = Position(
                    ticker=fill.ticker,
                    quantity=new_qty,
                    avg_cost=fill.fill_price,
                )
            else:
                # 部分平仓, 剩余仓位保留原 avg_cost
                updated_pos = Position(
                    ticker=fill.ticker,
                    quantity=new_qty,
                    avg_cost=old_avg,
                )

        # 所有校验通过后再提交状态变更
        self._cash = new_cash
        if updated_pos is None:
            self._positions.pop(fill.ticker, None)
        else:
            self._positions[fill.ticker] = updated_pos
        self._realized_pnl += realized_delta
        self._total_commission += fill.commission
        self._total_slippage += fill.slippage_cost
        self._last_prices[fill.ticker] = fill.market_price
        self._fills.append(fill)

    def mark_to_market(
        self,
        as_of_date: DateLike,
        prices: Mapping[str, float],
        *,
        strict_prices: bool = False,
    ) -> PortfolioSnapshot:
        """按给定价格截面计算组合净值并记录当日快照"""
        dt = normalize_date(as_of_date)
        for ticker, px in prices.items():
            if _is_finite_real(px) and px > 0:
                self._last_prices[ticker] = float(px)

        market_value = 0.0
        unrealized_pnl = 0.0
        for ticker, pos in self._positions.items():
            raw_px = prices.get(ticker)
            if raw_px is not None and _is_finite_real(raw_px) and raw_px > 0:
                mark_px = float(raw_px)
            elif strict_prices:
                raise EventFinanceError(
                    PortfolioErrorCode.MISSING_PRICE,
                    f"{dt} 盯市缺少持仓 {ticker} 有效价格",
                    capability="Portfolio.mark_to_market",
                )
            else:
                mark_px = self._last_prices.get(ticker, pos.avg_cost)

            market_value += pos.quantity * mark_px
            unrealized_pnl += pos.quantity * (mark_px - pos.avg_cost)

        snap = PortfolioSnapshot(
            date=dt,
            cash=self._cash,
            market_value=market_value,
            equity=self._cash + market_value,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_commission=self._total_commission,
            total_slippage=self._total_slippage,
            n_positions=len(self._positions),
        )
        if self._snapshots and self._snapshots[-1].date == dt:
            self._snapshots[-1] = snap
        else:
            self._snapshots.append(snap)
        return snap

    def mark_to_market_pit(
        self,
        pit: PITDataView,
        *,
        price_col: Literal["adj_close", "close"] = "adj_close",
        strict_prices: bool = False,
    ) -> PortfolioSnapshot:
        """基于 PITDataView 获取当前持仓收盘价并盯市"""
        if not self._positions:
            return self.mark_to_market(pit.now, {}, strict_prices=strict_prices)

        latest = pit.latest_prices(
            tickers=list(self._positions.keys()),
            columns=[price_col],
            universe_only=False,
        )
        prices: dict[str, float] = {}
        for ticker, val in latest[price_col].items():
            if pd.notna(val):
                prices[str(ticker)] = float(val)

        return self.mark_to_market(pit.now, prices, strict_prices=strict_prices)

    def equity_curve(self) -> pd.DataFrame:
        """导出每日净值与盈亏曲线 DataFrame (DatetimeIndex name='date')"""
        if not self._snapshots:
            return pd.DataFrame(
                columns=_EQUITY_COLUMNS,
                index=pd.DatetimeIndex([], dtype="datetime64[ns]", name="date"),
            )
        rows = [
            {
                "date": snap.date,
                "cash": snap.cash,
                "market_value": snap.market_value,
                "equity": snap.equity,
                "realized_pnl": snap.realized_pnl,
                "unrealized_pnl": snap.unrealized_pnl,
                "total_commission": snap.total_commission,
                "total_slippage": snap.total_slippage,
                "n_positions": snap.n_positions,
            }
            for snap in self._snapshots
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        return df.set_index("date").loc[:, _EQUITY_COLUMNS]

    def fills_frame(self) -> pd.DataFrame:
        """导出成交流水 DataFrame"""
        if not self._fills:
            return pd.DataFrame(columns=_FILL_COLUMNS)
        rows = [
            {
                "date": f.date,
                "ticker": f.ticker,
                "side": str(f.side),
                "quantity": f.quantity,
                "market_price": f.market_price,
                "fill_price": f.fill_price,
                "gross_amount": f.gross_amount,
                "commission": f.commission,
                "slippage_cost": f.slippage_cost,
                "net_cash_flow": f.net_cash_flow,
                "order_id": f.order_id,
            }
            for f in self._fills
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        return df.loc[:, _FILL_COLUMNS]

    def positions_frame(self) -> pd.DataFrame:
        """导出当前持仓快照 DataFrame (以 ticker 为索引)"""
        if not self._positions:
            return pd.DataFrame(
                columns=["quantity", "avg_cost"],
                index=pd.Index([], dtype="object", name="ticker"),
            )
        rows = [
            {
                "ticker": pos.ticker,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
            }
            for pos in sorted(self._positions.values(), key=lambda p: p.ticker)
        ]
        return pd.DataFrame(rows).set_index("ticker")
