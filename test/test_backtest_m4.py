"""测试 M4: Broker、Portfolio 与 BacktestEngine"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from event_finance_frame import (
    BacktestEngine,
    Broker,
    BrokerErrorCode,
    Clock,
    EventFinanceError,
    Fill,
    Order,
    OrderSide,
    PITDataView,
    Portfolio,
    PortfolioErrorCode,
    get_duckdb,
)
from event_finance_frame.calendar import load_calendar
from event_finance_frame.config import DB_STORAGE, DDL
from event_finance_frame.db.operation import read_sql


def _create_m4_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(read_sql(DDL / "prices.sql"))
    con.execute(read_sql(DDL / "universe.sql"))

    # AAA: 100 -> 110 -> 105 -> 120 (close = adj_close * 2, 用于区分 price_col)
    # BBB: 50 -> 55 -> None(停牌) -> 60
    # CCC: 20 -> 22 -> 24 -> 26 (Day 2 从 universe 调出)
    rows = [
        ("2024-01-02", "AAA", 100.0, 100.0, 100.0, 200.0, 100.0, 1000),
        ("2024-01-03", "AAA", 110.0, 110.0, 110.0, 220.0, 110.0, 1000),
        ("2024-01-04", "AAA", 105.0, 105.0, 105.0, 210.0, 105.0, 1000),
        ("2024-01-05", "AAA", 120.0, 120.0, 120.0, 240.0, 120.0, 1000),
        ("2024-01-02", "BBB", 50.0, 50.0, 50.0, 50.0, 50.0, 500),
        ("2024-01-03", "BBB", 55.0, 55.0, 55.0, 55.0, 55.0, 500),
        ("2024-01-05", "BBB", 60.0, 60.0, 60.0, 60.0, 60.0, 500),
        ("2024-01-02", "CCC", 20.0, 20.0, 20.0, 20.0, 20.0, 300),
        ("2024-01-03", "CCC", 22.0, 22.0, 22.0, 22.0, 22.0, 300),
        ("2024-01-04", "CCC", 24.0, 24.0, 24.0, 24.0, 24.0, 300),
        ("2024-01-05", "CCC", 26.0, 26.0, 26.0, 26.0, 26.0, 300),
    ]
    con.executemany(
        "INSERT INTO prices (date, ticker, open, high, low, close, adj_close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    load_calendar(con)

    con.executemany(
        "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("2024-01-02", "AAA", "0000000001", "Alpha", "NASDAQ", None, "Tech"),
            ("2024-01-02", "CCC", "0000000003", "Gamma", "NYSE", None, "Energy"),
            ("2024-01-03", "AAA", "0000000001", "Alpha", "NASDAQ", None, "Tech"),
            ("2024-01-03", "BBB", "0000000002", "Beta", "NYSE", None, "Finance"),
        ],
    )
    return con


def _assert_accounting_identity(portfolio: Portfolio) -> None:
    for snap in portfolio.snapshots:
        lhs = snap.realized_pnl + snap.unrealized_pnl - snap.total_commission
        rhs = snap.equity - portfolio.initial_cash
        assert lhs == pytest.approx(rhs, abs=1e-9)


def test_buy_and_hold_reconciliation_with_hand_calculation():
    """核心验收: 买入持有 (含手续费+滑点) 逐日结果与手算完全一致"""
    con = _create_m4_db()
    clk = Clock.from_duckdb(con)
    pit = PITDataView(con, clk)
    broker = Broker(commission_rate=0.001, min_commission=1.0, slippage_bps=10.0)
    portfolio = Portfolio(initial_cash=10_000.0)
    engine = BacktestEngine(clock=clk, pit=pit, broker=broker, portfolio=portfolio)

    def buy_and_hold(view: PITDataView, port: Portfolio) -> list[Order]:
        if view.now == date(2024, 1, 2):
            return [Order.buy("AAA", 50)]
        return []

    res = engine.run(buy_and_hold)
    assert res.n_days == 4
    assert res.start_date == date(2024, 1, 2)
    assert res.end_date == date(2024, 1, 5)
    assert len(res.fills) == 1
    _assert_accounting_identity(portfolio)

    curve = res.equity_curve
    expected_cash = 4989.995
    expected_rows = [
        ("2024-01-02", 5000.0, 9989.995, -5.0),
        ("2024-01-03", 5500.0, 10489.995, 495.0),
        ("2024-01-04", 5250.0, 10239.995, 245.0),
        ("2024-01-05", 6000.0, 10989.995, 995.0),
    ]
    for dt_str, mv, eq, upnl in expected_rows:
        row = curve.loc[dt_str]
        assert row["cash"] == pytest.approx(expected_cash, abs=1e-9)
        assert row["market_value"] == pytest.approx(mv, abs=1e-9)
        assert row["equity"] == pytest.approx(eq, abs=1e-9)
        assert row["unrealized_pnl"] == pytest.approx(upnl, abs=1e-9)
        assert row["realized_pnl"] == pytest.approx(0.0, abs=1e-9)
        assert row["total_commission"] == pytest.approx(5.005, abs=1e-9)
        assert row["total_slippage"] == pytest.approx(5.0, abs=1e-9)
        assert row["n_positions"] == 1

    # 追加 Day 4 全部卖出对账
    port2 = Portfolio(initial_cash=10_000.0)
    engine2 = BacktestEngine(clock=clk, pit=pit, broker=broker, portfolio=port2)

    class BuyThenExit:
        def on_bar(self, view: PITDataView, port: Portfolio) -> list[Order]:
            if view.now == date(2024, 1, 2):
                return [Order.buy("AAA", 50, order_id="entry")]
            if view.now == date(2024, 1, 5):
                return [Order.sell("AAA", port.quantity_of("AAA"), order_id="exit")]
            return []

    res2 = engine2.run(BuyThenExit(), reset_clock=True)
    _assert_accounting_identity(port2)
    assert len(res2.fills) == 2
    assert res2.fills.iloc[1]["fill_price"] == pytest.approx(119.88, abs=1e-9)
    assert port2.cash == pytest.approx(10978.001, abs=1e-9)
    assert port2.realized_pnl == pytest.approx(989.0, abs=1e-9)
    assert port2.total_commission == pytest.approx(10.999, abs=1e-9)
    assert port2.total_slippage == pytest.approx(11.0, abs=1e-9)
    assert res2.final_equity == pytest.approx(10978.001, abs=1e-9)
    assert res2.total_return == pytest.approx(978.001 / 10_000.0, abs=1e-9)
    assert port2.positions == {}
    assert port2.positions_frame().empty
    con.close()


def test_broker_cost_models_and_price_col():
    # 触发最低手续费 + 每股手续费
    broker = Broker(commission_rate=0.0005, min_commission=5.0, per_share_commission=0.02, slippage_bps=0.0)
    small_fill = broker.estimate_fill(Order.buy("AAA", 10), execution_date="2024-01-02", market_price=10.0)
    assert small_fill.commission == pytest.approx(5.0)

    large_fill = broker.estimate_fill(Order.buy("AAA", 1000), execution_date="2024-01-02", market_price=50.0)
    # 50_000 * 0.0005 + 1000 * 0.02 = 25 + 20 = 45
    assert large_fill.commission == pytest.approx(45.0)

    # 零成本模式
    zero_broker = Broker()
    zero_fill = zero_broker.estimate_fill(Order.sell("AAA", 10), execution_date="2024-01-02", market_price=50.0)
    assert zero_fill.commission == 0.0
    assert zero_fill.slippage_cost == 0.0
    assert zero_fill.net_cash_flow == 500.0

    # price_col='close' 切换 (数据库中 AAA close 是 adj_close 的 2 倍)
    con = _create_m4_db()
    clk = Clock.from_duckdb(con)
    pit = PITDataView(con, clk)
    close_broker = Broker(price_col="close")
    port = Portfolio(10_000.0)
    fills = close_broker.execute(Order.buy("AAA", 10), pit=pit, portfolio=port)
    assert fills[0].market_price == 200.0
    snap = port.mark_to_market_pit(pit, price_col="close")
    assert snap.market_value == 2000.0
    con.close()


def test_portfolio_position_accounting_and_short_flip():
    # 加仓、部分平仓、清仓
    port = Portfolio(initial_cash=20_000.0)
    brk = Broker(commission_rate=0.001)
    brk.execute_with_prices(
        Order.buy("AAA", 10),
        execution_date="2024-01-02",
        prices={"AAA": 100.0},
        portfolio=port,
    )
    brk.execute_with_prices(
        Order.buy("AAA", 10),
        execution_date="2024-01-03",
        prices={"AAA": 120.0},
        portfolio=port,
    )
    assert port.positions["AAA"].quantity == 20.0
    assert port.positions["AAA"].avg_cost == pytest.approx(110.0)

    # 部分减仓: 均价不变, 结算一半已实现盈亏
    brk.execute_with_prices(
        Order.sell("AAA", 5),
        execution_date="2024-01-04",
        prices={"AAA": 130.0},
        portfolio=port,
    )
    assert port.positions["AAA"].quantity == 15.0
    assert port.positions["AAA"].avg_cost == pytest.approx(110.0)
    assert port.realized_pnl == pytest.approx(5.0 * (130.0 - 110.0))
    snap = port.mark_to_market("2024-01-04", {"AAA": 130.0})
    assert snap.unrealized_pnl == pytest.approx(15.0 * (130.0 - 110.0))
    _assert_accounting_identity(port)

    # 同日重复盯市幂等覆盖
    snap2 = port.mark_to_market("2024-01-04", {"AAA": 135.0})
    assert len(port.snapshots) == 1
    assert snap2.market_value == pytest.approx(15.0 * 135.0)
    _assert_accounting_identity(port)

    # 做空与多空反手
    short_port = Portfolio(initial_cash=10_000.0, allow_short=True)
    short_brk = Broker(commission_rate=0.001, allow_short=True)
    short_brk.execute_with_prices(
        Order.sell("AAA", 10),
        execution_date="2024-01-02",
        prices={"AAA": 100.0},
        portfolio=short_port,
    )
    assert short_port.quantity_of("AAA") == -10.0
    short_port.mark_to_market("2024-01-02", {"AAA": 90.0})
    assert short_port.snapshots[-1].unrealized_pnl == pytest.approx(100.0)
    _assert_accounting_identity(short_port)

    # 反手买入 15 股 -> 平空 10 股 + 开多 5 股
    short_brk.execute_with_prices(
        Order.buy("AAA", 15),
        execution_date="2024-01-03",
        prices={"AAA": 90.0},
        portfolio=short_port,
    )
    assert short_port.positions["AAA"].quantity == 5.0
    assert short_port.positions["AAA"].avg_cost == pytest.approx(90.0)
    assert short_port.realized_pnl == pytest.approx(100.0)
    short_port.mark_to_market("2024-01-03", {"AAA": 95.0})
    _assert_accounting_identity(short_port)


def test_suspended_stock_mark_to_market_fallback_and_strict():
    con = _create_m4_db()
    clk = Clock.from_duckdb(con)
    pit = PITDataView(con, clk)
    brk = Broker()
    port = Portfolio(10_000.0)

    # 2024-01-02 买入 BBB @ 50
    brk.execute(Order.buy("BBB", 10), pit=pit, portfolio=port)
    port.mark_to_market_pit(pit)
    # 2024-01-03 BBB 收盘 55
    clk.step()
    port.mark_to_market_pit(pit)
    assert port.snapshots[-1].market_value == 550.0

    # 2024-01-04 BBB 停牌无价格 -> 默认沿用前值 55
    clk.step()
    snap = port.mark_to_market_pit(pit, strict_prices=False)
    assert snap.market_value == 550.0
    _assert_accounting_identity(port)

    with pytest.raises(EventFinanceError) as exc:
        port.mark_to_market_pit(pit, strict_prices=True)
    assert exc.value.code is PortfolioErrorCode.MISSING_PRICE
    con.close()


def test_broker_atomic_batch_sell_before_buy_and_delisted_sell():
    con = _create_m4_db()
    clk = Clock.from_duckdb(con)
    pit = PITDataView(con, clk)
    brk = Broker()
    port = Portfolio(initial_cash=1_000.0)

    # Day 1 满仓买入 50 股 CCC (@20 = 1000)
    brk.execute(Order.buy("CCC", 50), pit=pit, portfolio=port)
    assert port.cash == 0.0

    # Day 2 CCC 已被移出 universe 快照 (当前价格 22), 但仍可卖出并将资金在同一批次买入 AAA
    clk.step()
    assert "CCC" not in pit.universe_tickers()
    fills = brk.execute(
        [Order.buy("AAA", 10), Order.sell("CCC", 50)],
        pit=pit,
        portfolio=port,
    )
    assert [f.side for f in fills] == [OrderSide.SELL, OrderSide.BUY]
    assert port.quantity_of("CCC") == 0.0
    assert port.quantity_of("AAA") == 10.0
    assert port.cash == 0.0

    # 原子回滚: 先卖 5 股 AAA (+550), 再买 10 股 AAA (-1100) 现金不足 -> 整批失败且组合状态完全不变
    cash_before = port.cash
    pos_before = port.positions
    fills_before = port.fills
    with pytest.raises(EventFinanceError) as exc_cash:
        brk.execute([Order.sell("AAA", 5), Order.buy("AAA", 10)], pit=pit, portfolio=port)
    assert exc_cash.value.code is BrokerErrorCode.INSUFFICIENT_CASH
    assert port.cash == cash_before
    assert port.positions == pos_before
    assert port.fills == fills_before

    # 超卖触发 INSUFFICIENT_POSITION
    with pytest.raises(EventFinanceError) as exc_pos:
        brk.execute(Order.sell("AAA", 20), pit=pit, portfolio=port)
    assert exc_pos.value.code is BrokerErrorCode.INSUFFICIENT_POSITION
    assert port.positions == pos_before
    con.close()


def test_validation_errors_and_engine_reset():
    # Order / Broker / Portfolio 非法输入校验
    for kwargs in (
        {"ticker": "", "side": OrderSide.BUY, "quantity": 1},
        {"ticker": "AAA", "side": "HOLD", "quantity": 1},
        {"ticker": "AAA", "side": OrderSide.BUY, "quantity": 0},
        {"ticker": "AAA", "side": OrderSide.BUY, "quantity": True},
        {"ticker": "AAA", "side": OrderSide.BUY, "quantity": float("nan")},
    ):
        with pytest.raises(EventFinanceError) as exc:
            Order(**kwargs)  # type: ignore[arg-type]
        assert exc.value.code is BrokerErrorCode.INVALID_ORDER

    for bad_cfg in (
        {"commission_rate": -0.01},
        {"min_commission": True},
        {"slippage_bps": 10_000},
        {"price_col": "open"},
    ):
        with pytest.raises(EventFinanceError) as exc:
            Broker(**bad_cfg)  # type: ignore[arg-type]
        assert exc.value.code is BrokerErrorCode.INVALID_CONFIG

    for bad_cash in (-1.0, True, float("inf")):
        with pytest.raises(EventFinanceError) as exc:
            Portfolio(bad_cash)  # type: ignore[arg-type]
        assert exc.value.code is PortfolioErrorCode.INVALID_CASH

    with pytest.raises(EventFinanceError) as exc_fill:
        Portfolio(1000.0).apply_fill(
            Fill(
                date=date(2024, 1, 2),
                ticker="AAA",
                side=OrderSide.BUY,
                quantity=-1.0,
                market_price=10.0,
                fill_price=10.0,
                gross_amount=-10.0,
                commission=0.0,
                slippage_cost=0.0,
                net_cash_flow=10.0,
            )
        )
    assert exc_fill.value.code is PortfolioErrorCode.INVALID_FILL

    # Engine 校验与 reset_clock
    con = _create_m4_db()
    clk1 = Clock.from_duckdb(con)
    clk2 = Clock.from_duckdb(con)
    pit1 = PITDataView(con, clk1)
    with pytest.raises(ValueError):
        BacktestEngine(clock=clk2, pit=pit1, broker=Broker(), portfolio=Portfolio())

    # 空跑 reset_clock
    clk1.seek("2024-01-05")
    port = Portfolio(1000.0)
    eng = BacktestEngine(clock=clk1, pit=pit1, broker=Broker(), portfolio=port)
    res = eng.run(lambda v, p: [], reset_clock=True)
    assert res.n_days == 4
    assert res.final_equity == 1000.0
    assert res.total_return == 0.0
    assert port.fills_frame().empty
    con.close()


def test_real_lab_duckdb_buy_and_hold_smoke():
    if not DB_STORAGE.exists():
        pytest.skip(f"{DB_STORAGE} 不存在, 跳过真实库回测冒烟测试")

    with get_duckdb(path=DB_STORAGE, read_only=True) as con:
        clk = Clock.from_duckdb(con, start="2024-06-03", end="2024-06-07")
        pit = PITDataView(con, clk)
        broker = Broker(commission_rate=0.0005, slippage_bps=5.0)
        port = Portfolio(initial_cash=100_000.0)
        engine = BacktestEngine(clock=clk, pit=pit, broker=broker, portfolio=port)

        def strategy(view: PITDataView, p: Portfolio) -> list[Order]:
            if view.now == date(2024, 6, 3):
                return [Order.buy("AAPL", 100)]
            return []

        res = engine.run(strategy)
        assert res.n_days == 5
        assert len(res.fills) == 1
        assert not res.equity_curve.empty
        assert port.quantity_of("AAPL") == 100.0
        _assert_accounting_identity(port)
