"""测试因子注册发现、动量因子族、等权调仓器、绩效指标与 YAML 驱动回测"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from event_finance_frame import (
    BacktestEngine,
    BacktestResult,
    Broker,
    Clock,
    EventFinanceError,
    FactorBundleConfig,
    FactorConfig,
    FactorErrorCode,
    Order,
    PITDataView,
    Portfolio,
    ValidationErrorCode,
    combine_factor_scores,
    compute_factors_from_pit,
    discover_factors,
    equal_weight_targets,
    evaluate_backtest,
    get_factor,
    list_factors,
    load_backtest_config,
    load_factor_bundle,
    rebalance_to_weights,
    resolve_factor_config_path,
    run,
    run_backtest_from_config,
    select_top_k,
    validate_panel_index,
)
from event_finance_frame.calendar import load_calendar
from event_finance_frame.config import CONFIG, DB_STORAGE, DDL
from event_finance_frame.db.operation import read_sql
from event_finance_frame.factors.mom import (
    base_momfactor,
    mom_1m,
    mom_3_1,
    mom_6_1,
    mom_12_1,
)


def _make_log_return_panel(
    rows: list[tuple[str, str, float | None]],
) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "ticker", "log_return"])
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    return df.set_index(["date", "ticker"]).sort_index()


def _create_factor_backtest_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(read_sql(DDL / "prices.sql"))
    con.execute(read_sql(DDL / "universe.sql"))

    # 8 个连续交易日: AAA 持续上涨, BBB 先涨后跌, CCC 停牌一天(01-09)后温和上涨
    dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-02-01",
        "2024-02-02",
    ]
    aaa = [100.0, 106.0, 112.0, 116.0, 120.0, 124.0, 128.0, 132.0, 140.0, 145.0]
    bbb = [100.0, 99.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0]
    ccc = [50.0, 51.0, 52.0, 53.0, 54.0, None, 55.0, 56.0, 58.0, 59.0]

    px_rows = []
    for d, pa, pb, pc in zip(dates, aaa, bbb, ccc, strict=True):
        px_rows.append((d, "AAA", pa, pa, pa, pa, pa, 1000))
        px_rows.append((d, "BBB", pb, pb, pb, pb, pb, 1000))
        if pc is not None:
            px_rows.append((d, "CCC", pc, pc, pc, pc, pc, 500))

    con.executemany(
        "INSERT INTO prices (date, ticker, open, high, low, close, adj_close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        px_rows,
    )
    load_calendar(con)

    con.executemany(
        "INSERT INTO universe VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("2024-01-02", "AAA", "0000000001", "Alpha", "NASDAQ", None, "Tech"),
            ("2024-01-02", "BBB", "0000000002", "Beta", "NYSE", None, "Finance"),
            ("2024-01-02", "CCC", "0000000003", "Gamma", "NYSE", None, "Energy"),
        ],
    )
    return con


def test_factor_discovery_and_metadata():
    names_first = discover_factors()
    names_second = discover_factors()
    assert names_first == names_second
    for expected in ("mom_12_1", "mom_1m", "mom_3_1", "mom_6_1", "mom_base"):
        assert expected in names_first

    # 不误扫私有函数
    assert all(not n.startswith("_") for n in names_first)
    specs = {s.name: s for s in list_factors()}
    assert specs["mom_base"].metadata["source"] == "log_returns"
    assert callable(specs["mom_base"].metadata["lookback"])
    assert get_factor("mom_12_1").metadata == {"source": "log_returns", "lookback": 252}


def test_base_momfactor_accuracy_skip_missing_and_immutability():
    # AAA 连续 5 天, BBB 缺少 01-03(按自身记录计数), 01-05 含 NaN 脏值
    panel = _make_log_return_panel(
        [
            ("2024-01-02", "AAA", 0.1),
            ("2024-01-03", "AAA", 0.2),
            ("2024-01-04", "AAA", 0.3),
            ("2024-01-05", "AAA", 0.4),
            ("2024-01-08", "AAA", 0.5),
            ("2024-01-02", "BBB", 1.0),
            ("2024-01-04", "BBB", 2.0),
            ("2024-01-05", "BBB", np.nan),
            ("2024-01-08", "BBB", 3.0),
        ]
    )
    snapshot = panel.copy(deep=True)

    # window=2, skip=0
    res_w2 = base_momfactor(panel, window=2, skip=0)
    pd.testing.assert_frame_equal(panel, snapshot)
    assert res_w2.name == "mom_2_0"
    assert np.isnan(res_w2.loc[(pd.Timestamp("2024-01-02"), "AAA")])
    assert res_w2.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(0.3)
    assert res_w2.loc[(pd.Timestamp("2024-01-04"), "AAA")] == pytest.approx(0.5)
    # BBB 自身记录: 01-02(1.0), 01-04(2.0) -> 3.0; 含 NaN 窗口必须为 NaN
    assert res_w2.loc[(pd.Timestamp("2024-01-04"), "BBB")] == pytest.approx(3.0)
    assert np.isnan(res_w2.loc[(pd.Timestamp("2024-01-05"), "BBB")])
    assert np.isnan(res_w2.loc[(pd.Timestamp("2024-01-08"), "BBB")])

    # window=2, skip=1 -> 整体按每只股票自身记录往后 shift(1)
    res_skip = run("mom_base", panel, window=2, skip=1)
    assert np.isnan(res_skip.loc[(pd.Timestamp("2024-01-03"), "AAA")])
    assert res_skip.loc[(pd.Timestamp("2024-01-04"), "AAA")] == pytest.approx(0.3)
    assert res_skip.loc[(pd.Timestamp("2024-01-05"), "AAA")] == pytest.approx(0.5)
    assert res_skip.loc[(pd.Timestamp("2024-01-08"), "BBB")] == pytest.approx(np.nan, nan_ok=True)
    assert res_skip.loc[(pd.Timestamp("2024-01-05"), "BBB")] == pytest.approx(3.0)

    # 空面板保持合法 MultiIndex
    empty = panel.iloc[0:0]
    res_empty = base_momfactor(empty, window=2)
    assert res_empty.empty
    assert tuple(res_empty.index.names) == ("date", "ticker")


def test_base_momfactor_validation_and_presets():
    panel = _make_log_return_panel([("2024-01-02", "AAA", 0.1)])

    for bad_fn in (
        lambda: base_momfactor(panel, window=0),
        lambda: base_momfactor(panel, window=True),  # type: ignore[arg-type]
        lambda: base_momfactor(panel, window=2, skip=-1),
        lambda: base_momfactor(panel, window=2, skip=True),  # type: ignore[arg-type]
    ):
        with pytest.raises(EventFinanceError) as exc:
            bad_fn()
        assert exc.value.code is FactorErrorCode.INVALID_ARGUMENT

    with pytest.raises(EventFinanceError) as exc_col:
        base_momfactor(panel, window=2, col="missing_col")
    assert exc_col.value.code is FactorErrorCode.MISSING_COLUMN

    with pytest.raises(EventFinanceError) as exc_idx:
        run("mom_base", panel.reset_index(), window=2)
    assert exc_idx.value.code is ValidationErrorCode.INVALID_INDEX

    # 预设因子与 base_momfactor 严格一致
    dates = pd.bdate_range("2023-01-02", periods=260)
    rng = np.random.default_rng(7)
    long_df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "log_return": rng.normal(0.0, 0.01, size=len(dates)),
        }
    ).set_index(["date", "ticker"])
    pd.testing.assert_series_equal(mom_1m(long_df), base_momfactor(long_df, 21, skip=0), check_names=False)
    pd.testing.assert_series_equal(mom_3_1(long_df), base_momfactor(long_df, 42, skip=21), check_names=False)
    pd.testing.assert_series_equal(mom_6_1(long_df), base_momfactor(long_df, 105, skip=21), check_names=False)
    pd.testing.assert_series_equal(mom_12_1(long_df), base_momfactor(long_df, 231, skip=21), check_names=False)


def test_compute_factors_from_pit():
    con = _create_factor_backtest_db()
    clk = Clock.from_duckdb(con)
    clk.seek("2024-01-11")
    pit = PITDataView(con, clk)

    cfgs = [
        FactorConfig(name="mom_base", params={"window": 2, "skip": 1}, alias="mom_2_1"),
        FactorConfig(name="mom_base", params={"window": 3, "skip": 0}, alias="mom_3_0"),
    ]
    out = compute_factors_from_pit(pit, cfgs)
    validate_panel_index(out)
    assert list(out.columns) == ["mom_2_1", "mom_3_0"]
    assert out.index.get_level_values("date").max().date() == date(2024, 1, 11)

    # 检验 AAA 2024-01-11 的 mom_2_1 = ln(128/124) + ln(124/120) = ln(128/120)
    val = out.loc[(pd.Timestamp("2024-01-11"), "AAA"), "mom_2_1"]
    assert val == pytest.approx(math.log(128.0 / 120.0))

    with pytest.raises(EventFinanceError) as exc_dup:
        compute_factors_from_pit(pit, [FactorConfig("mom_1m"), FactorConfig("mom_1m")])
    assert exc_dup.value.code is FactorErrorCode.INVALID_CONFIG
    con.close()


def test_equal_weight_rebalance_cash_protection_and_metrics():
    con = _create_factor_backtest_db()
    clk = Clock.from_duckdb(con)
    pit = PITDataView(con, clk)
    broker = Broker(commission_rate=0.002, min_commission=1.0, slippage_bps=10.0)
    portfolio = Portfolio(initial_cash=10_000.0)

    # select_top_k: 过滤 NaN/inf 且按 ticker 升序打破平局
    scores = pd.Series([1.0, 2.0, 2.0, np.nan, np.inf], index=["CCC", "BBB", "AAA", "DDD", "EEE"])
    assert select_top_k(scores, 2) == ["AAA", "BBB"]
    assert select_top_k(scores, 2, ascending=True) == ["CCC", "AAA"]

    weights = equal_weight_targets(["AAA", "BBB", "AAA"])
    assert weights == {"AAA": 0.5, "BBB": 0.5}

    # 满仓 100% 权重 + 手续费 + 滑点, rebalance_to_weights 自动缩放买单确保 execute 不超支
    orders = rebalance_to_weights(weights, pit=pit, portfolio=portfolio, broker=broker)
    assert len(orders) == 2
    fills = broker.execute(orders, pit=pit, portfolio=portfolio)
    assert len(fills) == 2
    assert portfolio.cash >= 0.0
    portfolio.mark_to_market_pit(pit)

    # 推进至 2024-01-08 先持仓 CCC, 到 CCC 停牌日 (2024-01-09) 目标权重切为 AAA=1.0 时,
    # 停牌的 CCC 保持持有不报缺失错, BBB 正常卖出, 资金买入 AAA
    clk.seek("2024-01-08")
    orders_08 = rebalance_to_weights(
        {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2},
        pit=pit,
        portfolio=portfolio,
        broker=broker,
    )
    broker.execute(orders_08, pit=pit, portfolio=portfolio)
    ccc_qty = portfolio.quantity_of("CCC")
    assert ccc_qty > 0.0
    clk.step()  # 2024-01-09, CCC 缺价
    orders_09 = rebalance_to_weights({"AAA": 1.0}, pit=pit, portfolio=portfolio, broker=broker)
    broker.execute(orders_09, pit=pit, portfolio=portfolio)
    assert portfolio.quantity_of("CCC") == pytest.approx(ccc_qty)
    assert portfolio.quantity_of("BBB") == 0.0

    # 运行完整 BacktestEngine 并校验 evaluate_backtest
    port2 = Portfolio(initial_cash=10_000.0)
    clk.reset()
    engine = BacktestEngine(clock=clk, pit=pit, broker=broker, portfolio=port2)
    res: BacktestResult = engine.run(
        lambda v, p: [Order.buy("AAA", 50)] if v.now == date(2024, 1, 2) else []
    )
    m = evaluate_backtest(res)
    assert m.n_days == len(clk)
    assert m.n_trades == 1
    assert m.total_return == pytest.approx(res.total_return)
    assert m.max_drawdown <= 0.0
    assert m.annualized_turnover > 0.0
    assert m.total_commission > 0.0
    assert m.total_slippage > 0.0
    con.close()


def test_factor_bundle_yaml_and_combine_scores():
    # 内置因子组合目录解析 (支持短名与文件名)
    p1 = resolve_factor_config_path("mom_12_1")
    p2 = resolve_factor_config_path("mom_12_1.yaml")
    assert p1 == p2
    b1 = load_factor_bundle("mom_12_1")
    assert b1.combine == "raw_weighted"
    assert len(b1.factors) == 1 and b1.factors[0].name == "mom_12_1"

    b_blend = load_factor_bundle("mom_blend.yaml")
    assert b_blend.combine == "zscore_weighted"
    assert [f.output_name for f in b_blend.factors] == ["mom_12_1", "mom_6_1", "mom_20_5"]

    with pytest.raises(FileNotFoundError):
        resolve_factor_config_path("nonexistent_bundle")

    # 横截面合成模式: raw_weighted / zscore_weighted / rank_weighted
    cs = pd.DataFrame(
        {
            "f1": [10.0, 20.0, 30.0, np.nan],
            "f2": [1.0, 2.0, 4.0, 5.0],
        },
        index=pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker"),
    )
    bundle_raw = FactorBundleConfig(
        name="raw_mix",
        combine="raw_weighted",
        factors=(
            FactorConfig("mom_1m", alias="f1", weight=0.5),
            FactorConfig("mom_1m", alias="f2", weight=2.0),
        ),
    )
    raw_scores = combine_factor_scores(cs, bundle_raw)
    assert list(raw_scores.index) == ["AAA", "BBB", "CCC"]
    assert raw_scores["AAA"] == pytest.approx(10.0 * 0.5 + 1.0 * 2.0)

    bundle_z = FactorBundleConfig(
        name="z_mix",
        combine="zscore_weighted",
        factors=bundle_raw.factors,
    )
    z_scores = combine_factor_scores(cs, bundle_z)
    assert z_scores.mean() == pytest.approx(0.0, abs=1e-12)

    bundle_r = FactorBundleConfig(
        name="rank_mix",
        combine="rank_weighted",
        factors=bundle_raw.factors,
    )
    r_scores = combine_factor_scores(cs, bundle_r)
    assert r_scores["CCC"] > r_scores["BBB"] > r_scores["AAA"]


def test_separated_yaml_backtest_and_journal_txt_generation(tmp_path: Path):
    con = _create_factor_backtest_db()

    # 独立因子组合 YAML
    bundle_a = tmp_path / "bundle_mom2.yaml"
    bundle_a.write_text(
        yaml.safe_dump(
            {
                "name": "short_mom_combo",
                "combine": "zscore_weighted",
                "factors": [
                    {"name": "mom_base", "alias": "mom_2_0", "weight": 0.7, "params": {"window": 2, "skip": 0}},
                    {"name": "mom_base", "alias": "mom_3_0", "weight": 0.3, "params": {"window": 3, "skip": 0}},
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle_b = tmp_path / "bundle_mom_skip.yaml"
    bundle_b.write_text(
        yaml.safe_dump(
            {
                "name": "skip_mom",
                "combine": "raw_weighted",
                "factors": [
                    {"name": "mom_base", "alias": "mom_2_1", "weight": 1.0, "params": {"window": 2, "skip": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )

    log_txt = tmp_path / "out" / "rebalance_log.txt"
    cfg_factor = tmp_path / "backtest_factor.yaml"
    cfg_factor.write_text(
        yaml.safe_dump(
            {
                "data": {"start_date": "2024-01-05", "end_date": "2024-02-02", "price_col": "adj_close"},
                "strategy": {
                    "name": "factor_equal_weight",
                    "factor_config": str(bundle_a),
                    "top_k": 1,
                    "rebalance_freq": "monthly",
                },
                "broker": {"commission_rate": 0.0005, "slippage_bps": 5.0},
                "portfolio": {"initial_cash": 10000.0},
                "output": {"log_path": str(log_txt)},
            }
        ),
        encoding="utf-8",
    )

    res_f, m_f = run_backtest_from_config(cfg_factor, connection=con)
    assert m_f.n_days == 7
    assert m_f.n_trades >= 1
    assert "AAA" in res_f.portfolio.positions
    assert m_f.total_return > 0.0

    # 校验调仓日志事件与 .txt 文件内容
    assert len(res_f.events) >= 1
    first_ev = res_f.events[0]
    assert first_ev.date == date(2024, 1, 5)
    assert "AAA" in first_ev.ticker_contexts
    aaa_ctx = first_ev.ticker_contexts["AAA"]
    assert aaa_ctx.name == "Alpha"
    assert aaa_ctx.sub_industry == "Tech"
    assert aaa_ctx.adj_close == 116.0
    assert "mom_2_0" in aaa_ctx.factor_values
    assert aaa_ctx.rank == 1

    assert log_txt.is_file()
    txt_content = log_txt.read_text(encoding="utf-8")
    assert "short_mom_combo" in txt_content
    assert "2024-01-05" in txt_content
    assert "Alpha | Tech | NASDAQ" in txt_content
    assert "mom_2_0=" in txt_content
    assert "横截面排名=#1" in txt_content

    # 不改回测配置, 通过 factor_config_override 切换另一个因子组合 YAML
    log_override = tmp_path / "out" / "override.txt"
    res_ov, _ = run_backtest_from_config(
        cfg_factor,
        connection=con,
        factor_config_override=bundle_b,
        log_output_override=log_override,
    )
    assert res_ov.events[0].reason == "factor_ew:skip_mom"
    assert "skip_mom" in log_override.read_text(encoding="utf-8")

    # 2. equal_weight_all 全池 1/N 基准
    cfg_all = tmp_path / "all_ew.yaml"
    cfg_all.write_text(
        yaml.safe_dump(
            {
                "data": {"start_date": "2024-01-02", "end_date": "2024-02-02"},
                "strategy": {"name": "equal_weight_all", "rebalance_freq": "monthly"},
                "broker": {"commission_rate": 0.0, "slippage_bps": 0.0},
                "portfolio": {"initial_cash": 10000.0},
            }
        ),
        encoding="utf-8",
    )
    res_a, m_a = run_backtest_from_config(cfg_all, connection=con, write_log=False)
    assert m_a.n_days == 10
    assert set(res_a.portfolio.positions.keys()) == {"AAA", "BBB", "CCC"}
    assert len(res_a.events) == 2

    # 非法 YAML 配置校验
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(yaml.safe_dump({"strategy": {"name": "unknown"}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_backtest_config(bad_cfg)
    con.close()


def test_real_lab_duckdb_yaml_momentum_smoke(tmp_path: Path):
    cfg_path = CONFIG / "mom_equal_weight.yaml"
    cfg = load_backtest_config(cfg_path)
    assert cfg.strategy.factor_bundle is not None
    assert cfg.strategy.factor_bundle.name == "mom_12_1"

    if not DB_STORAGE.exists():
        pytest.skip(f"{DB_STORAGE} 不存在, 跳过真实库动量回测冒烟测试")

    from dataclasses import replace

    smoke_cfg = replace(
        cfg,
        data=replace(cfg.data, start_date="2024-05-01", end_date="2024-06-28"),
    )
    log_out = tmp_path / "real_smoke.txt"
    res, metrics = run_backtest_from_config(
        smoke_cfg,
        factor_config_override="mom_blend.yaml",
        log_output_override=log_out,
    )
    assert metrics.n_days > 30
    assert metrics.n_trades >= 30
    assert len(res.portfolio.positions) <= 30
    assert metrics.final_equity > 0.0
    assert len(res.events) >= 2
    assert log_out.is_file()
    assert "mom_blend" in log_out.read_text(encoding="utf-8")
