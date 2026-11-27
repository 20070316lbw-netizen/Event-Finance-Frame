"""YAML 配置驱动的回测策略、加载器与统一运行入口"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd
import yaml

from event_finance_frame.broker import Broker, Order
from event_finance_frame.clock import Clock
from event_finance_frame.config import ROOT
from event_finance_frame.db.operation import get_duckdb
from event_finance_frame.engine import BacktestEngine, BacktestResult
from event_finance_frame.factors import (
    FactorBundleConfig,
    FactorConfig,
    combine_factor_scores,
    compute_factors_from_pit,
    load_factor_bundle,
)
from event_finance_frame.journal import (
    BacktestJournal,
    PendingDecision,
    write_backtest_log,
)
from event_finance_frame.metrics import PerformanceMetrics, evaluate_backtest
from event_finance_frame.pit import PITDataView
from event_finance_frame.portfolio import Portfolio
from event_finance_frame.rebalance import (
    equal_weight_targets,
    rebalance_to_weights,
    select_top_k,
)

__all__ = [
    "BacktestDataConfig",
    "BacktestRunConfig",
    "BrokerRunConfig",
    "EqualWeightUniverseStrategy",
    "FactorEqualWeightStrategy",
    "OutputRunConfig",
    "PortfolioRunConfig",
    "StrategyConfig",
    "load_backtest_config",
    "run_backtest_from_config",
]

_VALID_STRATEGIES = frozenset({"factor_equal_weight", "equal_weight_all"})
_VALID_REBALANCE_FREQS = frozenset({"daily", "weekly", "monthly"})
_VALID_PRICE_COLS = frozenset({"adj_close", "close"})


@dataclass(frozen=True)
class BacktestDataConfig:
    db_path: str = "data/lab.duckdb"
    start_date: str | None = None
    end_date: str | None = None
    price_col: Literal["adj_close", "close"] = "adj_close"


@dataclass(frozen=True)
class StrategyConfig:
    name: Literal["factor_equal_weight", "equal_weight_all"] = "factor_equal_weight"
    factor_config: str | None = "mom_12_1.yaml"
    factor_bundle: FactorBundleConfig | None = None
    top_k: int = 30
    rebalance_freq: Literal["daily", "weekly", "monthly"] = "monthly"
    ascending: bool = False
    min_trade_value: float = 0.0
    allow_fractional: bool = True
    cash_buffer: float = 0.0


@dataclass(frozen=True)
class BrokerRunConfig:
    commission_rate: float = 0.0005
    min_commission: float = 0.0
    per_share_commission: float = 0.0
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class PortfolioRunConfig:
    initial_cash: float = 100_000.0


@dataclass(frozen=True)
class OutputRunConfig:
    log_path: str | None = None


@dataclass(frozen=True)
class BacktestRunConfig:
    data: BacktestDataConfig = field(default_factory=BacktestDataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    broker: BrokerRunConfig = field(default_factory=BrokerRunConfig)
    portfolio: PortfolioRunConfig = field(default_factory=PortfolioRunConfig)
    output: OutputRunConfig = field(default_factory=OutputRunConfig)


def _ensure_mapping(raw: Any, section: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"配置节 '{section}' 必须是 mapping 类型")
    return raw


def _validate_backtest_config(cfg: BacktestRunConfig) -> None:
    if not isinstance(cfg.data.db_path, str) or not cfg.data.db_path.strip():
        raise ValueError("data.db_path 不能为空")
    if cfg.data.price_col not in _VALID_PRICE_COLS:
        raise ValueError(f"不支持的 data.price_col: {cfg.data.price_col!r}")
    if cfg.data.start_date is not None:
        date.fromisoformat(cfg.data.start_date)
    if cfg.data.end_date is not None:
        date.fromisoformat(cfg.data.end_date)
    if (
        cfg.data.start_date is not None
        and cfg.data.end_date is not None
        and date.fromisoformat(cfg.data.start_date) > date.fromisoformat(cfg.data.end_date)
    ):
        raise ValueError("data.start_date 不能晚于 data.end_date")

    if cfg.strategy.name not in _VALID_STRATEGIES:
        raise ValueError(f"不支持的 strategy.name: {cfg.strategy.name!r}")
    if cfg.strategy.rebalance_freq not in _VALID_REBALANCE_FREQS:
        raise ValueError(f"不支持的 strategy.rebalance_freq: {cfg.strategy.rebalance_freq!r}")
    if (
        isinstance(cfg.strategy.top_k, bool)
        or not isinstance(cfg.strategy.top_k, int)
        or cfg.strategy.top_k < 1
    ):
        raise ValueError(f"strategy.top_k 必须是 >= 1 的整数, 实际为 {cfg.strategy.top_k!r}")
    if cfg.strategy.name == "factor_equal_weight" and cfg.strategy.factor_bundle is None:
        raise ValueError("strategy.name='factor_equal_weight' 时必须配置 strategy.factor_config")
    if cfg.output.log_path is not None and (
        not isinstance(cfg.output.log_path, str) or not cfg.output.log_path.strip()
    ):
        raise ValueError("output.log_path 必须是非空字符串或 null")


def load_backtest_config(
    path: str | Path,
    *,
    factor_config_override: str | Path | None = None,
) -> BacktestRunConfig:
    """从 YAML 文件加载回测配置, 并按 factor_config 解析外部因子组合 YAML"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, Mapping):
        raise TypeError(f"配置文件顶层必须是 YAML mapping: {config_path}")

    data_cfg = BacktestDataConfig(**_ensure_mapping(raw.get("data"), "data"))
    strat_raw = dict(_ensure_mapping(raw.get("strategy"), "strategy"))
    if "factor" in strat_raw:
        raise ValueError(
            "回测配置不再内嵌 strategy.factor; 请改用 strategy.factor_config 指定因子组合 YAML 文件"
        )

    strat_name = strat_raw.get("name", "factor_equal_weight")
    if strat_name not in _VALID_STRATEGIES:
        raise ValueError(f"不支持的 strategy.name: {strat_name!r}")

    if strat_name == "factor_equal_weight":
        ref = (
            factor_config_override
            if factor_config_override is not None
            else strat_raw.get("factor_config", "mom_12_1.yaml")
        )
        if ref is None:
            raise ValueError("strategy.name='factor_equal_weight' 时必须指定 strategy.factor_config")
        strat_raw["factor_config"] = str(ref)
        strat_raw["factor_bundle"] = load_factor_bundle(ref)
    else:
        strat_raw.setdefault("factor_config", None)
        strat_raw["factor_bundle"] = None

    strat_cfg = StrategyConfig(**strat_raw)
    broker_cfg = BrokerRunConfig(**_ensure_mapping(raw.get("broker"), "broker"))
    portfolio_cfg = PortfolioRunConfig(**_ensure_mapping(raw.get("portfolio"), "portfolio"))
    output_cfg = OutputRunConfig(**_ensure_mapping(raw.get("output"), "output"))

    cfg = BacktestRunConfig(
        data=data_cfg,
        strategy=strat_cfg,
        broker=broker_cfg,
        portfolio=portfolio_cfg,
        output=output_cfg,
    )
    _validate_backtest_config(cfg)
    return cfg


def _should_rebalance(
    current: date,
    last_rebalance: date | None,
    freq: Literal["daily", "weekly", "monthly"],
) -> bool:
    if last_rebalance is None or freq == "daily":
        return True
    if freq == "weekly":
        return current.isocalendar()[:2] != last_rebalance.isocalendar()[:2]
    return (current.year, current.month) != (last_rebalance.year, last_rebalance.month)


class EqualWeightUniverseStrategy:
    """全池 PIT 成分股 1/N 等权再平衡策略"""

    def __init__(
        self,
        broker: Broker,
        *,
        rebalance_freq: Literal["daily", "weekly", "monthly"] = "monthly",
        min_trade_value: float = 0.0,
        allow_fractional: bool = True,
        cash_buffer: float = 0.0,
    ) -> None:
        self._broker = broker
        self._rebalance_freq = rebalance_freq
        self._min_trade_value = min_trade_value
        self._allow_fractional = allow_fractional
        self._cash_buffer = cash_buffer
        self._last_rebalance: date | None = None
        self._journal: BacktestJournal | None = None

    def attach_journal(self, journal: BacktestJournal) -> None:
        self._journal = journal

    def on_bar(self, pit: PITDataView, portfolio: Portfolio) -> list[Order]:
        if not _should_rebalance(pit.now, self._last_rebalance, self._rebalance_freq):
            return []

        tickers = pit.universe_tickers()
        if not tickers:
            return []

        self._last_rebalance = pit.now
        weights = equal_weight_targets(tickers)
        if self._journal is not None:
            self._journal.set_decision(
                PendingDecision(
                    date=pit.now,
                    target_weights=weights,
                    selected_tickers=tuple(tickers),
                    reason="equal_weight_all",
                )
            )
        return rebalance_to_weights(
            weights,
            pit=pit,
            portfolio=portfolio,
            broker=self._broker,
            min_trade_value=self._min_trade_value,
            allow_fractional=self._allow_fractional,
            cash_buffer=self._cash_buffer,
        )


class FactorEqualWeightStrategy:
    """基于外部因子组合 YAML 横截面打分的 Top-K 等权调仓策略"""

    def __init__(
        self,
        factor_bundle: FactorBundleConfig | FactorConfig,
        broker: Broker,
        *,
        top_k: int = 30,
        rebalance_freq: Literal["daily", "weekly", "monthly"] = "monthly",
        ascending: bool = False,
        min_trade_value: float = 0.0,
        allow_fractional: bool = True,
        cash_buffer: float = 0.0,
    ) -> None:
        if isinstance(factor_bundle, FactorConfig):
            bundle = FactorBundleConfig(
                name=factor_bundle.output_name,
                factors=(factor_bundle,),
                combine="raw_weighted",
            )
        else:
            bundle = factor_bundle

        self._bundle = bundle
        self._broker = broker
        self._top_k = top_k
        self._rebalance_freq = rebalance_freq
        self._ascending = ascending
        self._min_trade_value = min_trade_value
        self._allow_fractional = allow_fractional
        self._cash_buffer = cash_buffer
        self._last_rebalance: date | None = None
        self._journal: BacktestJournal | None = None

    def attach_journal(self, journal: BacktestJournal) -> None:
        self._journal = journal

    def on_bar(self, pit: PITDataView, portfolio: Portfolio) -> list[Order]:
        if not _should_rebalance(pit.now, self._last_rebalance, self._rebalance_freq):
            return []

        panel = compute_factors_from_pit(
            pit,
            self._bundle.factors,
            price_col=self._broker.price_col,
        )
        if panel.empty:
            return []

        ts = pd.Timestamp(pit.now)
        if ts not in panel.index.get_level_values("date"):
            return []

        today_df = panel.xs(ts, level="date")
        combined = combine_factor_scores(today_df, self._bundle)
        selected = select_top_k(combined, self._top_k, ascending=self._ascending)
        if not selected:
            return []

        self._last_rebalance = pit.now
        weights = equal_weight_targets(selected)

        if self._journal is not None:
            ranks_series = combined.rank(method="min", ascending=self._ascending)
            ctx_tickers = set(selected) | set(portfolio.positions.keys())
            f_vals: dict[str, dict[str, float]] = {}
            c_scores: dict[str, float] = {}
            ranks_map: dict[str, int] = {}
            for ticker in ctx_tickers:
                if ticker in today_df.index:
                    row = today_df.loc[ticker]
                    f_vals[ticker] = {
                        str(col): float(row[col])
                        for col in today_df.columns
                        if pd.notna(row[col])
                    }
                if ticker in combined.index and pd.notna(combined.loc[ticker]):
                    c_scores[ticker] = float(combined.loc[ticker])
                if ticker in ranks_series.index and pd.notna(ranks_series.loc[ticker]):
                    ranks_map[ticker] = int(ranks_series.loc[ticker])

            self._journal.set_decision(
                PendingDecision(
                    date=pit.now,
                    target_weights=weights,
                    selected_tickers=tuple(selected),
                    factor_values=f_vals,
                    composite_scores=c_scores,
                    ranks=ranks_map,
                    reason=f"factor_ew:{self._bundle.name}",
                )
            )

        return rebalance_to_weights(
            weights,
            pit=pit,
            portfolio=portfolio,
            broker=self._broker,
            min_trade_value=self._min_trade_value,
            allow_fractional=self._allow_fractional,
            cash_buffer=self._cash_buffer,
        )


def _run_with_connection(
    cfg: BacktestRunConfig,
    connection: duckdb.DuckDBPyConnection,
    *,
    log_output_override: str | Path | None = None,
    write_log: bool = True,
) -> tuple[BacktestResult, PerformanceMetrics]:
    clock = Clock.from_duckdb(
        connection,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
    )
    pit = PITDataView(connection, clock)
    broker = Broker(
        commission_rate=cfg.broker.commission_rate,
        min_commission=cfg.broker.min_commission,
        per_share_commission=cfg.broker.per_share_commission,
        slippage_bps=cfg.broker.slippage_bps,
        price_col=cfg.data.price_col,
    )
    portfolio = Portfolio(initial_cash=cfg.portfolio.initial_cash)

    if cfg.strategy.name == "equal_weight_all":
        strategy: EqualWeightUniverseStrategy | FactorEqualWeightStrategy = (
            EqualWeightUniverseStrategy(
                broker,
                rebalance_freq=cfg.strategy.rebalance_freq,
                min_trade_value=cfg.strategy.min_trade_value,
                allow_fractional=cfg.strategy.allow_fractional,
                cash_buffer=cfg.strategy.cash_buffer,
            )
        )
    else:
        assert cfg.strategy.factor_bundle is not None
        strategy = FactorEqualWeightStrategy(
            cfg.strategy.factor_bundle,
            broker,
            top_k=cfg.strategy.top_k,
            rebalance_freq=cfg.strategy.rebalance_freq,
            ascending=cfg.strategy.ascending,
            min_trade_value=cfg.strategy.min_trade_value,
            allow_fractional=cfg.strategy.allow_fractional,
            cash_buffer=cfg.strategy.cash_buffer,
        )

    journal = BacktestJournal()
    engine = BacktestEngine(clock=clock, pit=pit, broker=broker, portfolio=portfolio)
    result = engine.run(strategy, journal=journal)
    metrics = evaluate_backtest(result)

    if write_log:
        raw_log = log_output_override if log_output_override is not None else cfg.output.log_path
        if raw_log is not None:
            out_path = Path(raw_log)
            if not out_path.is_absolute():
                out_path = ROOT / out_path
            write_backtest_log(result, metrics, out_path, config=cfg)

    return result, metrics


def run_backtest_from_config(
    config: str | Path | BacktestRunConfig,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
    factor_config_override: str | Path | None = None,
    log_output_override: str | Path | None = None,
    write_log: bool = True,
) -> tuple[BacktestResult, PerformanceMetrics]:
    """按回测 YAML (及可选覆盖的因子组合 YAML) 运行回测, 返回结果与绩效指标"""
    if isinstance(config, BacktestRunConfig):
        cfg = config
        if factor_config_override is not None and cfg.strategy.name == "factor_equal_weight":
            bundle = load_factor_bundle(factor_config_override)
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    factor_config=str(factor_config_override),
                    factor_bundle=bundle,
                ),
            )
    else:
        cfg = load_backtest_config(config, factor_config_override=factor_config_override)

    if connection is not None:
        return _run_with_connection(
            cfg,
            connection,
            log_output_override=log_output_override,
            write_log=write_log,
        )

    db_path = Path(cfg.data.db_path)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    with get_duckdb(path=db_path, read_only=True) as con:
        return _run_with_connection(
            cfg,
            con,
            log_output_override=log_output_override,
            write_log=write_log,
        )
