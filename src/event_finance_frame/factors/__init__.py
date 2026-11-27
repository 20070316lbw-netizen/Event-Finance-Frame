"""因子注册、自动发现、因子组合 YAML 加载与 PIT 计算框架"""

from __future__ import annotations

import importlib
import math
import pkgutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

import numpy as np
import pandas as pd
import yaml

from event_finance_frame.config import FACTOR_CONFIGS, ROOT
from event_finance_frame.error import (
    EventFinanceError,
    FactorErrorCode,
    RegistryErrorCode,
)
from event_finance_frame.pit import PITDataView
from event_finance_frame.register import (
    MethodSpec,
    get_method,
    list_methods,
    register,
    run,
    validate_panel_index,
)
from event_finance_frame.register.validators import Validator

__all__ = [
    "FactorBundleConfig",
    "FactorConfig",
    "combine_factor_scores",
    "compute_factors_from_pit",
    "discover_factors",
    "factor",
    "get_factor",
    "list_factors",
    "load_factor_bundle",
    "resolve_factor_config_path",
    "resolve_factor_lookback",
]

P = ParamSpec("P")
R = TypeVar("R")

_VALID_SOURCES = frozenset({"log_returns", "prices"})
_VALID_COMBINE_METHODS = frozenset({"zscore_weighted", "rank_weighted", "raw_weighted"})
_FACTOR_GROUP = "factors"


@dataclass(frozen=True)
class FactorConfig:
    """单个因子的运行配置"""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    alias: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子名称必须是非空字符串, 实际为 {self.name!r}",
                capability="FactorConfig",
            )
        if not isinstance(self.params, Mapping):
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子 {self.name} 的 params 必须是 mapping 类型",
                capability="FactorConfig",
            )
        if self.alias is not None and (not isinstance(self.alias, str) or not self.alias.strip()):
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子 {self.name} 的 alias 必须是非空字符串或 None",
                capability="FactorConfig",
            )
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
        ):
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子 {self.name} 的 weight 必须是有限实数, 实际为 {self.weight!r}",
                capability="FactorConfig",
            )
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "weight", float(self.weight))

    @property
    def output_name(self) -> str:
        return self.alias or self.name


@dataclass(frozen=True)
class FactorBundleConfig:
    """因子组合 YAML 配置 (描述多个子因子及其横截面合成方式)"""

    name: str
    factors: tuple[FactorConfig, ...]
    combine: Literal["zscore_weighted", "rank_weighted", "raw_weighted"] = "zscore_weighted"
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子组合 name 必须是非空字符串, 实际为 {self.name!r}",
                capability="FactorBundleConfig",
            )
        if not self.factors:
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子组合 {self.name} 至少需要包含一个因子",
                capability="FactorBundleConfig",
            )
        if self.combine not in _VALID_COMBINE_METHODS:
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"不支持的合成方式 {self.combine!r}, 允许值: {sorted(_VALID_COMBINE_METHODS)}",
                capability="FactorBundleConfig",
            )
        seen: set[str] = set()
        for f in self.factors:
            if f.output_name in seen:
                raise EventFinanceError(
                    FactorErrorCode.INVALID_CONFIG,
                    f"因子组合 {self.name} 中存在重复的输出列名: {f.output_name}",
                    capability="FactorBundleConfig",
                )
            seen.add(f.output_name)


def resolve_factor_config_path(ref: str | Path) -> Path:
    """解析因子组合 YAML 路径 (支持直接路径、相对仓库路径或 factors/configs 下的短名)"""
    raw = Path(ref)
    candidates: list[Path] = [raw]
    if not raw.is_absolute():
        candidates.extend([ROOT / raw, FACTOR_CONFIGS / raw])
        if raw.suffix not in {".yaml", ".yml"}:
            candidates.extend(
                [
                    FACTOR_CONFIGS / f"{raw}.yaml",
                    FACTOR_CONFIGS / f"{raw}.yml",
                ]
            )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"找不到因子组合配置文件: {ref} (已搜索当前路径、{ROOT} 与 {FACTOR_CONFIGS})"
    )


def load_factor_bundle(ref: str | Path) -> FactorBundleConfig:
    """加载并校验因子组合 YAML 配置文件"""
    path = resolve_factor_config_path(ref)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, Mapping):
        raise EventFinanceError(
            FactorErrorCode.INVALID_CONFIG,
            f"因子组合配置文件顶层必须是 YAML mapping: {path}",
            capability="load_factor_bundle",
        )

    raw_factors = raw.get("factors")
    if not isinstance(raw_factors, Sequence) or isinstance(raw_factors, (str, bytes)):
        raise EventFinanceError(
            FactorErrorCode.INVALID_CONFIG,
            f"因子组合配置 {path} 中的 factors 必须是列表",
            capability="load_factor_bundle",
        )

    parsed_factors: list[FactorConfig] = []
    for item in raw_factors:
        if not isinstance(item, Mapping):
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子组合配置 {path} 中的 factors 元素必须是 mapping",
                capability="load_factor_bundle",
            )
        parsed_factors.append(FactorConfig(**item))

    name = str(raw.get("name") or path.stem)
    combine = raw.get("combine", "zscore_weighted")
    description = str(raw.get("description") or "")
    return FactorBundleConfig(
        name=name,
        factors=tuple(parsed_factors),
        combine=combine,
        description=description,
    )


def combine_factor_scores(
    cross_section: pd.DataFrame,
    bundle: FactorBundleConfig,
) -> pd.Series:
    """对单日横截面各子因子列做无前视标准化与加权合成, 返回以 ticker 为索引的 Series"""
    cols = [f.output_name for f in bundle.factors]
    missing = [c for c in cols if c not in cross_section.columns]
    if missing:
        raise EventFinanceError(
            FactorErrorCode.MISSING_COLUMN,
            f"横截面数据缺少因子列: {missing}",
            capability="combine_factor_scores",
        )

    sub = cross_section.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if sub.empty:
        return pd.Series(
            dtype="float64",
            index=pd.Index([], name=cross_section.index.name or "ticker"),
            name=bundle.name,
        )

    if (
        len(bundle.factors) == 1
        and bundle.combine == "raw_weighted"
        and bundle.factors[0].weight == 1.0
    ):
        return sub[cols[0]].astype("float64").rename(bundle.name)

    total = pd.Series(0.0, index=sub.index, dtype="float64")
    for cfg in bundle.factors:
        col = sub[cfg.output_name].astype("float64")
        if bundle.combine == "zscore_weighted":
            std = float(col.std(ddof=1)) if len(col) > 1 else 0.0
            if std > 1e-12:
                norm = (col - float(col.mean())) / std
            else:
                norm = pd.Series(0.0, index=col.index, dtype="float64")
        elif bundle.combine == "rank_weighted":
            norm = col.rank(method="average", pct=True)
        else:
            norm = col
        total = total + cfg.weight * norm

    return total.rename(bundle.name)


def factor(
    name: str | None = None,
    *,
    source: Literal["log_returns", "prices"] = "log_returns",
    lookback: int | Callable[..., int],
    description: str = "",
    validator: Validator | None = validate_panel_index,
    data_arg: int | str = 0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """因子注册装饰器: 登记到全局注册表并记录数据源与回看窗口元信息"""
    if source not in _VALID_SOURCES:
        raise EventFinanceError(
            FactorErrorCode.UNSUPPORTED_SOURCE,
            f"不支持的因子数据源 {source!r}, 允许值: {sorted(_VALID_SOURCES)}",
            capability=name,
        )
    if not callable(lookback) and (
        isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1
    ):
        raise EventFinanceError(
            FactorErrorCode.INVALID_CONFIG,
            f"lookback 必须是 >= 1 的整数或可调用对象, 实际为 {lookback!r}",
            capability=name,
        )

    return register(
        name,
        description=description,
        group=_FACTOR_GROUP,
        validator=validator,
        data_arg=data_arg,
        metadata={"source": source, "lookback": lookback},
    )


def discover_factors(package_name: str = __name__) -> list[str]:
    """自动扫描并导入 factors 包下的全部子模块, 触发 @factor 注册 (幂等)"""
    package = importlib.import_module(package_name)
    if hasattr(package, "__path__"):
        for mod_info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            importlib.import_module(mod_info.name)
    return sorted(spec.name for spec in list_methods(group=_FACTOR_GROUP))


def get_factor(name: str) -> MethodSpec:
    """按名称获取已注册因子的 MethodSpec"""
    discover_factors()
    spec = get_method(name)
    if spec.group != _FACTOR_GROUP:
        raise EventFinanceError(
            RegistryErrorCode.METHOD_NOT_FOUND,
            f"因子尚未注册: {name}",
            capability=name,
        )
    return spec


def list_factors() -> list[MethodSpec]:
    """列出全部已注册的因子 (按名称升序)"""
    discover_factors()
    return sorted(list_methods(group=_FACTOR_GROUP), key=lambda spec: spec.name)


def resolve_factor_lookback(
    spec: MethodSpec,
    params: Mapping[str, Any] | None = None,
) -> int:
    """解析因子在给定参数下所需的 PIT 最小历史交易日数量"""
    raw = spec.metadata.get("lookback")
    if callable(raw):
        try:
            val = raw(**dict(params or {}))
        except (TypeError, ValueError) as exc:
            raise EventFinanceError(
                FactorErrorCode.INVALID_ARGUMENT,
                f"解析因子 {spec.name} 的 lookback 参数失败: {exc}",
                capability=spec.name,
            ) from exc
    else:
        val = raw

    if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        raise EventFinanceError(
            FactorErrorCode.INVALID_ARGUMENT,
            f"因子 {spec.name} 解析出的 lookback 必须是 >= 1 的整数, 实际为 {val!r}",
            capability=spec.name,
        )
    return val


def _empty_factor_panel(columns: Sequence[str]) -> pd.DataFrame:
    empty_index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([], dtype="datetime64[ns]", name="date"),
            pd.Index([], dtype="object", name="ticker"),
        ],
        names=["date", "ticker"],
    )
    return pd.DataFrame(columns=list(columns), index=empty_index)


def compute_factors_from_pit(
    pit: PITDataView,
    factor_configs: Sequence[FactorConfig],
    *,
    price_col: Literal["adj_close", "close"] = "adj_close",
    tickers: str | Sequence[str] | None = None,
    universe_only: bool | None = None,
) -> pd.DataFrame:
    """根据因子配置列表从 PITDataView 批量拉取所需历史数据并计算因子面板"""
    if not factor_configs:
        return _empty_factor_panel([])

    discover_factors()
    resolved: list[tuple[FactorConfig, MethodSpec, str, int]] = []
    seen_names: set[str] = set()
    max_lookbacks: dict[str, int] = {}

    for cfg in factor_configs:
        if cfg.output_name in seen_names:
            raise EventFinanceError(
                FactorErrorCode.INVALID_CONFIG,
                f"因子输出列名重复: {cfg.output_name}",
                capability="compute_factors_from_pit",
            )
        seen_names.add(cfg.output_name)

        spec = get_factor(cfg.name)
        source = spec.metadata.get("source")
        if source not in _VALID_SOURCES:
            raise EventFinanceError(
                FactorErrorCode.UNSUPPORTED_SOURCE,
                f"因子 {cfg.name} 声明了不支持的数据源: {source!r}",
                capability=cfg.name,
            )
        lookback = resolve_factor_lookback(spec, cfg.params)
        max_lookbacks[source] = max(max_lookbacks.get(source, 0), lookback)
        resolved.append((cfg, spec, source, lookback))

    source_frames: dict[str, pd.DataFrame] = {}
    for source, max_lb in max_lookbacks.items():
        if source == "log_returns":
            source_frames[source] = pit.log_returns(
                tickers=tickers,
                lookback=max_lb,
                price_col=price_col,
                universe_only=universe_only,
            )
        elif source == "prices":
            source_frames[source] = pit.prices(
                tickers=tickers,
                lookback=max_lb,
                universe_only=universe_only,
            )

    columns: dict[str, pd.Series] = {}
    for cfg, _, source, _ in resolved:
        series = run(cfg.name, source_frames[source], **cfg.params)
        if not isinstance(series, pd.Series):
            raise EventFinanceError(
                FactorErrorCode.INVALID_ARGUMENT,
                f"因子 {cfg.name} 必须返回 pd.Series, 实际为 {type(series).__name__}",
                capability=cfg.name,
            )
        columns[cfg.output_name] = series

    out = pd.DataFrame(columns)
    if out.empty:
        return _empty_factor_panel(list(columns.keys()))
    out.index.names = ["date", "ticker"]
    return out.sort_index()
