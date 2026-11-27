"""方法注册表: 统一登记、前置校验与按名调度"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from event_finance_frame.error import EventFinanceError, RegistryErrorCode
from event_finance_frame.register.validators import Validator, validate_panel_index

__all__ = [
    "REGISTRY",
    "MethodSpec",
    "get_method",
    "list_methods",
    "register",
    "run",
]

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    func: Callable[..., Any]
    description: str = ""
    group: str = "default"
    validator: Validator | None = None


REGISTRY: dict[str, MethodSpec] = {}


def _resolve_target_arg(
    sig: inspect.Signature,
    param_names: list[str],
    data_arg: int | str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if isinstance(data_arg, int) and 0 <= data_arg < len(args):
        return args[data_arg]

    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    if isinstance(data_arg, str):
        return bound.arguments[data_arg]
    return bound.arguments[param_names[data_arg]]


def register(
    name: str | None = None,
    *,
    description: str = "",
    group: str = "default",
    validator: Validator | None = validate_panel_index,
    data_arg: int | str = 0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """将函数登记到全局注册表, 并在调用前自动执行数据校验

    Args:
        name: 注册名称, 默认使用函数名 func.__name__
        description: 方法说明
        group: 命名空间分组 (如 "functional", "data", "backtest")
        validator: 前置校验器, 默认校验 (date, ticker) MultiIndex; 传 None 跳过校验
        data_arg: 待校验参数的位置索引或参数名, 默认为第 0 个参数
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        method_name = name or func.__name__
        if method_name in REGISTRY:
            raise EventFinanceError(
                RegistryErrorCode.DUPLICATE_METHOD,
                f"方法重复注册: {method_name}",
                capability=method_name,
            )

        if validator is None:
            wrapped = func
        else:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @wraps(func)
            def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
                target = _resolve_target_arg(sig, param_names, data_arg, args, kwargs)
                validator(target, capability=method_name)
                return func(*args, **kwargs)

        doc_lines = (func.__doc__ or "").strip().splitlines()
        resolved_description = description or (doc_lines[0] if doc_lines else "")

        REGISTRY[method_name] = MethodSpec(
            name=method_name,
            func=wrapped,
            description=resolved_description,
            group=group,
            validator=validator,
        )
        return wrapped

    return decorator


def get_method(name: str) -> MethodSpec:
    """按名称获取已注册的方法元数据"""
    if name not in REGISTRY:
        raise EventFinanceError(
            RegistryErrorCode.METHOD_NOT_FOUND,
            f"方法尚未注册: {name}",
            capability=name,
        )
    return REGISTRY[name]


def list_methods(group: str | None = None) -> list[MethodSpec]:
    """列出已注册的方法, 可按 group 过滤"""
    specs = list(REGISTRY.values())
    if group is not None:
        specs = [spec for spec in specs if spec.group == group]
    return specs


def run(name: str, *args: Any, **kwargs: Any) -> Any:
    """按名称调用已注册的方法"""
    return get_method(name).func(*args, **kwargs)
