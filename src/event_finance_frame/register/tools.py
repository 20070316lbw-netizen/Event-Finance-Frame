"""制作项目注册表的组件"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    func: Callable
    description: str


REGISTRY: dict[str, MethodSpec] = {}


def register(name: str, description: str = ""):
    def decorator(func: Callable):
        if name in REGISTRY:
            raise ValueError(f"方法重复注册{name}")

        REGISTRY[name] = MethodSpec(
            name=name,
            func=func,
            description=description,
        )
        return func

    return decorator


def run(name: str, *args, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"方法尚未注册: {name}")

    return REGISTRY[name].func(*args, **kwargs)
