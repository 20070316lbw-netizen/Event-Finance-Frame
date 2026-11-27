"""方法注册表: 用名字登记、校验并调用函数"""

from event_finance_frame.register.tools import (
    REGISTRY,
    MethodSpec,
    get_method,
    list_methods,
    register,
    run,
)
from event_finance_frame.register.validators import (
    PANEL_INDEX_NAMES,
    Validator,
    validate_panel_index,
)

__all__ = [
    "PANEL_INDEX_NAMES",
    "REGISTRY",
    "MethodSpec",
    "Validator",
    "get_method",
    "list_methods",
    "register",
    "run",
    "validate_panel_index",
]
