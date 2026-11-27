"""统一制作项目的自定义错误与错误码"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "BaseErrorCode",
    "ClockErrorCode",
    "DatabaseErrorCode",
    "EventFinanceError",
    "PITErrorCode",
    "RegistryErrorCode",
    "SourcesErrorCode",
    "ValidationErrorCode",
]


class BaseErrorCode(StrEnum): ...


class SourcesErrorCode(BaseErrorCode): ...


class DatabaseErrorCode(BaseErrorCode): ...


class ClockErrorCode(BaseErrorCode):
    EMPTY_CALENDAR = "CLK_EMPTY_CALENDAR"
    INVALID_DATE_RANGE = "CLK_INVALID_DATE_RANGE"
    INVALID_STEP = "CLK_INVALID_STEP"
    NON_TRADING_DAY = "CLK_NON_TRADING_DAY"
    OUT_OF_RANGE = "CLK_OUT_OF_RANGE"
    BACKWARD_SEEK = "CLK_BACKWARD_SEEK"


class PITErrorCode(BaseErrorCode):
    INVALID_ARGUMENT = "PIT_INVALID_ARGUMENT"
    FUTURE_ACCESS_DENIED = "PIT_FUTURE_ACCESS_DENIED"


class RegistryErrorCode(BaseErrorCode):
    DUPLICATE_METHOD = "REG_DUPLICATE_METHOD"
    METHOD_NOT_FOUND = "REG_METHOD_NOT_FOUND"


class ValidationErrorCode(BaseErrorCode):
    INVALID_INPUT_TYPE = "VAL_INVALID_INPUT_TYPE"
    INVALID_INDEX = "VAL_INVALID_INDEX"
    INVALID_DATE_LEVEL = "VAL_INVALID_DATE_LEVEL"
    DUPLICATE_INDEX = "VAL_DUPLICATE_INDEX"
    UNSORTED_INDEX = "VAL_UNSORTED_INDEX"


class EventFinanceError(Exception):
    """项目统一异常

    调用方法:
    ```python
    raise EventFinanceError(
        SourcesErrorCode.INVALID_FORMAT,
        "SEC 数据缺少 facts 字段",
        capability="load_companyfacts",
    )
    ```
    """

    def __init__(
        self,
        code: BaseErrorCode,
        message: str,
        *,
        capability: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.capability = capability

    def __str__(self) -> str:
        text = f"[{self.code}] {super().__str__()}"
        if self.capability:
            text += f" (capability={self.capability})"
        return text
