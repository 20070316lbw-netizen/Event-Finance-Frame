"""统一制作项目的自定义错误

暂时占位, 先不做编写
"""

from enum import StrEnum


class BaseErrorCode(StrEnum): ...


class SourcesErrorCode(BaseErrorCode): ...


class DatabaseErrorCode(BaseErrorCode): ...


class EventFinanceError(Exception):
    """调用方法:
    ```
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
