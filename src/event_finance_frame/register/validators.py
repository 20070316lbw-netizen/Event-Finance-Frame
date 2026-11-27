"""注册表前置数据校验器"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import pandas as pd

from event_finance_frame.error import EventFinanceError, ValidationErrorCode

__all__ = ["PANEL_INDEX_NAMES", "Validator", "validate_panel_index"]

PANEL_INDEX_NAMES: tuple[str, str] = ("date", "ticker")


class Validator(Protocol):
    def __call__(self, data: Any, *, capability: str | None = None) -> None: ...


def _is_valid_date_level(date_level: pd.Index) -> bool:
    if pd.api.types.is_datetime64_any_dtype(date_level):
        return not bool(date_level.isna().any())
    if len(date_level) == 0:
        return False
    return all(isinstance(v, date) for v in date_level)


def validate_panel_index(data: Any, *, capability: str | None = None) -> None:
    """校验面板数据是否满足 (date, ticker) MultiIndex 契约

    校验项:
        1. 输入必须是 pandas.DataFrame 或 pandas.Series
        2. Index 必须是 MultiIndex, 且层级名称依次为 ('date', 'ticker')
        3. date 层必须为日期/时间类型 (datetime64 或 datetime.date, 不含缺失值)
        4. Index 不允许存在重复项
        5. Index 必须按升序排列 (is_monotonic_increasing)
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise EventFinanceError(
            ValidationErrorCode.INVALID_INPUT_TYPE,
            f"期望输入为 DataFrame 或 Series, 实际为 {type(data).__name__}",
            capability=capability,
        )

    index = data.index
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != PANEL_INDEX_NAMES:
        raise EventFinanceError(
            ValidationErrorCode.INVALID_INDEX,
            f"期望 MultiIndex 名称为 {PANEL_INDEX_NAMES}, 实际为 {type(index).__name__}({tuple(index.names)})",
            capability=capability,
        )

    date_level = index.get_level_values("date")
    if not _is_valid_date_level(date_level):
        raise EventFinanceError(
            ValidationErrorCode.INVALID_DATE_LEVEL,
            f"date 层必须为有效的日期类型 (datetime64 或 datetime.date), 实际 dtype={date_level.dtype}",
            capability=capability,
        )

    if index.has_duplicates:
        raise EventFinanceError(
            ValidationErrorCode.DUPLICATE_INDEX,
            "Index 存在重复的 (date, ticker) 键",
            capability=capability,
        )

    if not index.is_monotonic_increasing:
        raise EventFinanceError(
            ValidationErrorCode.UNSORTED_INDEX,
            "Index 未按 (date, ticker) 升序排列, 请先调用 .sort_index()",
            capability=capability,
        )
