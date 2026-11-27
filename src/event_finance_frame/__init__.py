"""Event Finance Frame: 基于 DuckDB 的标普500时点数据与回测框架"""

from event_finance_frame import (
    calendar,
    clock,
    db,
    error,
    models,
    pipeline,
    pit,
    register,
)
from event_finance_frame.calendar import TradingCalendar, build_calendar, load_calendar
from event_finance_frame.clock import Clock
from event_finance_frame.db import get_duckdb, operation_sql, read_sql
from event_finance_frame.error import (
    BaseErrorCode,
    ClockErrorCode,
    DatabaseErrorCode,
    EventFinanceError,
    PITErrorCode,
    RegistryErrorCode,
    SourcesErrorCode,
    ValidationErrorCode,
)
from event_finance_frame.pipeline import build_lab
from event_finance_frame.pit import PITDataView
from event_finance_frame.register import (
    REGISTRY,
    MethodSpec,
    get_method,
    list_methods,
    run,
    validate_panel_index,
)

__all__ = [
    "REGISTRY",
    "BaseErrorCode",
    "Clock",
    "ClockErrorCode",
    "DatabaseErrorCode",
    "EventFinanceError",
    "MethodSpec",
    "PITDataView",
    "PITErrorCode",
    "RegistryErrorCode",
    "SourcesErrorCode",
    "TradingCalendar",
    "ValidationErrorCode",
    "build_calendar",
    "build_lab",
    "calendar",
    "clock",
    "db",
    "error",
    "get_duckdb",
    "get_method",
    "list_methods",
    "load_calendar",
    "models",
    "operation_sql",
    "pipeline",
    "pit",
    "read_sql",
    "register",
    "run",
    "validate_panel_index",
]
