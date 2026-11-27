"""Event Finance Frame: 基于 DuckDB 的标普500时点数据与回测框架"""

from event_finance_frame import calendar, db, error, models, pipeline, register
from event_finance_frame.calendar import build_calendar, load_calendar
from event_finance_frame.db import get_duckdb, operation_sql, read_sql
from event_finance_frame.error import (
    BaseErrorCode,
    DatabaseErrorCode,
    EventFinanceError,
    RegistryErrorCode,
    SourcesErrorCode,
    ValidationErrorCode,
)
from event_finance_frame.pipeline import build_lab
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
    "DatabaseErrorCode",
    "EventFinanceError",
    "MethodSpec",
    "RegistryErrorCode",
    "SourcesErrorCode",
    "ValidationErrorCode",
    "build_calendar",
    "build_lab",
    "calendar",
    "db",
    "error",
    "get_duckdb",
    "get_method",
    "list_methods",
    "load_calendar",
    "models",
    "operation_sql",
    "pipeline",
    "read_sql",
    "register",
    "run",
    "validate_panel_index",
]
