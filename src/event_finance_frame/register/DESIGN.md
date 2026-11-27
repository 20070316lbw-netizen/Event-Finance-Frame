# 注册表与公开 API 设计

本模块 (`src/event_finance_frame/register`) 负责统一登记项目的可调用函数、在调用前自动校验输入数据契约，并配合 `src/eff` 门面层提供类似 PyTorch 风格的简洁导入体验。

## 1. 对外导入方式 (`eff` 门面 + `event_finance_frame` 核心包)

保留 `src/event_finance_frame` 作为核心实现包，同时通过 `pyproject.toml` 的 `[tool.uv.build-backend] module-name = ["event_finance_frame", "eff"]` 暴露轻量门面 `src/eff`：

```python
import eff
from eff import register, run, build_lab, EventFinanceError
from eff import data, db, errors, registry
```

静态导出（`__init__.py` + `__all__`）保证 IDE 补全与类型检查完整可用；`@register` 返回经 `functools.wraps` 包装后的函数，使得**直接静态导入调用**和**通过 `eff.run(name, ...)` 动态调度**均会触发相同的前置校验。

## 2. 面板数据前置校验 (`validate_panel_index`)

通过 `@register` 登记的方法默认开启 `validate_panel_index` 校验（非面板函数可传 `validator=None` 跳过）：

| 校验项 | 要求 | 失败错误码 (`ValidationErrorCode`) |
|---|---|---|
| 输入类型 | `pandas.DataFrame` 或 `pandas.Series` | `VAL_INVALID_INPUT_TYPE` |
| 索引结构 | `MultiIndex`，且层级名依次为 `("date", "ticker")` | `VAL_INVALID_INDEX` |
| `date` 层类型 | `datetime64` 或 Python `datetime.date`（不含缺失值） | `VAL_INVALID_DATE_LEVEL` |
| 索引唯一性 | 不存在重复的 `(date, ticker)` 键 | `VAL_DUPLICATE_INDEX` |
| 索引顺序 | 已按 `(date, ticker)` 升序排列 (`is_monotonic_increasing`) | `VAL_UNSORTED_INDEX` |

## 3. 统一异常处理 (`src/event_finance_frame/error.py`)

所有注册表错误（`RegistryErrorCode`：`REG_DUPLICATE_METHOD`、`REG_METHOD_NOT_FOUND`）与校验错误（`ValidationErrorCode`）均统一抛出 `EventFinanceError`，并在 `capability` 字段中携带方法名，便于定位问题。
