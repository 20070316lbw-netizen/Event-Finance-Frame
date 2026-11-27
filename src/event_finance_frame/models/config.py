"""模型训练配置加载与组件工厂"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from event_finance_frame.models.loss import ICLoss, PairwiseRankLoss
from event_finance_frame.models.model import MLP

_VALID_DATA_SOURCES = {"synthetic", "duckdb"}
_VALID_PRICE_COLS = {"adj_close", "close"}
_VALID_MODELS = {"mlp"}
_VALID_LOSSES = {"pairwise_rank", "ic"}
_VALID_OPTIMIZERS = {"adam", "adamw", "sgd"}
_VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    # synthetic 参数
    n_days: int = 40
    n_stocks: int = 32
    n_features: int = 5
    noise_std: float = 0.1
    # duckdb 参数
    db_path: str = "data/lab.duckdb"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    feature_windows: tuple[int, ...] = (5, 10, 20, 60)
    label_horizon: int = 5
    label_gap: int = 1
    price_col: str = "adj_close"
    train_ratio: float = 0.8


@dataclass(frozen=True)
class ModelConfig:
    name: str = "mlp"
    hidden: tuple[int, ...] = (32, 16)


@dataclass(frozen=True)
class LossConfig:
    name: str = "pairwise_rank"
    margin: float = 0.0
    eps: float = 1e-8


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-2
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    log_every: int = 5
    output_path: str | None = None


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _ensure_mapping(raw: Any, section: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"配置节 '{section}' 必须是 mapping 类型")
    return raw


def _to_int_tuple(raw: Any, name: str) -> tuple[int, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(int(v) for v in raw)
    raise ValueError(f"配置项 '{name}' 必须是整数列表")


def _validate_config(cfg: RunConfig) -> None:
    if cfg.device not in _VALID_DEVICES:
        raise ValueError(f"不支持的 device: {cfg.device}")
    if cfg.data.source not in _VALID_DATA_SOURCES:
        raise ValueError(f"不支持的 data.source: {cfg.data.source}")
    if cfg.data.price_col not in _VALID_PRICE_COLS:
        raise ValueError(f"不支持的 data.price_col: {cfg.data.price_col}")
    if not 0.0 < cfg.data.train_ratio < 1.0:
        raise ValueError(f"data.train_ratio 必须在 (0, 1) 之间, 实际为 {cfg.data.train_ratio}")
    if cfg.data.n_days < 2 or cfg.data.n_stocks < 2 or cfg.data.n_features < 1:
        raise ValueError("synthetic 数据参数 n_days/n_stocks >= 2 且 n_features >= 1")
    if cfg.data.label_horizon < 1 or cfg.data.label_gap < 0:
        raise ValueError("label_horizon 必须 >= 1 且 label_gap 必须 >= 0")
    if any(w < 1 for w in cfg.data.feature_windows):
        raise ValueError("data.feature_windows 中的窗口必须全部 >= 1")
    if cfg.model.name not in _VALID_MODELS:
        raise ValueError(f"不支持的 model.name: {cfg.model.name}")
    if any(h < 1 for h in cfg.model.hidden):
        raise ValueError("model.hidden 中的隐藏层宽度必须全部 >= 1")
    if cfg.loss.name not in _VALID_LOSSES:
        raise ValueError(f"不支持的 loss.name: {cfg.loss.name}")
    if cfg.optimizer.name not in _VALID_OPTIMIZERS:
        raise ValueError(f"不支持的 optimizer.name: {cfg.optimizer.name}")
    if cfg.optimizer.lr <= 0:
        raise ValueError(f"optimizer.lr 必须 > 0, 实际为 {cfg.optimizer.lr}")
    if cfg.train.epochs < 1 or cfg.train.log_every < 1:
        raise ValueError("train.epochs 和 train.log_every 必须 >= 1")
    if cfg.train.output_path is not None:
        if not isinstance(cfg.train.output_path, str):
            raise TypeError("train.output_path 必须是字符串或 null")
        if not cfg.train.output_path.strip():
            raise ValueError("train.output_path 不能为空字符串")


def load_config(path: str | Path) -> RunConfig:
    """从 YAML 文件加载并校验模型运行配置"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, Mapping):
        raise TypeError(f"配置文件顶层必须是 YAML mapping: {config_path}")

    data_raw = dict(_ensure_mapping(raw.get("data"), "data"))
    if "feature_windows" in data_raw:
        data_raw["feature_windows"] = _to_int_tuple(
            data_raw["feature_windows"], "data.feature_windows"
        )
    data_cfg = DataConfig(**data_raw)

    model_raw = dict(_ensure_mapping(raw.get("model"), "model"))
    if "hidden" in model_raw:
        model_raw["hidden"] = _to_int_tuple(model_raw["hidden"], "model.hidden")
    model_cfg = ModelConfig(**model_raw)

    loss_cfg = LossConfig(**_ensure_mapping(raw.get("loss"), "loss"))
    optimizer_cfg = OptimizerConfig(**_ensure_mapping(raw.get("optimizer"), "optimizer"))
    train_cfg = TrainConfig(**_ensure_mapping(raw.get("train"), "train"))

    cfg = RunConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        data=data_cfg,
        model=model_cfg,
        loss=loss_cfg,
        optimizer=optimizer_cfg,
        train=train_cfg,
    )
    _validate_config(cfg)
    return cfg


def resolve_device(device: str) -> torch.device:
    """解析运行设备; auto 默认优先 CUDA, 否则使用 CPU"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in _VALID_DEVICES:
        return torch.device(device)
    raise ValueError(f"不支持的 device: {device}")


def build_model(cfg: ModelConfig, n_features: int) -> torch.nn.Module:
    if cfg.name == "mlp":
        return MLP(n_features=n_features, hidden=cfg.hidden)
    raise ValueError(f"不支持的 model.name: {cfg.name}")


def build_loss(cfg: LossConfig) -> torch.nn.Module:
    if cfg.name == "pairwise_rank":
        return PairwiseRankLoss(margin=cfg.margin)
    if cfg.name == "ic":
        return ICLoss(eps=cfg.eps)
    raise ValueError(f"不支持的 loss.name: {cfg.loss.name if hasattr(cfg, 'loss') else cfg.name}")


def build_optimizer(
    cfg: OptimizerConfig, params: Iterable[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    if cfg.name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"不支持的 optimizer.name: {cfg.name}")
