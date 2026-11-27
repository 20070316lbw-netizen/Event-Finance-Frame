"""模型权重与元数据的保存和加载"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from event_finance_frame.models.config import (
    ModelConfig,
    RunConfig,
    build_model,
    resolve_device,
)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    cfg: RunConfig,
    feature_names: Sequence[str],
    metrics: Mapping[str, float | int] | None = None,
) -> Path:
    """保存模型权重及配置元数据到 checkpoint 文件"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_name": cfg.model.name,
        "n_features": len(feature_names),
        "hidden": list(cfg.model.hidden),
        "feature_names": list(feature_names),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, out)
    return out


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """从 checkpoint 文件安全加载模型并切换为 eval 模式"""
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model_cfg = ModelConfig(
        name=str(payload["model_name"]),
        hidden=tuple(int(h) for h in payload["hidden"]),
    )
    model = build_model(model_cfg, n_features=int(payload["n_features"]))
    model.load_state_dict(payload["state_dict"])
    model.to(resolve_device(device))
    model.eval()

    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, metadata
