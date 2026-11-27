"""models 模块单元测试与配置/简单训练验证"""

from __future__ import annotations

import pytest
import torch

from event_finance_frame.config import CONFIG
from event_finance_frame.models import (
    MLP,
    ICLoss,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PairwiseRankLoss,
    Train,
    build_loss,
    build_model,
    build_optimizer,
    load_checkpoint,
    load_config,
    resolve_device,
    save_checkpoint,
)


def test_mlp_output_shape():
    model = MLP(n_features=6, hidden=(16, 8))
    x = torch.randn(10, 6)
    out = model(x)
    assert out.shape == (10,)

    linear_only = MLP(n_features=4, hidden=())
    assert linear_only(torch.randn(5, 4)).shape == (5,)


def test_pairwise_rank_loss_ordering_and_edge_cases():
    loss_fn = PairwiseRankLoss()
    targets = torch.tensor([0.1, 0.2, 0.5])
    good_scores = torch.tensor([-1.0, 0.0, 2.0])
    bad_scores = torch.tensor([2.0, 0.0, -1.0])

    assert loss_fn(good_scores, targets) < loss_fn(bad_scores, targets)

    # 所有目标值相同时应返回 0 且保留梯度图
    tied_scores = torch.tensor([1.0, 2.0], requires_grad=True)
    tied_loss = loss_fn(tied_scores, torch.tensor([0.0, 0.0]))
    assert tied_loss.item() == 0.0
    tied_loss.backward()
    assert tied_scores.grad is not None


def test_ic_loss_bounds():
    loss_fn = ICLoss()
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.isclose(loss_fn(y, y), torch.tensor(-1.0), atol=1e-5)
    assert torch.isclose(loss_fn(-y, y), torch.tensor(1.0), atol=1e-5)


def test_train_simple_cross_sectional_fit():
    torch.manual_seed(42)
    n_days, n_stocks, n_features = 20, 16, 5
    n_rows = n_days * n_stocks

    x = torch.randn(n_rows, n_features)
    true_w = torch.tensor([1.5, -1.0, 0.5, 0.0, 0.8])
    y = x @ true_w + 0.1 * torch.randn(n_rows)
    batches = [torch.arange(d * n_stocks, (d + 1) * n_stocks) for d in range(n_days)]

    model = MLP(n_features=n_features, hidden=(16, 8))
    loss_fn = PairwiseRankLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Train(model=model, loss=loss_fn, optimizer=optimizer)

    history = trainer.fit(x, y, batches=batches, epochs=30)
    assert len(history) == 30
    assert history[-1] < history[0]

    preds = trainer.predict(x)
    assert preds.shape == (n_rows,)
    corr = torch.corrcoef(torch.stack([preds, y]))[0, 1].item()
    assert corr > 0.8


def test_load_repo_configs():
    smoke_cfg = load_config(CONFIG / "mlp_smoke.yaml")
    assert smoke_cfg.data.source == "synthetic"
    assert smoke_cfg.model.hidden == (32, 16)
    assert smoke_cfg.train.epochs == 30
    assert smoke_cfg.train.output_path == "checkpoints/mlp_smoke.pt"

    rank_cfg = load_config(CONFIG / "mlp_rank.yaml")
    assert rank_cfg.data.source == "duckdb"
    assert rank_cfg.data.feature_windows == (5, 10, 20, 60)
    assert rank_cfg.data.price_col == "adj_close"
    assert rank_cfg.train.output_path == "checkpoints/mlp_rank.pt"


def test_load_config_validation_errors(tmp_path):
    bad_source = tmp_path / "bad_source.yaml"
    bad_source.write_text("data:\n  source: csv\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data.source"):
        load_config(bad_source)

    bad_ratio = tmp_path / "bad_ratio.yaml"
    bad_ratio.write_text("data:\n  train_ratio: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train_ratio"):
        load_config(bad_ratio)

    not_mapping = tmp_path / "not_mapping.yaml"
    not_mapping.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(not_mapping)


def test_model_factories_and_device():
    assert resolve_device("cpu") == torch.device("cpu")
    auto_dev = resolve_device("auto")
    assert auto_dev.type in {"cuda", "cpu"}

    model = build_model(ModelConfig(name="mlp", hidden=(8,)), n_features=4)
    assert isinstance(model, MLP)

    assert isinstance(build_loss(LossConfig(name="pairwise_rank")), PairwiseRankLoss)
    assert isinstance(build_loss(LossConfig(name="ic")), ICLoss)

    opt = build_optimizer(
        OptimizerConfig(name="adamw", lr=1e-3, weight_decay=1e-4),
        model.parameters(),
    )
    assert isinstance(opt, torch.optim.AdamW)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    cfg = load_config(CONFIG / "mlp_smoke.yaml")
    model = build_model(cfg.model, n_features=5)
    model.eval()
    x = torch.randn(4, 5)
    with torch.no_grad():
        orig_pred = model(x)

    ckpt_path = tmp_path / "nested" / "model.pt"
    saved = save_checkpoint(
        ckpt_path,
        model,
        cfg=cfg,
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        metrics={"final_val_ic": 0.5},
    )
    assert saved == ckpt_path
    assert ckpt_path.exists()

    loaded_model, metadata = load_checkpoint(ckpt_path, device="cpu")
    assert loaded_model.training is False
    with torch.no_grad():
        loaded_pred = loaded_model(x)
    assert torch.allclose(orig_pred, loaded_pred)

    assert metadata["model_name"] == "mlp"
    assert metadata["n_features"] == 5
    assert metadata["hidden"] == [32, 16]
    assert metadata["feature_names"] == ["f0", "f1", "f2", "f3", "f4"]
    assert metadata["metrics"] == {"final_val_ic": 0.5}
