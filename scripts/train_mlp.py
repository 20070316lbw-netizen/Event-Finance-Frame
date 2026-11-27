"""根据 YAML 配置运行横截面排序模型训练与评估"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from loguru import logger

from event_finance_frame.config import CONFIG, ROOT
from event_finance_frame.models import (
    DataConfig,
    Train,
    build_loss,
    build_model,
    build_optimizer,
    load_config,
    resolve_device,
    save_checkpoint,
)


@dataclass
class CrossSectionalDataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    train_batches: list[torch.Tensor]
    x_val: torch.Tensor
    y_val: torch.Tensor
    val_batches: list[torch.Tensor]
    feature_names: list[str]
    train_dates: int
    val_dates: int


def _build_batches_from_dates(dates: pd.Series) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    # dates 须已按 (date, ticker) 排好序且索引连续
    for idx in dates.groupby(dates, sort=False).groups.values():
        batches.append(torch.tensor(list(idx), dtype=torch.long))
    return batches


def load_synthetic_dataset(cfg: DataConfig, seed: int) -> CrossSectionalDataset:
    gen = torch.Generator().manual_seed(seed)
    n_days = cfg.n_days
    n_stocks = cfg.n_stocks
    n_features = cfg.n_features
    n_rows = n_days * n_stocks

    x = torch.randn(n_rows, n_features, generator=gen)
    true_w = torch.randn(n_features, generator=gen)
    y = x @ true_w + cfg.noise_std * torch.randn(n_rows, generator=gen)

    n_train_days = max(1, min(n_days - 1, int(n_days * cfg.train_ratio)))
    n_val_days = n_days - n_train_days
    split_row = n_train_days * n_stocks

    x_train, y_train = x[:split_row], y[:split_row]
    x_val, y_val = x[split_row:], y[split_row:]

    train_batches = [
        torch.arange(d * n_stocks, (d + 1) * n_stocks, dtype=torch.long)
        for d in range(n_train_days)
    ]
    val_batches = [
        torch.arange(d * n_stocks, (d + 1) * n_stocks, dtype=torch.long)
        for d in range(n_val_days)
    ]
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return CrossSectionalDataset(
        x_train=x_train,
        y_train=y_train,
        train_batches=train_batches,
        x_val=x_val,
        y_val=y_val,
        val_batches=val_batches,
        feature_names=feature_names,
        train_dates=n_train_days,
        val_dates=n_val_days,
    )


def load_duckdb_dataset(cfg: DataConfig) -> CrossSectionalDataset:
    db_path = Path(cfg.db_path)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB 数据库不存在: {db_path}")

    start_dt = date.fromisoformat(cfg.start_date)
    end_dt = date.fromisoformat(cfg.end_date)
    if end_dt <= start_dt:
        raise ValueError(f"end_date ({end_dt}) 必须晚于 start_date ({start_dt})")

    max_window = max(cfg.feature_windows) if cfg.feature_windows else 1
    buf_start = start_dt - timedelta(days=max(max_window * 2 + 30, 30))
    buf_end = end_dt + timedelta(days=(cfg.label_horizon + cfg.label_gap) * 2 + 15)

    price_expr = "p.close" if cfg.price_col == "close" else "p.adj_close"
    window_exprs = [
        f"price / lag(price, {w}) OVER w - 1 AS ret_{w}" for w in cfg.feature_windows
    ]
    feature_cols = ["log_ret_1", *[f"ret_{w}" for w in cfg.feature_windows]]
    select_features = ",\n            ".join(["ln(price) - ln(lag(price, 1) OVER w) AS log_ret_1", *window_exprs])

    query = f"""
    WITH min_snap AS (
        SELECT coalesce(max(snapshot_date), DATE '1900-01-01') AS dt
        FROM universe
        WHERE snapshot_date <= ?
    ),
    candidate_tickers AS (
        SELECT DISTINCT u.ticker
        FROM universe u, min_snap
        WHERE u.snapshot_date >= min_snap.dt
          AND u.snapshot_date <= ?
    ),
    px AS (
        SELECT
            p.date,
            p.ticker,
            CASE WHEN {price_expr} > 0 THEN {price_expr} END AS price
        FROM prices p
        JOIN candidate_tickers ct ON p.ticker = ct.ticker
        WHERE p.date BETWEEN ? AND ?
    ),
    feats AS (
        SELECT
            date,
            ticker,
            {select_features},
            lead(date, {cfg.label_gap + cfg.label_horizon}) OVER w AS exit_date,
            lead(price, {cfg.label_gap + cfg.label_horizon}) OVER w
                / lead(price, {cfg.label_gap}) OVER w - 1 AS forward_return
        FROM px
        WINDOW w AS (PARTITION BY ticker ORDER BY date)
    ),
    trading_dates AS (
        SELECT DISTINCT date
        FROM feats
        WHERE date BETWEEN ? AND ?
    ),
    pit_snap AS (
        SELECT
            td.date,
            (
                SELECT max(u.snapshot_date)
                FROM universe u
                WHERE u.snapshot_date <= td.date
            ) AS snapshot_date
        FROM trading_dates td
    )
    SELECT
        f.date,
        f.ticker,
        f.exit_date,
        f.forward_return,
        {", ".join(f"f.{c}" for c in feature_cols)}
    FROM feats f
    JOIN pit_snap ps ON f.date = ps.date
    JOIN universe u ON u.snapshot_date = ps.snapshot_date AND u.ticker = f.ticker
    ORDER BY f.date, f.ticker
    """

    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            query,
            [start_dt, end_dt, buf_start, buf_end, start_dt, end_dt],
        ).df()

    if df.empty:
        raise ValueError("DuckDB 查询返回空数据集, 请检查日期范围和数据库内容")

    needed = [*feature_cols, "forward_return", "exit_date"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)

    # 过滤横截面股票数不足 2 的日期
    counts = df.groupby("date")["ticker"].transform("count")
    df = df.loc[counts >= 2].reset_index(drop=True)

    # 按日横截面标准化特征 (零前视偏差)
    for col in feature_cols:
        grp = df.groupby("date")[col]
        mean = grp.transform("mean")
        std = grp.transform("std").clip(lower=1e-8).fillna(1.0)
        df[col] = ((df[col] - mean) / std).clip(-5.0, 5.0)

    unique_dates = np.sort(df["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError(f"有效交易日数量不足以划分训练/验证集: {len(unique_dates)}")

    n_train_dates = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * cfg.train_ratio)))
    val_start_date = unique_dates[n_train_dates]

    # Purging: 剔除标签 exit_date 跨越验证集起始日的训练样本
    train_df = df.loc[(df["date"] < val_start_date) & (df["exit_date"] < val_start_date)].reset_index(
        drop=True
    )
    val_df = df.loc[df["date"] >= val_start_date].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("切分与 purging 后训练集或验证集为空, 请扩大日期范围或调整 train_ratio")

    x_train = torch.tensor(train_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = torch.tensor(train_df["forward_return"].to_numpy(dtype=np.float32))
    x_val = torch.tensor(val_df[feature_cols].to_numpy(dtype=np.float32))
    y_val = torch.tensor(val_df["forward_return"].to_numpy(dtype=np.float32))

    return CrossSectionalDataset(
        x_train=x_train,
        y_train=y_train,
        train_batches=_build_batches_from_dates(train_df["date"]),
        x_val=x_val,
        y_val=y_val,
        val_batches=_build_batches_from_dates(val_df["date"]),
        feature_names=feature_cols,
        train_dates=int(train_df["date"].nunique()),
        val_dates=int(val_df["date"].nunique()),
    )


def mean_cross_sectional_ic(
    scores: torch.Tensor,
    targets: torch.Tensor,
    batches: list[torch.Tensor],
    eps: float = 1e-8,
) -> float:
    """计算按日横截面平均 Pearson IC"""
    ics: list[float] = []
    for rows in batches:
        if rows.numel() < 2:
            continue
        s = scores[rows] - scores[rows].mean()
        t = targets[rows] - targets[rows].mean()
        denom = float((s.pow(2).sum().sqrt() * t.pow(2).sum().sqrt()).clamp_min(eps).item())
        if denom > eps:
            ics.append(float((s * t).sum().item() / denom))
    return sum(ics) / len(ics) if ics else 0.0


def run_from_config(
    config_path: Path,
    output_override: Path | None = None,
) -> Path | None:
    cfg = load_config(config_path)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    logger.info("加载配置: {} (source={}, device={})", config_path, cfg.data.source, device)
    if cfg.data.source == "synthetic":
        ds = load_synthetic_dataset(cfg.data, cfg.seed)
    else:
        ds = load_duckdb_dataset(cfg.data)

    x_train = ds.x_train.to(device)
    y_train = ds.y_train.to(device)
    x_val = ds.x_val.to(device)
    y_val = ds.y_val.to(device)
    train_batches = [b.to(device) for b in ds.train_batches]
    val_batches = [b.to(device) for b in ds.val_batches]

    logger.info(
        "数据集就绪: features={} ({}), train={} rows / {} days, val={} rows / {} days",
        len(ds.feature_names),
        ", ".join(ds.feature_names),
        x_train.shape[0],
        ds.train_dates,
        x_val.shape[0],
        ds.val_dates,
    )

    model = build_model(cfg.model, n_features=x_train.shape[1]).to(device)
    loss_fn = build_loss(cfg.loss).to(device)
    optimizer = build_optimizer(cfg.optimizer, model.parameters())
    trainer = Train(model=model, loss=loss_fn, optimizer=optimizer)

    final_loss = 0.0
    final_train_ic = 0.0
    final_val_ic = 0.0
    for epoch in range(1, cfg.train.epochs + 1):
        final_loss = trainer.fit_epoch(x_train, y_train, train_batches)
        if epoch == 1 or epoch % cfg.train.log_every == 0 or epoch == cfg.train.epochs:
            train_scores = trainer.predict(x_train)
            val_scores = trainer.predict(x_val)
            final_train_ic = mean_cross_sectional_ic(train_scores, y_train, train_batches)
            final_val_ic = mean_cross_sectional_ic(val_scores, y_val, val_batches)
            logger.info(
                "epoch {:>3d}/{} | train_loss={:.6f} | train_ic={:+.4f} | val_ic={:+.4f}",
                epoch,
                cfg.train.epochs,
                final_loss,
                final_train_ic,
                final_val_ic,
            )

    logger.info(
        "训练完成: final_loss={:.6f}, final_train_ic={:+.4f}, final_val_ic={:+.4f}",
        final_loss,
        final_train_ic,
        final_val_ic,
    )

    raw_output = output_override if output_override is not None else cfg.train.output_path
    if raw_output is None:
        return None

    out_path = Path(raw_output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    saved_path = save_checkpoint(
        out_path,
        model,
        cfg=cfg,
        feature_names=ds.feature_names,
        metrics={
            "final_loss": final_loss,
            "final_train_ic": final_train_ic,
            "final_val_ic": final_val_ic,
            "train_rows": int(x_train.shape[0]),
            "val_rows": int(x_val.shape[0]),
        },
    )
    logger.info("已保存模型权重: {}", saved_path)
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行事件金融框架模型训练")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG / "mlp_smoke.yaml",
        help="YAML 配置文件路径 (默认: config/mlp_smoke.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="模型权重输出路径 (默认读取配置中的 train.output_path)",
    )
    args = parser.parse_args()
    run_from_config(args.config, output_override=args.output)


if __name__ == "__main__":
    main()
