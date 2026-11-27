"""根据回测 YAML 与因子组合 YAML 运行回测, 打印绩效摘要并生成调仓日志 .txt"""

from __future__ import annotations

import argparse
from pathlib import Path

from event_finance_frame.config import CONFIG, ROOT
from event_finance_frame.runner import load_backtest_config, run_backtest_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="运行分离式 YAML 配置驱动的 PIT 等权回测")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=CONFIG / "mom_equal_weight.yaml",
        help="回测配置文件路径 (默认: config/mom_equal_weight.yaml)",
    )
    parser.add_argument(
        "-f",
        "--factor-config",
        type=str,
        default=None,
        help="可选: 覆盖回测配置中的因子组合 YAML (如 mom_blend.yaml 或文件路径)",
    )
    parser.add_argument(
        "-o",
        "--log-output",
        type=Path,
        default=None,
        help="可选: 覆盖回测调仓日志 .txt 输出路径",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不输出 .txt 调仓日志文件",
    )
    args = parser.parse_args()

    cfg = load_backtest_config(args.config, factor_config_override=args.factor_config)
    result, m = run_backtest_from_config(
        cfg,
        log_output_override=args.log_output,
        write_log=not args.no_log,
    )

    factor_desc = (
        f"{cfg.strategy.factor_bundle.name} ({cfg.strategy.factor_config})"
        if cfg.strategy.factor_bundle is not None
        else "N/A (1/N全池)"
    )
    summary_lines = [
        f"策略: {cfg.strategy.name} | 因子组合: {factor_desc}",
        f"回测区间: {m.start_date} -> {m.end_date} ({m.n_days} 个交易日)",
        f"初始资金: {m.initial_cash:,.2f} -> 期末权益: {m.final_equity:,.2f}",
        f"累计收益: {m.total_return:+.2%} | 年化收益 (CAGR): {m.cagr:+.2%}",
        f"年化波动: {m.annualized_volatility:.2%} | Sharpe: {m.sharpe_ratio:.2f} | 最大回撤: {m.max_drawdown:.2%}",
        f"调仓次数: {len(result.events)} | 成交笔数: {m.n_trades} | 年化换手率: {m.annualized_turnover:.2%}",
        f"交易成本: 手续费 {m.total_commission:,.2f} + 滑点 {m.total_slippage:,.2f}",
        f"期末持仓数: {len(result.portfolio.positions)}",
    ]

    if not args.no_log:
        raw_log = args.log_output if args.log_output is not None else cfg.output.log_path
        if raw_log is not None:
            out_path = Path(raw_log)
            if not out_path.is_absolute():
                out_path = ROOT / out_path
            summary_lines.append(f"调仓日志已写入: {out_path}")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
