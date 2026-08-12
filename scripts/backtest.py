#!/usr/bin/env python3
"""CLI entry-point for backtesting: python -m scripts.backtest --config backtest.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from packages.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestReport,
    CsvDataFeed,
    MACDCrossover,
)


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with p.open() as f:
        return yaml.safe_load(f)


def run_backtest(config_path: str, output_dir: str = "output") -> None:
    cfg = load_config(config_path)

    # Build BacktestConfig from YAML
    bt_cfg_dict = cfg.get("backtest", {})
    config = BacktestConfig(**bt_cfg_dict)

    # Data feed
    data_path = cfg.get("data", {}).get("path", "")
    if not data_path:
        print("ERROR: data.path required in YAML config", file=sys.stderr)
        sys.exit(1)
    data_feed = CsvDataFeed(filepath=data_path, symbol=config.symbol)

    # Strategy
    strategy_cfg = cfg.get("strategy", {})
    strategy_type = strategy_cfg.get("type", "macd_crossover")

    if strategy_type == "macd_crossover":
        strategy = MACDCrossover(
            name=strategy_cfg.get("name", "macd"),
            fast_period=strategy_cfg.get("fast_period", 12),
            slow_period=strategy_cfg.get("slow_period", 26),
            signal_period=strategy_cfg.get("signal_period", 9),
            min_confidence=strategy_cfg.get("min_confidence", 0.6),
            position_size=strategy_cfg.get("position_size", 0.1),
        )
    else:
        print(f"ERROR: unknown strategy type: {strategy_type}", file=sys.stderr)
        sys.exit(1)

    # Run engine
    engine = BacktestEngine(config=config)
    result = engine.run(data_feed, strategy)

    # Output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = BacktestReport(result)

    # JSON report
    json_path = str(Path(output_dir) / "report.json")
    report.to_json_file(json_path)
    print(f"JSON report written to: {json_path}")

    # HTML report
    html_path = str(Path(output_dir) / "report.html")
    report.to_html_file(html_path)
    print(f"HTML report written to: {html_path}")

    # CSV trade log
    csv_path = str(Path(output_dir) / "trades.csv")
    report.to_csv(csv_path)
    print(f"Trade log written to: {csv_path}")

    # Print key metrics to stdout
    metrics = result.metrics
    print(f"\n{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital:  ${config.initial_capital:,.2f}")
    print(f"Final Equity:     ${result.final_equity:,.2f}")
    print(f"Total Return:     {metrics.get('total_return', 0)*100:.2f}%")
    print(f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown:     {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"Total Trades:     {result.total_trades}")
    print(f"{'='*50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    args = parser.parse_args()

    run_backtest(args.config, args.output)


if __name__ == "__main__":
    main()
