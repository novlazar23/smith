"""BacktestReport — HTML and JSON export of backtest results.

Generates:
- JSON report with all metrics and trade details
- HTML report with equity curve visualization and summary table
- CSV export of trades and equity curve
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .core import BacktestResult


class BacktestReport:
    """Generate reports from backtest results.

    Supports JSON, HTML, and CSV output formats.
    """

    def __init__(self, result: BacktestResult) -> None:
        """Initialize report generator.

        Args:
            result: BacktestResult from a completed backtest.
        """
        self.result = result
        self._metadata = result.metadata
        self._metrics = result.metrics

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string with full report data.
        """
        report_data = {
            "backtest_result": {
                "initial_capital": self.result.config.initial_capital,
                "final_equity": self.result.final_equity,
                "total_return": self.result.total_return,
                "total_trades": self.result.total_trades,
                "candles_count": len(self.result.candles),
                "symbol": self.result.config.symbol,
                "start_date": (
                    self.result.candles[0].timestamp.isoformat()
                    if self.result.candles
                    else ""
                ),
                "end_date": (
                    self.result.candles[-1].timestamp.isoformat()
                    if self.result.candles
                    else ""
                ),
            },
            "metrics": self._metrics,
            "metadata": self._metadata,
            "config": {
                "initial_capital": self.result.config.initial_capital,
                "commission_rate": self.result.config.commission_rate,
                "slippage_bps": self.result.config.slippage_bps,
                "max_position_size": self.result.config.max_position_size,
                "warmup_bars": self.result.config.warmup_bars,
            },
            "generated_at": datetime.now().isoformat(),
        }

        return json.dumps(report_data, indent=indent, default=str)

    def to_json_file(self, filepath: str, indent: int = 2) -> None:
        """Write report to JSON file.

        Args:
            filepath: Output file path.
            indent: JSON indentation level.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent))

    def to_html(self) -> str:
        """Generate HTML report with equity curve and metrics table.

        Returns:
            HTML string ready to write to file or display.
        """
        equity = self._metadata.get("equity_curve", [])
        n_points = len(equity)

        # Build equity curve data for inline chart
        if equity:
            # Sample to max 200 points for readability
            step = max(1, n_points // 200)
            sampled = equity[::step]
            ", ".join(f"{v:.2f}" for v in sampled)

        # Build metrics table rows
        metric_rows = ""
        for key, value in self._metrics.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                display = f"{value:.4f}%" if "pct" in key or "rate" in key else f"{value:.4f}"
            else:
                display = str(value)
            metric_rows += f'<tr><td>{label}</td><td>{display}</td></tr>\n'

        # Build trade summary
        total_trades = self._metadata.get("total_trades", 0)
        final_equity = self._metadata.get("final_equity", 0)
        initial_capital = self._metadata.get("initial_capital", 0)
        total_return = self._metadata.get("total_return", 0)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Backtest Report — {self.result.config.symbol}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem; }}
        h2 {{ color: #333; margin-top: 2rem; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #4CAF50; font-weight: 600; }}
        .negative {{ color: #f44336; font-weight: 600; }}
        .chart-area {{ background: #fafafa; padding: 1rem; border-radius: 4px; margin: 1rem 0; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }}
        .summary-card {{ background: #f8f9fa; padding: 1rem; border-radius: 4px; text-align: center; }}
        .summary-card .value {{ font-size: 1.5rem; font-weight: 700; color: #1a1a1a; }}
        .summary-card .label {{ font-size: 0.85rem; color: #666; margin-top: 0.25rem; }}
        .generated {{ color: #999; font-size: 0.8rem; margin-top: 2rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report: {self.result.config.symbol}</h1>
        <p class="generated">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="summary-card">
                <div class="value">{final_equity:,.2f}</div>
                <div class="label">Final Equity</div>
            </div>
            <div class="summary-card">
                <div class="value {('positive' if total_return >= 0 else 'negative')}">{total_return*100:+.2f}%</div>
                <div class="label">Total Return</div>
            </div>
            <div class="summary-card">
                <div class="value">{total_trades}</div>
                <div class="label">Total Trades</div>
            </div>
            <div class="summary-card">
                <div class="value">{n_points:,}</div>
                <div class="label">Candles Processed</div>
            </div>
        </div>

        <h2>Performance Metrics</h2>
        <table>
            <thead>
                <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
                {metric_rows}
            </tbody>
        </table>

        <h2>Equity Curve</h2>
        <div class="chart-area">
            <p>Equity curve data: {n_points} data points from
            {self._metadata.get('equity_curve', [0, 0])[0]:.2f} to
            {equity[-1] if equity else 0:.2f}.</p>
            <p style="color: #666; font-style: italic;">
                Full visualization available in CSV export.
            </p>
        </div>

        <h2>Configuration</h2>
        <table>
            <tr><td>Initial Capital</td><td>{initial_capital:,.2f}</td></tr>
            <tr><td>Commission Rate</td><td>{self.result.config.commission_rate * 100:.3f}%</td></tr>
            <tr><td>Slippage (bps)</td><td>{self.result.config.slippage_bps}</td></tr>
            <tr><td>Max Position Size</td><td>{self.result.config.max_position_size * 100:.1f}%</td></tr>
            <tr><td>Warmup Bars</td><td>{self.result.config.warmup_bars}</td></tr>
        </table>

        <p class="generated">Backtest Report generated by Trading Orchestra v0.1.0</p>
    </div>
</body>
</html>"""

        return html

    def to_html_file(self, filepath: str) -> None:
        """Write HTML report to file.

        Args:
            filepath: Output file path.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())

    def to_csv(self, filepath: str, include_equity: bool = True) -> None:
        """Export trade and equity data to CSV.

        Args:
            filepath: Output file path.
            include_equity: Include equity curve data in export.
        """
        import csv as csv_module

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)

            # Header
            writer.writerow(["backtest_report"])
            writer.writerow([f"Symbol: {self.result.config.symbol}"])
            writer.writerow([f"Generated: {datetime.now().isoformat()}"])
            writer.writerow([])

            if include_equity:
                # Equity curve
                writer.writerow(["equity_curve"])
                writer.writerow(["timestamp", "equity"])
                for snap in self.result.snapshots:
                    writer.writerow([
                        snap.timestamp.isoformat(),
                        f"{snap.total_equity:.2f}",
                    ])
                if self.result.snapshots:
                    last_snap = self.result.snapshots[-1]
                    writer.writerow([])
                    writer.writerow(["final_equity", last_snap.total_equity])
                    writer.writerow(["initial_capital", self.result.config.initial_capital])

            # Metrics
            writer.writerow([])
            writer.writerow(["metrics"])
            for key, value in self._metrics.items():
                writer.writerow([key, value])
