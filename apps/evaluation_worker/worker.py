"""Backtest evaluation worker — computes performance metrics from trade lists."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BacktestEvaluator:
    """Evaluates backtest results and computes performance metrics."""

    def evaluate(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 100000.0,
    ) -> dict[str, Any]:
        """Evaluates a list of trades and returns computed performance metrics.

        Each trade dict must contain: symbol, direction, entry_price, exit_price,
        quantity, timestamp, commission, pnl.

        Args:
            trades: List of trade dicts to evaluate.
            initial_capital: Starting capital for return calculations.

        Returns:
            Dict with computed metrics including win_rate, profit_factor,
            sharpe_ratio, max_drawdown, return_pct, and a enriched trade_list.
        """
        if not trades:
            return self._empty_result(initial_capital)

        # Sort trades chronologically
        sorted_trades = sorted(trades, key=lambda t: t["timestamp"])

        # Enrich trades with computed metrics
        enriched_trades: list[dict[str, Any]] = []
        for trade in sorted_trades:
            pnl = trade.get("pnl", 0.0)
            enriched: dict[str, Any] = dict(trade)
            enriched["computed_pnl"] = pnl
            enriched["is_winner"] = pnl > 0.0
            enriched["is_loser"] = pnl < 0.0
            enriched_trades.append(enriched)

        total_trades = len(enriched_trades)
        winning_trades = sum(1 for t in enriched_trades if t["is_winner"])
        losing_trades = sum(1 for t in enriched_trades if t["is_loser"])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        wins = [t["computed_pnl"] for t in enriched_trades if t["is_winner"]]
        losses = [abs(t["computed_pnl"]) for t in enriched_trades if t["is_loser"]]

        total_pnl = sum(t["computed_pnl"] for t in enriched_trades)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        total_wins = sum(wins) if wins else 0.0
        total_losses = sum(losses) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        # Build equity curve from trades in chronological order
        equity_curve: list[float] = [initial_capital]
        for t in enriched_trades:
            equity_curve.append(equity_curve[-1] + t["computed_pnl"])

        # Max drawdown from equity curve
        max_drawdown = self._compute_max_drawdown(equity_curve)

        # Sharpe ratio (annualized)
        sharpe_ratio = self._compute_sharpe_ratio(equity_curve)

        # Return percentage
        final_equity = equity_curve[-1]
        return_pct = (final_equity - initial_capital) / initial_capital * 100.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 6),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 6),
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "max_drawdown": round(max_drawdown, 6),
            "sharpe_ratio": round(sharpe_ratio, 6),
            "return_pct": round(return_pct, 2),
            "trade_list": enriched_trades,
        }

    def generate_report(self, evaluation: dict[str, Any]) -> str:
        """Generates a text report from evaluation results.

        Args:
            evaluation: Dict as returned by evaluate().

        Returns:
            Formatted string with key metrics.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("BACKTEST EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append("--- TRADE SUMMARY ---")
        lines.append(f"  Total Trades:    {evaluation.get('total_trades', 0)}")
        lines.append(f"  Winning Trades:  {evaluation.get('winning_trades', 0)}")
        lines.append(f"  Losing Trades:   {evaluation.get('losing_trades', 0)}")
        lines.append("")

        lines.append("--- PERFORMANCE ---")
        lines.append(f"  Win Rate:        {evaluation.get('win_rate', 0.0):.2%}")
        lines.append(f"  Total PnL:       {evaluation.get('total_pnl', 0.0):.2f}")
        lines.append(f"  Avg Win:         {evaluation.get('avg_win', 0.0):.2f}")
        lines.append(f"  Avg Loss:        {evaluation.get('avg_loss', 0.0):.2f}")
        lines.append(f"  Avg Trade PnL:   {evaluation.get('avg_trade_pnl', 0.0):.2f}")
        lines.append(f"  Return %:        {evaluation.get('return_pct', 0.0):.2f}%")
        lines.append("")

        lines.append("--- RISK METRICS ---")
        lines.append(f"  Profit Factor:   {evaluation.get('profit_factor', 0.0):.4f}")
        lines.append(f"  Max Drawdown:    {evaluation.get('max_drawdown', 0.0):.4%}")
        lines.append(f"  Sharpe Ratio:    {evaluation.get('sharpe_ratio', 0.0):.4f}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[float]) -> float:
        """Computes peak-to-trough maximum drawdown from an equity curve."""
        if len(equity_curve) < 2:
            return 0.0

        equity_arr = np.array(equity_curve, dtype=np.float64)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max
        drawdowns = np.where(running_max == 0, 0.0, drawdowns)
        return float(np.max(drawdowns))

    @staticmethod
    def _compute_sharpe_ratio(equity_curve: list[float]) -> float:
        """Computes annualized Sharpe ratio from an equity curve.

        Uses daily returns: mean(daily_returns) / std(daily_returns) * sqrt(252).
        Returns 0.0 if std is zero or not enough data.
        """
        if len(equity_curve) < 3:
            return 0.0

        equity_arr = np.array(equity_curve, dtype=np.float64)
        daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        daily_returns = np.where(np.isfinite(daily_returns), daily_returns, 0.0)

        if len(daily_returns) < 2:
            return 0.0

        mean_ret = float(np.mean(daily_returns))
        std_ret = float(np.std(daily_returns, ddof=1))

        if std_ret == 0.0:
            return 0.0

        return mean_ret / std_ret * float(np.sqrt(252))

    @staticmethod
    def _empty_result(initial_capital: float) -> dict[str, Any]:
        """Returns default metrics for an empty trade list."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "return_pct": 0.0,
            "trade_list": [],
        }
