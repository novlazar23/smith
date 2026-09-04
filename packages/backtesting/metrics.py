"""Performance metrics for backtest results.

Calculates standard financial metrics:
- Return metrics: total, annualized, cumulative
- Risk metrics: Sharpe, Sortino, Calmar, max drawdown
- Trade metrics: win rate, profit factor, avg win/loss
- Risk-adjusted metrics: information ratio, beta, alpha
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

# ─── Core Metrics ───────────────────────────────────────────────────────────


@dataclass
class BacktestMetrics:
    """Container for all backtest performance metrics."""

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_win_return: float = 0.0
    avg_loss_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_holding_days: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Risk-adjusted
    information_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0

    # Config metadata
    initial_capital: float = 0.0
    final_equity: float = 0.0
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as a flat dict."""
        return {
            "total_return_pct": round(self.total_return * 100, 4),
            "annualized_return_pct": round(self.annualized_return * 100, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown * 100, 4),
            "max_drawdown_duration_bars": self.max_drawdown_duration_bars,
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 4),
            "avg_trade_return_pct": round(self.avg_trade_return * 100, 4),
            "avg_win_return_pct": round(self.avg_win_return * 100, 4),
            "avg_loss_return_pct": round(self.avg_loss_return * 100, 4),
            "best_trade_return_pct": round(self.best_trade * 100, 4),
            "worst_trade_return_pct": round(self.worst_trade * 100, 4),
            "avg_holding_days": round(self.avg_holding_days, 2),
            "largest_win": round(self.largest_win, 4),
            "largest_loss": round(self.largest_loss, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "information_ratio": round(self.information_ratio, 4),
            "beta": round(self.beta, 4),
            "alpha": round(self.alpha * 100, 4),
        }


# ─── Calculation Functions ──────────────────────────────────────────────────


def calculate_backtest_metrics(
    equity_curve: list[float],
    trades: list[dict[str, Any]],
    risk_free_rate: float = 0.0,
    benchmark_returns: list[float] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    trading_days_per_year: int = 365,
    **kwargs: Any,
) -> BacktestMetrics:
    """Calculate all backtest metrics from equity curve and trades.

    Args:
        equity_curve: List of equity values over time.
        trades: List of trade dicts with keys: pnl, quantity, price, timestamp.
        risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%).
        benchmark_returns: Optional list of benchmark returns for alpha/beta.
        start_date: Start date for annualization.
        end_date: End date for annualization.
        trading_days_per_year: Days used for annualization (252 for equities, 365 for crypto).

    Returns:
        BacktestMetrics with all computed values.
    """
    metrics = BacktestMetrics()
    metrics.initial_capital = equity_curve[0] if equity_curve else 0
    metrics.final_equity = equity_curve[-1] if equity_curve else 0

    # Time period
    if start_date and end_date:
        days = (end_date - start_date).days
        metrics.start_date = start_date.isoformat()
        metrics.end_date = end_date.isoformat()
        metrics.trading_days = days
        years = days / 365.0 if days > 0 else 1
    else:
        years = len(equity_curve) / trading_days_per_year if equity_curve else 1
        years = max(years, 1 / trading_days_per_year)

    # ── Return metrics ──────────────────────────────────────────────────
    if metrics.initial_capital > 0:
        metrics.total_return = (metrics.final_equity - metrics.initial_capital) / metrics.initial_capital

    metrics.cumulative_return = metrics.total_return
    if years > 0 and metrics.initial_capital > 0:
        metrics.annualized_return = (
            (1 + metrics.total_return) ** (1 / years) - 1
        )

    # ── Returns series ──────────────────────────────────────────────────
    returns = _compute_returns(equity_curve)

    if returns:
        # Sharpe ratio
        metrics.sharpe_ratio = calculate_sharpe_ratio(
            returns, risk_free_rate=risk_free_rate
        )

        # Sortino ratio
        metrics.sortino_ratio = calculate_sortino_ratio(
            returns, risk_free_rate=risk_free_rate
        )

        # Drawdown metrics
        drawdown = calculate_drawdown(equity_curve)
        metrics.max_drawdown = max(drawdown) if drawdown else 0
        metrics.max_drawdown_duration_bars = _max_drawdown_duration(drawdown)

        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

    # ── Trade metrics ───────────────────────────────────────────────────
    if trades:
        metrics.total_trades = len(trades)

        pnls = [t.get("pnl", 0) for t in trades]
        # Trade-Return als Anteil am Handelsnotional (0.05 = +5 %);
        # Win-Rate/Profit-Factor bleiben PnL-basiert (Absolutbeträge).
        trade_pairs: list[tuple[float, float]] = [
            (t.get("pnl", 0), (t.get("price", 0.0) - t["entry_price"]) / t["entry_price"])
            for t in trades
            if t.get("entry_price")
        ]
        rets = [r for _, r in trade_pairs]
        metrics.avg_trade_return = sum(rets) / len(rets) if rets else 0
        metrics.best_trade = max(rets) if rets else 0
        metrics.worst_trade = min(rets) if rets else 0

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics.win_rate = len(wins) / len(pnls) if pnls else 0
        win_rets = [r for p, r in trade_pairs if p > 0]
        loss_rets = [r for p, r in trade_pairs if p <= 0]
        metrics.avg_win_return = sum(win_rets) / len(win_rets) if win_rets else 0
        metrics.avg_loss_return = sum(loss_rets) / len(loss_rets) if loss_rets else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        metrics.largest_win = max(wins) if wins else 0
        metrics.largest_loss = min(losses) if losses else 0

        # Consecutive wins/losses
        metrics.max_consecutive_wins = _max_consecutive(pnls, lambda p: p > 0)
        metrics.max_consecutive_losses = _max_consecutive(pnls, lambda p: p <= 0)

        # Average holding period
        holding_days = [t.get("holding_days", 1) for t in trades]
        metrics.avg_holding_days = (
            sum(holding_days) / len(holding_days) if holding_days else 0
        )

    # ── Risk-adjusted metrics ───────────────────────────────────────────
    if benchmark_returns and len(returns) == len(benchmark_returns):
        metrics.beta = _calculate_beta(returns, benchmark_returns)
        metrics.alpha = (
            metrics.annualized_return - risk_free_rate
            - metrics.beta * (np.mean(benchmark_returns) * trading_days_per_year - risk_free_rate)
        ) if benchmark_returns else 0.0

        # Information ratio
        active_returns = [r - b for r, b in zip(returns, benchmark_returns, strict=False)]
        active_std = float(np.std(active_returns)) if active_returns else 1
        if active_std > 0:
            metrics.information_ratio = (
                sum(active_returns) / len(active_returns) / active_std
            )

    return metrics


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Calculate annualized Sharpe ratio.

    Sharpe = (R_p - R_f) / σ_p * sqrt(periods_per_year)
    """
    if not returns or len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = [r - daily_rf for r in returns]

    mean_excess = sum(excess_returns) / len(excess_returns)
    std_excess = float(np.std(excess_returns, ddof=1))

    # Epsilon-Guard: (nahezu) konstante Equity-Kurven (z. B. 0 Trades,
    # nur Kosten-Effekte) liefern numerisch degenerate Standardabweichungen
    # im Float64-Unterlaufbereich → ohne Guard ergibt sich ein
    # unbeschränkt großes Sharpe statt "kein Risiko = kein Verhältnis".
    if std_excess < 1e-12:
        return 0.0

    return (mean_excess / std_excess) * (periods_per_year ** 0.5)


def calculate_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Calculate annualized Sortino ratio.

    Sortino = (R_p - R_f) / σ_downside * sqrt(periods_per_year)
    Only downside deviation is used.
    """
    if not returns or len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = [r - daily_rf for r in returns]

    mean_excess = sum(excess_returns) / len(excess_returns)

    # Downside deviation (only negative returns)
    negative_returns = [r for r in excess_returns if r < 0]
    if not negative_returns:
        return float("inf") if mean_excess > 0 else 0.0

    downside_variance = sum(r ** 2 for r in negative_returns) / len(excess_returns)
    downside_deviation = downside_variance ** 0.5

    # Epsilon-Guard, s. calculate_sharpe_ratio (degenerate Kurven).
    if downside_deviation < 1e-12:
        return 0.0

    return (mean_excess / downside_deviation) * (periods_per_year ** 0.5)


def calculate_drawdown(equity_curve: list[float]) -> list[float]:
    """Calculate drawdown series from equity curve.

    Drawdown at time t = (peak[0:t] - equity[t]) / peak[0:t]

    Returns list of drawdown values (all >= 0).
    """
    if not equity_curve:
        return []

    drawdowns = []
    peak = equity_curve[0]

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        drawdowns.append(dd)

    return drawdowns


def _compute_returns(equity_curve: list[float]) -> list[float]:
    """Compute periodic returns from equity curve."""
    if len(equity_curve) < 2:
        return []
    return [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] if equity_curve[i - 1] != 0 else 0.0
        for i in range(1, len(equity_curve))
    ]


def _max_drawdown_duration(drawdowns: list[float]) -> int:
    """Calculate the maximum drawdown duration in bars.

    One unit is one entry of the drawdown series, i.e. one bar of the
    equity curve (5m, 1m, or whatever the feed was resampled to) — not
    calendar days.
    """
    if not drawdowns:
        return 0

    max_duration = 0
    current_duration = 0

    for dd in drawdowns:
        if dd > 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def _max_consecutive(values: list[float], predicate) -> int:
    """Find maximum consecutive values satisfying predicate."""
    if not values:
        return 0

    max_count = 0
    current_count = 0

    for v in values:
        if predicate(v):
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


def _calculate_beta(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Calculate portfolio beta against benchmark."""
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 2:
        return 1.0

    s = np.array(strategy_returns[:n])
    b = np.array(benchmark_returns[:n])

    cov = np.cov(s, b)[0, 1]
    var_b = np.var(b, ddof=1)

    if var_b == 0:
        return 1.0

    return float(cov / var_b)
