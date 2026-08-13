"""Backtest Engine — Candle-by-Candle execution using PaperExecutor.

Iterates over historical candles, calls strategy on_bar(), submits orders
through PaperExecutor, tracks equity curve, and computes metrics.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from packages.paper import PaperAccount, PaperExecutor, TradeDirection

from .core import BacktestConfig, BacktestResult, Candle, PortfolioSnapshot, Trade
from .datafeed import DataFeed, compute_indicators
from .metrics import calculate_backtest_metrics


def _ensure_aware(dt: datetime | str) -> datetime:
    """Ensure a datetime is timezone-aware (default to UTC).

    Accepts either a datetime or an ISO-8601 string.
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


class BacktestEngine:
    """Main backtesting engine — iterates candles, evaluates strategy, executes orders.

    Uses PaperExecutor from packages/paper for simulated order execution
    with slippage and commission.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        executor: PaperExecutor | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.executor = executor or PaperExecutor(
            initial_cash=self.config.initial_capital,
            default_slippage_pct=self.config.slippage_bps / 10_000,
            default_commission_pct=self.config.commission_rate,
            max_position_size_pct=self.config.max_position_size,
        )
        self._account: PaperAccount | None = None
        self._snapshots: list[PortfolioSnapshot] = []
        self._trades: list[Trade] = []
        self._candles_processed: int = 0

    def run(
        self,
        data_feed: DataFeed,
        strategy: Any,
        warmup_bars: int | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run backtest over the given data feed and strategy.

        Args:
            data_feed: DataFeed providing historical candles.
            strategy: Strategy with on_bar(candle) -> signal method.
            warmup_bars: Bars to skip for warmup (uses config if None).
            **kwargs: Extra data for indicators (e.g., benchmark_returns).

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        warmup = warmup_bars or self.config.warmup_bars
        candles = data_feed.get_candles(self.config.symbol)

        if not candles:
            raise ValueError("No candles in data feed")

        # Compute indicators and attach to candles
        candles = compute_indicators(candles)

        # Setup account
        self._account = self.executor.create_account("backtest")

        # Warmup phase — process candles but don't trade
        for i in range(min(warmup, len(candles))):
            strategy.on_bar(candles[i])

        # Trading phase — candle by candle
        equity_curve: list[float] = []
        equity_curve.append(self._account.equity)

        for i in range(warmup, len(candles)):
            candle = candles[i]
            signal = strategy.on_bar(candle)

            if signal is not None:
                self._execute_signal(signal, candle)

            # Record equity at end of bar
            equity_curve.append(self._account.equity)

            # Track state
            snapshot = PortfolioSnapshot(
                timestamp=candle.timestamp,
                initial_capital=self.config.initial_capital,
                cash=self._account.cash,
                market_value=self._account.equity - self._account.cash,
                total_equity=self._account.equity,
            )
            self._snapshots.append(snapshot)
            self._candles_processed += 1

        # Collect trade data for metrics
        trade_data: list[dict[str, Any]] = []
        for pos in self._account.positions.values():
            if pos.realized_pnl != 0 or pos.quantity == 0:
                closed_at = _ensure_aware(pos.closed_at) if pos.closed_at else None
                opened_at = _ensure_aware(pos.opened_at) if pos.opened_at else None
                trade_data.append({
                    "pnl": pos.realized_pnl,
                    "quantity": pos.quantity,
                    "price": pos.avg_price,
                    "timestamp": (
                        closed_at.isoformat() if closed_at
                        else _ensure_aware(candles[-1].timestamp).isoformat()
                    ),
                    "holding_days": (
                        (closed_at - opened_at).days
                        if closed_at and opened_at
                        else 1
                    ),
                })

        # For trades that are still open, estimate unrealized PnL
        current_price = candles[-1].close if candles else 0
        current_ts = _ensure_aware(candles[-1].timestamp) if candles else None
        for pos in self._account.positions.values():
            if pos.quantity > 0:
                opened_at = _ensure_aware(pos.opened_at) if pos.opened_at else None
                holding_days = (current_ts - opened_at).days if opened_at and current_ts else 1
                trade_data.append({
                    "pnl": pos.unrealized_pnl,
                    "quantity": pos.quantity,
                    "price": current_price,
                    "timestamp": current_ts.isoformat() if current_ts else "",
                    "holding_days": holding_days,
                    "unrealized": True,
                })

        # PaperAccount tracks positions, not individual trade objects

        # Calculate metrics
        benchmark_returns = kwargs.get("benchmark_returns")
        start_dt = _ensure_aware(candles[0].timestamp)
        end_dt = _ensure_aware(candles[-1].timestamp)
        metrics = calculate_backtest_metrics(
            equity_curve=equity_curve,
            trades=trade_data,
            risk_free_rate=0.02,
            benchmark_returns=benchmark_returns,
            start_date=start_dt,
            end_date=end_dt,
        )

        return BacktestResult(
            config=self.config,
            candles=candles,
            snapshots=self._snapshots,
            trades=self._trades,
            metrics=metrics.to_dict(),
            metadata={
                "equity_curve": equity_curve,
                "candles_processed": self._candles_processed,
                "total_trades": len(trade_data),
                "final_equity": self._account.equity,
                "initial_capital": self.config.initial_capital,
            },
        )

    def _execute_signal(
        self,
        signal: Any,
        candle: Candle,
    ) -> None:
        """Execute a strategy signal through the PaperExecutor.

        Args:
            signal: StrategySignal with action, confidence, position_size.
            candle: Current candle with price data.
        """
        if self._account is None:
            return

        if not hasattr(signal, "action"):
            return

        action = signal.action
        action_str = str(action) if not hasattr(action, "value") else action.value

        price = candle.close
        equity = self._account.equity

        if action_str in ("buy", "BUY"):
            size_pct = getattr(signal, "position_size", 0.1)
            if size_pct <= 0:
                size_pct = 0.1

            max_notional = equity * size_pct
            quantity = max_notional / price if price > 0 else 0

            if quantity <= 0:
                return

            try:
                self.executor.submit_order(
                    account=self._account,
                    instrument=self.config.symbol,
                    direction=TradeDirection.BUY,
                    quantity=quantity,
                    price=price,
                )
                self._trades.append(
                    Trade(
                        trade_id=str(uuid.uuid4()),
                        instrument=self.config.symbol,
                        side="buy",
                        quantity=quantity,
                        price=price,
                        commission=0.0,
                        timestamp=datetime.now(UTC),
                    )
                )
            except ValueError:
                pass

        elif action_str in ("sell", "SELL"):
            if self.config.symbol in self._account.positions:
                try:
                    self.executor.close_position(
                        account=self._account,
                        instrument=self.config.symbol,
                    )
                    self._trades.append(
                        Trade(
                            trade_id=str(uuid.uuid4()),
                            instrument=self.config.symbol,
                            side="sell",
                            quantity=0,
                            price=price,
                            commission=0.0,
                            timestamp=datetime.now(UTC),
                        )
                    )
                except ValueError:
                    pass
