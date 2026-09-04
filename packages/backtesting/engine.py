"""Backtest Engine — Candle-by-Candle execution using PaperExecutor.

Iterates over historical candles, calls strategy on_bar(), submits orders
through PaperExecutor, tracks equity curve, and computes metrics.
"""

from __future__ import annotations

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
        self._round_trips: list[dict[str, Any]] = []
        self._cycle_start: dict[str, tuple[int, datetime]] = {}
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
        self._round_trips = []
        self._cycle_start = {}

        # Warmup phase — process candles but don't trade
        for i in range(min(warmup, len(candles))):
            strategy.on_bar(candles[i])

        # Trading phase — candle by candle
        equity_curve: list[float] = []
        equity_curve.append(self.config.initial_capital)

        for i in range(warmup, len(candles)):
            candle = candles[i]
            signal = strategy.on_bar(candle)

            if signal is not None:
                self._execute_signal(signal, candle, i)

            self._apply_risk_exits(candle, i)

            # Record equity at end of bar (positions marked at the bar close)
            equity = self._marked_equity(candle)
            equity_curve.append(equity)

            # Track state
            snapshot = PortfolioSnapshot(
                timestamp=candle.timestamp,
                initial_capital=self.config.initial_capital,
                cash=self._account.cash,
                market_value=equity - self._account.cash,
                total_equity=equity,
            )
            self._snapshots.append(snapshot)
            self._candles_processed += 1

        # Collect trade data for metrics: closed round trips + open positions
        trade_data: list[dict[str, Any]] = []
        for rt in self._round_trips:
            trade_data.append(
                {
                    "pnl": rt["pnl"],
                    "entry_price": rt["entry_price"],
                    "quantity": rt["quantity"],
                    "price": rt["exit_price"],
                    "timestamp": rt["exit_time"],
                    "holding_days": rt["holding_days"],
                }
            )

        # For trades that are still open, mark them at the final close
        current_price = candles[-1].close if candles else 0
        current_ts = _ensure_aware(candles[-1].timestamp) if candles else None
        for symbol, pos in self._account.positions.items():
            if pos.quantity > 0:
                price = current_price if symbol == self.config.symbol else pos.avg_price
                cycle = self._cycle_start.get(symbol)
                opened_at = _ensure_aware(cycle[1]) if cycle else current_ts
                holding_days = (
                    (current_ts - opened_at).total_seconds() / 86400.0
                    if current_ts and opened_at
                    else 0.0
                )
                trade_data.append(
                    {
                        "pnl": (price - pos.avg_price) * pos.quantity,
                        "entry_price": pos.avg_price,
                        "quantity": pos.quantity,
                        "price": price,
                        "timestamp": current_ts.isoformat() if current_ts else "",
                        "holding_days": round(holding_days, 4),
                        "unrealized": True,
                    }
                )

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
                "final_equity": self._marked_equity(candles[-1]),
                "initial_capital": self.config.initial_capital,
                "round_trips": [dict(rt) for rt in self._round_trips],
                "open_positions": [
                    {
                        "symbol": symbol,
                        "quantity": pos.quantity,
                        "avg_price": pos.avg_price,
                        "entry_bar": self._cycle_start[symbol][0] if symbol in self._cycle_start else None,
                        "holding_bars": (
                            (len(candles) - 1) - self._cycle_start[symbol][0]
                            if symbol in self._cycle_start
                            else None
                        ),
                        "unrealized_pnl": (
                            (current_price if symbol == self.config.symbol else pos.avg_price)
                            - pos.avg_price
                        )
                        * pos.quantity,
                    }
                    for symbol, pos in self._account.positions.items()
                    if pos.quantity > 0
                ],
            }
        )

    def _marked_equity(self, candle: Candle) -> float:
        """Kontostand mit Positionen zum Schlusskurs der Kerze bewertet."""
        assert self._account is not None
        equity = self._account.cash
        for symbol, pos in self._account.positions.items():
            price = candle.close if symbol == self.config.symbol else pos.avg_price
            equity += pos.quantity * price
        return equity

    def _apply_risk_exits(self, candle: Candle, bar_index: int) -> None:
        """Risko-Schicht: Stop-Loss und Max-Haltezeit auf offenen Positionen.

        Läuft nach dem Strategie-Signal derselben Kerze und schließt die
        Position zum Schlusskurs, wenn

        - ``close <= avg_price * (1 - stop_loss_pct)`` (Stop-Loss), oder
        - die Position ``max_holding_bars`` Bars offen ist (Max-Haltezeit).

        Beide Regeln sind deaktiviert, wenn der jeweilige Config-Wert None
        ist. Der Exit-Grund wird im Round-Trip-Record als ``exit_reason``
        gespeichert.
        """
        if self._account is None:
            return
        pos = self._account.positions.get(self.config.symbol)
        if pos is None or pos.quantity <= 0:
            return
        stop = self.config.stop_loss_pct
        if (
            stop is not None
            and pos.avg_price > 0
            and candle.close <= pos.avg_price * (1.0 - stop)
        ):
            self._close_position_at_market(candle, bar_index, f"stop_loss_{stop:.2f}")
            return
        max_bars = self.config.max_holding_bars
        if max_bars is not None:
            cycle = self._cycle_start.get(self.config.symbol)
            if cycle is not None and bar_index - cycle[0] >= max_bars:
                self._close_position_at_market(candle, bar_index, f"max_holding_{max_bars}")

    def _close_position_at_market(
        self, candle: Candle, bar_index: int, reason: str
    ) -> None:
        """Schließt die komplette Position zum Schlusskurs (Markterfüllung).

        Der PnL wird zum Marktpreis realisiert (Slippage/Commission wie bei
        jeder Order); ein Round-Trip-Record mit Bar-Timestamps wird angelegt.
        """
        if self._account is None:
            return
        pos = self._account.positions.get(self.config.symbol)
        if pos is None or pos.quantity <= 0:
            return
        quantity = pos.quantity
        avg_price = pos.avg_price
        cycle = self._cycle_start.pop(self.config.symbol, None)
        entry_bar, entry_ts = cycle if cycle else (bar_index, candle.timestamp)
        try:
            trade = self.executor.submit_order(
                account=self._account,
                instrument=self.config.symbol,
                direction=TradeDirection.SELL,
                quantity=quantity,
                price=candle.close,
            )
        except ValueError:
            self._cycle_start[self.config.symbol] = (entry_bar, entry_ts)
            return
        if trade.filled_quantity <= 0:
            self._cycle_start[self.config.symbol] = (entry_bar, entry_ts)
            return
        pnl = (trade.filled_price - avg_price) * trade.filled_quantity
        holding_days = (candle.timestamp - entry_ts).total_seconds() / 86400.0
        self._round_trips.append(
            {
                "instrument": self.config.symbol,
                "entry_bar": entry_bar,
                "exit_bar": bar_index,
                "entry_time": entry_ts,
                "exit_time": candle.timestamp,
                "entry_price": avg_price,
                "exit_price": trade.filled_price,
                "quantity": trade.filled_quantity,
                "pnl": pnl,
                "holding_bars": bar_index - entry_bar,
                "holding_days": round(holding_days, 4),
                "exit_reason": reason,
            }
        )
        self._trades.append(
            Trade(
                trade_id=trade.trade_id,
                instrument=self.config.symbol,
                side="sell",
                quantity=trade.filled_quantity,
                price=trade.filled_price,
                commission=trade.commission,
                timestamp=candle.timestamp,
                pnl=pnl,
            )
        )

    def _execute_signal(
        self,
        signal: Any,
        candle: Candle,
        bar_index: int,
    ) -> None:
        """Execute a strategy signal through the PaperExecutor.

        Args:
            signal: StrategySignal with action, confidence, position_size.
            candle: Current candle with price data.
            bar_index: Index der Kerze im Feed (für Round-Trip-Tracking).
        """
        if self._account is None:
            return

        if not hasattr(signal, "action"):
            return

        action = signal.action
        action_str = str(action) if not hasattr(action, "value") else action.value

        price = candle.close
        equity = self._marked_equity(candle)

        if action_str in ("buy", "BUY"):
            had_position = (
                self.config.symbol in self._account.positions
                and self._account.positions[self.config.symbol].quantity > 0
            )
            if had_position and not self.config.allow_pyramiding:
                # Flatsize: eine Position pro Symbol, kein Nachkauf
                return

            size_pct = getattr(signal, "position_size", 0.1)
            if size_pct <= 0:
                size_pct = 0.1

            max_notional = equity * size_pct
            quantity = max_notional / price if price > 0 else 0

            if quantity <= 0:
                return

            try:
                trade = self.executor.submit_order(
                    account=self._account,
                    instrument=self.config.symbol,
                    direction=TradeDirection.BUY,
                    quantity=quantity,
                    price=price,
                )
            except ValueError:
                return
            if trade.filled_quantity > 0 and not had_position:
                self._cycle_start[self.config.symbol] = (bar_index, candle.timestamp)
            if trade.filled_quantity > 0:
                self._trades.append(
                    Trade(
                        trade_id=trade.trade_id,
                        instrument=self.config.symbol,
                        side="buy",
                        quantity=trade.filled_quantity,
                        price=trade.filled_price,
                        commission=trade.commission,
                        timestamp=candle.timestamp,
                    )
                )

        elif action_str in ("sell", "SELL"):
            if self.config.symbol in self._account.positions:
                self._close_position_at_market(candle, bar_index, "signal")
