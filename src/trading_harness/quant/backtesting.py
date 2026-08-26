"""Backtesting Engine (Phase 8).

Einfache Backtesting-Engine für Handelsstrategien auf historischen Daten.
Trackt PnL, Drawdown, Win Rate, Sharpe Ratio.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Handelssignal."""
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class BacktestTrade:
    """Einzelner Trade im Backtest."""
    entry_time: str
    entry_price: float
    direction: str  # "long" oder "short"
    size: float = 1.0
    exit_time: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Ergebnis eines Backtests."""
    symbol: str
    timeframe: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_pnl: float
    profit_factor: float
    trades: list[BacktestTrade]
    equity_curve: list[float]


class BacktestEngine:
    """Einfache Backtesting-Engine — nur stdlib."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(
        self,
        candles: list[dict],
        strategy: Callable[[list[dict], int], Signal],
        symbol: str = "",
        timeframe: str = "1m",
    ) -> BacktestResult:
        """Führt Backtest mit gegebener Strategie aus.

        Args:
            candles: OHLCV-Kerzenreihe
            strategy: Funktion (candles, index) → Signal
            symbol: Symbol-Name
            timeframe: Timeframe

        Returns:
            BacktestResult mit allen Statistiken
        """
        if len(candles) < 2:
            return self._empty_result(symbol, timeframe)

        trades: list[BacktestTrade] = []
        equity_curve: list[float] = [self.initial_capital]
        capital = self.initial_capital
        current_trade: BacktestTrade | None = None
        peak_equity = capital

        for i in range(1, len(candles)):
            signal = strategy(candles, i)
            price = candles[i]["close"]
            time_str = candles[i].get("time", f"t_{i}")

            if current_trade is None:
                # Entry
                if signal == Signal.LONG:
                    size = (capital * self.risk_per_trade) / self.stop_loss_pct
                    current_trade = BacktestTrade(
                        entry_time=time_str, entry_price=price,
                        direction="long", size=size,
                    )
                elif signal == Signal.SHORT:
                    size = (capital * self.risk_per_trade) / self.stop_loss_pct
                    current_trade = BacktestTrade(
                        entry_time=time_str, entry_price=price,
                        direction="short", size=size,
                    )
            else:
                # Exit Check
                exit_signal = self._check_exit(current_trade, price, time_str)
                if exit_signal:
                    current_trade.exit_time = time_str
                    current_trade.exit_price = price
                    if current_trade.direction == "long":
                        current_trade.pnl = (price - current_trade.entry_price) * current_trade.size
                    else:
                        current_trade.pnl = (current_trade.entry_price - price) * current_trade.size
                    current_trade.pnl_pct = current_trade.pnl / capital if capital > 0 else 0.0
                    capital += current_trade.pnl
                    trades.append(current_trade)
                    current_trade = None

            equity_curve.append(capital)
            peak_equity = max(peak_equity, capital)

        # Close open position at last candle
        if current_trade:
            last_price = candles[-1]["close"]
            last_time = candles[-1].get("time", f"t_{len(candles)-1}")
            current_trade.exit_time = last_time
            current_trade.exit_price = last_price
            if current_trade.direction == "long":
                current_trade.pnl = (last_price - current_trade.entry_price) * current_trade.size
            else:
                current_trade.pnl = (current_trade.entry_price - last_price) * current_trade.size
            current_trade.pnl_pct = current_trade.pnl / capital if capital > 0 else 0.0
            capital += current_trade.pnl
            trades.append(current_trade)
            equity_curve.append(capital)

        return self._compute_stats(trades, equity_curve, symbol, timeframe)

    def _check_exit(self, trade: BacktestTrade, price: float, time_str: str) -> bool:
        """Prüft ob Stop-Loss oder Take-Profit erreicht."""
        if trade.direction == "long":
            if price <= trade.entry_price * (1 - self.stop_loss_pct):
                return True
            if price >= trade.entry_price * (1 + self.take_profit_pct):
                return True
        else:
            if price >= trade.entry_price * (1 + self.stop_loss_pct):
                return True
            if price <= trade.entry_price * (1 - self.take_profit_pct):
                return True
        return False

    def _compute_stats(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        symbol: str,
        timeframe: str,
    ) -> BacktestResult:
        """Berechnet Backtest-Statistiken."""
        total = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / total if total > 0 else 0.0
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = total_pnl / self.initial_capital if self.initial_capital > 0 else 0.0

        # Max Drawdown
        peak = equity_curve[0] if equity_curve else self.initial_capital
        max_dd = 0.0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Sharpe Ratio (simplified, assuming risk-free rate = 0)
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                r = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(r)
        sharpe = 0.0
        if returns:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_r = math.sqrt(var_r) if var_r > 0 else 1.0
            sharpe = mean_r / std_r

        avg_pnl = total_pnl / total if total > 0 else 0.0
        sum_gains = sum(t.pnl for t in winning)
        sum_losses = abs(sum(t.pnl for t in losing))
        profit_factor = min(sum_gains / sum_losses, 10.0) if sum_losses > 0 else (10.0 if sum_gains > 0 else 0.0)

        return BacktestResult(
            symbol=symbol, timeframe=timeframe,
            total_trades=total, winning_trades=len(winning),
            losing_trades=len(losing), win_rate=win_rate,
            total_pnl=total_pnl, total_pnl_pct=total_pnl_pct,
            max_drawdown=max_dd, sharpe_ratio=sharpe,
            avg_trade_pnl=avg_pnl, profit_factor=profit_factor,
            trades=trades, equity_curve=equity_curve,
        )

    def _empty_result(self, symbol: str, timeframe: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol, timeframe=timeframe,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_pnl=0.0, total_pnl_pct=0.0,
            max_drawdown=0.0, sharpe_ratio=0.0, avg_trade_pnl=0.0,
            profit_factor=0.0, trades=[], equity_curve=[self.initial_capital],
        )

    def simple_moving_average_strategy(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
    ) -> Callable[[list[dict], int], Signal]:
        """Factory für SMA-Crossover-Strategie."""
        def strategy(candles: list[dict], index: int) -> Signal:
            if index < slow_period:
                return Signal.HOLD
            closes = [c["close"] for c in candles]
            fast_ma = sum(closes[index - fast_period + 1 : index + 1]) / fast_period
            slow_ma = sum(closes[index - slow_period + 1 : index + 1]) / slow_period
            if fast_ma > slow_ma:
                return Signal.LONG
            elif fast_ma < slow_ma:
                return Signal.SHORT
            return Signal.HOLD
        return strategy

    def rsi_strategy(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> Callable[[list[dict], int], Signal]:
        """Factory für RSI-Strategie."""
        def strategy(candles: list[dict], index: int) -> Signal:
            if index < period + 1:
                return Signal.HOLD
            closes = [c["close"] for c in candles]
            gains = []
            losses = []
            for i in range(index - period + 1, index + 1):
                change = closes[i] - closes[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0.0)
                else:
                    gains.append(0.0)
                    losses.append(abs(change))
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            if rsi < oversold:
                return Signal.LONG
            elif rsi > overbought:
                return Signal.SHORT
            return Signal.HOLD
        return strategy
