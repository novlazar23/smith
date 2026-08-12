"""RSI Baseline.

Uses RSI (Relative Strength Index) to generate signals.
- RSI < 30 (oversold) → predict UP
- RSI > 70 (overbought) → predict DOWN
- RSI 30-70 → predict RANGE
"""

from __future__ import annotations

from typing import Any

from .base import Baseline, BaselinePrediction


class RSIBaseline(Baseline):
    """RSI-based baseline.

    Attributes:
        period: RSI calculation period (default 14).
        oversold: RSI threshold for oversold (default 30).
        overbought: RSI threshold for overbought (default 70).
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def baseline_id(self) -> str:
        return "rsi"

    def _compute_rsi(self, prices: list[float]) -> float:
        """Compute RSI from price history."""
        if len(prices) < self.period + 1:
            return 50.0  # Neutral when not enough data

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [max(0, d) for d in deltas[-self.period :]]
        losses = [max(0, -d) for d in deltas[-self.period :]]

        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def predict(
        self,
        features: dict[str, float],
        historical_prices: list[float] | None = None,
        **kwargs: Any,
    ) -> BaselinePrediction:
        rsi = (
            self._compute_rsi(historical_prices)
            if historical_prices
            else 50.0
        )

        if rsi < self.oversold:
            prob_up = min(0.95, 0.5 + (self.oversold - rsi) / 100)
            prob_down = max(0.02, 0.5 - (self.oversold - rsi) / 100)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"RSI: Oversold ({rsi:.1f}) → UP"
        elif rsi > self.overbought:
            prob_down = min(0.95, 0.5 + (rsi - self.overbought) / 100)
            prob_up = max(0.02, 0.5 - (rsi - self.overbought) / 100)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"RSI: Overbought ({rsi:.1f}) → DOWN"
        else:
            prob_up = 0.33
            prob_down = 0.33
            prob_range = 0.34
            signal = f"RSI: Neutral ({rsi:.1f})"

        return BaselinePrediction(
            baseline_id=self.baseline_id,
            baseline_version=self.baseline_version,
            probabilities={"UP": prob_up, "DOWN": prob_down, "RANGE": prob_range},
            confidence=0.5 + min(0.45, abs(rsi - 50) / 100),
            signal=signal,
        )
