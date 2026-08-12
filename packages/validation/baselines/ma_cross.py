"""Moving Average Crossover Baseline.

Uses short-term vs long-term MA crossover to generate directional signals.
- Short MA above Long MA → predict UP
- Short MA below Long MA → predict DOWN
- Difference magnitude → confidence
"""

from __future__ import annotations

from typing import Any

from .base import Baseline, BaselinePrediction


class MACrossBaseline(Baseline):
    """MA crossover baseline with configurable windows.

    Attributes:
        short_window: Short MA window (default 20).
        long_window: Long MA window (default 50).
        min_confidence: Floor for confidence when signals are weak.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        min_confidence: float = 0.5,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.min_confidence = min_confidence

    @property
    def baseline_id(self) -> str:
        return "ma_cross"

    def predict(
        self,
        features: dict[str, float],
        historical_prices: list[float] | None = None,
        **kwargs: Any,
    ) -> BaselinePrediction:
        if not historical_prices or len(historical_prices) < self.long_window:
            # Not enough data — default to neutral
            return BaselinePrediction(
                baseline_id=self.baseline_id,
                baseline_version=self.baseline_version,
                probabilities={"UP": 0.33, "DOWN": 0.33, "RANGE": 0.34},
                confidence=self.min_confidence,
                signal="MA_CROSS: Insufficient data",
            )

        short_ma = (
            sum(historical_prices[-self.short_window:]) / self.short_window
        )
        long_ma = sum(historical_prices[-self.long_window:]) / self.long_window

        spread = (short_ma - long_ma) / long_ma if long_ma != 0 else 0.0

        if spread > 0:
            prob_up = min(0.95, 0.5 + abs(spread) * 10)
            prob_down = max(0.02, 0.5 - abs(spread) * 10)
            prob_range = 1.0 - prob_up - prob_down
            signal = f"MA_CROSS: Bullish (spread={spread:.4f})"
        else:
            prob_down = min(0.95, 0.5 + abs(spread) * 10)
            prob_up = max(0.02, 0.5 - abs(spread) * 10)
            prob_range = 1.0 - prob_up - prob_down
            signal = f"MA_CROSS: Bearish (spread={spread:.4f})"

        prob_range = max(0.0, prob_range)

        return BaselinePrediction(
            baseline_id=self.baseline_id,
            baseline_version=self.baseline_version,
            probabilities={"UP": prob_up, "DOWN": prob_down, "RANGE": prob_range},
            confidence=self.min_confidence + min(0.45, abs(spread) * 5),
            signal=signal,
        )
