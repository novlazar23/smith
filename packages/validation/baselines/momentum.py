"""Momentum Baseline.

Uses recent return direction and magnitude to generate signals.
- Positive return → predict UP with confidence proportional to return
- Negative return → predict DOWN with confidence proportional to return
- Near-zero return → predict RANGE
"""

from __future__ import annotations

from typing import Any

from .base import Baseline, BaselinePrediction


class MomentumBaseline(Baseline):
    """Recent momentum baseline.

    Attributes:
        lookback: Number of periods to look back for return computation.
    """

    def __init__(self, lookback: int = 10) -> None:
        self.lookback = lookback

    @property
    def baseline_id(self) -> str:
        return "momentum"

    def predict(
        self,
        features: dict[str, float],
        historical_prices: list[float] | None = None,
        **kwargs: Any,
    ) -> BaselinePrediction:
        if not historical_prices or len(historical_prices) < self.lookback + 1:
            return BaselinePrediction(
                baseline_id=self.baseline_id,
                baseline_version=self.baseline_version,
                probabilities={"UP": 0.33, "DOWN": 0.33, "RANGE": 0.34},
                confidence=0.5,
                signal="MOMENTUM: Insufficient data",
            )

        start_price = historical_prices[-(self.lookback + 1)]
        end_price = historical_prices[-1]
        ret = (
            (end_price - start_price) / start_price if start_price != 0 else 0.0
        )

        if ret > 0.001:  # Small threshold to avoid noise
            prob_up = min(0.95, 0.5 + abs(ret) * 50)
            prob_down = max(0.02, 0.5 - abs(ret) * 50)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"MOMENTUM: Positive ({ret:.4f})"
        elif ret < -0.001:
            prob_down = min(0.95, 0.5 + abs(ret) * 50)
            prob_up = max(0.02, 0.5 - abs(ret) * 50)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"MOMENTUM: Negative ({ret:.4f})"
        else:
            prob_up = 0.33
            prob_down = 0.33
            prob_range = 0.34
            signal = f"MOMENTUM: Neutral ({ret:.4f})"

        return BaselinePrediction(
            baseline_id=self.baseline_id,
            baseline_version=self.baseline_version,
            probabilities={"UP": prob_up, "DOWN": prob_down, "RANGE": prob_range},
            confidence=0.5 + min(0.45, abs(ret) * 25),
            signal=signal,
        )
