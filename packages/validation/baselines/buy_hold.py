"""Buy & Hold Baseline.

Always predicts UP with maximum confidence.
This is the simplest possible benchmark — if agents can't beat this
on a bull market, they're not useful.
"""

from __future__ import annotations

from typing import Any

from .base import Baseline, BaselinePrediction


class BuyHoldBaseline(Baseline):
    """Always predicts UP with confidence 1.0.

    This baseline serves as the absolute floor: any agent that performs
    worse than 100% UP prediction on a bull market is worse than doing nothing.
    """

    @property
    def baseline_id(self) -> str:
        return "buy_hold"

    def predict(
        self,
        features: dict[str, float],
        historical_prices: list[float] | None = None,
        **kwargs: Any,
    ) -> BaselinePrediction:
        return BaselinePrediction(
            baseline_id=self.baseline_id,
            baseline_version=self.baseline_version,
            probabilities={"UP": 1.0, "DOWN": 0.0, "RANGE": 0.0},
            confidence=1.0,
            signal="BUY_AND_HOLD: Always long",
        )
