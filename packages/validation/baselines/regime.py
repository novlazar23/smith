"""Regime Baseline.

Determines trend vs mean-reverting regime and predicts accordingly.
- Trend regime: predicts in direction of trend
- Mean-reverting regime: predicts reversal of recent move
"""

from __future__ import annotations

from typing import Any

from .base import Baseline, BaselinePrediction


class RegimeBaseline(Baseline):
    """Regime-aware baseline.

    Attributes:
        trend_window: Window for trend detection (default 50).
        mr_window: Window for mean-reversion detection (default 20).
        trend_threshold: Slope threshold for trend classification.
    """

    def __init__(
        self,
        trend_window: int = 50,
        mr_window: int = 20,
        trend_threshold: float = 0.0005,
    ) -> None:
        self.trend_window = trend_window
        self.mr_window = mr_window
        self.trend_threshold = trend_threshold

    @property
    def baseline_id(self) -> str:
        return "regime"

    def _detect_regime(
        self, prices: list[float]
    ) -> tuple[str, float]:
        """Detect current regime.

        Returns:
            (regime_type, confidence) where regime_type is "trend_up",
            "trend_down", or "mean_revert", and confidence is [0,1].
        """
        if len(prices) < self.trend_window + 1:
            return ("unknown", 0.5)

        # Linear regression slope for trend detection
        n = len(prices)
        x = list(range(n))
        x_mean = sum(x) / n
        y = prices[-self.trend_window :]
        y_mean = sum(y) / len(y)

        numerator = sum(
            (x[i] - x_mean) * (y[i] - y_mean) for i in range(len(y))
        )
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(y)))

        slope = numerator / denominator if denominator != 0 else 0.0

        if abs(slope) > self.trend_threshold:
            trend = "trend_up" if slope > 0 else "trend_down"
            confidence = min(0.95, 0.5 + abs(slope) * 10000)
            return (trend, confidence)
        else:
            # Mean-reverting: predict reversal of most recent move
            mr_price = prices[-self.mr_window]
            recent_return = (
                (prices[-1] - mr_price) / mr_price
                if mr_price != 0
                else 0.0
            )
            mr_dir = -1 if recent_return > 0 else 1  # Reversal
            return (
                "mean_revert" + ("_down" if mr_dir < 0 else "_up"),
                min(0.8, 0.5 + abs(recent_return) * 10),
            )

    def predict(
        self,
        features: dict[str, float],
        historical_prices: list[float] | None = None,
        **kwargs: Any,
    ) -> BaselinePrediction:
        if not historical_prices:
            return BaselinePrediction(
                baseline_id=self.baseline_id,
                baseline_version=self.baseline_version,
                probabilities={"UP": 0.33, "DOWN": 0.33, "RANGE": 0.34},
                confidence=0.5,
                signal="REGIME: Insufficient data",
            )

        regime, confidence = self._detect_regime(historical_prices)

        if regime == "unknown":
            return BaselinePrediction(
                baseline_id=self.baseline_id,
                baseline_version=self.baseline_version,
                probabilities={"UP": 0.33, "DOWN": 0.33, "RANGE": 0.34},
                confidence=confidence,
                signal="REGIME: Insufficient data",
            )

        if "trend_up" in regime:
            prob_up = min(0.9, 0.5 + confidence * 0.4)
            prob_down = max(0.05, 0.5 - confidence * 0.4)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"REGIME: Trend UP (conf={confidence:.2f})"
        elif "trend_down" in regime:
            prob_down = min(0.9, 0.5 + confidence * 0.4)
            prob_up = max(0.05, 0.5 - confidence * 0.4)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"REGIME: Trend DOWN (conf={confidence:.2f})"
        elif "mean_revert_up" in regime:
            prob_up = min(0.85, 0.5 + confidence * 0.35)
            prob_down = max(0.05, 0.5 - confidence * 0.35)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"REGIME: Mean-revert UP (conf={confidence:.2f})"
        else:
            prob_down = min(0.85, 0.5 + confidence * 0.35)
            prob_up = max(0.05, 0.5 - confidence * 0.35)
            prob_range = max(0.0, 1.0 - prob_up - prob_down)
            signal = f"REGIME: Mean-revert DOWN (conf={confidence:.2f})"

        return BaselinePrediction(
            baseline_id=self.baseline_id,
            baseline_version=self.baseline_version,
            probabilities={"UP": prob_up, "DOWN": prob_down, "RANGE": prob_range},
            confidence=confidence,
            signal=signal,
        )
