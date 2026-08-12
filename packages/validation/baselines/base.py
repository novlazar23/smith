"""Base class for all baseline strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class BaselinePrediction:
    """Prediction from a baseline strategy.

    Attributes:
        baseline_id: Identifier for the baseline (e.g. "buy_hold", "ma_cross").
        baseline_version: Version string.
        probabilities: Direction probabilities {UP, DOWN, RANGE}.
        confidence: Confidence in [0, 1].
        signal: Human-readable signal description.
    """

    model_config = ConfigDict(frozen=True)

    baseline_id: str
    baseline_version: str
    probabilities: dict[str, float]
    confidence: float
    signal: str = ""


class Baseline(ABC):
    """Abstract base for all baseline strategies.

    Subclasses implement `predict()` which takes a feature dict and optional
    historical price data, and returns a BaselinePrediction.
    """

    @property
    @abstractmethod
    def baseline_id(self) -> str:
        """Return the baseline identifier."""

    @property
    def baseline_version(self) -> str:
        return "1.0.0"

    @abstractmethod
    def predict(
        self,
        features: dict[str, float],
        historical_prices: list[float] | None = None,
        **kwargs: Any,
    ) -> BaselinePrediction:
        """Generate a prediction for the given features and prices.

        Args:
            features: Feature dict (may contain any baseline-specific keys).
            historical_prices: Optional list of recent closing prices.
            **kwargs: Additional context (e.g. instrument, horizon).

        Returns:
            BaselinePrediction with probabilities, confidence, signal.
        """
        ...
