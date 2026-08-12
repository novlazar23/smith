"""Base class for all calibration methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ConfigDict


@dataclass(frozen=True)
class CalibrationSample:
    """A single calibration sample.

    Attributes:
        raw_confidence: Agent's raw confidence score (before calibration).
        actual: Binary actual outcome (1.0 = correct/UP, 0.0 = incorrect/DOWN).
        direction: The agent's predicted direction (e.g. "UP", "DOWN").
        sample_id: Optional sample identifier.
    """

    raw_confidence: float
    actual: float
    direction: str
    sample_id: str = ""


@dataclass(frozen=True)
class CalibrationResult:
    """Result of calibration on a dataset.

    Attributes:
        method: Name of the calibration method.
        ece_before: Expected Calibration Error before calibration.
        ece_after: Expected Calibration Error after calibration.
        num_samples: Number of calibration samples used.
    """

    model_config = ConfigDict(frozen=True)

    method: str
    ece_before: float
    ece_after: float
    num_samples: int

    @property
    def improvement(self) -> float:
        """Improvement in ECE (positive means better)."""
        return self.ece_before - self.ece_after


class Calibrator(ABC):
    """Abstract base for all calibrators.

    Subclasses implement:
    - `fit`: Learn calibration parameters from calibration set.
    - `calibrate`: Apply calibration to raw probabilities.
    - `is_fitted`: Check if calibration parameters are learned.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the calibration method name."""

    @abstractmethod
    def fit(self, samples: Sequence[CalibrationSample]) -> None:
        """Learn calibration parameters from the calibration set.

        Args:
            samples: Calibration samples with raw confidence and actual outcomes.
        """

    @abstractmethod
    def calibrate(self, raw_confidence: float) -> float:
        """Calibrate a single raw confidence value.

        Args:
            raw_confidence: Raw confidence in [0, 1].

        Returns:
            Calibrated probability in [0, 1].

        Raises:
            RuntimeError: If not fitted yet.
        """

    @abstractmethod
    def calibrate_batch(
        self, raw_confidences: Sequence[float]
    ) -> list[float]:
        """Calibrate a batch of raw confidence values."""

    @property
    def is_fitted(self) -> bool:
        """Whether calibration parameters have been learned."""
        return False


def compute_ece(
    confidences: Sequence[float],
    actuals: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins confidences into n_bins equal-width intervals and computes
    weighted average of |accuracy - confidence| per bin.

    Args:
        confidences: Predicted probabilities/confidences.
        actuals: Binary actual outcomes.
        n_bins: Number of bins for histogram.

    Returns:
        ECE value (lower is better).
    """
    confidences = list(confidences)
    actuals = list(actuals)

    if not confidences or not actuals:
        return 0.0

    if len(confidences) != len(actuals):
        raise ValueError("confidences and actuals must have same length")

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        indices = [j for j in range(total) if lo <= confidences[j] < hi]
        if i == n_bins - 1:
            # Include the right boundary for the last bin
            indices = [j for j in range(total) if lo <= confidences[j] <= hi]

        if not indices:
            continue

        bin_conf = sum(confidences[j] for j in indices) / len(indices)
        bin_acc = sum(actuals[j] for j in indices) / len(indices)
        ece += len(indices) / total * abs(bin_acc - bin_conf)

    return ece
