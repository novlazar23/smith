"""Platt Scaling Calibration.

Fits a logistic regression model: P(calibrated) = sigmoid(a * logit(raw) + b).
Two parameters: a (slope) and b (intercept), learned by maximizing likelihood.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, log

from .base import CalibrationResult, CalibrationSample, Calibrator, compute_ece


class PlattCalibrator(Calibrator):
    """Platt scaling calibrator: sigmoid(a * logit(p) + b).

    Attributes:
        a: Slope parameter (learned from data).
        b: Intercept parameter (learned from data).
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self._fitted = False

    @property
    def method_name(self) -> str:
        return "platt"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            return 1.0 / (1.0 + exp(-x))
        else:
            ez = exp(x)
            return ez / (1.0 + ez)

    @staticmethod
    def _logit(p: float) -> float:
        """Numerically stable logit function."""
        p = max(1e-7, min(1 - 1e-7, p))
        return log(p / (1 - p))

    def fit(self, samples: Sequence[CalibrationSample]) -> None:
        """Fit Platt scaling parameters using gradient descent on log-loss.

        Args:
            samples: Calibration samples with raw confidence and actual outcomes.
        """
        if len(samples) < 2:
            raise ValueError("Need at least 2 samples for Platt scaling")

        # Prepare training data: logit(raw_confidence) -> actual
        train_x = [self._logit(s.raw_confidence) for s in samples]
        train_y = [s.actual for s in samples]
        n = len(train_x)

        # Gradient descent
        lr = 0.01
        n_iter = 1000

        # Initialize
        a, b = 1.0, 0.0

        for _ in range(n_iter):
            grad_a = 0.0
            grad_b = 0.0
            for i in range(n):
                pred = self._sigmoid(a * train_x[i] + b)
                error = pred - train_y[i]
                grad_a += error * train_x[i]
                grad_b += error
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self.a = a
        self.b = b
        self._fitted = True

    def calibrate(self, raw_confidence: float) -> float:
        """Apply Platt scaling to a single confidence value."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        logit_val = self._logit(max(1e-7, min(1 - 1e-7, raw_confidence)))
        return self._sigmoid(self.a * logit_val + self.b)

    def calibrate_batch(self, raw_confidences: Sequence[float]) -> list[float]:
        """Calibrate a batch of confidence values."""
        return [self.calibrate(c) for c in raw_confidences]

    def evaluate(self, samples: Sequence[CalibrationSample]) -> CalibrationResult:
        """Evaluate calibration quality before and after fitting.

        Args:
            samples: Calibration samples.

        Returns:
            CalibrationResult with ECE before/after.
        """
        self.fit(samples)
        raw_confs = [s.raw_confidence for s in samples]
        actuals = [s.actual for s in samples]

        ece_before = compute_ece(raw_confs, actuals)

        cal_confs = self.calibrate_batch(raw_confs)
        ece_after = compute_ece(cal_confs, actuals)

        return CalibrationResult(
            method=self.method_name,
            ece_before=ece_before,
            ece_after=ece_after,
            num_samples=len(samples),
        )
