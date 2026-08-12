"""Temperature Scaling Calibration.

Applies a single scalar temperature to the softmax:
P(calibrated) = softmax(logit(raw) / T).
One parameter: T (temperature), learned by minimizing NLL.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, log

from .base import CalibrationResult, CalibrationSample, Calibrator, compute_ece


class TemperatureCalibrator(Calibrator):
    """Temperature scaling calibrator.

    Attributes:
        temperature: Temperature parameter (learned from data, default 1.0 = no scaling).
        _fitted: Whether fit has been called.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._fitted = False

    @property
    def method_name(self) -> str:
        return "temperature"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _softmax(self, logits: list[float], temperature: float) -> list[float]:
        """Compute softmax with temperature scaling.

        For binary case: P(up) = 1 / (1 + exp((logit_up - logit_down) / T)).
        """
        # Scale logits by temperature
        scaled = [logit / temperature for logit in logits]
        max_val = max(scaled)
        exp_vals = [exp(s - max_val) for s in scaled]
        total = sum(exp_vals)
        return [e / total for e in exp_vals]

    @staticmethod
    def _logit(p: float) -> float:
        """Numerically stable logit function."""
        p = max(1e-7, min(1 - 1e-7, p))
        return log(p / (1 - p))

    def fit(self, samples: Sequence[CalibrationSample]) -> None:
        """Fit temperature parameter by minimizing NLL via gradient descent.

        Args:
            samples: Calibration samples with raw confidence and actual outcomes.
        """
        if len(samples) < 2:
            raise ValueError("Need at least 2 samples for temperature scaling")

        lr = 0.001
        n_iter = 2000

        # Binary calibration: each sample has raw_confidence (confidence in prediction)
        # and actual (1.0 = correct, 0.0 = incorrect)
        # We model: logit calibrated = logit(raw) / T
        # and train T to minimize NLL

        logits = [self._logit(s.raw_confidence) for s in samples]
        target_outcomes = [s.actual for s in samples]

        temperature_value = 1.0  # Initialize

        for _ in range(n_iter):
            # Compute gradient of NLL w.r.t. T
            grad = 0.0
            n = len(samples)
            for i in range(n):
                scaled_logit = logits[i] / temperature_value
                pred = 1.0 / (1.0 + exp(-scaled_logit))  # sigmoid
                grad += (pred - target_outcomes[i]) * (-logits[i] / (temperature_value * temperature_value))

            temperature_value -= lr * grad / n

            # Clamp temperature to reasonable range
            temperature_value = max(0.1, min(10.0, temperature_value))

        self.temperature = temperature_value
        self._fitted = True

    def calibrate(self, raw_confidence: float) -> float:
        """Apply temperature scaling to a single confidence value."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        logit_val = self._logit(max(1e-7, min(1 - 1e-7, raw_confidence)))
        scaled = logit_val / self.temperature
        return 1.0 / (1.0 + exp(-scaled))

    def calibrate_batch(self, raw_confidences: Sequence[float]) -> list[float]:
        """Calibrate a batch of confidence values."""
        return [self.calibrate(c) for c in raw_confidences]

    def evaluate(self, samples: Sequence[CalibrationSample]) -> CalibrationResult:
        """Evaluate calibration quality before and after fitting."""
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
