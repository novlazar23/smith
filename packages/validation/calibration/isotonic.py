"""Isotonic Regression Calibration.

Fits a monotonically non-decreasing step function to raw confidences.
No parametric assumptions — fully non-parametric.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import CalibrationResult, CalibrationSample, Calibrator, compute_ece


class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibrator.

    Attributes:
        _x: Sorted unique raw confidence values.
        _y: Corresponding calibrated values (monotonically non-decreasing).
        _fitted: Whether fit has been called.
    """

    def __init__(self) -> None:
        self._x: list[float] = []
        self._y: list[float] = []
        self._fitted = False

    @property
    def method_name(self) -> str:
        return "isotonic"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @staticmethod
    def _pool_adjacent_violators(
        x: list[float], y: list[float]
    ) -> tuple[list[float], list[float]]:
        """Pool Adjacent Violators Algorithm (PAVA).

        Enforces monotonically non-decreasing constraint on y vs x.
        """
        n = len(x)
        x_out = list(x)
        y_out = list(y)
        weight = [1.0] * n

        i = 1
        while i < n:
            if y_out[i] < y_out[i - 1]:
                # Violation — pool with previous
                # Find the start of the current pooled block
                j = i - 1
                while j > 0 and weight[j] > 0:
                    j -= 1
                j += 1
                # Pool from j to i
                total_weight = sum(weight[j : i + 1])
                total_y = sum(y_out[k] * weight[k] for k in range(j, i + 1))
                new_y = total_y / total_weight
                new_weight = total_weight

                for k in range(j, i + 1):
                    y_out[k] = new_y
                    weight[k] = new_weight

                # Reset to re-check
                i = j
            i += 1

        return x_out, y_out

    def fit(self, samples: Sequence[CalibrationSample]) -> None:
        """Fit isotonic regression from calibration samples.

        Args:
            samples: Calibration samples with raw confidence and actual outcomes.
        """
        if len(samples) < 2:
            raise ValueError("Need at least 2 samples for isotonic regression")

        # Sort by raw confidence
        sorted_samples = sorted(samples, key=lambda s: s.raw_confidence)
        x = [s.raw_confidence for s in sorted_samples]
        y = [s.actual for s in sorted_samples]

        # Run PAVA
        x, y = self._pool_adjacent_violators(x, y)

        # Store unique x,y pairs for lookup
        self._x = x
        self._y = y
        self._fitted = True

    def calibrate(self, raw_confidence: float) -> float:
        """Apply isotonic calibration via nearest-neighbor lookup."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")

        # Binary search for nearest x value
        lo, hi = 0, len(self._x) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._x[mid] < raw_confidence:
                lo = mid + 1
            else:
                hi = mid

        # Clamp to range
        if raw_confidence <= self._x[0]:
            return self._y[0]
        if raw_confidence >= self._x[-1]:
            return self._y[-1]

        # Linear interpolation between nearest neighbors
        idx = lo
        x0, x1 = self._x[idx - 1], self._x[idx]
        y0, y1 = self._y[idx - 1], self._y[idx]

        if x1 == x0:
            return y0

        t = (raw_confidence - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

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
