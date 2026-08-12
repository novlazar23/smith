"""Calibration Methods for Historical Validation.

Three calibration methods for transforming raw agent confidences:
- Platt Scaling: logistic regression on log-odds
- Isotonic Regression: monotonic piecewise constant fit
- Temperature Scaling: single scalar softmax temperature

All calibrators accept raw probabilities and produce calibrated probabilities.
"""

from __future__ import annotations

from .base import CalibrationResult, CalibrationSample, Calibrator
from .isotonic import IsotonicCalibrator
from .platt import PlattCalibrator
from .temperature import TemperatureCalibrator

__all__ = [
    "CalibrationResult",
    "CalibrationSample",
    "Calibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "TemperatureCalibrator",
]
