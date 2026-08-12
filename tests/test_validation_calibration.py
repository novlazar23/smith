"""Tests for calibration package — Platt, Isotonic, Temperature Scaling."""

from __future__ import annotations

import math

import pytest

from packages.validation.calibration.base import (
    CalibrationResult,
    CalibrationSample,
    compute_ece,
)
from packages.validation.calibration.isotonic import IsotonicCalibrator
from packages.validation.calibration.platt import PlattCalibrator
from packages.validation.calibration.temperature import TemperatureCalibrator


class TestComputeECE:
    """Tests for compute_ece function."""

    def test_known_values_uniform_bins(self):
        """ECE with perfectly calibrated predictions should be 0."""
        # Perfect calibration: each confidence equals the actual rate
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        actuals = [0.1, 0.3, 0.5, 0.7, 0.9]
        ece = compute_ece(confidences, actuals)
        # With 5 points in 10 bins, each bin has at most one point
        # Perfect calibration means ECE ≈ 0 (exactly 0 for these discrete values)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_known_values_miscalibrated(self):
        """ECE computation with known miscalibrated values."""
        # All predictions say 0.9 but half are wrong
        confidences = [0.9] * 10
        actuals = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ece = compute_ece(confidences, actuals)
        # All 10 points fall in the [0.9, 1.0] bin
        # bin_conf = 0.9, bin_acc = 0.5
        # ECE = 1.0 * |0.5 - 0.9| = 0.4
        assert ece == pytest.approx(0.4, abs=1e-10)

    def test_empty_input_returns_zero(self):
        """Empty inputs should return ECE = 0."""
        assert compute_ece([], []) == 0.0
        assert compute_ece([], [0.5]) == 0.0
        assert compute_ece([0.5], []) == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_ece([0.5, 0.6], [0.5])

    def test_single_point(self):
        """Single point should produce valid ECE."""
        ece = compute_ece([0.8], [1.0])
        assert 0.0 <= ece <= 1.0

    def test_multibin_distribution(self):
        """ECE across multiple bins with known values."""
        # 20 points: 10 in [0.0, 0.5), 10 in [0.5, 1.0]
        confidences = [0.25] * 10 + [0.75] * 10
        actuals = [1.0] * 8 + [0.0] * 2 + [1.0] * 7 + [0.0] * 3
        ece = compute_ece(confidences, actuals)
        # Bin [0.0, 0.5): 8 correct / 10 = acc 0.8, conf 0.25 → contribution = 0.5 * |0.8-0.25| = 0.275
        # Bin [0.5, 1.0]: 7 correct / 10 = acc 0.7, conf 0.75 → contribution = 0.5 * |0.7-0.75| = 0.025
        expected = 0.275 + 0.025  # = 0.3
        assert ece == pytest.approx(expected, abs=1e-10)


class TestPlattCalibrator:
    """Tests for PlattCalibrator."""

    def test_basic_fit_and_calibrate(self):
        """Platt calibrator should fit and produce outputs in [0, 1]."""
        cal = PlattCalibrator()
        samples = [
            CalibrationSample(0.3, 0.0, "DOWN", "s1"),
            CalibrationSample(0.7, 1.0, "UP", "s2"),
            CalibrationSample(0.5, 0.0, "DOWN", "s3"),
            CalibrationSample(0.8, 1.0, "UP", "s4"),
            CalibrationSample(0.6, 1.0, "UP", "s5"),
        ]
        cal.fit(samples)
        assert cal.is_fitted
        # Output must be in [0, 1]
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = cal.calibrate(conf)
            assert 0.0 <= result <= 1.0

    def test_raises_runtime_error_before_fit(self):
        """Unfitted Platt calibrator should raise RuntimeError."""
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            cal.calibrate(0.5)

    def test_improves_ece(self):
        """Platt calibration should reduce or maintain ECE."""
        # Create intentionally miscalibrated data
        samples = [
            CalibrationSample(0.9, 0.0, "DOWN", f"s{i}")
            for i in range(30)
        ] + [
            CalibrationSample(0.2, 1.0, "UP", f"s{i}")
            for i in range(30)
        ] + [
            CalibrationSample(0.5, 0.5, "MIXED", f"s{i}")
            for i in range(40)
        ]
        cal = PlattCalibrator()
        result = cal.evaluate(samples)
        # ECE should not worsen after calibration
        assert result.ece_after <= result.ece_before + 0.01
        assert result.num_samples == 100
        assert result.improvement == result.ece_before - result.ece_after

    def test_numerical_edge_cases(self):
        """Edge cases: very low/high confidence values."""
        cal = PlattCalibrator()
        samples = [
            CalibrationSample(0.01, 0.0, "DOWN", "s1"),
            CalibrationSample(0.99, 1.0, "UP", "s2"),
        ]
        cal.fit(samples)
        # Test boundary values
        assert 0.0 <= cal.calibrate(0.001) <= 1.0
        assert 0.0 <= cal.calibrate(0.999) <= 1.0

    def test_sigmoid_stability(self):
        """Sigmoid function should be numerically stable."""
        cal = PlattCalibrator()
        assert cal._sigmoid(100) == pytest.approx(1.0, abs=1e-15)
        assert cal._sigmoid(-100) == pytest.approx(0.0, abs=1e-15)
        assert cal._sigmoid(0) == pytest.approx(0.5, abs=1e-15)

    def test_logit_stability(self):
        """Logit function should clamp edge probabilities."""
        assert math.isfinite(PlattCalibrator._logit(0.0))
        assert math.isfinite(PlattCalibrator._logit(1.0))
        assert math.isfinite(PlattCalibrator._logit(0.5))

    def test_requires_minimum_samples(self):
        """Must have at least 2 samples."""
        cal = PlattCalibrator()
        with pytest.raises(ValueError, match="at least 2"):
            cal.fit([CalibrationSample(0.5, 1.0, "UP", "s1")])


class TestIsotonicCalibrator:
    """Tests for IsotonicCalibrator."""

    def test_monotonicity(self):
        """Calibrated outputs must be monotonically non-decreasing with raw confidence."""
        cal = IsotonicCalibrator()
        samples = [
            CalibrationSample(0.1, 0.0, "DOWN", "s1"),
            CalibrationSample(0.2, 0.0, "DOWN", "s2"),
            CalibrationSample(0.3, 0.2, "DOWN", "s3"),
            CalibrationSample(0.4, 0.5, "MIXED", "s4"),
            CalibrationSample(0.5, 0.6, "MIXED", "s5"),
            CalibrationSample(0.6, 0.7, "UP", "s6"),
            CalibrationSample(0.7, 0.8, "UP", "s7"),
            CalibrationSample(0.8, 0.9, "UP", "s8"),
            CalibrationSample(0.9, 1.0, "UP", "s9"),
        ]
        cal.fit(samples)
        # Test monotonicity
        prev = -1.0
        for conf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            val = cal.calibrate(conf)
            assert val >= prev, f"Monotonicity violated at {conf}: {val} < {prev}"
            prev = val

    def test_boundary_values(self):
        """Boundary values should return edge calibrated values."""
        cal = IsotonicCalibrator()
        samples = [
            CalibrationSample(0.3, 0.0, "DOWN", "s1"),
            CalibrationSample(0.7, 1.0, "UP", "s2"),
        ]
        cal.fit(samples)
        # Below min should return edge value
        val_low = cal.calibrate(0.1)
        assert val_low == cal.calibrate(0.3)
        # Above max should return edge value
        val_high = cal.calibrate(0.9)
        assert val_high == cal.calibrate(0.7)

    def test_improves_ece(self):
        """Isotonic calibration should reduce or maintain ECE."""
        samples = [
            CalibrationSample(0.9, 0.0, "DOWN", f"s{i}")
            for i in range(30)
        ] + [
            CalibrationSample(0.2, 1.0, "UP", f"s{i}")
            for i in range(30)
        ] + [
            CalibrationSample(0.5, 0.5, "MIXED", f"s{i}")
            for i in range(40)
        ]
        cal = IsotonicCalibrator()
        result = cal.evaluate(samples)
        assert result.ece_after <= result.ece_before + 0.01
        assert result.improvement == result.ece_before - result.ece_after

    def test_raises_runtime_error_before_fit(self):
        """Unfitted isotonic calibrator should raise RuntimeError."""
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            cal.calibrate(0.5)

    def test_outputs_in_range(self):
        """All calibrated outputs must be in [0, 1]."""
        cal = IsotonicCalibrator()
        samples = [
            CalibrationSample(0.1, 0.0, "DOWN", "s1"),
            CalibrationSample(0.9, 1.0, "UP", "s2"),
            CalibrationSample(0.5, 0.5, "MIXED", "s3"),
        ]
        cal.fit(samples)
        for conf in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = cal.calibrate(conf)
            assert 0.0 <= result <= 1.0, f"Output {result} out of range for conf {conf}"

    def test_requires_minimum_samples(self):
        """Must have at least 2 samples."""
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError, match="at least 2"):
            cal.fit([CalibrationSample(0.5, 1.0, "UP", "s1")])

    def test_interpolation(self):
        """Values between calibration points should be interpolated."""
        cal = IsotonicCalibrator()
        samples = [
            CalibrationSample(0.2, 0.0, "DOWN", "s1"),
            CalibrationSample(0.8, 1.0, "UP", "s2"),
        ]
        cal.fit(samples)
        # Midpoint should interpolate between 0 and 1
        mid = cal.calibrate(0.5)
        assert 0.0 < mid < 1.0


class TestTemperatureCalibrator:
    """Tests for TemperatureCalibrator."""

    def test_basic_fit_and_calibrate(self):
        """Temperature calibrator should fit and produce outputs in [0, 1]."""
        cal = TemperatureCalibrator()
        samples = [
            CalibrationSample(0.3, 0.0, "DOWN", "s1"),
            CalibrationSample(0.7, 1.0, "UP", "s2"),
            CalibrationSample(0.5, 0.0, "DOWN", "s3"),
            CalibrationSample(0.8, 1.0, "UP", "s4"),
            CalibrationSample(0.6, 1.0, "UP", "s5"),
        ]
        cal.fit(samples)
        assert cal.is_fitted
        assert 0.1 <= cal.temperature <= 10.0
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = cal.calibrate(conf)
            assert 0.0 <= result <= 1.0

    def test_temperature_1_when_well_calibrated(self):
        """T should stay near 1.0 when data is already well-calibrated."""
        # Perfect calibration: confidence matches actual
        samples = [
            CalibrationSample(0.3, 0.3, "DOWN", "s1"),
            CalibrationSample(0.5, 0.5, "MIXED", "s2"),
            CalibrationSample(0.7, 0.7, "UP", "s3"),
            CalibrationSample(0.9, 0.9, "UP", "s4"),
            CalibrationSample(0.2, 0.2, "DOWN", "s5"),
            CalibrationSample(0.4, 0.4, "DOWN", "s6"),
            CalibrationSample(0.6, 0.6, "MIXED", "s7"),
            CalibrationSample(0.8, 0.8, "UP", "s8"),
        ]
        cal = TemperatureCalibrator()
        cal.fit(samples)
        # Temperature should remain close to 1.0
        assert abs(cal.temperature - 1.0) < 0.5

    def test_improves_ece(self):
        """Temperature calibration should reduce or maintain ECE."""
        samples = [
            CalibrationSample(0.9, 0.0, "DOWN", f"s{i}")
            for i in range(30)
        ] + [
            CalibrationSample(0.2, 1.0, "UP", f"s{i}")
            for i in range(30)
        ] + [
            CalibrationSample(0.5, 0.5, "MIXED", f"s{i}")
            for i in range(40)
        ]
        cal = TemperatureCalibrator()
        result = cal.evaluate(samples)
        assert result.ece_after <= result.ece_before + 0.01
        assert result.improvement == result.ece_before - result.ece_after

    def test_raises_runtime_error_before_fit(self):
        """Unfitted temperature calibrator should raise RuntimeError."""
        cal = TemperatureCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            cal.calibrate(0.5)

    def test_clamps_temperature_range(self):
        """Temperature should stay within [0.1, 10.0]."""
        samples = [
            CalibrationSample(0.01, 0.0, "DOWN", "s1"),
            CalibrationSample(0.99, 1.0, "UP", "s2"),
        ]
        cal = TemperatureCalibrator()
        cal.fit(samples)
        assert 0.1 <= cal.temperature <= 10.0

    def test_outputs_in_range(self):
        """All calibrated outputs must be in [0, 1]."""
        cal = TemperatureCalibrator()
        samples = [
            CalibrationSample(0.1, 0.0, "DOWN", "s1"),
            CalibrationSample(0.9, 1.0, "UP", "s2"),
        ]
        cal.fit(samples)
        for conf in [0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]:
            result = cal.calibrate(conf)
            assert 0.0 <= result <= 1.0, f"Output {result} out of range for conf {conf}"

    def test_requires_minimum_samples(self):
        """Must have at least 2 samples."""
        cal = TemperatureCalibrator()
        with pytest.raises(ValueError, match="at least 2"):
            cal.fit([CalibrationSample(0.5, 1.0, "UP", "s1")])


class TestCommonProperties:
    """Tests common to all calibrator types."""

    @pytest.fixture
    def calibration_samples(self):
        return [
            CalibrationSample(0.1, 0.0, "DOWN", "s1"),
            CalibrationSample(0.2, 0.0, "DOWN", "s2"),
            CalibrationSample(0.3, 0.1, "DOWN", "s3"),
            CalibrationSample(0.4, 0.3, "DOWN", "s4"),
            CalibrationSample(0.5, 0.5, "MIXED", "s5"),
            CalibrationSample(0.6, 0.7, "UP", "s6"),
            CalibrationSample(0.7, 0.8, "UP", "s7"),
            CalibrationSample(0.8, 0.9, "UP", "s8"),
            CalibrationSample(0.9, 1.0, "UP", "s9"),
            CalibrationSample(0.95, 1.0, "UP", "s10"),
        ]

    def test_method_name_property(self):
        """Each calibrator must have a unique method_name."""
        assert PlattCalibrator().method_name == "platt"
        assert IsotonicCalibrator().method_name == "isotonic"
        assert TemperatureCalibrator().method_name == "temperature"

    def test_is_fitted_property(self):
        """is_fitted should be False before fit and True after."""
        for cal in [PlattCalibrator(), IsotonicCalibrator(), TemperatureCalibrator()]:
            assert cal.is_fitted is False
            samples = [
                CalibrationSample(0.3, 0.0, "DOWN", "s1"),
                CalibrationSample(0.7, 1.0, "UP", "s2"),
            ]
            cal.fit(samples)
            assert cal.is_fitted is True

    def test_calibrate_batch(self):
        """Batch calibration should produce same results as individual calls."""
        samples = [
            CalibrationSample(0.3, 0.0, "DOWN", "s1"),
            CalibrationSample(0.7, 1.0, "UP", "s2"),
            CalibrationSample(0.5, 0.5, "MIXED", "s3"),
            CalibrationSample(0.8, 1.0, "UP", "s4"),
        ]
        raw_confs = [0.2, 0.4, 0.6, 0.8]
        for cal in [PlattCalibrator(), IsotonicCalibrator(), TemperatureCalibrator()]:
            cal.fit(samples)
            batch_result = cal.calibrate_batch(raw_confs)
            assert len(batch_result) == len(raw_confs)
            for i, conf in enumerate(raw_confs):
                assert batch_result[i] == pytest.approx(cal.calibrate(conf), abs=1e-10)

    def test_ece_decreases_after_calibration(self):
        """ECE should decrease (or stay same) after any calibration."""
        samples = [
            CalibrationSample(0.9, 0.0, "DOWN", f"s{i}")
            for i in range(40)
        ] + [
            CalibrationSample(0.2, 1.0, "UP", f"s{i}")
            for i in range(40)
        ] + [
            CalibrationSample(0.5, 0.5, "MIXED", f"s{i}")
            for i in range(20)
        ]
        for cal in [PlattCalibrator(), IsotonicCalibrator(), TemperatureCalibrator()]:
            result = cal.evaluate(samples)
            assert result.ece_after <= result.ece_before + 0.02, (
                f"{cal.method_name}: ECE increased from {result.ece_before:.4f} "
                f"to {result.ece_after:.4f}"
            )


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_improvement_calculation(self):
        """improvement should be ece_before - ece_after."""
        result = CalibrationResult(
            method="platt",
            ece_before=0.3,
            ece_after=0.1,
            num_samples=100,
        )
        assert result.improvement == pytest.approx(0.2)

    def test_negative_improvement(self):
        """ECE can increase slightly; improvement can be negative."""
        result = CalibrationResult(
            method="platt",
            ece_before=0.1,
            ece_after=0.15,
            num_samples=100,
        )
        assert result.improvement == pytest.approx(-0.05)

    def test_frozen_dataclass(self):
        """CalibrationResult should be immutable."""
        result = CalibrationResult(
            method="platt",
            ece_before=0.3,
            ece_after=0.1,
            num_samples=100,
        )
        with pytest.raises(Exception):
            result.method = "new_method"  # pyright: ignore


class TestCalibrationSample:
    """Tests for CalibrationSample frozen dataclass."""

    def test_frozen_immutable(self):
        """CalibrationSample should be immutable."""
        sample = CalibrationSample(0.5, 1.0, "UP", "s1")
        with pytest.raises(Exception):
            sample.raw_confidence = 0.6  # pyright: ignore

    def test_default_values(self):
        """sample_id defaults to empty string."""
        sample = CalibrationSample(0.5, 1.0, "UP")
        assert sample.sample_id == ""

    def test_hashable(self):
        """Frozen dataclass should be hashable."""
        sample = CalibrationSample(0.5, 1.0, "UP", "s1")
        hash(sample)  # Should not raise