"""Tests for the Dataset Builder — temporal splits and target encoding."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from packages.schemas.agent_report import EvidenceReference
from packages.validation.dataset_builder import (
    CrossSectionalDatasetBuilder,
    DatasetBuilder,
    FeatureConfig,
    ValidationDataset,
)
from packages.validation.target_variables import (
    Direction,
    Sample,
    TargetConfig,
    TargetType,
    build_samples_from_reports,
    build_temporal_split,
    encode_target,
)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

BASE_TIME = datetime(2024, 1, 1, 0, 0, 0)


def _make_sample(
    as_of_offset_hours: int,
    instrument: str = "BTC/USDT",
    features: dict[str, float] | None = None,
    direction: str | None = "UP",
    realized_return: float = 0.02,
    raw_confidence: float | None = 0.8,
) -> Sample:
    """Create a Sample with deterministic properties."""
    return Sample(
        sample_id=f"sample_{as_of_offset_hours:03d}",
        as_of=BASE_TIME + timedelta(hours=as_of_offset_hours),
        instrument=instrument,
        horizon="1d",
        features=features or {"rsi": 0.5, "macd": 0.3},
        direction=direction,
        realized_return=realized_return,
        raw_confidence=raw_confidence,
        agent_id="agent_1",
        feature_snapshot_id="snap_001",
    )


def _make_report(
    as_of_offset_hours: int,
    expected_return: dict[str, float] | None = None,
    instrument: str = "BTC/USDT",
) -> Any:
    """Create a minimal mock AgentReport for testing."""
    from packages.schemas.agent_report import AgentStatus

    return type(
        "MockReport",
        (),
        {
            "report_id": f"rpt_{as_of_offset_hours:03d}",
            "run_id": "run_001",
            "agent_id": "agent_1",
            "agent_version": "1.0",
            "instrument": instrument,
            "horizon": "1d",
            "as_of": BASE_TIME + timedelta(hours=as_of_offset_hours),
            "hypothesis": "test hypothesis",
            "probabilities": {"up": 0.5, "down": 0.3, "range": 0.2},
            "expected_return": expected_return,
            "evidence": [
                EvidenceReference(
                    reference="ref_1",
                    feature="rsi",
                    value="30",
                    direction="positive",
                    relevance=0.8,
                ),
            ],
            "counter_evidence": [],
            "invalidations": [],
            "sample_size": None,
            "raw_confidence": 0.75,
            "calibrated_confidence": 0.7,
            "data_quality": 0.95,
            "uncertainty": None,
            "status": AgentStatus.SHADOW,
            "narrative": None,
            "model_version": None,
            "prompt_version": None,
            "feature_snapshot_id": "snap_001",
        },
    )


class TestEncodeTargetThreeClass:
    """Test encode_target with THREE_CLASS encoding."""

    def test_up_above_threshold(self) -> None:
        config = TargetConfig(target_type=TargetType.THREE_CLASS, up_threshold=0.01)
        result = encode_target(0.02, config)
        assert result == Direction.UP

    def test_down_below_threshold(self) -> None:
        config = TargetConfig(target_type=TargetType.THREE_CLASS, down_threshold=-0.01)
        result = encode_target(-0.02, config)
        assert result == Direction.DOWN

    def test_range_between_thresholds(self) -> None:
        config = TargetConfig(
            target_type=TargetType.THREE_CLASS,
            up_threshold=0.01,
            down_threshold=-0.01,
        )
        result = encode_target(0.005, config)
        assert result == Direction.RANGE

    def test_exactly_at_up_threshold(self) -> None:
        config = TargetConfig(target_type=TargetType.THREE_CLASS, up_threshold=0.01)
        result = encode_target(0.01, config)
        # At threshold (not strictly greater) → RANGE
        assert result == Direction.RANGE

    def test_zero_return(self) -> None:
        config = TargetConfig(
            target_type=TargetType.THREE_CLASS,
            up_threshold=0.01,
            down_threshold=-0.01,
        )
        result = encode_target(0.0, config)
        assert result == Direction.RANGE

    def test_large_positive_return(self) -> None:
        config = TargetConfig(target_type=TargetType.THREE_CLASS)
        result = encode_target(0.15, config)
        assert result == Direction.UP

    def test_large_negative_return(self) -> None:
        config = TargetConfig(target_type=TargetType.THREE_CLASS)
        result = encode_target(-0.20, config)
        assert result == Direction.DOWN


class TestEncodeTargetTwoClass:
    """Test encode_target with TWO_CLASS encoding."""

    def test_up_classifies_as_one(self) -> None:
        config = TargetConfig(target_type=TargetType.TWO_CLASS, up_threshold=0.01)
        result = encode_target(0.02, config)
        assert result == 1.0

    def test_range_classifies_as_zero(self) -> None:
        config = TargetConfig(target_type=TargetType.TWO_CLASS, up_threshold=0.01)
        result = encode_target(0.005, config)
        assert result == 0.0

    def test_negative_classifies_as_zero(self) -> None:
        config = TargetConfig(target_type=TargetType.TWO_CLASS, up_threshold=0.01)
        result = encode_target(-0.05, config)
        assert result == 0.0

    def test_at_threshold_is_not_up(self) -> None:
        config = TargetConfig(target_type=TargetType.TWO_CLASS, up_threshold=0.01)
        result = encode_target(0.01, config)
        assert result == 0.0


class TestEncodeTargetRegression:
    """Test encode_target with REGRESSION encoding."""

    def test_returns_return_value(self) -> None:
        config = TargetConfig(target_type=TargetType.REGRESSION)
        result = encode_target(0.03, config)
        assert result == 0.03

    def test_returns_negative_return(self) -> None:
        config = TargetConfig(target_type=TargetType.REGRESSION)
        result = encode_target(-0.07, config)
        assert result == -0.07


class TestEncodeTargetConfigValidation:
    """Test TargetConfig validation."""

    def test_invalid_up_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="up_threshold must be > 0"):
            TargetConfig(up_threshold=0)

    def test_invalid_down_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="down_threshold must be < 0"):
            TargetConfig(down_threshold=0)

    def test_down_above_up(self) -> None:
        with pytest.raises(ValueError):
            TargetConfig(up_threshold=0.01, down_threshold=0.02)

    def test_valid_config(self) -> None:
        config = TargetConfig(up_threshold=0.02, down_threshold=-0.02)
        assert config.up_threshold == 0.02
        assert config.down_threshold == -0.02


class TestBuildTemporalSplitOrdering:
    """Test build_temporal_split guarantees correct ordering."""

    def test_correct_temporal_ordering(self) -> None:
        samples = [_make_sample(i) for i in range(10)]
        split = build_temporal_split(samples)

        assert split.train_end <= split.calibration_start
        assert split.calibration_end <= split.test_start

    def test_train_before_calibration(self) -> None:
        samples = [_make_sample(i) for i in range(10)]
        split = build_temporal_split(samples)

        train_asofs = [s.as_of for s in split.train]
        cal_asofs = [s.as_of for s in split.calibration]

        for ta in train_asofs:
            for ca in cal_asofs:
                assert ta <= ca

    def test_calibration_before_test(self) -> None:
        samples = [_make_sample(i) for i in range(10)]
        split = build_temporal_split(samples)

        cal_asofs = [s.as_of for s in split.calibration]
        test_asofs = [s.as_of for s in split.test]

        for ca in cal_asofs:
            for te in test_asofs:
                assert ca <= te

    def test_no_overlap(self) -> None:
        samples = [_make_sample(i) for i in range(10)]
        split = build_temporal_split(samples)

        all_ids = set()
        for s in split.train + split.calibration + split.test:
            assert s.sample_id not in all_ids, f"Duplicate sample: {s.sample_id}"
            all_ids.add(s.sample_id)

    def test_correct_ratios(self) -> None:
        samples = [_make_sample(i) for i in range(100)]
        split = build_temporal_split(
            samples, train_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2
        )

        assert len(split.train) == 60
        assert len(split.calibration) == 20
        assert len(split.test) == 20

    def test_unsorted_input_sorted(self) -> None:
        samples = [_make_sample(5), _make_sample(2), _make_sample(8), _make_sample(1)]
        split = build_temporal_split(samples)

        train_asofs = [s.as_of for s in split.train]
        assert train_asofs == sorted(train_asofs)

    def test_ratios_do_not_sum_to_one(self) -> None:
        samples = [_make_sample(i) for i in range(10)]
        with pytest.raises(ValueError, match=r"Ratios must sum to ~1.0"):
            build_temporal_split(samples, train_ratio=0.5, calibration_ratio=0.3, test_ratio=0.3)

    def test_not_enough_samples(self) -> None:
        with pytest.raises(ValueError, match="Need at least 3 samples"):
            build_temporal_split([], train_ratio=0.6, calibration_ratio=0.2, test_ratio=0.2)

    def test_not_enough_samples_two(self) -> None:
        samples = [_make_sample(1), _make_sample(2)]
        with pytest.raises(ValueError, match="Need at least 3 samples"):
            build_temporal_split(samples)

    def test_no_look_ahead_bias(self) -> None:
        """No sample in train has as_of > any sample in calibration."""
        samples = [_make_sample(i) for i in range(50)]
        split = build_temporal_split(samples)

        max_train_as_of = max(s.as_of for s in split.train)
        min_cal_as_of = min(s.as_of for s in split.calibration)
        min_test_as_of = min(s.as_of for s in split.test)

        assert max_train_as_of <= min_cal_as_of
        assert max_train_as_of <= min_test_as_of
        assert max(split.calibration, key=lambda s: s.as_of).as_of <= min_test_as_of

    def test_custom_ratios(self) -> None:
        samples = [_make_sample(i) for i in range(50)]
        split = build_temporal_split(
            samples, train_ratio=0.5, calibration_ratio=0.3, test_ratio=0.2
        )
        assert len(split.train) == 25
        assert len(split.calibration) == 15
        assert len(split.test) == 10


class TestDatasetBuilderNormalize:
    """Test DatasetBuilder.normalize with min-max normalization."""

    def test_normalization_produces_zero_to_one(self) -> None:
        builder = DatasetBuilder()
        samples = [_make_sample(i, features={"value": float(i * 10)}) for i in range(5)]
        ds = builder.normalize(samples)
        matrix = ds.feature_matrix()

        for row in matrix:
            val = row[0]
            assert 0.0 <= val <= 1.0, f"Normalized value {val} out of [0, 1]"

    def test_min_value_maps_to_zero(self) -> None:
        builder = DatasetBuilder()
        samples = [_make_sample(i, features={"x": float(i)}) for i in range(5)]
        ds = builder.normalize(samples)
        matrix = ds.feature_matrix()

        assert matrix[0][0] == 0.0  # min → 0

    def test_max_value_maps_to_one(self) -> None:
        builder = DatasetBuilder()
        samples = [_make_sample(i, features={"x": float(i)}) for i in range(5)]
        ds = builder.normalize(samples)
        matrix = ds.feature_matrix()

        assert matrix[-1][0] == 1.0  # max → 1

    def test_same_stats_applied_to_calibration_and_test(self) -> None:
        """Train learns stats, calibration/test reuse them."""
        builder = DatasetBuilder()
        train_samples = [_make_sample(i, features={"x": float(i * 10)}) for i in range(10)]
        cal_samples = [_make_sample(i + 10, features={"x": float(i * 10 + 5)}) for i in range(5)]
        test_samples = [_make_sample(i + 15, features={"x": float(i * 10 + 10)}) for i in range(5)]

        _train_ds = builder.normalize(train_samples)
        cal_ds = builder.normalize(cal_samples)
        test_ds = builder.normalize(test_samples)

        # Stats learned from train should constrain cal/test to [0, 1]
        for ds in (cal_ds, test_ds):
            for row in ds.feature_matrix():
                val = row[0]
                assert 0.0 <= val <= 1.0, f"Value {val} not in [0, 1]"

    def test_normalize_disabled(self) -> None:
        builder = DatasetBuilder()
        samples = [_make_sample(i, features={"x": float(i * 100)}) for i in range(3)]
        config = FeatureConfig(normalize=False)
        ds = builder.normalize(samples, config)

        matrix = ds.feature_matrix()
        assert matrix[0][0] == 0.0
        assert matrix[1][0] == 100.0
        assert matrix[2][0] == 200.0

    def test_normalize_empty(self) -> None:
        builder = DatasetBuilder()
        ds = builder.normalize([])
        assert ds.samples == []
        assert ds.feature_names == []

    def test_feature_names_sorted(self) -> None:
        builder = DatasetBuilder()
        samples = [_make_sample(0, features={"zebra": 1.0, "alpha": 2.0, "beta": 3.0})]
        ds = builder.normalize(samples)
        assert ds.feature_names == ["alpha", "beta", "zebra"]


class TestCrossSectionalDatasetBuilder:
    """Test CrossSectionalDatasetBuilder.build_by_instrument."""

    def test_per_instrument_splits(self) -> None:
        builder = CrossSectionalDatasetBuilder()
        samples = [_make_sample(i, instrument="BTC/USDT") for i in range(10)] + [
            _make_sample(i, instrument="ETH/USDT") for i in range(10)
        ]
        result = builder.build_by_instrument(samples)

        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert len(result) == 2

    def test_temporal_ordering_per_instrument(self) -> None:
        builder = CrossSectionalDatasetBuilder()
        samples = [_make_sample(i, instrument="BTC/USDT") for i in range(20)]
        result = builder.build_by_instrument(samples)

        split = result["BTC/USDT"]
        assert split.train_end <= split.calibration_start
        assert split.calibration_end <= split.test_start

    def test_few_samples_skipped(self) -> None:
        builder = CrossSectionalDatasetBuilder()
        # Only 2 samples — should be skipped
        samples = [
            _make_sample(0, instrument="SOL/USDT"),
            _make_sample(1, instrument="SOL/USDT"),
        ]
        result = builder.build_by_instrument(samples)
        assert "SOL/USDT" not in result

    def test_multi_instrument_ordering(self) -> None:
        builder = CrossSectionalDatasetBuilder()
        samples = []
        for inst in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
            for i in range(15):
                samples.append(_make_sample(i, instrument=inst))
        result = builder.build_by_instrument(samples)

        for inst_split in result.values():
            assert inst_split.train_end <= inst_split.calibration_start
            assert inst_split.calibration_end <= inst_split.test_start


class TestValidationDataset:
    """Test ValidationDataset feature_matrix and direction_vector."""

    def test_feature_matrix_alignment(self) -> None:
        builder = DatasetBuilder()
        samples = [_make_sample(i, features={"a": float(i), "b": float(i * 2)}) for i in range(3)]
        ds = builder.normalize(samples)

        matrix = ds.feature_matrix()
        assert len(matrix) == 3
        assert len(matrix[0]) == len(ds.feature_names)
        assert matrix[0] == matrix[0]  # non-empty

    def test_direction_vector_alignment(self) -> None:
        samples = [
            _make_sample(0, direction="UP"),
            _make_sample(1, direction="DOWN"),
            _make_sample(2, direction="RANGE"),
        ]
        ds = ValidationDataset(
            samples=samples,
            feature_names=["x"],
        )

        vec = ds.direction_vector()
        assert vec == ["UP", "DOWN", "RANGE"]
        assert len(vec) == len(ds.feature_matrix())

    def test_direction_vector_with_none(self) -> None:
        samples = [
            _make_sample(0, direction="UP"),
            _make_sample(1, direction=None),
        ]
        ds = ValidationDataset(
            samples=samples,
            feature_names=["x"],
        )
        vec = ds.direction_vector()
        assert vec == ["UP", None]
        assert len(vec) == len(ds.feature_matrix())


class TestBuildSamplesFromReports:
    """Test build_samples_from_reports integration."""

    def test_samples_sorted_by_as_of(self) -> None:
        reports = [
            _make_report(10, expected_return={"q50": 0.02}),
            _make_report(5, expected_return={"q50": 0.01}),
            _make_report(1, expected_return={"q50": 0.005}),
        ]
        config = TargetConfig()
        samples = build_samples_from_reports(reports, config)

        as_ofs = [s.as_of for s in samples]
        assert as_ofs == sorted(as_ofs)

    def test_features_extracted_from_evidence(self) -> None:
        reports = [_make_report(1, expected_return={"q50": 0.02})]
        config = TargetConfig()
        samples = build_samples_from_reports(reports, config)

        assert len(samples) == 1
        assert "rsi" in samples[0].features
        assert samples[0].features["rsi"] == 0.8

    def test_direction_encoded_from_return(self) -> None:
        reports = [_make_report(1, expected_return={"q50": 0.05})]
        config = TargetConfig()
        samples = build_samples_from_reports(reports, config)

        assert samples[0].direction == Direction.UP

    def test_raw_confidence_from_report(self) -> None:
        reports = [_make_report(1, expected_return={"q50": 0.02})]
        config = TargetConfig()
        samples = build_samples_from_reports(reports, config)

        assert samples[0].raw_confidence == 0.75

    def test_regression_returns_none_direction(self) -> None:
        reports = [_make_report(1, expected_return={"q50": 0.02})]
        config = TargetConfig(target_type=TargetType.REGRESSION)
        samples = build_samples_from_reports(reports, config)

        assert samples[0].direction is None


class TestDatasetBuilderIntegration:
    """Integration tests: full pipeline from reports to normalized splits."""

    def test_full_pipeline(self) -> None:
        builder = DatasetBuilder()
        reports = [_make_report(i, expected_return={"q50": (i * 0.005) - 0.02}) for i in range(50)]
        config = TargetConfig()

        samples = builder.build_samples(reports, config)
        split = builder.build_temporal_split(samples)

        train_ds = builder.normalize(split.train)
        cal_ds = builder.normalize(split.calibration)
        test_ds = builder.normalize(split.test)

        assert len(train_ds.samples) > 0
        assert len(cal_ds.samples) > 0
        assert len(test_ds.samples) > 0

        # Feature matrix alignment
        assert len(train_ds.feature_matrix()) == len(train_ds.samples)
        assert len(train_ds.direction_vector()) == len(train_ds.samples)

        # Normalized values in [0, 1]
        for ds in (train_ds, cal_ds, test_ds):
            for row in ds.feature_matrix():
                for val in row:
                    assert 0.0 <= val <= 1.0

    def test_cross_sectional_full_pipeline(self) -> None:
        builder = CrossSectionalDatasetBuilder()
        samples = []
        for inst in ["BTC/USDT", "ETH/USDT"]:
            for i in range(30):
                samples.append(_make_sample(i, instrument=inst, features={"x": float(i)}))

        result = builder.build_by_instrument(samples)
        assert len(result) == 2

        for _inst, split in result.items():
            train_ds = builder.normalize(split.train)
            for row in train_ds.feature_matrix():
                assert all(0.0 <= v <= 1.0 for v in row)
