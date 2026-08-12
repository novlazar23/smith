"""Target Variable Encoding for Historical Validation.

Encodes trading outcomes into classification/regression targets:
- 3-class: UP/DOWN/RANGE based on return thresholds
- 2-class: directional (UP vs not-UP)
- Regression: realized return values
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.schemas import AgentReport


class TargetType(StrEnum):
    """Target encoding type."""

    THREE_CLASS = "three_class"
    TWO_CLASS = "two_class"
    REGRESSION = "regression"


class Direction(StrEnum):
    """Directional label."""

    UP = "UP"
    DOWN = "DOWN"
    RANGE = "RANGE"


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for target encoding.

    Attributes:
        target_type: Encoding type.
        up_threshold: Return threshold for UP classification (e.g. 0.01 = 1%).
        down_threshold: Return threshold for DOWN classification (e.g. -0.01 = -1%).
        horizon: Price horizon for computing returns (e.g. "1d", "4h", "1w").
    """

    target_type: TargetType = TargetType.THREE_CLASS
    up_threshold: float = 0.01
    down_threshold: float = -0.01
    horizon: str = "1d"

    def __post_init__(self) -> None:
        if self.up_threshold <= 0:
            raise ValueError("up_threshold must be > 0")
        if self.down_threshold >= 0:
            raise ValueError("down_threshold must be < 0")
        if self.down_threshold >= self.up_threshold:
            raise ValueError("down_threshold must be < up_threshold")


@dataclass(frozen=True)
class Sample:
    """A single training sample.

    Attributes:
        sample_id: Unique identifier.
        as_of: Timestamp when features were available (analysis time).
        instrument: Trading instrument (e.g. "BTC/USDT").
        horizon: Price horizon (e.g. "1d").
        features: Dict mapping feature names to float values.
        direction: True direction label (from Direction enum or None for regression).
        realized_return: Realized return over the horizon.
        raw_confidence: Agent's raw confidence score (may differ from calibrated).
        agent_id: Which agent produced the sample.
        feature_snapshot_id: Link to feature snapshot for audit.
    """

    sample_id: str
    as_of: datetime
    instrument: str
    horizon: str
    features: dict[str, float] = field(default_factory=dict)
    direction: str | None = None
    realized_return: float = 0.0
    raw_confidence: float | None = None
    agent_id: str = ""
    feature_snapshot_id: str | None = None


@dataclass(frozen=True)
class DatasetSplit:
    """A train/calibration/test split with temporal ordering guarantee.

    Attributes:
        train: Training samples.
        calibration: Calibration samples (MUST be between train and test).
        test: Test samples.
        train_end: Latest as_of in training set.
        calibration_start: Earliest as_of in calibration set.
        calibration_end: Latest as_of in calibration set.
        test_start: Earliest as_of in test set.
    """

    train: list[Sample]
    calibration: list[Sample]
    test: list[Sample]
    train_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    test_start: datetime

    def __post_init__(self) -> None:
        """Verify temporal ordering: train_end <= calibration_start <= calibration_end < test_start."""
        if self.train_end > self.calibration_start:
            raise ValueError(
                f"Temporal ordering violation: train_end ({self.train_end}) "
                f"> calibration_start ({self.calibration_start})"
            )
        if self.calibration_end > self.test_start:
            raise ValueError(
                f"Temporal ordering violation: calibration_end ({self.calibration_end}) "
                f"> test_start ({self.test_start})"
            )


def encode_target(
    return_value: float,
    config: TargetConfig,
) -> str | float:
    """Encode a realized return into a target label.

    Args:
        return_value: Realized return over the horizon.
        config: Target encoding configuration.

    Returns:
        For THREE_CLASS: "UP", "DOWN", or "RANGE".
        For TWO_CLASS: 1.0 if UP, else 0.0.
        For REGRESSION: the return_value itself.
    """
    if config.target_type == TargetType.REGRESSION:
        return return_value
    if config.target_type == TargetType.TWO_CLASS:
        return 1.0 if return_value > config.up_threshold else 0.0
    # THREE_CLASS
    if return_value > config.up_threshold:
        return Direction.UP
    if return_value < config.down_threshold:
        return Direction.DOWN
    return Direction.RANGE


def build_samples_from_reports(
    reports: Sequence[AgentReport],
    target_config: TargetConfig,
) -> list[Sample]:
    """Build Sample objects from AgentReports with realized returns.

    Extracts: probabilities -> confidence, evidence -> features, as_of -> sample timestamp.
    Returns samples with direction labels encoded from the return distribution.

    Args:
        reports: Sequence of AgentReport objects.
        target_config: Configuration for target encoding.

    Returns:
        List of Sample objects, sorted by as_of timestamp.
    """
    samples: list[Sample] = []
    for report in reports:
        # Extract features from evidence
        features: dict[str, float] = {}
        for ev in report.evidence:
            features[ev.feature] = ev.relevance

        # Derive raw confidence from probabilities
        raw_confidence: float | None = report.raw_confidence
        if raw_confidence is None and report.probabilities:
            raw_confidence = float(max(report.probabilities.values()))

        # Derive realized return from expected_return distribution (q50 = median)
        realized_return = 0.0
        if report.expected_return is not None:
            realized_return = float(report.expected_return.get("q50", 0.0))

        # Encode direction label
        encoded = encode_target(realized_return, target_config)
        direction: str | None = None
        if isinstance(encoded, str):
            direction = encoded

        sample = Sample(
            sample_id=f"sample_{report.report_id}",
            as_of=report.as_of,
            instrument=report.instrument,
            horizon=report.horizon,
            features=features,
            direction=direction,
            realized_return=realized_return,
            raw_confidence=raw_confidence,
            agent_id=report.agent_id,
            feature_snapshot_id=report.feature_snapshot_id,
        )
        samples.append(sample)

    samples.sort(key=lambda s: s.as_of)
    return samples


def build_temporal_split(
    samples: list[Sample],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> DatasetSplit:
    """Split samples into train/calibration/test by temporal order.

    CRITICAL: Samples are sorted by as_of, then split sequentially.
    This guarantees no look-ahead bias — later data never appears in earlier splits.

    Args:
        samples: List of Sample objects (will be sorted by as_of internally).
        train_ratio: Fraction for training (e.g. 0.6).
        calibration_ratio: Fraction for calibration (e.g. 0.2).
        test_ratio: Fraction for test set (e.g. 0.2).

    Returns:
        DatasetSplit with ordered train/calibration/test.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or not enough samples.
    """
    epsilon = 0.01
    total = train_ratio + calibration_ratio + test_ratio
    if abs(total - 1.0) > epsilon:
        raise ValueError(
            f"Ratios must sum to ~1.0 (got {total:.4f}): "
            f"train={train_ratio}, calibration={calibration_ratio}, test={test_ratio}"
        )

    if len(samples) < 3:
        raise ValueError(f"Need at least 3 samples for a temporal split, got {len(samples)}")

    sorted_samples = sorted(samples, key=lambda s: s.as_of)
    n = len(sorted_samples)

    train_end_idx = int(n * train_ratio)
    cal_end_idx = int(n * (train_ratio + calibration_ratio))

    # Ensure each split has at least one sample
    if train_end_idx < 1:
        train_end_idx = 1
    if cal_end_idx - train_end_idx < 1:
        cal_end_idx = train_end_idx + 1
    if n - cal_end_idx < 1:
        cal_end_idx = n - 1

    train_samples = sorted_samples[:train_end_idx]
    cal_samples = sorted_samples[train_end_idx:cal_end_idx]
    test_samples = sorted_samples[cal_end_idx:]

    return DatasetSplit(
        train=train_samples,
        calibration=cal_samples,
        test=test_samples,
        train_end=train_samples[-1].as_of,
        calibration_start=cal_samples[0].as_of,
        calibration_end=cal_samples[-1].as_of,
        test_start=test_samples[0].as_of,
    )
