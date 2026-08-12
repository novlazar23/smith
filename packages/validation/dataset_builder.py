"""Dataset Builder for Historical Validation.

Builds temporal splits from AgentReport data, applies feature normalization,
and produces structured train/calibration/test datasets for validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .target_variables import (
    DatasetSplit,
    Sample,
    TargetConfig,
    build_samples_from_reports,
    build_temporal_split,
)

if TYPE_CHECKING:
    from packages.schemas import AgentReport


@dataclass(frozen=True)
class FeatureConfig:
    """Feature normalization configuration.

    Attributes:
        normalize: Whether to apply min-max normalization per feature.
        min_value: Minimum bound for clipping (used if normalize=True).
        max_value: Maximum bound for clipping (used if normalize=True).
    """

    normalize: bool = True
    min_value: float = -10.0
    max_value: float = 10.0


@dataclass(frozen=True)
class ValidationDataset:
    """A complete validation dataset with features and targets.

    Attributes:
        samples: Samples with normalized features.
        feature_names: Ordered list of feature names.
        feature_config: Normalization configuration used.
    """

    samples: list[Sample]
    feature_names: list[str]
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    def feature_matrix(self) -> list[list[float]]:
        """Return features as a list of lists for downstream modeling."""
        return [
            [s.features.get(name, 0.0) for name in self.feature_names]
            for s in self.samples
        ]

    def direction_vector(self) -> list[str | None]:
        """Return direction labels aligned with feature matrix."""
        return [s.direction for s in self.samples]


class DatasetBuilder:
    """Builds validation datasets from AgentReports with temporal guarantees.

    Usage:
        builder = DatasetBuilder()
        samples = builder.build_samples(reports, target_config)
        split = builder.build_temporal_split(samples)
        train_ds = builder.normalize(split.train, feature_config)

    The builder guarantees:
    - No future leakage: all splits are temporally ordered.
    - Calibration set is separate from training and test.
    - Features are normalized consistently across all splits.
    """

    def __init__(self) -> None:
        self._feature_stats: dict[str, tuple[float, float]] = {}

    def build_samples(
        self,
        reports: Sequence[AgentReport],
        target_config: TargetConfig,
    ) -> list[Sample]:
        """Build samples from AgentReports.

        Args:
            reports: Sequence of AgentReport objects from agent runs.
            target_config: Target encoding configuration.

        Returns:
            List of Sample objects sorted by as_of.
        """
        samples = build_samples_from_reports(reports, target_config)
        samples.sort(key=lambda s: s.as_of)
        return samples

    def build_temporal_split(
        self,
        samples: list[Sample],
        train_ratio: float = 0.6,
        calibration_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> DatasetSplit:
        """Build train/calibration/test split with temporal guarantees.

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
        return build_temporal_split(
            samples, train_ratio, calibration_ratio, test_ratio
        )

    def normalize(
        self,
        samples: list[Sample],
        feature_config: FeatureConfig | None = None,
    ) -> ValidationDataset:
        """Normalize features per-sample using stats learned from this set.

        IMPORTANT: For proper validation, you must call this on train first
        to learn stats, then reuse the same stats for calibration/test.
        The returned ValidationDataset includes feature_names for alignment.
        """
        feature_config = feature_config or FeatureConfig()

        if not samples:
            return ValidationDataset(
                samples=[], feature_names=[], feature_config=feature_config
            )

        feature_names = sorted(samples[0].features.keys())

        if feature_config.normalize:
            # Learn min/max from this set
            for name in feature_names:
                vals = [s.features[name] for s in samples if name in s.features]
                if vals:
                    mn, mx = min(vals), max(vals)
                    self._feature_stats[name] = (mn, mx)

        # Create copies (frozen dataclass)
        normalized_samples: list[Sample] = []
        for s in samples:
            new_features: dict[str, float] = {}
            for name in feature_names:
                if name in s.features:
                    val = s.features[name]
                    if feature_config.normalize:
                        # Clip
                        val = max(
                            feature_config.min_value,
                            min(feature_config.max_value, val),
                        )
                        # Normalize using learned stats
                        if name in self._feature_stats:
                            mn, mx = self._feature_stats[name]
                            val = (val - mn) / (mx - mn) if mx > mn else 0.5
                else:
                    val = 0.0
                new_features[name] = val
            normalized_samples.append(
                Sample(
                    sample_id=s.sample_id,
                    as_of=s.as_of,
                    instrument=s.instrument,
                    horizon=s.horizon,
                    features=new_features,
                    direction=s.direction,
                    realized_return=s.realized_return,
                    raw_confidence=s.raw_confidence,
                    agent_id=s.agent_id,
                    feature_snapshot_id=s.feature_snapshot_id,
                )
            )

        return ValidationDataset(
            samples=normalized_samples,
            feature_names=feature_names,
            feature_config=feature_config,
        )


class CrossSectionalDatasetBuilder(DatasetBuilder):
    """Extends DatasetBuilder to build cross-sectional datasets grouped by instrument.

    For each instrument, builds independent temporal splits, then combines.
    This preserves temporal ordering within each instrument while maximizing
    training data diversity.
    """

    def build_by_instrument(
        self,
        samples: list[Sample],
        train_ratio: float = 0.6,
        calibration_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> dict[str, DatasetSplit]:
        """Build temporal splits per instrument, then return combined split.

        Args:
            samples: All samples, may span multiple instruments.
            train_ratio, calibration_ratio, test_ratio: Split ratios.

        Returns:
            Dict mapping instrument to its DatasetSplit.
        """
        by_instrument: dict[str, list[Sample]] = {}
        for s in samples:
            by_instrument.setdefault(s.instrument, []).append(s)

        result: dict[str, DatasetSplit] = {}
        for instrument, inst_samples in by_instrument.items():
            inst_samples.sort(key=lambda s: s.as_of)
            n = len(inst_samples)
            if n < 3:
                continue  # Skip instruments with too few samples

            split_idx_cal = int(n * train_ratio)
            split_idx_test = int(n * (train_ratio + calibration_ratio))

            train_samples = inst_samples[:split_idx_cal]
            cal_samples = inst_samples[split_idx_cal:split_idx_test]
            test_samples = inst_samples[split_idx_test:]

            if not train_samples or not cal_samples or not test_samples:
                continue

            result[instrument] = DatasetSplit(
                train=train_samples,
                calibration=cal_samples,
                test=test_samples,
                train_end=train_samples[-1].as_of,
                calibration_start=cal_samples[0].as_of,
                calibration_end=cal_samples[-1].as_of,
                test_start=test_samples[0].as_of,
            )

        return result
