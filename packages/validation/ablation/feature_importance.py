"""Feature Importance via Permutation.

Measures how much each feature contributes to model performance
by permuting (shuffling) one feature at a time and measuring
the degradation in score.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .base import AblationAnalyzer, AblationResult


@dataclass(frozen=True)
class FeatureDataset:
    """A dataset with features and actual outcomes.

    Attributes:
        feature_names: Ordered list of feature names.
        feature_matrix: List of feature dicts per sample.
        actuals: True outcomes.
        sample_ids: Optional sample IDs.
    """

    feature_names: list[str]
    feature_matrix: list[dict[str, float]]
    actuals: list[str]
    sample_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_matrix:
            raise ValueError("feature_matrix must not be empty")
        n = len(self.feature_matrix)
        if len(self.actuals) != n:
            raise ValueError(
                f"actuals length ({len(self.actuals)}) != "
                f"feature_matrix length ({n})"
            )
        if len(self.sample_ids) != n:
            object.__setattr__(self, "sample_ids", [f"s{i}" for i in range(n)])


class FeatureImportanceAnalyzer(AblationAnalyzer):
    """Feature importance via permutation testing.

    For each feature:
    1. Score the full dataset.
    2. Permute (shuffle) that feature column.
    3. Score the permuted dataset.
    4. marginal = score_full - score_permuted.
    """

    def __init__(
        self,
        score_fn: Callable[[list[dict[str, float]], list[str]], float] | None = None,
        n_permutations: int = 10,
        seed: int = 42,
        higher_is_better: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        self.score_fn = score_fn or self._default_brier_score
        self.n_permutations = n_permutations
        self.seed = seed
        self.higher_is_better = higher_is_better

    @property
    def method_name(self) -> str:
        return "feature_importance"

    @staticmethod
    def _default_brier_score(
        predictions: list[dict[str, float]], actuals: list[str]
    ) -> float:
        """Mean Brier score. Lower is better."""
        total = 0.0
        n = len(predictions)
        if n == 0:
            return 0.0
        for pred, actual in zip(predictions, actuals, strict=True):
            score = 0.0
            for cls in ("UP", "DOWN", "RANGE"):
                predicted = pred.get(cls, 0.0)
                expected = 1.0 if actual == cls else 0.0
                score += (predicted - expected) ** 2
            total += score
        return total / n

    def run(
        self,
        full_score: float,
        ablated_score: float,
        agent_id: str,
        confidence: float = 1.0,
    ) -> AblationResult:
        """Run a single feature importance comparison.

        Uses ablated_score as the permuted score.
        For Brier score: lower permuted = feature was helpful.
        """
        if self.higher_is_better:
            marginal = full_score - ablated_score
        else:
            marginal = ablated_score - full_score

        if marginal > 1e-9:
            direction = "helpful"
        elif marginal < -1e-9:
            direction = "harmful"
        else:
            direction = "neutral"

        return AblationResult(
            method=self.method_name,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=marginal,
            direction=direction,
            confidence=confidence,
        )

    def analyze(
        self,
        dataset: FeatureDataset,
        model_predictions: list[dict[str, float]],
    ) -> list[AblationResult]:
        """Run feature importance analysis via permutation.

        For each feature, permutes the column n_permutations times
        and measures average score degradation.

        Args:
            dataset: Dataset with features and actuals.
            model_predictions: Predictions from the model being evaluated.

        Returns:
            List of AblationResult, one per feature, sorted by |marginal| descending.
        """
        random.seed(self.seed)
        n_samples = len(dataset.feature_matrix)
        full_score = self.score_fn(model_predictions, dataset.actuals)

        results: list[AblationResult] = []

        for feature_name in dataset.feature_names:
            # Collect feature values for permutation
            feature_values = [
                dataset.feature_matrix[i].get(feature_name, 0.0)
                for i in range(n_samples)
            ]

            # Average over multiple permutations
            avg_permuted_score = 0.0
            for _ in range(self.n_permutations):
                permuted = list(feature_values)
                random.shuffle(permuted)

                # Create permuted feature matrix
                permuted_matrix = []
                for i in range(n_samples):
                    row = dict(dataset.feature_matrix[i])
                    row[feature_name] = permuted[i]
                    permuted_matrix.append(row)

                permuted_score = self.score_fn(permuted_matrix, dataset.actuals)
                avg_permuted_score += permuted_score

            avg_permuted_score /= self.n_permutations

            results.append(self.run(full_score, avg_permuted_score, feature_name))

        # Sort by absolute marginal contribution (most important first)
        results.sort(key=lambda r: abs(r.marginal_contribution), reverse=True)

        return results
