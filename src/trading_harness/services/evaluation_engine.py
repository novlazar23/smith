"""Evaluation Engine (Docker-Ready Phase).

Berechnet Brier Score, Calibration, Walk-Forward und andere Evaluations-Metriken
für Agenten-Vorhersagen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prediction:
    """Eine einzelne Vorhersage eines Agenten."""
    agent_id: str
    symbol: str
    direction: str  # "long", "short", "neutral"
    confidence: float  # 0.0 - 1.0
    timestamp: str
    actual_direction: str = ""
    actual_return: float = 0.0


@dataclass
class EvaluationResult:
    """Ergebnis einer Evaluierung."""
    agent_id: str
    metric_name: str
    value: float
    sample_size: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationBucket:
    """Calibration-Bucket für Brier Score Analyse."""
    predicted_probability: float
    actual_frequency: float
    count: int


class EvaluationEngine:
    """Berechnet Evaluations-Metriken — nur stdlib."""

    def __init__(self) -> None:
        pass

    def brier_score(self, predictions: list[Prediction]) -> float:
        """Berechnet Brier Score für binäre Vorhersagen.

        Brier Score = (1/N) * sum((predicted - actual)^2)
        Lower is better (0.0 = perfect, 1.0 = worst)
        """
        if not predictions:
            return 0.0

        valid = [p for p in predictions if p.actual_direction]
        if not valid:
            return 0.0

        total = 0.0
        for p in valid:
            predicted = 1.0 if p.direction == p.actual_direction else 0.0
            actual = 1.0
            total += (p.confidence - actual) ** 2 if predicted == 1.0 else (0.0 - actual) ** 2

        return total / len(valid)

    def directional_accuracy(self, predictions: list[Prediction]) -> float:
        """Berechnet Directional Accuracy (Trefferquote)."""
        valid = [p for p in predictions if p.actual_direction]
        if not valid:
            return 0.0

        correct = sum(1 for p in valid if p.direction == p.actual_direction)
        return correct / len(valid)

    def calibration(
        self,
        predictions: list[Prediction],
        n_buckets: int = 10,
    ) -> list[CalibrationBucket]:
        """Berechnet Calibration — wie gut sind die Konfidenzen kalibriert.

        Teilt Vorhersagen in Buckets nach Konfidenz auf und vergleicht
        mit tatsächlicher Trefferquote.
        """
        valid = [p for p in predictions if p.actual_direction]
        if not valid:
            return []

        buckets: list[CalibrationBucket] = []
        bucket_size = 1.0 / n_buckets

        for i in range(n_buckets):
            low = i * bucket_size
            high = (i + 1) * bucket_size
            bucket_predictions = [
                p for p in valid if low <= p.confidence < high
            ]
            if not bucket_predictions:
                continue

            predicted_prob = (low + high) / 2
            actual_freq = sum(
                1 for p in bucket_predictions
                if p.direction == p.actual_direction
            ) / len(bucket_predictions)

            buckets.append(CalibrationBucket(
                predicted_probability=predicted_prob,
                actual_frequency=actual_freq,
                count=len(bucket_predictions),
            ))

        return buckets

    def calibration_error(
        self,
        predictions: list[Prediction],
        n_buckets: int = 10,
    ) -> float:
        """Berechnet Expected Calibration Error (ECE)."""
        buckets = self.calibration(predictions, n_buckets)
        if not buckets:
            return 0.0

        total = sum(b.count for b in buckets)
        ece = sum(
            b.count * abs(b.predicted_probability - b.actual_frequency)
            for b in buckets
        ) / total

        return ece

    def profit_factor(self, predictions: list[Prediction]) -> float:
        """Berechnet Profit Factor (sum gains / sum losses)."""
        valid = [p for p in predictions if p.actual_return != 0.0]
        if not valid:
            return 0.0

        gains = sum(p.actual_return for p in valid if p.actual_return > 0)
        losses = abs(sum(p.actual_return for p in valid if p.actual_return < 0))

        if losses == 0:
            return 10.0 if gains > 0 else 0.0
        return min(gains / losses, 10.0)

    def expectancy(self, predictions: list[Prediction]) -> float:
        """Berechnet Erwartungswert pro Trade."""
        valid = [p for p in predictions if p.actual_return != 0.0]
        if not valid:
            return 0.0
        return sum(p.actual_return for p in valid) / len(valid)

    def max_drawdown(self, predictions: list[Prediction]) -> float:
        """Berechnet Maximum Drawdown basierend auf kumuliertem Return."""
        valid = [p for p in predictions if p.actual_return != 0.0]
        if not valid:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for p in valid:
            cumulative += p.actual_return
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return max_dd

    def walk_forward(
        self,
        predictions: list[Prediction],
        train_ratio: float = 0.7,
        n_splits: int = 5,
    ) -> list[dict[str, Any]]:
        """Walk-Forward Analyse — teilt Daten in Train/Test auf."""
        if len(predictions) < n_splits * 2:
            return []

        chunk_size = len(predictions) // n_splits
        results = []

        for i in range(n_splits):
            start = i * chunk_size
            end = min(start + chunk_size, len(predictions))
            split = predictions[start:end]

            train_size = int(len(split) * train_ratio)
            train = split[:train_size]
            test = split[train_size:]

            if not test:
                continue

            train_acc = self.directional_accuracy(train) if train else 0.0
            test_acc = self.directional_accuracy(test) if test else 0.0

            results.append({
                "split": i,
                "train_size": len(train),
                "test_size": len(test),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "degradation": train_acc - test_acc,
            })

        return results

    def sharpe_ratio(
        self,
        predictions: list[Prediction],
        risk_free_rate: float = 0.0,
    ) -> float:
        """Berechnet Sharpe Ratio."""
        valid = [p for p in predictions if p.actual_return != 0.0]
        if len(valid) < 2:
            return 0.0

        returns = [p.actual_return for p in valid]
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(var_r) if var_r > 0 else 1.0

        return (mean_r - risk_free_rate) / std_r

    def evaluate_agent(
        self,
        agent_id: str,
        predictions: list[Prediction],
    ) -> list[EvaluationResult]:
        """Führt vollständige Evaluierung für einen Agenten durch."""
        results = []

        # Brier Score
        bs = self.brier_score(predictions)
        results.append(EvaluationResult(
            agent_id=agent_id, metric_name="brier_score",
            value=bs, sample_size=len(predictions),
        ))

        # Directional Accuracy
        da = self.directional_accuracy(predictions)
        results.append(EvaluationResult(
            agent_id=agent_id, metric_name="directional_accuracy",
            value=da, sample_size=len(predictions),
        ))

        # Calibration Error
        ce = self.calibration_error(predictions)
        results.append(EvaluationResult(
            agent_id=agent_id, metric_name="calibration_error",
            value=ce, sample_size=len(predictions),
        ))

        # Profit Factor
        pf = self.profit_factor(predictions)
        results.append(EvaluationResult(
            agent_id=agent_id, metric_name="profit_factor",
            value=pf, sample_size=len(predictions),
        ))

        # Expectancy
        exp = self.expectancy(predictions)
        results.append(EvaluationResult(
            agent_id=agent_id, metric_name="expectancy",
            value=exp, sample_size=len(predictions),
        ))

        # Sharpe Ratio
        sr = self.sharpe_ratio(predictions)
        results.append(EvaluationResult(
            agent_id=agent_id, metric_name="sharpe_ratio",
            value=sr, sample_size=len(predictions),
        ))

        return results
