"""Feature Importance Engine (Phase 7).

Berechnet Feature Importance via Korrelation, Mutual Information
und Feature-Ranking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FeatureImportance:
    """Einzelnes Feature mit Importance-Score."""
    name: str
    importance: float
    correlation: float = 0.0
    rank: int = 0


@dataclass
class FeatureImportanceResult:
    """Ergebnis der Feature-Importance-Berechnung."""
    features: list[FeatureImportance]
    top_features: list[str]
    feature_groups: dict[str, float]


class FeatureImportanceEngine:
    """Berechnet Feature Importance — nur stdlib."""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def compute(
        self,
        features: dict[str, list[float]],
        target: list[float],
    ) -> FeatureImportanceResult:
        """Berechnet Feature Importance basierend auf Korrelation mit Ziel.

        Args:
            features: Feature-Name → Liste von Werten
            target: Zielwert-Liste

        Returns:
            FeatureImportanceResult mit gerankten Features
        """
        if not features or not target:
            return FeatureImportanceResult(features=[], top_features=[], feature_groups={})

        n = len(target)
        importances: list[FeatureImportance] = []

        for name, values in features.items():
            if len(values) != n or n < 2:
                continue
            corr = self._pearson_correlation(values, target)
            importance = abs(corr)
            importances.append(FeatureImportance(
                name=name, importance=importance, correlation=corr,
            ))

        # Sort by importance
        importances.sort(key=lambda f: f.importance, reverse=True)
        for i, f in enumerate(importances, 1):
            f.rank = i

        top = [f.name for f in importances if f.importance >= self.threshold]
        groups = self._group_importance(importances)

        return FeatureImportanceResult(
            features=importances, top_features=top, feature_groups=groups,
        )

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Pearson-Korrelation — nur stdlib."""
        n = len(x)
        if n < 2:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        if var_x == 0 or var_y == 0:
            return 0.0
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        return cov / math.sqrt(var_x * var_y)

    def _group_importance(self, importances: list[FeatureImportance]) -> dict[str, float]:
        """Berechnet durchschnittliche Importance pro Feature-Gruppe."""
        groups: dict[str, list[float]] = {}
        for f in importances:
            parts = f.name.split("_", 1)
            group = parts[0] if len(parts) > 1 else "other"
            groups.setdefault(group, []).append(f.importance)
        return {g: sum(vals) / len(vals) for g, vals in groups.items()}

    def select_features(
        self,
        features: dict[str, list[float]],
        target: list[float],
        max_features: int = 10,
    ) -> list[str]:
        """Wählt die wichtigsten Features aus."""
        result = self.compute(features, target)
        return result.top_features[:max_features]

    def mutual_information_approx(
        self,
        x: list[float],
        y: list[float],
        bins: int = 10,
    ) -> float:
        """Ungefähre Mutual Information — Binning-Ansatz."""
        n = len(x)
        if n < 2:
            return 0.0
        # Create bins
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        # Count joint and marginal distributions
        joint = [[0] * bins for _ in range(bins)]
        marginal_x = [0] * bins
        marginal_y = [0] * bins

        for i in range(n):
            xi = min(int((x[i] - x_min) / x_range * (bins - 1)), bins - 1)
            yi = min(int((y[i] - y_min) / y_range * (bins - 1)), bins - 1)
            joint[xi][yi] += 1
            marginal_x[xi] += 1
            marginal_y[yi] += 1

        # Compute MI
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint[i][j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                    p_xy = joint[i][j] / n
                    p_x = marginal_x[i] / n
                    p_y = marginal_y[j] / n
                    mi += p_xy * math.log(p_xy / (p_x * p_y))
        return mi
