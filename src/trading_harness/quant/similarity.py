"""Similarity Engine für OHLCV-Daten (Phase 5).

Findet historisch ähnliche Marktbedingungen mittels Euclidean Distance
auf normalisierten Preissequenzen mit Sliding Window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SimilarMatch:
    """Ein ähnlicher historischer Ausschnitt."""
    start_index: int
    end_index: int
    distance: float  # Euclidean Distance (niedriger = ähnlicher)
    correlation: float  # Pearson-Korrelation (-1 bis 1)
    candles: list[dict] = field(default_factory=list)  # optional: die Originalkerzen


@dataclass
class SimilarityResult:
    """Ergebnis der Similarity-Suche."""
    query_length: int
    matches: list[SimilarMatch]
    best_distance: float | None
    best_correlation: float | None


def _normalize(values: list[float]) -> list[float]:
    """Min-Max Normalisierung auf [0, 1]."""
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    rng = max_v - min_v
    if rng == 0:
        return [0.0] * len(values)
    return [(v - min_v) / rng for v in values]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean Distance zwischen zwei normalisierten Vektoren."""
    if len(a) != len(b) or not a:
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _pearson_correlation(a: list[float], b: list[float]) -> float:
    """Pearson-Korrelation zwischen zwei Vektoren."""
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    if var_a == 0 or var_b == 0:
        return 0.0
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    return cov / math.sqrt(var_a * var_b)


class SimilarityEngine:
    """Findet historisch ähnliche Muster — nur stdlib."""

    def __init__(
        self,
        window_size: int = 20,
        top_k: int = 5,
        normalize: bool = True,
    ) -> None:
        self.window_size = window_size
        self.top_k = top_k
        self.normalize = normalize

    def find_similar(
        self,
        query: list[dict],
        history: list[dict],
    ) -> SimilarityResult:
        """Findet die top_k ähnlichsten Fenster in der Historie.

        Args:
            query: Die Such-Sequenz (OHLCV-Kerzen)
            history: Die historischen Daten (OHLCV-Kerzen)

        Returns:
            SimilarityResult mit den ähnlichsten Fenstern
        """
        if len(query) < 2 or len(history) < self.window_size:
            return SimilarityResult(
                query_length=len(query),
                matches=[],
                best_distance=None,
                best_correlation=None,
            )

        query_closes = [c["close"] for c in query]
        if self.normalize:
            query_closes = _normalize(query_closes)

        history_closes = [c["close"] for c in history]
        if self.normalize:
            history_norm = _normalize(history_closes)
        else:
            history_norm = history_closes

        candidates: list[SimilarMatch] = []

        for i in range(len(history_norm) - self.window_size + 1):
            window = history_norm[i : i + self.window_size]

            # Adjust query length to match window
            q = query_closes[: self.window_size] if len(query_closes) >= self.window_size else query_closes
            w = window[: len(q)]

            if len(q) < 2 or len(w) < 2:
                continue

            dist = _euclidean_distance(q, w)
            corr = _pearson_correlation(q, w)

            candidates.append(SimilarMatch(
                start_index=i,
                end_index=i + self.window_size - 1,
                distance=dist,
                correlation=corr,
            ))

        # Sort by distance (ascending)
        candidates.sort(key=lambda m: m.distance)
        top = candidates[: self.top_k]

        return SimilarityResult(
            query_length=len(query),
            matches=top,
            best_distance=top[0].distance if top else None,
            best_correlation=top[0].correlation if top else None,
        )

    def find_similar_with_candles(
        self,
        query: list[dict],
        history: list[dict],
    ) -> SimilarityResult:
        """Wie find_similar, aber fügt die Originalkerzen bei."""
        result = self.find_similar(query, history)
        for match in result.matches:
            match.candles = history[match.start_index : match.end_index + 1]
        return result

    def compute_distance_matrix(
        self,
        sequences: list[list[dict]],
    ) -> list[list[float]]:
        """Berechnet Distanz-Matrix zwischen mehreren Sequenzen."""
        n = len(sequences)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        closes_list: list[list[float]] = []
        for seq in sequences:
            c = [x["close"] for x in seq]
            if self.normalize:
                c = _normalize(c)
            closes_list.append(c)

        for i in range(n):
            for j in range(i + 1, n):
                # Align to same length
                min_len = min(len(closes_list[i]), len(closes_list[j]))
                if min_len < 2:
                    d = float("inf")
                else:
                    d = _euclidean_distance(
                        closes_list[i][:min_len],
                        closes_list[j][:min_len],
                    )
                matrix[i][j] = d
                matrix[j][i] = d
        return matrix
