"""Tests für Similarity Engine."""
from __future__ import annotations

import math

import pytest

from trading_harness.quant.similarity import (
    SimilarityEngine,
    SimilarityResult,
    _euclidean_distance,
    _normalize,
    _pearson_correlation,
)


def _make_candles(prices: list[float]) -> list[dict]:
    return [
        {"time": f"2026-01-01T{i:04d}:00:00Z", "open": p, "high": p * 1.01,
         "low": p * 0.99, "close": p, "volume": 1000.0}
        for i, p in enumerate(prices)
    ]


class TestHelpers:
    def test_normalize_basic(self):
        result = _normalize([1.0, 2.0, 3.0])
        assert result == [0.0, 0.5, 1.0]

    def test_normalize_constant(self):
        result = _normalize([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]

    def test_normalize_empty(self):
        assert _normalize([]) == []

    def test_euclidean_identical(self):
        assert _euclidean_distance([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_euclidean_different(self):
        d = _euclidean_distance([0.0, 0.0], [1.0, 1.0])
        assert d == pytest.approx(math.sqrt(2))

    def test_pearson_perfect_positive(self):
        assert _pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_pearson_perfect_negative(self):
        assert _pearson_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_pearson_insufficient_data(self):
        assert _pearson_correlation([1.0], [2.0]) == 0.0


class TestSimilarityEngine:
    def test_find_similar_identical_pattern(self):
        # Same pattern repeated
        pattern = [100.0, 102.0, 104.0, 103.0, 101.0]
        history = pattern + [90.0, 88.0, 87.0, 89.0, 91.0] + pattern
        candles = _make_candles(history)
        engine = SimilarityEngine(window_size=5, top_k=1)
        result = engine.find_similar(_make_candles(pattern), candles)
        assert isinstance(result, SimilarityResult)
        assert len(result.matches) >= 1
        assert result.best_distance is not None

    def test_find_similar_returns_top_k(self):
        history = list(range(100, 200))
        candles = _make_candles([float(x) for x in history])
        query = _make_candles([100.0, 102.0, 104.0, 103.0, 101.0])
        engine = SimilarityEngine(window_size=5, top_k=3)
        result = engine.find_similar(query, candles)
        assert len(result.matches) <= 3

    def test_find_similar_insufficient_history(self):
        query = _make_candles([100.0, 101.0])
        history = _make_candles([100.0])
        engine = SimilarityEngine(window_size=20)
        result = engine.find_similar(query, history)
        assert result.matches == []

    def test_find_similar_short_query(self):
        query = _make_candles([100.0])
        history = _make_candles([float(i) for i in range(50)])
        engine = SimilarityEngine(window_size=5)
        result = engine.find_similar(query, history)
        assert result.matches == []

    def test_find_similar_best_distance_zero_for_identical(self):
        pattern = [100.0, 101.0, 102.0, 103.0, 104.0]
        candles = _make_candles(pattern)
        engine = SimilarityEngine(window_size=5, top_k=1, normalize=False)
        result = engine.find_similar(candles, candles)
        assert result.best_distance == pytest.approx(0.0, abs=0.01)

    def test_find_similar_correlation_bounded(self):
        history = [float(i) for i in range(100)]
        candles = _make_candles(history)
        query = _make_candles([50.0, 51.0, 52.0, 53.0, 54.0])
        engine = SimilarityEngine(window_size=5, top_k=5)
        result = engine.find_similar(query, candles)
        for m in result.matches:
            assert -1.0 <= m.correlation <= 1.0

    def test_find_similar_with_candles_populates(self):
        pattern = [100.0, 101.0, 102.0, 103.0, 104.0]
        history = [float(i) for i in range(100)]
        candles = _make_candles(history)
        query = _make_candles(pattern)
        engine = SimilarityEngine(window_size=5, top_k=2)
        result = engine.find_similar_with_candles(query, candles)
        for m in result.matches:
            assert len(m.candles) > 0

    def test_compute_distance_matrix(self):
        seqs = [
            _make_candles([100.0, 101.0, 102.0]),
            _make_candles([100.0, 101.0, 102.0]),
            _make_candles([200.0, 201.0, 202.0]),
        ]
        engine = SimilarityEngine(normalize=True)
        matrix = engine.compute_distance_matrix(seqs)
        assert len(matrix) == 3
        assert matrix[0][0] == 0.0
        assert matrix[0][1] == pytest.approx(0.0, abs=0.01)  # identical normalized

    def test_deterministic(self):
        history = [float(i) for i in range(100)]
        candles = _make_candles(history)
        query = _make_candles([50.0, 51.0, 52.0, 53.0, 54.0])
        engine = SimilarityEngine(window_size=5, top_k=3)
        r1 = engine.find_similar(query, candles)
        r2 = engine.find_similar(query, candles)
        assert len(r1.matches) == len(r2.matches)
        for m1, m2 in zip(r1.matches, r2.matches):
            assert m1.start_index == m2.start_index
            assert m1.distance == pytest.approx(m2.distance)
