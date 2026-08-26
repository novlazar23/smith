"""Tests für Evaluation Engine."""
from __future__ import annotations

import pytest
from trading_harness.services.evaluation_engine import (
    CalibrationBucket,
    EvaluationEngine,
    EvaluationResult,
    Prediction,
)


def _make_predictions(
    n: int = 20,
    accuracy: float = 0.6,
) -> list[Prediction]:
    """Erstellt Test-Vorhersagen mit gegebener Genauigkeit."""
    import random
    random.seed(42)
    preds = []
    for i in range(n):
        direction = "long" if random.random() < 0.5 else "short"
        actual = direction if random.random() < accuracy else ("short" if direction == "long" else "long")
        confidence = 0.5 + random.random() * 0.4
        preds.append(Prediction(
            agent_id="agent-1",
            symbol="BTCUSDT",
            direction=direction,
            confidence=confidence,
            timestamp=f"2026-01-01T{i:04d}:00:00Z",
            actual_direction=actual,
            actual_return=random.uniform(-0.02, 0.03),
        ))
    return preds


class TestEvaluationEngine:
    def test_brier_score_perfect(self):
        engine = EvaluationEngine()
        preds = [
            Prediction("a", "BTC", "long", 1.0, "t1", "long", 0.01),
            Prediction("a", "BTC", "short", 1.0, "t2", "short", 0.01),
        ]
        assert engine.brier_score(preds) == pytest.approx(0.0)

    def test_brier_score_worst(self):
        engine = EvaluationEngine()
        preds = [
            Prediction("a", "BTC", "long", 0.9, "t1", "short", -0.01),
            Prediction("a", "BTC", "short", 0.9, "t2", "long", -0.01),
        ]
        bs = engine.brier_score(preds)
        assert bs > 0.5

    def test_directional_accuracy(self):
        engine = EvaluationEngine()
        preds = _make_predictions(20, accuracy=0.7)
        da = engine.directional_accuracy(preds)
        assert 0.5 < da < 0.9

    def test_calibration(self):
        engine = EvaluationEngine()
        preds = _make_predictions(50, accuracy=0.6)
        buckets = engine.calibration(preds, n_buckets=5)
        assert len(buckets) > 0
        for b in buckets:
            assert 0.0 <= b.actual_frequency <= 1.0

    def test_calibration_error(self):
        engine = EvaluationEngine()
        preds = _make_predictions(50, accuracy=0.6)
        ece = engine.calibration_error(preds, n_buckets=5)
        assert ece >= 0.0

    def test_profit_factor(self):
        engine = EvaluationEngine()
        preds = [
            Prediction("a", "BTC", "long", 0.8, "t1", "long", 0.05),
            Prediction("a", "BTC", "long", 0.8, "t2", "long", 0.03),
            Prediction("a", "BTC", "long", 0.8, "t3", "short", -0.02),
        ]
        pf = engine.profit_factor(preds)
        assert pf > 1.0

    def test_expectancy(self):
        engine = EvaluationEngine()
        preds = [
            Prediction("a", "BTC", "long", 0.8, "t1", "long", 0.05),
            Prediction("a", "BTC", "long", 0.8, "t2", "short", -0.02),
        ]
        exp = engine.expectancy(preds)
        assert exp == pytest.approx(0.015)

    def test_max_drawdown(self):
        engine = EvaluationEngine()
        preds = [
            Prediction("a", "BTC", "long", 0.8, "t1", "long", 0.05),
            Prediction("a", "BTC", "long", 0.8, "t2", "short", -0.08),
            Prediction("a", "BTC", "long", 0.8, "t3", "long", 0.10),
        ]
        dd = engine.max_drawdown(preds)
        assert dd >= 0.0

    def test_walk_forward(self):
        engine = EvaluationEngine()
        preds = _make_predictions(100, accuracy=0.6)
        wf = engine.walk_forward(preds, n_splits=5)
        assert len(wf) > 0
        for split in wf:
            assert "train_accuracy" in split
            assert "test_accuracy" in split

    def test_sharpe_ratio(self):
        engine = EvaluationEngine()
        preds = _make_predictions(50, accuracy=0.6)
        sr = engine.sharpe_ratio(preds)
        assert isinstance(sr, float)

    def test_evaluate_agent(self):
        engine = EvaluationEngine()
        preds = _make_predictions(30, accuracy=0.6)
        results = engine.evaluate_agent("agent-1", preds)
        assert len(results) == 6
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_empty_predictions(self):
        engine = EvaluationEngine()
        assert engine.brier_score([]) == 0.0
        assert engine.directional_accuracy([]) == 0.0
        assert engine.calibration([]) == []
        results = engine.evaluate_agent("a", [])
        assert len(results) == 6  # Returns results with zero values

    def test_deterministic(self):
        engine = EvaluationEngine()
        preds = _make_predictions(20)
        r1 = engine.brier_score(preds)
        r2 = engine.brier_score(preds)
        assert r1 == r2
