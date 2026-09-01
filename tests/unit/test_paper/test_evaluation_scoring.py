"""Tests for evaluation worker — scoring, agent evaluation, resolution."""

from __future__ import annotations

import pytest
from apps.evaluation_worker.worker import (
    AgentEvaluator,
    ResolutionEngine,
    ScoringEngine,
)


class TestScoringEngineBrier:
    """Test Brier score computation."""

    def test_perfect_predictions(self) -> None:
        scores = ScoringEngine.compute_brier_score([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])
        assert scores == 0.0

    def test_worst_predictions(self) -> None:
        scores = ScoringEngine.compute_brier_score([0.0, 0.0, 1.0, 1.0], [1, 1, 0, 0])
        assert scores == 1.0

    def test_partial_prediction(self) -> None:
        # (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18
        scores = ScoringEngine.compute_brier_score([0.7, 0.3], [1.0, 0.0])
        assert scores == pytest.approx(0.09)

    def test_empty_list(self) -> None:
        assert ScoringEngine.compute_brier_score([], []) == 0.0

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            ScoringEngine.compute_brier_score([0.5], [1, 0])


class TestScoringEngineLogLoss:
    """Test log loss computation."""

    def test_perfect_predictions(self) -> None:
        eps = 1e-15
        scores = ScoringEngine.compute_log_loss([1.0 - eps, eps], [1.0, 0.0])
        assert scores == pytest.approx(0.0, abs=1e-10)

    def test_uncertain_predictions(self) -> None:
        # Both predictions at 0.5 → high log loss
        scores = ScoringEngine.compute_log_loss([0.5, 0.5], [1.0, 0.0])
        assert scores == pytest.approx(0.693147)  # -log(0.5)

    def test_empty_list(self) -> None:
        assert ScoringEngine.compute_log_loss([], []) == 0.0


class TestScoringEngineConfusionMatrix:
    """Test confusion matrix computation."""

    def test_perfect_classification(self) -> None:
        cm = ScoringEngine.compute_confusion_matrix([1, 1, 0, 0], [1, 1, 0, 0])
        assert cm == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    def test_mixed_classification(self) -> None:
        cm = ScoringEngine.compute_confusion_matrix([1, 0, 1, 0], [1, 1, 1, 0])
        # pred=[1,0,1,0], actual=[1,1,1,0]
        # idx0: pred=1,actual=1→TP; idx1: pred=0,actual=1→FN;
        # idx2: pred=1,actual=1→TP; idx3: pred=0,actual=0→TN
        assert cm == {"tp": 2, "fp": 0, "tn": 1, "fn": 1}

    def test_all_wrong(self) -> None:
        cm = ScoringEngine.compute_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
        # All predictions wrong but counts are per-class
        assert cm == {"tp": 0, "fp": 2, "tn": 0, "fn": 2}


class TestScoringEnginePrecisionRecall:
    """Test precision, recall, F1 computation."""

    def test_perfect_scores(self) -> None:
        scores = ScoringEngine.compute_precision_recall_f1(
            [1, 1, 0, 0], [1, 1, 0, 0]
        )
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1"] == 1.0

    def test_no_positives_predicted(self) -> None:
        scores = ScoringEngine.compute_precision_recall_f1([0, 0], [1, 0])
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0


class TestScoringEngineScoreAgent:
    """Test comprehensive agent scoring."""

    def test_score_agent_basic(self) -> None:
        scoring = ScoringEngine()
        result = scoring.score_agent(
            [0.8, 0.3, 0.9, 0.2],
            [1, 0, 1, 0],
        )
        assert "brier_score" in result
        assert "log_loss" in result
        assert result["n_samples"] == 4
        assert "classification" in result

    def test_score_agent_with_custom_threshold(self) -> None:
        scoring = ScoringEngine()
        result = scoring.score_agent(
            [0.8, 0.3, 0.9, 0.2],
            [1, 0, 1, 0],
            thresholds=[0.3, 0.5, 0.7],
        )
        assert "threshold_0.3" in result["classification"]
        assert "threshold_0.5" in result["classification"]
        assert "threshold_0.7" in result["classification"]


class TestAgentEvaluator:
    """Test agent evaluation and comparison."""

    def test_evaluate_agent(self) -> None:
        evaluator = AgentEvaluator()
        result = evaluator.evaluate_agent(
            agent_id="test_agent",
            predictions=[0.8, 0.3, 0.9, 0.2],
            actuals=[1, 0, 1, 0],
            metadata={"window": "2024-01"},
        )
        assert result["agent_id"] == "test_agent"
        assert result["evaluation"]["brier_score"] == pytest.approx(0.045)
        assert "metadata" in result

    def test_compare_agents(self) -> None:
        evaluator = AgentEvaluator()
        results = [
            {
                "evaluation": {"brier_score": 0.1},
                "agent_id": "agent_a",
            },
            {
                "evaluation": {"brier_score": 0.05},
                "agent_id": "agent_b",
            },
        ]
        ranked = evaluator.compare_agents(results, metric="brier_score")
        assert ranked[0]["agent_id"] == "agent_b"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2

    def test_champion_challenger_promote(self) -> None:
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="champion",
            challenger_id="challenger",
            champion_scores={"brier_score": 0.15},
            challenger_scores={"brier_score": 0.10},
            required_improvement=0.01,
        )
        assert result["promote_challenger"] is True
        assert result["improvements"]["brier_score"] is True

    def test_champion_challenger_keep(self) -> None:
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="champion",
            challenger_id="challenger",
            champion_scores={"brier_score": 0.05},
            challenger_scores={"brier_score": 0.10},
            required_improvement=0.01,
        )
        assert result["promote_challenger"] is False


class TestResolutionEngine:
    """Test prediction resolution."""

    def test_resolve_correct(self) -> None:
        engine = ResolutionEngine()
        result = engine.resolve_prediction(
            {"id": "p1", "direction": "LONG", "confidence": 0.8},
            {"direction": "LONG", "timestamp": "2024-01-02"},
        )
        assert result["correct"] is True
        assert result["score"] == pytest.approx(0.8)

    def test_resolve_incorrect(self) -> None:
        engine = ResolutionEngine()
        result = engine.resolve_prediction(
            {"id": "p2", "direction": "LONG", "confidence": 0.7},
            {"direction": "SHORT", "timestamp": "2024-01-02"},
        )
        assert result["correct"] is False
        assert result["score"] == pytest.approx(0.3)

    def test_resolve_batch(self) -> None:
        engine = ResolutionEngine()
        predictions = [
            {"id": "p1", "direction": "LONG", "confidence": 0.8},
            {"id": "p2", "direction": "SHORT", "confidence": 0.6},
        ]
        outcomes = [
            {"direction": "LONG", "timestamp": "2024-01-02"},
            {"direction": "SHORT", "timestamp": "2024-01-03"},
        ]
        result = engine.resolve_batch(predictions, outcomes)
        assert result["total"] == 2
        assert result["correct"] == 2
        assert result["accuracy"] == 1.0
        assert "resolved" in result
        assert len(result["resolved"]) == 2
