"""Tests for evaluation worker module — BacktestEvaluator, ScoringEngine, AgentEvaluator, ResolutionEngine."""

from __future__ import annotations

import numpy as np
import pytest
from apps.evaluation_worker.worker import (
    AgentEvaluator,
    BacktestEvaluator,
    ResolutionEngine,
    ScoringEngine,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_trades() -> list[dict]:
    """Mixed winning and losing trades for evaluation."""
    return [
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 150.0,
            "exit_price": 160.0,
            "quantity": 10,
            "timestamp": "2024-01-03T10:00:00",
            "commission": 1.0,
            "pnl": 99.0,
        },
        {
            "symbol": "GOOG",
            "direction": "short",
            "entry_price": 2800.0,
            "exit_price": 2750.0,
            "quantity": 5,
            "timestamp": "2024-01-02T10:00:00",
            "commission": 2.0,
            "pnl": 248.0,
        },
        {
            "symbol": "TSLA",
            "direction": "long",
            "entry_price": 200.0,
            "exit_price": 180.0,
            "quantity": 10,
            "timestamp": "2024-01-04T10:00:00",
            "commission": 1.0,
            "pnl": -201.0,
        },
    ]


@pytest.fixture
def predictions() -> list[float]:
    """Probabilistic predictions for scoring tests."""
    return [0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.85]


@pytest.fixture
def actuals_binary() -> list[int]:
    """Actual binary outcomes."""
    return [1, 0, 1, 0, 1, 1, 0, 1]


# ── BacktestEvaluator tests ───────────────────────────────────────────────────


class TestBacktestEvaluator:
    """Tests for BacktestEvaluator class."""

    def test_empty_trade_list(self) -> None:
        """Empty trade list returns zeroed metrics."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate([])
        assert result["total_trades"] == 0
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["total_pnl"] == 0.0
        assert result["profit_factor"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["return_pct"] == 0.0
        assert result["trade_list"] == []

    def test_single_winning_trade(self) -> None:
        """Single winning trade produces correct metrics."""
        evaluator = BacktestEvaluator()
        trades = [{
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "quantity": 10,
            "timestamp": "2024-01-01T00:00:00",
            "commission": 1.0,
            "pnl": 99.0,
        }]
        result = evaluator.evaluate(trades)
        assert result["total_trades"] == 1
        assert result["winning_trades"] == 1
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 1.0
        assert result["total_pnl"] == 99.0
        assert result["profit_factor"] == 0.0  # no losses → 0

    def test_single_losing_trade(self) -> None:
        """Single losing trade produces correct metrics."""
        evaluator = BacktestEvaluator()
        trades = [{
            "symbol": "GOOG",
            "direction": "short",
            "entry_price": 2800.0,
            "exit_price": 2900.0,
            "quantity": 5,
            "timestamp": "2024-01-01T00:00:00",
            "commission": 2.0,
            "pnl": -502.0,
        }]
        result = evaluator.evaluate(trades)
        assert result["total_trades"] == 1
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 1
        assert result["win_rate"] == 0.0
        assert result["total_pnl"] == -502.0

    def test_mixed_trades_metrics(self, sample_trades: list[dict]) -> None:
        """Mixed trades produce expected win/loss counts."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades)
        assert result["total_trades"] == 3
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert abs(result["win_rate"] - 2 / 3) < 1e-5

    def test_mixed_trades_total_pnl(self, sample_trades: list[dict]) -> None:
        """Total PnL is the sum of individual trade PnLs."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades)
        expected_pnl = 99.0 + 248.0 - 201.0
        assert result["total_pnl"] == expected_pnl

    def test_mixed_trades_profit_factor(self, sample_trades: list[dict]) -> None:
        """Profit factor = total_wins / total_losses."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades)
        total_wins = 99.0 + 248.0
        total_losses = 201.0
        expected_pf = total_wins / total_losses
        assert abs(result["profit_factor"] - expected_pf) < 0.01

    def test_return_pct_calculation(self, sample_trades: list[dict]) -> None:
        """Return percentage matches expected formula."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades, initial_capital=100000.0)
        expected = (99.0 + 248.0 - 201.0) / 100000.0 * 100.0
        assert abs(result["return_pct"] - expected) < 0.1

    def test_max_drawdown_bounds(self, sample_trades: list[dict]) -> None:
        """Max drawdown is in [0, 1]."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades)
        assert 0.0 <= result["max_drawdown"] <= 1.0

    def test_sharpe_ratio_type(self, sample_trades: list[dict]) -> None:
        """Sharpe ratio is a finite float."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades)
        assert isinstance(result["sharpe_ratio"], float)
        assert np.isfinite(result["sharpe_ratio"]) or result["sharpe_ratio"] == 0.0

    def test_equity_curve_length(self, sample_trades: list[dict]) -> None:
        """Equity curve has initial_capital + len(trades) entries."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades, initial_capital=50000.0)
        # equity_curve is internal; verify trade_list enriched
        assert len(result["trade_list"]) == 3
        for trade in result["trade_list"]:
            assert "is_winner" in trade
            assert "is_loser" in trade
            assert "computed_pnl" in trade

    def test_enriched_trades_correct_labels(self, sample_trades: list[dict]) -> None:
        """Enriched trades have correct winner/loser labels."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades)
        {t["symbol"]: t["pnl"] for t in sample_trades}
        for enriched in result["trade_list"]:
            pnl = enriched["computed_pnl"]
            if pnl > 0:
                assert enriched["is_winner"] is True
                assert enriched["is_loser"] is False
            elif pnl < 0:
                assert enriched["is_winner"] is False
                assert enriched["is_loser"] is True
            else:
                assert enriched["is_winner"] is False
                assert enriched["is_loser"] is False

    def test_trade_list_sorted_by_timestamp(self, sample_trades: list[dict]) -> None:
        """Trades in result are sorted chronologically."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(sample_trades)
        timestamps = [t["timestamp"] for t in result["trade_list"]]
        assert timestamps == sorted(timestamps)


class TestBacktestEvaluatorEdgeCases:
    """Edge cases for BacktestEvaluator."""

    def test_all_winning_trades(self) -> None:
        """All wins: profit_factor should be 0 (no losses)."""
        evaluator = BacktestEvaluator()
        trades = [
            {
                "symbol": "A",
                "direction": "long",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "quantity": 10,
                "timestamp": f"2024-01-{i:02d}T00:00:00",
                "commission": 1.0,
                "pnl": 99.0,
            }
            for i in range(1, 6)
        ]
        result = evaluator.evaluate(trades)
        assert result["winning_trades"] == 5
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 1.0
        assert result["profit_factor"] == 0.0

    def test_all_losing_trades(self) -> None:
        """All losses: profit_factor should be 0."""
        evaluator = BacktestEvaluator()
        trades = [
            {
                "symbol": "B",
                "direction": "long",
                "entry_price": 100.0,
                "exit_price": 90.0,
                "quantity": 10,
                "timestamp": f"2024-01-{i:02d}T00:00:00",
                "commission": 1.0,
                "pnl": -101.0,
            }
            for i in range(1, 6)
        ]
        result = evaluator.evaluate(trades)
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 5
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0

    def test_zero_pnl_trade(self) -> None:
        """Trade with zero PnL is neither winner nor loser."""
        evaluator = BacktestEvaluator()
        trades = [{
            "symbol": "C",
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 100.0,
            "quantity": 10,
            "timestamp": "2024-01-01T00:00:00",
            "commission": 0.0,
            "pnl": 0.0,
        }]
        result = evaluator.evaluate(trades)
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["trade_list"][0]["is_winner"] is False
        assert result["trade_list"][0]["is_loser"] is False

    def test_high_drawdown_scenario(self) -> None:
        """Large loss followed by recovery shows max drawdown."""
        evaluator = BacktestEvaluator()
        trades = [
            {"symbol": "A", "direction": "long", "entry_price": 100.0, "exit_price": 110.0,
             "quantity": 100, "timestamp": "2024-01-01T00:00:00", "commission": 0, "pnl": 990.0},
            {"symbol": "B", "direction": "long", "entry_price": 110.0, "exit_price": 50.0,
             "quantity": 100, "timestamp": "2024-01-02T00:00:00", "commission": 0, "pnl": -5990.0},
            {"symbol": "C", "direction": "long", "entry_price": 50.0, "exit_price": 100.0,
             "quantity": 100, "timestamp": "2024-01-03T00:00:00", "commission": 0, "pnl": 4990.0},
        ]
        result = evaluator.evaluate(trades, initial_capital=100000.0)
        # After loss, equity drops to 94000, so drawdown >= 6000/100000 = 0.06
        assert result["max_drawdown"] > 0.05


class TestBacktestEvaluatorReport:
    """Tests for BacktestEvaluator.generate_report."""

    def test_report_header(self, sample_trades: list[dict]) -> None:
        evaluator = BacktestEvaluator()
        evaluation = evaluator.evaluate(sample_trades)
        report = evaluator.generate_report(evaluation)
        assert "BACKTEST EVALUATION REPORT" in report
        assert report.startswith("=" * 60)

    def test_report_sections(self, sample_trades: list[dict]) -> None:
        evaluator = BacktestEvaluator()
        evaluation = evaluator.evaluate(sample_trades)
        report = evaluator.generate_report(evaluation)
        assert "--- TRADE SUMMARY ---" in report
        assert "--- PERFORMANCE ---" in report
        assert "--- RISK METRICS ---" in report

    def test_report_includes_all_metrics(self, sample_trades: list[dict]) -> None:
        evaluator = BacktestEvaluator()
        evaluation = evaluator.evaluate(sample_trades)
        report = evaluator.generate_report(evaluation)
        expected_labels = [
            "Total Trades", "Winning Trades", "Losing Trades",
            "Win Rate", "Total PnL", "Avg Win", "Avg Loss",
            "Avg Trade PnL", "Return %", "Profit Factor",
            "Max Drawdown", "Sharpe Ratio",
        ]
        for label in expected_labels:
            assert label in report, f"Missing label in report: {label}"

    def test_report_empty_evaluation(self) -> None:
        evaluator = BacktestEvaluator()
        report = evaluator.generate_report({})
        assert "Total Trades:    0" in report
        assert "Win Rate:        0.00%" in report


# ── ScoringEngine tests ───────────────────────────────────────────────────────


class TestScoringEngine:
    """Tests for ScoringEngine class."""

    def test_compute_brier_score_perfect(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Perfect predictions (close to actuals) yield low Brier score."""
        # Predictions near actuals
        preds = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        actuals = [1, 0, 1, 0, 1, 1, 0, 1]
        score = ScoringEngine.compute_brier_score(preds, actuals)
        assert score == 0.0

    def test_compute_brier_score_random(self) -> None:
        """Random predictions yield Brier score around 0.25."""
        preds = [0.5] * 10
        actuals = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        score = ScoringEngine.compute_brier_score(preds, actuals)
        # Each error is (0.5 - 1)^2 or (0.5 - 0)^2 = 0.25
        assert score == pytest.approx(0.25)

    def test_compute_brier_score_empty(self) -> None:
        """Empty inputs return 0.0."""
        score = ScoringEngine.compute_brier_score([], [])
        assert score == 0.0

    def test_compute_brier_score_length_mismatch(self) -> None:
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            ScoringEngine.compute_brier_score([0.5, 0.3], [1])

    def test_compute_log_loss_perfect(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Perfect predictions yield very low log loss."""
        preds = [1.0 - 1e-15, 1e-15, 1.0 - 1e-15, 1e-15, 1.0 - 1e-15, 1.0 - 1e-15, 1e-15, 1.0 - 1e-15]
        actuals = [1, 0, 1, 0, 1, 1, 0, 1]
        loss = ScoringEngine.compute_log_loss(preds, actuals)
        assert loss < 0.1

    def test_compute_log_loss_empty(self) -> None:
        """Empty inputs return 0.0."""
        loss = ScoringEngine.compute_log_loss([], [])
        assert loss == 0.0

    def test_compute_log_loss_length_mismatch(self) -> None:
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            ScoringEngine.compute_log_loss([0.5], [1, 0])

    def test_compute_confusion_matrix_perfect(self) -> None:
        """Perfect classification yields tp=n, fp=0, tn=0, fn=0."""
        preds = [1, 1, 1, 0, 0]
        actuals = [1, 1, 1, 0, 0]
        cm = ScoringEngine.compute_confusion_matrix(preds, actuals)
        assert cm == {"tp": 3, "fp": 0, "tn": 2, "fn": 0}

    def test_compute_confusion_matrix_worst(self) -> None:
        """Worst classification (all wrong) yields fp=n, tn=0."""
        preds = [1, 1, 1, 1, 1]
        actuals = [0, 0, 0, 0, 0]
        cm = ScoringEngine.compute_confusion_matrix(preds, actuals)
        assert cm == {"tp": 0, "fp": 5, "tn": 0, "fn": 0}

    def test_compute_confusion_matrix_mixed(self) -> None:
        """Mixed classification yields correct counts."""
        preds = [1, 0, 1, 0, 1]
        actuals = [1, 0, 0, 0, 1]
        cm = ScoringEngine.compute_confusion_matrix(preds, actuals)
        assert cm == {"tp": 2, "fp": 1, "tn": 2, "fn": 0}

    def test_compute_confusion_matrix_length_mismatch(self) -> None:
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            ScoringEngine.compute_confusion_matrix([1, 0], [1])

    def test_compute_precision_recall_f1_perfect(self) -> None:
        """Perfect classification yields precision=recall=f1=1.0."""
        preds = [1, 0, 1, 0, 1]
        actuals = [1, 0, 1, 0, 1]
        metrics = ScoringEngine.compute_precision_recall_f1(preds, actuals)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_compute_precision_recall_f1_no_positives(self) -> None:
        """All negative: precision=recall=f1=0.0."""
        preds = [0, 0, 0, 0]
        actuals = [0, 0, 0, 0]
        metrics = ScoringEngine.compute_precision_recall_f1(preds, actuals)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_compute_precision_recall_f1_imbalanced(self) -> None:
        """Imbalanced: more negatives than positives."""
        preds = [0, 0, 0, 0, 0]
        actuals = [1, 0, 0, 0, 0]
        cm = ScoringEngine.compute_confusion_matrix(preds, actuals)
        assert cm["fn"] == 1
        assert cm["tn"] == 4


class TestScoringEngineScoreAgent:
    """Tests for ScoringEngine.score_agent."""

    def test_score_agent_default_threshold(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Score agent with default threshold produces all required keys."""
        engine = ScoringEngine()
        result = engine.score_agent(predictions, actuals_binary)
        assert "brier_score" in result
        assert "log_loss" in result
        assert "n_samples" in result
        assert result["n_samples"] == len(predictions)
        assert "classification" in result
        assert "threshold_0.5" in result["classification"]

    def test_score_agent_custom_threshold(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Score agent with custom threshold includes it."""
        engine = ScoringEngine()
        result = engine.score_agent(predictions, actuals_binary, thresholds=[0, 1])
        assert "threshold_0.0" in result["classification"]
        assert "threshold_1.0" in result["classification"]

    def test_score_agent_brier_value_range(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Brier score is in [0, 1]."""
        engine = ScoringEngine()
        result = engine.score_agent(predictions, actuals_binary)
        assert 0.0 <= result["brier_score"] <= 1.0


# ── AgentEvaluator tests ──────────────────────────────────────────────────────


class TestAgentEvaluator:
    """Tests for AgentEvaluator class."""

    def test_evaluate_agent(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Evaluate agent returns structured result."""
        evaluator = AgentEvaluator()
        result = evaluator.evaluate_agent("agent-1", predictions, actuals_binary)
        assert result["agent_id"] == "agent-1"
        assert "evaluation" in result
        assert "metadata" in result
        assert result["evaluation"]["n_samples"] == len(predictions)

    def test_evaluate_agent_with_metadata(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Evaluate agent with custom metadata."""
        evaluator = AgentEvaluator()
        meta = {"window": "2024-Q1", "market": "US"}
        result = evaluator.evaluate_agent("agent-2", predictions, actuals_binary, metadata=meta)
        assert result["metadata"] == meta

    def test_compare_agents(self, predictions: list[float], actuals_binary: list[int]) -> None:
        """Compare agents sorts by specified metric."""
        evaluator = AgentEvaluator()
        # Agent with predictions that are all 0.5 → poor
        agent_results = [
            evaluator.evaluate_agent("good", [1.0] * 5, [1] * 5),
            evaluator.evaluate_agent("bad", [0.5] * 5, [1] * 5),
        ]
        sorted_results = evaluator.compare_agents(agent_results, metric="brier_score")
        # good agent should have lower brier score → ranked first
        assert sorted_results[0]["agent_id"] == "good"
        assert sorted_results[1]["agent_id"] == "bad"
        assert sorted_results[0]["rank"] == 1
        assert sorted_results[1]["rank"] == 2

    def test_compare_agents_empty(self) -> None:
        """Compare empty agent list returns empty."""
        evaluator = AgentEvaluator()
        result = evaluator.compare_agents([])
        assert result == []


class TestChampionChallenger:
    """Tests for champion-challenger comparison logic."""

    def test_challenger_better(self) -> None:
        """Challenger better on ALL metrics → promote."""
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="c1",
            challenger_id="c2",
            champion_scores={"brier_score": 0.3, "max_drawdown": 0.15},
            challenger_scores={"brier_score": 0.1, "max_drawdown": 0.05},
        )
        # Both metrics improved → promote
        assert result["promote_challenger"] is True
        assert result["champion_id"] == "c1"
        assert result["challenger_id"] == "c2"
        assert result["improvements"]["brier_score"] is True
        assert result["improvements"]["max_drawdown"] is True

    def test_challenger_worse(self) -> None:
        """Challenger clearly worse → no promote."""
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="c1",
            challenger_id="c2",
            champion_scores={"brier_score": 0.1, "accuracy": 0.9},
            challenger_scores={"brier_score": 0.3, "accuracy": 0.7},
        )
        assert result["promote_challenger"] is False

    def test_challenger_requires_improvement_threshold(self) -> None:
        """Challenger not improved enough → no promote."""
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="c1",
            challenger_id="c2",
            champion_scores={"brier_score": 0.3},
            challenger_scores={"brier_score": 0.29},
            required_improvement=0.05,
        )
        assert result["promote_challenger"] is False
        assert result["required_improvement"] == 0.05

    def test_improvements_dict(self) -> None:
        """Improvements dict correctly flags each metric."""
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="c1",
            challenger_id="c2",
            champion_scores={"brier_score": 0.1, "f1": 0.5},
            challenger_scores={"brier_score": 0.3, "f1": 0.4},
        )
        # brier_score worse: (0.1-0.3)/0.1 = -2.0 → False
        assert result["improvements"]["brier_score"] is False
        # f1 improves 20%: (0.5-0.4)/0.5 = 0.2 > 0.01 → True
        assert result["improvements"]["f1"] is True
        # Mixed results → no promote
        assert result["promote_challenger"] is False

    def test_zero_champion_value(self) -> None:
        """Champion with zero metric: any positive challenger is worse."""
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="c1",
            challenger_id="c2",
            champion_scores={"penalty": 0.0},
            challenger_scores={"penalty": 0.0},
        )
        # Both zero → not better
        assert result["promote_challenger"] is False

    def test_empty_scores(self) -> None:
        """Empty scores dict → no promote."""
        evaluator = AgentEvaluator()
        result = evaluator.champion_challenger(
            champion_id="c1",
            challenger_id="c2",
            champion_scores={},
            challenger_scores={},
        )
        assert result["promote_challenger"] is False


# ── ResolutionEngine tests ────────────────────────────────────────────────────


class TestResolutionEngine:
    """Tests for ResolutionEngine class."""

    def test_resolve_correct_prediction(self) -> None:
        """Correctly predicted direction yields score == confidence."""
        engine = ResolutionEngine()
        pred = {"id": "p1", "direction": "bullish", "confidence": 0.8}
        outcome = {"direction": "bullish", "timestamp": "2024-01-01"}
        result = engine.resolve_prediction(pred, outcome)
        assert result["correct"] is True
        assert result["score"] == pytest.approx(0.8)

    def test_resolve_incorrect_prediction(self) -> None:
        """Incorrectly predicted direction yields score == 1 - confidence."""
        engine = ResolutionEngine()
        pred = {"id": "p2", "direction": "bullish", "confidence": 0.8}
        outcome = {"direction": "bearish", "timestamp": "2024-01-01"}
        result = engine.resolve_prediction(pred, outcome)
        assert result["correct"] is False
        assert result["score"] == pytest.approx(0.2)

    def test_resolve_missing_direction(self) -> None:
        """Missing direction defaults to empty string."""
        engine = ResolutionEngine()
        pred = {"id": "p3", "confidence": 0.5}
        outcome = {"direction": "bullish", "timestamp": "2024-01-01"}
        result = engine.resolve_prediction(pred, outcome)
        assert result["correct"] is False

    def test_resolve_batch(self) -> None:
        """Resolve batch returns aggregated stats."""
        engine = ResolutionEngine()
        predictions = [
            {"id": "p1", "direction": "bullish", "confidence": 0.9},
            {"id": "p2", "direction": "bearish", "confidence": 0.7},
            {"id": "p3", "direction": "bullish", "confidence": 0.6},
        ]
        outcomes = [
            {"direction": "bullish", "timestamp": "2024-01-01"},
            {"direction": "bullish", "timestamp": "2024-01-02"},
            {"direction": "bearish", "timestamp": "2024-01-03"},
        ]
        result = engine.resolve_batch(predictions, outcomes)  # type: ignore[reportGeneralTypeIssues]
        assert result["total"] == 3
        assert result["correct"] == 1  # only p1 is correct
        assert result["accuracy"] == pytest.approx(1 / 3)
        assert "resolved" in result
        assert len(result["resolved"]) == 3

    def test_resolve_batch_empty(self) -> None:
        """Empty batch returns zero stats."""
        engine = ResolutionEngine()
        result = engine.resolve_batch([], [])  # type: ignore[reportGeneralTypeIssues]
        assert result["total"] == 0
        assert result["correct"] == 0
        assert result["accuracy"] == 0.0
        assert result["resolved"] == []

    def test_resolve_batch_length_mismatch(self) -> None:
        """Mismatched batch lengths should be handled by zip strict."""
        engine = ResolutionEngine()
        predictions = [
            {"id": "p1", "direction": "bullish", "confidence": 0.9},
        ]
        outcomes = [
            {"direction": "bullish", "timestamp": "2024-01-01"},
            {"direction": "bullish", "timestamp": "2024-01-02"},
        ]
        # zip with strict=True raises ValueError on length mismatch
        with pytest.raises(ValueError):
            engine.resolve_batch(predictions, outcomes)

    def test_resolution_score_range(self) -> None:
        """Resolution scores are in [0, 1]."""
        engine = ResolutionEngine()
        for confidence in [0.1, 0.5, 0.9]:
            pred = {"id": "x", "direction": "bullish", "confidence": confidence}
            # Correct
            r = engine.resolve_prediction(pred, {"direction": "bullish", "timestamp": "t"})
            assert 0.0 <= r["score"] <= 1.0
            # Incorrect
            r = engine.resolve_prediction(pred, {"direction": "bearish", "timestamp": "t"})
            assert 0.0 <= r["score"] <= 1.0
