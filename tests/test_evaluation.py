from datetime import UTC, datetime

from tests._test_utils import OutcomeGenerator
from trading_harness.models import MarketRegime, OutcomeRecord
from trading_harness.services.evaluation import (
    EvaluationService,
    _compute_classification_metrics,
    _compute_ece,
    _compute_mfe_mae,
)


def _make_outcome(**overrides):
    defaults = {
        "prediction_id": "pred-1",
        "agent_id": "agent-1",
        "run_id": "run-1",
        "snapshot_id": "snap-1",
        "symbol": "BTCUSDT",
        "direction_predicted": "LONG",
        "direction_actual": "LONG",
        "confidence_predicted": 0.75,
        "entry_price": 100.0,
        "exit_price": 105.0,
        "holding_period_bars": 5,
        "realized_pnl": 50.0,
        "regime": MarketRegime.STRONG_BULL,
        "timestamp": datetime.now(UTC),
    }
    defaults.update(overrides)
    return OutcomeRecord(**defaults)


# ---------------------------------------------------------------------------
# OutcomeGenerator
# ---------------------------------------------------------------------------


def test_outcome_generator_generate():
    gen = OutcomeGenerator()
    o = gen.generate(
        prediction_id="p1",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="BTCUSDT",
        direction_predicted="LONG",
        direction_actual="LONG",
        confidence_predicted=0.8,
        entry_price=100.0,
        exit_price=110.0,
        realized_pnl=100.0,
    )
    assert o.id.startswith("outcome-")
    assert o.entry_price == 100.0
    assert o.exit_price == 110.0
    assert o.realized_pnl == 100.0
    assert o.mfe > 0


def test_outcome_generator_negative_return():
    gen = OutcomeGenerator()
    o = gen.generate(
        prediction_id="p2",
        agent_id="a1",
        run_id="r1",
        snapshot_id="s1",
        symbol="ETHUSDT",
        direction_predicted="LONG",
        direction_actual="SHORT",
        confidence_predicted=0.6,
        entry_price=50.0,
        exit_price=40.0,
        realized_pnl=-50.0,
    )
    assert o.mae > 0
    assert o.mfe == 0.0


def test_outcome_generator_invalid_prices():
    gen = OutcomeGenerator()
    try:
        gen.generate(
            prediction_id="p3",
            agent_id="a1",
            run_id="r1",
            snapshot_id="s1",
            symbol="BTC",
            direction_predicted="LONG",
            direction_actual="LONG",
            confidence_predicted=0.5,
            entry_price=0,
            exit_price=100,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero entry_price")


def test_outcome_generator_add():
    gen = OutcomeGenerator()
    o = _make_outcome()
    result = gen.add(o)
    assert result is o
    assert gen.get(o.id) is o


def test_outcome_generator_by_agent():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(agent_id="a1"))
    gen.add(_make_outcome(agent_id="a2"))
    gen.add(_make_outcome(agent_id="a1"))
    assert len(gen.by_agent("a1")) == 2
    assert len(gen.by_agent("a2")) == 1


def test_outcome_generator_by_run():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(run_id="r1"))
    gen.add(_make_outcome(run_id="r2"))
    gen.add(_make_outcome(run_id="r1"))
    assert len(gen.by_run("r1")) == 2
    assert len(gen.by_run("r2")) == 1


def test_outcome_generator_by_regime():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(regime=MarketRegime.STRONG_BULL))
    gen.add(_make_outcome(regime=MarketRegime.CRASH))
    gen.add(_make_outcome(regime=MarketRegime.STRONG_BULL))
    assert len(gen.by_regime(MarketRegime.STRONG_BULL)) == 2
    assert len(gen.by_regime(MarketRegime.CRASH)) == 1


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_all_correct():
    outcomes = [
        _make_outcome(direction_actual="LONG", realized_pnl=50.0),
        _make_outcome(direction_actual="LONG", realized_pnl=30.0),
        _make_outcome(direction_actual="LONG", realized_pnl=20.0),
    ]
    result = _compute_classification_metrics(outcomes)
    assert result.total == 3
    assert result.correct == 3
    assert result.directional_accuracy == 1.0
    assert result.true_positives == 3
    assert result.false_positives == 0
    assert result.brier_score > 0  # Even perfect predictions have non-zero Brier


def test_compute_metrics_all_wrong():
    outcomes = [
        _make_outcome(
            direction_predicted="SHORT",
            direction_actual="LONG",
            confidence_predicted=0.8,
        ),
        _make_outcome(
            direction_predicted="SHORT",
            direction_actual="LONG",
            confidence_predicted=0.7,
        ),
    ]
    result = _compute_classification_metrics(outcomes)
    assert result.correct == 0
    assert result.directional_accuracy == 0.0
    assert result.false_negatives == 2  # predicted short, actual long


def test_compute_metrics_empty():
    result = _compute_classification_metrics([])
    assert result.total == 0
    assert result.brier_score == 0.0
    assert result.calibration_error == 0.0
    assert result.expectancy == 0.0


def test_compute_expectancy():
    outcomes = [
        _make_outcome(direction_actual="LONG", realized_pnl=100.0),
        _make_outcome(direction_actual="LONG", realized_pnl=80.0),
        _make_outcome(direction_actual="SHORT", realized_pnl=-50.0),
        _make_outcome(direction_actual="SHORT", realized_pnl=-60.0),
    ]
    result = _compute_classification_metrics(outcomes)
    # 50% win rate, avg win=90, avg loss=55
    # expectancy = 0.5*90 - 0.5*55 = 17.5
    assert result.expectancy > 0
    assert abs(result.expectancy - 17.5) < 0.01


# ---------------------------------------------------------------------------
# MFE/MAE
# ---------------------------------------------------------------------------


def test_compute_mfe_mae():
    outcomes = [
        _make_outcome(realized_pnl=50.0, mfe=0.05, mae=0.02),
        _make_outcome(realized_pnl=-30.0, mfe=0.01, mae=0.03),
        _make_outcome(realized_pnl=100.0, mfe=0.10, mae=0.01),
    ]
    stats = _compute_mfe_mae(outcomes)
    assert stats["avg_mfe"] > 0
    assert stats["avg_mae"] > 0
    assert stats["max_mfe"] == 0.10


def test_compute_mfe_mae_empty():
    stats = _compute_mfe_mae([])
    assert stats == {
        "avg_mfe": 0.0,
        "avg_mae": 0.0,
        "max_mfe": 0.0,
        "max_mae": 0.0,
    }


# ---------------------------------------------------------------------------
# Calibration (ECE)
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration():
    outcomes = [
        _make_outcome(direction_actual="LONG", confidence_predicted=1.0),
        _make_outcome(direction_actual="LONG", confidence_predicted=1.0),
        _make_outcome(direction_actual="LONG", confidence_predicted=1.0),
        _make_outcome(direction_actual="LONG", confidence_predicted=1.0),
        _make_outcome(direction_actual="LONG", confidence_predicted=1.0),
    ]
    ece = _compute_ece(outcomes)
    assert ece == 0.0


def test_ece_miscalibrated():
    # High confidence but half wrong
    outcomes = [
        _make_outcome(direction_actual="LONG", confidence_predicted=0.95),
        _make_outcome(direction_actual="LONG", confidence_predicted=0.95),
        _make_outcome(direction_actual="SHORT", confidence_predicted=0.95),
        _make_outcome(direction_actual="SHORT", confidence_predicted=0.95),
    ]
    ece = _compute_ece(outcomes)
    assert ece > 0  # Should detect miscalibration


def test_ece_empty():
    assert _compute_ece([]) == 0.0


# ---------------------------------------------------------------------------
# EvaluationService
# ---------------------------------------------------------------------------


def test_evaluate_agent():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(direction_actual="LONG", realized_pnl=50.0))
    gen.add(_make_outcome(direction_actual="LONG", realized_pnl=30.0))
    gen.add(_make_outcome(direction_actual="SHORT", realized_pnl=-20.0))

    svc = EvaluationService(gen)
    result = svc.evaluate_agent("agent-1")

    assert result["observations"] == 3
    assert "brier_score" in result["metrics"]
    assert "directional_accuracy" in result["metrics"]
    assert "expectancy" in result["metrics"]
    assert "confusion_matrix" in result
    assert result["confusion_matrix"]["tp"] == 2
    assert result["confusion_matrix"]["fp"] == 1


def test_evaluate_agent_empty():
    gen = OutcomeGenerator()
    svc = EvaluationService(gen)
    result = svc.evaluate_agent("agent-999")
    assert result["observations"] == 0
    assert result["metrics"] == {}


def test_evaluate_agent_with_run_filter():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(run_id="r1"))
    gen.add(_make_outcome(run_id="r1"))
    gen.add(_make_outcome(run_id="r2"))

    svc = EvaluationService(gen)
    result_r1 = svc.evaluate_agent("agent-1", run_id="r1")
    assert result_r1["observations"] == 2

    result_r2 = svc.evaluate_agent("agent-1", run_id="r2")
    assert result_r2["observations"] == 1


def test_evaluate_regime_performance():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(regime=MarketRegime.STRONG_BULL, realized_pnl=100.0))
    gen.add(_make_outcome(regime=MarketRegime.STRONG_BULL, realized_pnl=80.0))
    gen.add(_make_outcome(regime=MarketRegime.CRASH, realized_pnl=-200.0))

    svc = EvaluationService(gen)
    result = svc.evaluate_regime_performance("agent-1", MarketRegime.STRONG_BULL)
    assert result["observations"] == 2
    assert result["metrics"]["total_pnl"] == 180.0

    result_crash = svc.evaluate_regime_performance("agent-1", MarketRegime.CRASH)
    assert result_crash["observations"] == 1


def test_evaluate_drawdown():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(realized_pnl=100.0, timestamp=datetime(2025, 1, 1, tzinfo=UTC)))
    gen.add(_make_outcome(realized_pnl=50.0, timestamp=datetime(2025, 1, 2, tzinfo=UTC)))
    gen.add(_make_outcome(realized_pnl=-200.0, timestamp=datetime(2025, 1, 3, tzinfo=UTC)))

    svc = EvaluationService(gen)
    result = svc.evaluate_drawdown("agent-1")
    assert result["observations"] == 3
    assert result["max_drawdown"] > 0
    assert result["peak_equity"] > 0


def test_evaluate_drawdown_empty():
    gen = OutcomeGenerator()
    svc = EvaluationService(gen)
    result = svc.evaluate_drawdown("agent-999")
    assert result["max_drawdown"] == 0.0


def test_evaluate_out_of_sample():
    gen = OutcomeGenerator()
    svc = EvaluationService(gen)

    # Train set: 2 correct, 1 wrong
    train = [
        _make_outcome(direction_actual="LONG", realized_pnl=50.0),
        _make_outcome(direction_actual="LONG", realized_pnl=40.0),
        _make_outcome(direction_predicted="SHORT", direction_actual="LONG", confidence_predicted=0.3, realized_pnl=10.0),
    ]
    # Test set: same quality ratio — 2 correct, 1 wrong
    test = [
        _make_outcome(direction_actual="LONG", realized_pnl=30.0),
        _make_outcome(direction_actual="LONG", realized_pnl=25.0),
        _make_outcome(direction_predicted="SHORT", direction_actual="LONG", confidence_predicted=0.3, realized_pnl=5.0),
    ]

    result = svc.evaluate_out_of_sample("agent-1", train, test)
    assert result["train_observations"] == 3
    assert result["test_observations"] == 3
    assert "oos_pass" in result
    assert result["oos_pass"] is True


def test_evaluate_out_of_sample_insufficient_test():
    gen = OutcomeGenerator()
    svc = EvaluationService(gen)

    train = [_make_outcome(direction_actual="LONG", realized_pnl=50.0)]
    test = []

    result = svc.evaluate_out_of_sample("agent-1", train, test)
    assert result["oos_pass"] is False


def test_evaluate_walk_forward():
    gen = OutcomeGenerator()
    svc = EvaluationService(gen)

    # Create enough outcomes for 2+ windows
    outcomes = []
    for i in range(120):
        o = _make_outcome(
            direction_actual="LONG",
            realized_pnl=10.0 if i % 3 != 0 else -20.0,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        outcomes.append(o)

    result = svc.evaluate_walk_forward("agent-1", outcomes, window_size=20, step_size=10)
    assert result["agent_id"] == "agent-1"
    assert result["num_windows"] > 0
    assert "windows" in result
    assert "stable" in result
    assert "avg_stability" in result


def test_evaluate_walk_forward_insufficient_data():
    gen = OutcomeGenerator()
    svc = EvaluationService(gen)

    few_outcomes = [
        _make_outcome(direction_actual="LONG", realized_pnl=10.0),
        _make_outcome(direction_actual="SHORT", realized_pnl=-5.0),
    ]
    result = svc.evaluate_walk_forward("agent-1", few_outcomes, window_size=50)
    assert result["stable"] is False
    assert result["reason"] == "INSUFFICIENT_DATA"


def test_evaluation_results_storage():
    gen = OutcomeGenerator()
    gen.add(_make_outcome())
    svc = EvaluationService(gen)
    svc.evaluate_agent("agent-1")

    results = svc.get_results("agent-1")
    assert len(results) >= 1
    assert results[0].metric_name == "aggregate_evaluation"
    assert results[0].agent_id == "agent-1"


def test_evaluation_results_filter_by_agent():
    gen = OutcomeGenerator()
    gen.add(_make_outcome(agent_id="agent-1"))
    gen.add(_make_outcome(agent_id="agent-2"))

    svc = EvaluationService(gen)
    svc.evaluate_agent("agent-1")
    svc.evaluate_agent("agent-2")

    results_1 = svc.get_results("agent-1")
    results_2 = svc.get_results("agent-2")
    assert len(results_1) >= 1
    assert len(results_2) >= 1