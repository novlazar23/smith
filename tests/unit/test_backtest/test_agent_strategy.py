"""Tests für die AgentEnsembleStrategy (Warmup, Rhythmus, Gate-Mapping, Cache)."""

from __future__ import annotations

import pytest
from apps.backtest.agent_strategy import dominant_direction, extract_per_agent
from packages.backtesting.strategies import SignalAction
from tests.unit.test_backtest.conftest import (
    FakeReport,
    StubPipeline,
    drive,
    make_candles,
    make_pipeline_result,
    make_strategy,
    trending_up,
)


class TestWindowWarmup:
    """Keine Evaluation, bevor das Fenster min_candles Kerzen enthält."""

    def test_no_evaluation_before_min_candles(self) -> None:
        strategy = make_strategy(make_pipeline_result())
        drive(strategy, trending_up(29))
        assert strategy.evaluations == []
        assert strategy.consensus_cache == {}
        assert strategy.signal_events == []

    def test_first_evaluation_exactly_at_min_candles(self) -> None:
        strategy = make_strategy(make_pipeline_result())
        candles = trending_up(30)
        drive(strategy, candles)
        assert len(strategy.evaluations) == 1
        assert strategy.evaluations[0].timestamp == candles[29].timestamp

    def test_candle_limit_capped_window(self) -> None:
        strategy = make_strategy(make_pipeline_result(), candle_limit=50, min_candles=30)
        drive(strategy, trending_up(80))
        # erste Evaluation bei Bar 30 (1-basiert), dann alle 5 → 30..80
        assert len(strategy.evaluations) == (80 - 30) // 5 + 1


class TestEvaluateEvery:
    """Der Evaluations-Rhythmus wird eingehalten."""

    def test_every_fifth_bar(self) -> None:
        strategy = make_strategy(make_pipeline_result(), evaluate_every=5, min_candles=30)
        candles = trending_up(100)
        drive(strategy, candles)
        # 1-basierte Bars 30, 35, ..., 100 → 15 Evaluations
        assert len(strategy.evaluations) == 15
        expected_bars = [29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
        assert [e.timestamp for e in strategy.evaluations] == [candles[i].timestamp for i in expected_bars]

    def test_evaluate_every_two(self) -> None:
        strategy = make_strategy(make_pipeline_result(), evaluate_every=2, min_candles=10)
        drive(strategy, trending_up(20))
        # 1-basierte Bars 10, 12, 14, 16, 18, 20
        assert len(strategy.evaluations) == 6


class TestGateMapping:
    """LONG/SHORT/NO_TRADE mit Gate → Signal (long-only-Semantik wie plan_trade)."""

    def test_long_above_gate_buys_with_notional_fraction(self) -> None:
        strategy = make_strategy(make_pipeline_result("LONG_BIAS", 0.9), min_confidence=0.3)
        signals = drive(strategy, trending_up(35))
        assert len(signals) == 2  # Bars 30, 35
        signal = signals[0]
        assert signal.action == SignalAction.BUY
        assert signal.position_size == pytest.approx(2000.0 / 100_000.0)
        assert signal.confidence == pytest.approx(0.9)
        assert signal.metadata["decision"] == "LONG_BIAS"
        assert signal.metadata["confidence"] == pytest.approx(0.9)

    def test_long_below_gate_no_signal(self) -> None:
        strategy = make_strategy(make_pipeline_result("LONG_BIAS", 0.2), min_confidence=0.3)
        signals = drive(strategy, trending_up(35))
        assert signals == []
        assert len(strategy.evaluations) == 2
        assert all(e.signal_emitted is False for e in strategy.evaluations)

    def test_long_exactly_at_gate_trades(self) -> None:
        strategy = make_strategy(make_pipeline_result("LONG_BIAS", 0.3), min_confidence=0.3)
        signals = drive(strategy, trending_up(30))
        assert len(signals) == 1

    def test_short_bias_sells(self) -> None:
        strategy = make_strategy(make_pipeline_result("SHORT_BIAS", 0.9), min_confidence=0.3)
        signals = drive(strategy, trending_up(35))
        assert len(signals) == 2
        assert all(s.action == SignalAction.SELL for s in signals)

    def test_short_bias_below_gate_no_signal(self) -> None:
        strategy = make_strategy(make_pipeline_result("SHORT_BIAS", 0.1), min_confidence=0.3)
        signals = drive(strategy, trending_up(35))
        assert signals == []

    def test_no_trade_decision_never_signals(self) -> None:
        strategy = make_strategy(make_pipeline_result("NO_TRADE", 0.95), min_confidence=0.3)
        signals = drive(strategy, trending_up(35))
        assert signals == []

    def test_range_decision_never_signals(self) -> None:
        strategy = make_strategy(make_pipeline_result("RANGE", 0.95), min_confidence=0.3)
        signals = drive(strategy, trending_up(35))
        assert signals == []


class TestEvaluationsAndCache:
    """Evaluations werden mit decision/confidence rekordiert, Cache gefüllt."""

    def test_evaluations_recorded(self) -> None:
        strategy = make_strategy(make_pipeline_result("LONG_BIAS", 0.7))
        candles = trending_up(40)
        drive(strategy, candles)
        assert len(strategy.evaluations) == 3  # Bars 30, 35, 40
        for evaluation, expected_ts in zip(
            strategy.evaluations,
            (candles[29].timestamp, candles[34].timestamp, candles[39].timestamp),
            strict=True,
        ):
            assert evaluation.timestamp == expected_ts
            assert evaluation.decision == "LONG_BIAS"
            assert evaluation.confidence == pytest.approx(0.7)

    def test_consensus_cache_filled_at_bar_indexes(self) -> None:
        strategy = make_strategy(make_pipeline_result("LONG_BIAS", 0.7))
        drive(strategy, trending_up(40))
        assert set(strategy.consensus_cache) == {29, 34, 39}
        decision, confidence, per_agent = strategy.consensus_cache[29]
        assert decision == "LONG_BIAS"
        assert confidence == pytest.approx(0.7)
        assert per_agent == {}

    def test_none_consensus_gives_zero_confidence(self) -> None:
        stub = StubPipeline(make_pipeline_result("NO_TRADE", 0.0))
        stub.results[0].consensus = None
        strategy = make_strategy(stub)
        drive(strategy, trending_up(30))
        assert strategy.evaluations[0].confidence == 0.0
        assert strategy.signal_events == []


class TestPerAgentExtraction:
    """Per-Agent-Details aus First-Round-Reports (AgentReport-Felder)."""

    def test_extract_from_fake_reports(self) -> None:
        strategy = make_strategy(
            make_pipeline_result(
                "LONG_BIAS",
                0.9,
                reports=[
                    FakeReport("anomaly", up=0.6, down=0.1, range_prob=0.3, raw_confidence=0.75),
                    FakeReport("historical_analogy", up=0.2, down=0.7, range_prob=0.1, raw_confidence=0.6),
                    FakeReport("chart", up=0.4, down=0.4, range_prob=0.2, raw_confidence=0.5),
                ],
            )
        )
        drive(strategy, trending_up(30))
        per_agent = strategy.evaluations[0].per_agent
        assert set(per_agent) == {"anomaly", "historical_analogy", "chart"}
        assert per_agent["anomaly"]["direction"] == "LONG"
        assert per_agent["anomaly"]["confidence"] == pytest.approx(0.75)
        assert per_agent["anomaly"]["status"] == "active"
        assert per_agent["historical_analogy"]["direction"] == "SHORT"
        assert per_agent["chart"]["direction"] == "LONG"  # up == down → erster Kandidat

    def test_placeholder_reports_skipped(self) -> None:
        strategy = make_strategy(make_pipeline_result("LONG_BIAS", 0.9, reports=[object(), object()]))
        drive(strategy, trending_up(30))
        assert strategy.evaluations[0].per_agent == {}

    def test_dominant_direction_unknown_on_empty(self) -> None:
        assert dominant_direction({}) == "UNKNOWN"
        assert extract_per_agent([]) == {}

    def test_replay_with_gate_uses_cache_only(self) -> None:
        results = [
            make_pipeline_result("LONG_BIAS", 0.5),
            make_pipeline_result("SHORT_BIAS", 0.9),
            make_pipeline_result("NO_TRADE", 0.9),
            make_pipeline_result("LONG_BIAS", 0.2),
        ]
        strategy = make_strategy(results, min_confidence=0.0)
        stub: StubPipeline = strategy._pipeline_factory()
        drive(strategy, trending_up(55))  # 6 Evaluations
        calls_after = len(stub.calls)
        replayed_low = strategy.replay_with_gate(0.3)
        replayed_high = strategy.replay_with_gate(0.6)
        # 6 Zyklen: L.5, S.9, NT.9, L.2, L.5, S.9
        assert len(replayed_low) == 4  # L.5, S.9, L.5, S.9
        assert len(replayed_high) == 2  # nur die beiden S.9
        actions = [action for _, action, _ in replayed_low]
        assert actions == ["BUY", "SELL", "BUY", "SELL"]
        confidences = [confidence for _, _, confidence in replayed_high]
        assert confidences == [0.9, 0.9]
        assert len(stub.calls) == calls_after  # keine Pipeline-Rekomputation

    def test_to_dict_summary(self) -> None:
        strategy = make_strategy(
            make_pipeline_result(
                "LONG_BIAS",
                0.8,
                reports=[FakeReport("anomaly", raw_confidence=0.7)],
            )
        )
        drive(strategy, trending_up(40))
        summary = strategy.to_dict()
        assert summary["n_evaluations"] == 3
        assert summary["decision_distribution"] == {"LONG_BIAS": 3}
        assert summary["mean_confidence"] == pytest.approx(0.8)
        assert summary["per_agent"]["anomaly"]["evaluations"] == 3
        assert summary["per_agent"]["anomaly"]["mean_confidence"] == pytest.approx(0.7)
        assert summary["n_buy_signals"] == 3
        assert summary["instrument"] == "BTC/USDT"
        assert strategy.evaluations_to_dicts()[0]["decision"] == "LONG_BIAS"


class TestRunIdAndInputs:
    """Pipeline-Call-Parameter (run_id, instrument, market_data)."""

    def test_pipeline_called_with_backtest_run_id_and_window(self) -> None:
        stub = StubPipeline(make_pipeline_result())
        strategy = make_strategy(stub)
        candles = make_candles(30, step=1.0)
        drive(strategy, candles)
        call = stub.calls[0]
        assert call["instrument"] == "BTC/USDT"
        assert call["run_id"] == f"backtest-BTC/USDT-{candles[29].timestamp.isoformat()}"
        market_data = call["market_data"]
        assert len(market_data["close"]) == 30  # volles Fenster
        assert float(market_data["close"][-1]) == pytest.approx(candles[29].close)
        assert float(market_data["close"][0]) == pytest.approx(candles[0].close)
