"""Tests für den Runner (Engine-Run, extra_metrics, Buckets, Gate-Sweep)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from apps.backtest.agent_strategy import AgentEnsembleStrategy
from apps.backtest.runner import (
    CONFIDENCE_BUCKETS,
    confidence_buckets,
    extra_metrics,
    gate_sweep,
    run_backtest,
)
from packages.backtesting.core import BacktestConfig
from packages.backtesting.datafeed import MemoryDataFeed
from packages.orchestrator.pipeline import OrchestratorPipelineResult
from tests.unit.test_backtest.conftest import (
    BTC,
    FakeReport,
    StubPipeline,
    make_pipeline_result,
    make_strategy,
    trending_up,
)

N_BARS = 300


def _uptrend_feed() -> MemoryDataFeed:
    return MemoryDataFeed(trending_up(N_BARS, price0=100.0, step=1.0))


def _strategy_for(
    results: OrchestratorPipelineResult | list[OrchestratorPipelineResult],
    **overrides: Any,
) -> AgentEnsembleStrategy:
    return make_strategy(results, candle_limit=50, min_candles=30, evaluate_every=5, **overrides)


class TestRunBacktest:
    """Voller Engine-Run auf synthetischem Uptrend."""

    def test_uptrend_run_executes_buys_and_tracks_equity(self) -> None:
        # Positionen werden mark-to-market zum Bar-Schluss bewertet und
        # SHORT-/Glattstellungssignale schließen zum Marktpreis. Im Uptrend
        # steigt die Equity einer long-only Strategie daher über das
        # Startkapital (Kursgewinn der gehaltenen Position).
        strategy = _strategy_for(make_pipeline_result("LONG_BIAS", 0.9), min_confidence=0.3)
        result = run_backtest(_uptrend_feed(), lambda: strategy, BacktestConfig(symbol=BTC), "test")
        assert result.metadata["strategy"] is strategy
        assert result.metadata["initial_capital"] == 100_000.0
        assert result.metadata["equity_curve"][0] == 100_000.0
        buys = [trade for trade in result.trades if trade.side == "buy"]
        assert len(buys) > 10
        # Mark-to-Market: Uptrend → Kursgewinn schlägt Kosten, Equity steigt
        assert result.metadata["final_equity"] > 100_000.0

    def test_equity_curve_length(self) -> None:
        strategy = _strategy_for(make_pipeline_result("LONG_BIAS", 0.9))
        result = run_backtest(_uptrend_feed(), lambda: strategy, None, "test")
        # warmup = candle_limit = 50 → 1 Initialwert + (N - warmup) Bars
        assert len(result.metadata["equity_curve"]) == N_BARS - 50 + 1

    def test_extra_metrics_keys_and_values(self) -> None:
        strategy = _strategy_for(make_pipeline_result("LONG_BIAS", 0.9), min_confidence=0.3)
        result = run_backtest(_uptrend_feed(), lambda: strategy, None, "test")
        extra = extra_metrics(result, strategy)
        # Evaluations: 1-basierte Bars 30..300, alle 5 → 55
        assert extra["gate"] == 0.3
        assert extra["n_evaluations"] == 55
        assert extra["gate_pass_rate"] == 1.0
        assert extra["decision_distribution"] == {"LONG_BIAS": 55}
        assert extra["mean_confidence"] == pytest.approx(0.9)
        for key in ("final_equity", "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                    "win_rate_pct", "total_trades", "profit_factor"):
            assert key in extra


class TestConfidenceBuckets:
    """Bucket-Grenzen und Trade-Zuordnung über die Entry-Konfidenz."""

    def test_six_buckets_with_correct_boundaries(self) -> None:
        strategy = _strategy_for(make_pipeline_result("LONG_BIAS", 0.9))
        result = run_backtest(_uptrend_feed(), lambda: strategy, None, "test")
        buckets = confidence_buckets(result, strategy)
        assert len(buckets) == 6
        assert [(b["low"], b["high"]) for b in buckets] == list(CONFIDENCE_BUCKETS)
        assert buckets[-1]["bucket"].endswith("1.0]")

    def test_all_evaluations_in_high_bucket(self) -> None:
        strategy = _strategy_for(make_pipeline_result("LONG_BIAS", 0.9))
        result = run_backtest(_uptrend_feed(), lambda: strategy, None, "test")
        buckets = confidence_buckets(result, strategy)
        assert sum(b["n_evaluations"] for b in buckets) == 55
        high = buckets[-1]
        assert high["n_evaluations"] == 55
        # ein offener Long-Round-Trip (BUY ohne SELL), Entry-Konfidenz 0.9
        assert high["n_trades"] == 1
        assert high["win_rate"] == 1.0
        assert high["avg_pnl"] is not None
        assert high["avg_pnl"] > 0
        for bucket in buckets[:-1]:
            assert bucket["n_evaluations"] == 0
            assert bucket["n_trades"] == 0
            assert bucket["win_rate"] is None

    def test_bucket_attribution_by_entry_confidence(self) -> None:
        # zykloide Konfidenzen → mehrere Bucket-Belegung; 0.25 fällt unter alle Buckets
        results = [
            make_pipeline_result("LONG_BIAS", c) for c in (0.25, 0.35, 0.45, 0.55, 0.65, 0.85)
        ]
        strategy = _strategy_for(results, min_confidence=0.0)
        result = run_backtest(_uptrend_feed(), lambda: strategy, None, "test")
        buckets = confidence_buckets(result, strategy)
        by_low = {b["low"]: b for b in buckets}
        assert by_low[0.3]["n_evaluations"] >= 1  # 0.35
        assert by_low[0.4]["n_evaluations"] >= 1  # 0.45
        assert by_low[0.5]["n_evaluations"] >= 1  # 0.55
        assert by_low[0.6]["n_evaluations"] >= 1  # 0.65
        assert by_low[0.8]["n_evaluations"] >= 1  # 0.85
        # 55 Evaluations, Zyklenlänge 6 → 0.25 (Zyklenposition 0) kommt 9-mal vor
        count_025 = sum(1 for k in range(55) if k % 6 == 0)
        assert sum(b["n_evaluations"] for b in buckets) == 55 - count_025
        # Round-Trips sind auf die Buckets ihrer Entry-Konfidenz verteilt
        assert sum(b["n_trades"] for b in buckets) >= 1


class TestGateSweep:
    """Gate-Sweep: gecachter Konsens, keine Pipeline-Rekomputation."""

    def _factory(self, stub: StubPipeline) -> Callable[[], AgentEnsembleStrategy]:
        def factory() -> AgentEnsembleStrategy:
            return _strategy_for(stub, min_confidence=0.0)

        return factory

    def test_lower_gate_more_trades(self) -> None:
        confidences = (0.25, 0.35, 0.45, 0.55, 0.65)
        stub = StubPipeline([make_pipeline_result("LONG_BIAS", c) for c in confidences])
        feed = MemoryDataFeed(trending_up(200, price0=100.0, step=1.0))
        rows = gate_sweep(feed, self._factory(stub), [0.2, 0.3, 0.5, 0.7])
        assert len(rows) == 4
        by_gate = {row["gate"]: row for row in rows}
        assert by_gate[0.2]["trades"] > by_gate[0.3]["trades"]
        assert by_gate[0.3]["trades"] > by_gate[0.5]["trades"]
        assert by_gate[0.5]["trades"] > by_gate[0.7]["trades"] == 0
        # gate_pass_rate fällt mit dem Gate (Anteil der Konsense ≥ gate)
        assert by_gate[0.2]["gate_pass_rate"] == 1.0
        assert by_gate[0.3]["gate_pass_rate"] == pytest.approx(0.8)
        assert by_gate[0.7]["gate_pass_rate"] == 0.0
        assert by_gate[0.2]["total_return_pct"] is not None
        assert by_gate[0.2]["win_rate"] is not None  # Uptrend → gewinnender Trip

    def test_warm_strategy_skips_pipeline_phase(self) -> None:
        stub = StubPipeline([make_pipeline_result("LONG_BIAS", c) for c in (0.35, 0.55)])
        feed = MemoryDataFeed(trending_up(100, price0=100.0, step=1.0))
        strategy = _strategy_for(stub, min_confidence=0.0)
        run_backtest(feed, lambda: strategy, None, "warm")
        calls_after_warm = len(stub.calls)
        rows = gate_sweep(feed, self._factory(stub), [0.3, 0.5], warm_strategy=strategy)
        assert len(stub.calls) == calls_after_warm  # Phase 2: keine Pipeline-Aufrufe
        assert len(rows) == 2
        assert rows[0]["trades"] > rows[1]["trades"]

    def test_sweep_rows_carry_required_columns(self) -> None:
        stub = StubPipeline(make_pipeline_result("LONG_BIAS", 0.5))
        feed = MemoryDataFeed(trending_up(120, price0=100.0, step=1.0))
        rows = gate_sweep(feed, self._factory(stub), [0.4, 0.6])
        for row in rows:
            for key in ("gate", "trades", "win_rate", "total_return_pct",
                        "sharpe_ratio", "max_drawdown_pct", "gate_pass_rate"):
                assert key in row
        assert rows[0]["trades"] > 0
        assert rows[1]["trades"] == 0

    def test_sweep_inherits_entry_rules(self) -> None:
        # Entry-Pflicht-Agent votet SHORT: der Sweep (Replay) bekommt keine BUYs
        stub = StubPipeline(
            make_pipeline_result(
                "LONG_BIAS",
                0.9,
                reports=[FakeReport("trend", up=0.2, down=0.6, range_prob=0.2)],
            )
        )
        feed = MemoryDataFeed(trending_up(120, price0=100.0, step=1.0))

        def with_entry_filter() -> AgentEnsembleStrategy:
            return _strategy_for(stub, min_confidence=0.3, entry_required_agents=("trend",))

        rows = gate_sweep(feed, with_entry_filter, [0.3])
        assert rows[0]["trades"] == 0

        # Ohne Entry-Filter liefert derselbe Cache BUYs
        plain_rows = gate_sweep(feed, lambda: _strategy_for(stub, min_confidence=0.3), [0.3])
        assert plain_rows[0]["trades"] > 0
