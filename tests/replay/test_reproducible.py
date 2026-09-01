"""Tests: replay reproducibility — same data → same results.

Property: Running the same historical data through the replay pipeline
produces deterministic, identical results across runs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from apps.ingestion.replay_engine import ReplayConfig, ReplayEngine
from packages.domain.market_data import MultiTimeframeAggregator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int) -> datetime:
    return datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=offset_minutes)


def _make_events(count: int) -> list[dict[str, Any]]:
    """Deterministic candle events."""
    events: list[dict[str, Any]] = []
    for i in range(count):
        events.append({
            "event_id": f"evt-{i:04d}",
            "type": "candle",
            "instrument": "BTC/USDT",
            "venue": "binance",
            "event_time": _ts(i).isoformat(),
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 10.0 + i * 0.5,
            "sequence": i + 1,
        })
    return events


def _write_jsonl(events: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(events, default=str))


# ---------------------------------------------------------------------------
# Test: ReplayEngine produces identical results on repeated runs
# ---------------------------------------------------------------------------

class TestReplayDeterminism:
    """Two replay runs with the same data produce identical decisions."""

    def test_run_twice_identical(self, tmp_path: Path) -> None:
        events = _make_events(50)
        input_file = tmp_path / "events.json"
        _write_jsonl(events, input_file)

        config = ReplayConfig(
            input_path=str(input_file),
            validate_events=False,
        )
        engine1 = ReplayEngine(config)
        engine2 = ReplayEngine(config)

        result1 = engine1.run()
        result2 = engine2.run()

        assert len(result1) == len(result2)
        for a, b in zip(result1, result2, strict=True):
            assert a == b, "Replay results differ across runs"

    def test_run_twice_identical_with_validation(self, tmp_path: Path) -> None:
        events = _make_events(20)
        input_file = tmp_path / "events.json"
        _write_jsonl(events, input_file)

        config = ReplayConfig(
            input_path=str(input_file),
            validate_events=True,
        )
        engine1 = ReplayEngine(config)
        engine2 = ReplayEngine(config)

        valid1, _ = engine1.validate_and_filter(events)
        valid2, _ = engine2.validate_and_filter(events)

        assert valid1 == valid2, "Validation results differ across runs"

    def test_sort_is_deterministic(self) -> None:
        """Sorting events produces a stable, deterministic order."""
        engine = ReplayEngine(ReplayConfig(input_path="/tmp"))

        # Events intentionally out of order
        events = [
            {"event_time": "2025-01-01T00:03:00+00:00", "id": "third"},
            {"event_time": "2025-01-01T00:01:00+00:00", "id": "first"},
            {"event_time": "2025-01-01T00:02:00+00:00", "id": "second"},
        ]

        sorted1 = engine.sort_events(events)
        sorted2 = engine.sort_events(events)

        assert sorted1 == sorted2
        assert sorted1[0]["id"] == "first"
        assert sorted1[1]["id"] == "second"
        assert sorted1[2]["id"] == "third"


# ---------------------------------------------------------------------------
# Tests: MultiTimeframeAggregator determinism
# ---------------------------------------------------------------------------

class TestAggregatorDeterminism:
    """Same input candles → same aggregation every time."""

    def test_aggregate_deterministic(self) -> None:
        candles = [
            {
                "open_time": _ts(i),
                "close_time": _ts(i + 1),
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "close": 101.0 + i,
                "volume": 10.0 + i * 0.5,
                "trade_count": i + 1,
            }
            for i in range(60)
        ]

        agg = MultiTimeframeAggregator(base_timeframe="1m")
        result1 = agg.aggregate(candles, "1h", "BTC/USDT", "binance")
        result2 = agg.aggregate(candles, "1h", "BTC/USDT", "binance")

        assert result1.open == result2.open
        assert result1.high == result2.high
        assert result1.low == result2.low
        assert result1.close == result2.close
        assert result1.volume == result2.volume

    def test_aggregate_all_deterministic(self) -> None:
        candles = [
            {
                "open_time": _ts(i),
                "close_time": _ts(i + 1),
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "close": 101.0 + i,
                "volume": 10.0 + i * 0.5,
                "trade_count": i + 1,
            }
            for i in range(100)
        ]

        agg = MultiTimeframeAggregator(base_timeframe="1m")
        results1 = agg.aggregate_all(candles, "BTC/USDT", "binance")
        results2 = agg.aggregate_all(candles, "BTC/USDT", "binance")

        for tf in results1:
            r1 = results1[tf]
            r2 = results2[tf]
            assert r1.open == r2.open
            assert r1.close == r2.close

    def test_sma_deterministic(self) -> None:
        candles = [
            {
                "open_time": _ts(i),
                "close_time": _ts(i + 1),
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "close": 101.0 + i,
                "volume": 10.0 + i * 0.5,
                "trade_count": i + 1,
            }
            for i in range(40)
        ]

        agg = MultiTimeframeAggregator(base_timeframe="1m")
        sma1 = agg.compute_simple_moving_average(candles, "close", window=10)
        sma2 = agg.compute_simple_moving_average(candles, "close", window=10)

        import numpy as np
        np.testing.assert_array_equal(sma1, sma2)

    def test_ema_deterministic(self) -> None:
        candles = [
            {
                "open_time": _ts(i),
                "close_time": _ts(i + 1),
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "close": 101.0 + i,
                "volume": 10.0 + i * 0.5,
                "trade_count": i + 1,
            }
            for i in range(40)
        ]

        agg = MultiTimeframeAggregator(base_timeframe="1m")
        ema1 = agg.compute_exponential_moving_average(candles, "close", span=10)
        ema2 = agg.compute_exponential_moving_average(candles, "close", span=10)

        import numpy as np
        np.testing.assert_array_equal(ema1, ema2)


# ---------------------------------------------------------------------------
# Test: Same input JSON → same engine output
# ---------------------------------------------------------------------------

class TestReplayFromDisk:
    """Reading events from disk and replaying must be reproducible."""

    def test_replay_from_json_file(self, tmp_path: Path) -> None:
        events = _make_events(30)
        input_file = tmp_path / "input.json"
        _write_jsonl(events, input_file)

        config = ReplayConfig(
            input_path=str(input_file),
            validate_events=False,
        )
        engine = ReplayEngine(config)
        results = engine.run()

        # Should have all 30 events
        assert len(results) == 30

        # Second run must match
        engine2 = ReplayEngine(config)
        results2 = engine2.run()
        assert results == results2

    def test_replay_events_sorted_by_time(self, tmp_path: Path) -> None:
        """Events are sorted by event_time after loading."""
        events = _make_events(10)
        # Shuffle to test sorting
        import random
        shuffled = list(events)
        random.seed(42)  # deterministic shuffle
        random.shuffle(shuffled)

        input_file = tmp_path / "shuffled.json"
        _write_jsonl(shuffled, input_file)

        config = ReplayConfig(
            input_path=str(input_file),
            validate_events=False,
        )
        engine = ReplayEngine(config)
        results = engine.run()

        # Verify sorted
        for i in range(len(results) - 1):
            assert results[i]["event_time"] <= results[i + 1]["event_time"]
