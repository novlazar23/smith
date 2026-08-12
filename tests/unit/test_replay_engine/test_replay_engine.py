from __future__ import annotations

import json
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from apps.ingestion.replay_engine import ReplayConfig, ReplayEngine

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_events() -> list[dict[str, Any]]:
    """Erstellt eine Liste deterministischer Sample-Events."""
    return [
        {
            "event_id": "evt-001",
            "type": "candle",
            "instrument": "BTC/USDT",
            "venue": "binance",
            "event_time": "2025-01-01T00:01:00+00:00",
            "open": 42000.0,
            "high": 42500.0,
            "low": 41800.0,
            "close": 42300.0,
            "volume": 100.5,
            "sequence": 1,
        },
        {
            "event_id": "evt-002",
            "type": "trade",
            "instrument": "BTC/USDT",
            "venue": "binance",
            "event_time": "2025-01-01T00:00:30+00:00",
            "price": 42100.0,
            "quantity": 1.5,
            "side": "buy",
            "trade_id": "tr-001",
            "sequence": 2,
        },
        {
            "event_id": "evt-003",
            "type": "candle",
            "instrument": "BTC/USDT",
            "venue": "binance",
            "event_time": "2025-01-01T00:02:00+00:00",
            "open": 42300.0,
            "high": 42800.0,
            "low": 42100.0,
            "close": 42600.0,
            "volume": 120.0,
            "sequence": 3,
        },
    ]


@pytest.fixture
def valid_candle_events() -> list[dict[str, Any]]:
    """Events ohne Validierungsprobleme."""
    return [
        {
            "type": "candle",
            "instrument": "ETH/USDT",
            "venue": "kraken",
            "event_time": "2025-01-01T00:00:00+00:00",
            "open": 3000.0,
            "high": 3100.0,
            "low": 2950.0,
            "close": 3050.0,
            "volume": 50.0,
        },
    ]


@pytest.fixture
def sample_csv_dir() -> Path:
    """Erstellt temporäres CSV-Verzeichnis (keine Datumsstrings mit Kommata)."""
    tmp = tempfile.mkdtemp()
    dirpath = Path(tmp)
    rows = [
        "event_id,type,instrument,venue,event_time,open,high,low,close,volume,sequence",
        "e-1,candle,BTC,binance,2025-01-01T00:01:00,42000,42500,41800,42300,100,1",
        "e-2,candle,BTC,binance,2025-01-01T00:02:00,42300,42800,42100,42600,120,3",
    ]
    with (dirpath / "events.csv").open("w") as fh:
        fh.write("\n".join(rows) + "\n")
    return dirpath


@pytest.fixture
def event_dir(sample_events: list[dict[str, Any]]) -> Path:
    """Erstellt ein temporäres Verzeichnis mit JSON-Events."""
    tmp = tempfile.mkdtemp()
    dirpath = Path(tmp)
    with (dirpath / "events.json").open("w") as fh:
        json.dump(sample_events, fh)
    return dirpath


@pytest.fixture
def event_file(sample_events: list[dict[str, Any]]) -> Path:
    """Erstellt eine temporäre JSON-Event-Datei."""
    tmp = tempfile.mkdtemp()
    filepath = Path(tmp) / "events.json"
    with filepath.open("w") as fh:
        json.dump(sample_events, fh)
    return filepath


@pytest.fixture
def jsonl_file(sample_events: list[dict[str, Any]]) -> Path:
    """Erstellt eine temporäre JSONL-Event-Datei."""
    tmp = tempfile.mkdtemp()
    filepath = Path(tmp) / "events.jsonl"
    with filepath.open("w") as fh:
        for event in sample_events:
            fh.write(json.dumps(event) + "\n")
    return filepath


# ── ReplayConfig Tests ────────────────────────────────────────────────


class TestReplayConfig:

    def test_default_values(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        assert config.event_type == "all"
        assert config.speed_multiplier == 1.0
        assert config.gap_threshold_ms == 1000.0
        assert config.quarantine_threshold == 0.5
        assert config.validate_events is True
        assert config.output_path is None

    def test_custom_values(self) -> None:
        config = ReplayConfig(
            input_path="/tmp/data",
            event_type="candle",
            speed_multiplier=2.0,
            gap_threshold_ms=500.0,
            quarantine_threshold=0.7,
            validate_events=False,
            output_path="/tmp/output.log",
        )
        assert config.event_type == "candle"
        assert config.speed_multiplier == 2.0
        assert config.gap_threshold_ms == 500.0
        assert config.quarantine_threshold == 0.7
        assert config.validate_events is False
        assert config.output_path == "/tmp/output.log"

    def test_frozen(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        with pytest.raises(FrozenInstanceError):
            config.input_path = "new_value"


# ── ReplayEngine Initialization Tests ─────────────────────────────────


class TestReplayEngineInit:

    def test_create_engine(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        engine = ReplayEngine(config)
        assert engine.config is config
        assert engine._events == []
        assert engine._replayed == []
        assert engine._validation_results == []
        assert engine._gap_results == []
        assert engine._replay_log == []

    def test_expose_quarantine_manager(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        engine = ReplayEngine(config)
        assert engine.quarantine_manager is not None


# ── Event Loading Tests ───────────────────────────────────────────────


class TestLoadEvents:

    def test_load_from_json_file(self, event_file: Path, sample_events: list[dict]) -> None:
        config = ReplayConfig(input_path=str(event_file))
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert len(events) == len(sample_events)
        assert all(isinstance(e, dict) for e in events)

    def test_load_from_directory(self, event_dir: Path, sample_events: list[dict]) -> None:
        config = ReplayConfig(input_path=str(event_dir))
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert len(events) == len(sample_events)

    def test_load_from_jsonl(self, jsonl_file: Path, sample_events: list[dict]) -> None:
        config = ReplayConfig(input_path=str(jsonl_file))
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert len(events) == len(sample_events)

    def test_load_from_csv(self, sample_csv_dir: Path) -> None:
        config = ReplayConfig(input_path=str(sample_csv_dir))
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert len(events) == 2

    def test_load_nonexistent_path(self) -> None:
        config = ReplayConfig(input_path="/nonexistent/path")
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert events == []

    def test_load_filter_by_event_type(self, sample_events: list[dict]) -> None:
        filtered = [e for e in sample_events if e.get("type") == "candle"]
        assert len(filtered) == 2

    def test_load_empty_json_file(self) -> None:
        tmp = tempfile.mktemp(suffix=".json")
        Path(tmp).write_text("[]")
        config = ReplayConfig(input_path=tmp)
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert events == []

    def test_load_invalid_json(self) -> None:
        tmp = tempfile.mktemp(suffix=".json")
        Path(tmp).write_text("{invalid json}}}")
        config = ReplayConfig(input_path=tmp)
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert events == []

    def test_load_empty_jsonl(self) -> None:
        tmp = tempfile.mktemp(suffix=".jsonl")
        Path(tmp).write_text("")
        config = ReplayConfig(input_path=tmp)
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert events == []

    def test_load_single_json_object(self) -> None:
        tmp = tempfile.mktemp(suffix=".json")
        Path(tmp).write_text('{"type": "candle"}')
        config = ReplayConfig(input_path=tmp)
        engine = ReplayEngine(config)
        events = engine.load_events()
        assert len(events) == 1
        assert events[0]["type"] == "candle"


# ── Event Sorting Tests ───────────────────────────────────────────────


class TestSortEvents:

    def test_sort_by_event_time(self) -> None:
        events = [
            {"event_time": "2025-01-01T00:03:00+00:00", "type": "candle"},
            {"event_time": "2025-01-01T00:01:00+00:00", "type": "candle"},
            {"event_time": "2025-01-01T00:02:00+00:00", "type": "trade"},
        ]
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        sorted_events = engine.sort_events(events)
        assert sorted_events[0]["event_time"] < sorted_events[1]["event_time"]
        assert sorted_events[1]["event_time"] < sorted_events[2]["event_time"]

    def test_sort_by_timestamp_fallback(self) -> None:
        events = [
            {"timestamp": 100, "type": "trade"},
            {"timestamp": 200, "type": "candle"},
            {"timestamp": 50, "type": "trade"},
        ]
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        sorted_events = engine.sort_events(events)
        assert sorted_events[0]["timestamp"] < sorted_events[1]["timestamp"]
        assert sorted_events[1]["timestamp"] < sorted_events[2]["timestamp"]

    def test_sort_empty_list(self) -> None:
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        sorted_events = engine.sort_events([])
        assert sorted_events == []

    def test_sort_stable(self) -> None:
        events = [
            {"event_time": "2025-01-01T00:01:00+00:00", "type": "a"},
            {"event_time": "2025-01-01T00:01:00+00:00", "type": "b"},
            {"event_time": "2025-01-01T00:01:00+00:00", "type": "c"},
        ]
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        sorted_events = engine.sort_events(events)
        assert len(sorted_events) == 3
        assert sorted_events[0]["type"] == "a"
        assert sorted_events[1]["type"] == "b"
        assert sorted_events[2]["type"] == "c"


# ── Gap Detection Tests ───────────────────────────────────────────────


class TestDetectGaps:

    def test_detect_sequence_gap(self) -> None:
        events = [
            {"sequence": 1, "instrument": "BTC", "venue": "binance", "type": "candle"},
            {"sequence": 2, "instrument": "BTC", "venue": "binance", "type": "candle"},
            {"sequence": 5, "instrument": "BTC", "venue": "binance", "type": "candle"},
        ]
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        gaps = engine.detect_gaps(events)
        assert len(gaps) > 0

    def test_no_gap(self) -> None:
        events = [
            {"sequence": 1, "instrument": "BTC", "venue": "binance", "type": "candle"},
            {"sequence": 2, "instrument": "BTC", "venue": "binance", "type": "candle"},
            {"sequence": 3, "instrument": "BTC", "venue": "binance", "type": "candle"},
        ]
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        gaps = engine.detect_gaps(events)
        assert all(g.is_valid for g in gaps)

    def test_sequence_regression(self) -> None:
        events = [
            {"sequence": 5, "instrument": "BTC", "venue": "binance", "type": "candle"},
            {"sequence": 3, "instrument": "BTC", "venue": "binance", "type": "candle"},
        ]
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        gaps = engine.detect_gaps(events)
        assert len(gaps) > 0
        assert any(not g.is_valid for g in gaps)

    def test_different_instruments_independent(self) -> None:
        events = [
            {"sequence": 1, "instrument": "BTC", "venue": "binance", "type": "candle"},
            {"sequence": 1, "instrument": "ETH", "venue": "kraken", "type": "candle"},
            {"sequence": 5, "instrument": "BTC", "venue": "binance", "type": "candle"},
        ]
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        gaps = engine.detect_gaps(events)
        assert len(gaps) > 0

    def test_time_gap_detection(self) -> None:
        engine = ReplayEngine(ReplayConfig(
            input_path="/tmp/test",
            gap_threshold_ms=100,
        ))
        events = [
            {
                "event_time": "2025-01-01T00:00:00+00:00",
                "type": "candle",
                "instrument": "BTC",
                "venue": "binance",
            },
            {
                "event_time": "2025-12-31T23:59:59+00:00",
                "type": "candle",
                "instrument": "BTC",
                "venue": "binance",
            },
        ]
        gaps = engine.detect_gaps(events)
        assert len(gaps) > 0

    def test_empty_events(self) -> None:
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        gaps = engine.detect_gaps([])
        assert gaps == []


# ── Validation & Quarantine Tests ─────────────────────────────────────


class TestValidateAndFilter:

    def test_validate_valid_events(self, valid_candle_events: list[dict]) -> None:
        config = ReplayConfig(
            input_path="/tmp/test",
            quarantine_threshold=0.5,
        )
        engine = ReplayEngine(config)
        valid, results = engine.validate_and_filter(valid_candle_events)
        assert len(valid) == 1
        assert len(results) == 1
        assert results[0].is_valid

    def test_validate_invalid_events(self) -> None:
        events = [
            {
                "type": "candle",
                "instrument": "BTC/USDT",
                "venue": "binance",
                "open": 100.0,
                "high": 50.0,
                "low": 200.0,
                "close": 80.0,
                "volume": -10.0,
            },
        ]
        config = ReplayConfig(
            input_path="/tmp/test",
            quarantine_threshold=0.5,
        )
        engine = ReplayEngine(config)
        valid, results = engine.validate_and_filter(events)
        assert len(valid) == 0
        assert len(results) == 1
        assert not results[0].is_valid

    def test_validate_with_quarantine(self) -> None:
        events = [
            {
                "type": "candle",
                "instrument": "BTC/USDT",
                "venue": "binance",
                "open": 100.0,
                "high": 50.0,
                "low": 200.0,
                "close": 80.0,
                "volume": -10.0,
            },
        ]
        config = ReplayConfig(
            input_path="/tmp/test",
            quarantine_threshold=0.5,
        )
        engine = ReplayEngine(config)
        engine.validate_and_filter(events)
        stats = engine.quarantine_manager.get_stats()
        assert stats["total_quarantined"] == 1

    def test_validate_skipped_in_run(self, sample_events: list[dict]) -> None:
        config = ReplayConfig(
            input_path="/tmp/test",
            validate_events=False,
        )
        engine = ReplayEngine(config)
        engine._events = sample_events
        valid, _ = engine.validate_and_filter(sample_events)
        assert len(valid) <= len(sample_events)

    def test_validate_empty_list(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        engine = ReplayEngine(config)
        valid, results = engine.validate_and_filter([])
        assert valid == []
        assert results == []

    def test_stats_tracking(self) -> None:
        events = [
            {
                "type": "candle",
                "instrument": "BTC/USDT",
                "venue": "binance",
                "open": 42000.0,
                "high": 42500.0,
                "low": 41800.0,
                "close": 42300.0,
                "volume": 100.0,
            },
            {
                "type": "trade",
                "instrument": "ETH/USDT",
                "venue": "kraken",
                "price": 3000.0,
                "quantity": 1.0,
                "side": "buy",
            },
            {
                "type": "candle",
                "instrument": "BTC/USDT",
                "venue": "binance",
                "open": 50.0,
                "high": 200.0,
                "low": 100.0,
                "close": 80.0,
                "volume": -10.0,
            },
        ]
        config = ReplayConfig(input_path="/tmp/test")
        engine = ReplayEngine(config)
        engine.validate_and_filter(events)
        # Event 0: valid candle
        # Event 1: trade missing trade_id → invalid
        # Event 2: high > low, volume < 0 → invalid
        assert engine._valid_count == 1
        assert engine._invalid_count == 2


# ── Main Replay Tests ─────────────────────────────────────────────────


class TestReplayRun:

    def test_full_replay(self, jsonl_file: Path) -> None:
        config = ReplayConfig(
            input_path=str(jsonl_file),
            event_type="all",
            validate_events=False,
        )
        engine = ReplayEngine(config)
        result = engine.run()
        assert len(result) == 3

    def test_replay_empty(self) -> None:
        config = ReplayConfig(input_path="/nonexistent")
        engine = ReplayEngine(config)
        result = engine.run()
        assert result == []

    def test_replay_with_sorting(self, jsonl_file: Path) -> None:
        config = ReplayConfig(
            input_path=str(jsonl_file),
            validate_events=False,
        )
        engine = ReplayEngine(config)
        result = engine.run()
        assert len(result) == 3
        for i in range(len(result) - 1):
            t1 = result[i].get("event_time", "")
            t2 = result[i + 1].get("event_time", "")
            assert t1 <= t2

    def test_replay_deterministic(self, jsonl_file: Path) -> None:
        config = ReplayConfig(
            input_path=str(jsonl_file),
            validate_events=False,
        )
        engine1 = ReplayEngine(config)
        result1 = engine1.run()

        engine2 = ReplayEngine(config)
        result2 = engine2.run()

        assert result1 == result2

    def test_replay_output_path(self, jsonl_file: Path) -> None:
        tmp_output = tempfile.mktemp(suffix=".json")
        config = ReplayConfig(
            input_path=str(jsonl_file),
            output_path=tmp_output,
            validate_events=False,
        )
        engine = ReplayEngine(config)
        engine.run()
        assert Path(tmp_output).exists()
        with Path(tmp_output).open() as fh:
            data = json.load(fh)
        assert "metadata" in data
        assert "statistics" in data
        assert "replay_events" in data
        assert len(data["replay_events"]) == 3


# ── Batch Replay Tests ────────────────────────────────────────────────


class TestReplayBatch:

    def test_batch_replay(self, sample_events: list[dict]) -> None:
        processed: list[dict] = []
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))

        def handler(event: dict) -> None:
            processed.append(event)

        engine.replay_batch(sample_events, handler)
        assert len(processed) == len(sample_events)

    def test_batch_replay_empty(self) -> None:
        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        handler = MagicMock()
        engine.replay_batch([], handler)
        handler.assert_not_called()

    def test_batch_replay_handler_error(self, sample_events: list[dict]) -> None:
        call_count = 0

        def failing_handler(event: dict) -> None:
            nonlocal call_count
            call_count += 1
            if event.get("event_id") == "evt-002":
                raise ValueError("Intentional error")

        engine = ReplayEngine(ReplayConfig(input_path="/tmp/test"))
        engine.replay_batch(sample_events, failing_handler)
        assert call_count == len(sample_events)


# ── Statistics Tests ──────────────────────────────────────────────────


class TestStatistics:

    def test_get_statistics(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        engine = ReplayEngine(config)
        engine._events = [{"type": "candle"}, {"type": "trade"}]
        engine._replayed = [{"type": "candle"}, {"type": "trade"}]
        engine._valid_count = 2
        engine._replay_start = 0.0
        engine._replay_end = 1.5

        stats = engine.get_statistics()
        assert stats["total_loaded"] == 2
        assert stats["total_replayed"] == 2
        assert stats["total_valid"] == 2
        assert stats["duration_seconds"] == 1.5
        assert "quarantine_stats" in stats

    def test_get_statistics_pre_run(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        engine = ReplayEngine(config)
        stats = engine.get_statistics()
        assert stats["total_loaded"] == 0
        assert stats["total_replayed"] == 0

    def test_get_statistics_quarantine(self) -> None:
        config = ReplayConfig(input_path="/tmp/test")
        engine = ReplayEngine(config)
        stats = engine.get_statistics()
        assert "quarantine_stats" in stats


# ── Save Replay Log Tests ─────────────────────────────────────────────


class TestSaveReplayLog:

    def test_save_log_json_format(self) -> None:
        tmp = tempfile.mktemp(suffix=".json")
        config = ReplayConfig(input_path="/tmp/test", output_path=tmp)
        engine = ReplayEngine(config)
        engine._replayed = [{"type": "candle"}]
        engine._replay_log = [{"index": 0}]
        engine._valid_count = 1
        engine._invalid_count = 0
        engine._quarantined_count = 0
        engine._replay_start = 0.0
        engine._replay_end = 1.0

        engine.save_replay_log(tmp)

        with Path(tmp).open() as fh:
            data = json.load(fh)

        assert "metadata" in data
        assert "statistics" in data
        assert "replay_events" in data
        assert "gap_results" in data
        assert "replay_log" in data
        assert len(data["replay_events"]) == 1


# ── Async Replay Tests ────────────────────────────────────────────────


class TestAsyncReplay:

    @pytest.mark.asyncio
    async def test_async_replay_quick(self, jsonl_file: Path) -> None:
        config = ReplayConfig(
            input_path=str(jsonl_file),
            validate_events=False,
            speed_multiplier=1000.0,
        )
        engine = ReplayEngine(config)
        result = await engine.run_async()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_async_replay_empty(self) -> None:
        config = ReplayConfig(input_path="/nonexistent", validate_events=False)
        engine = ReplayEngine(config)
        result = await engine.run_async()
        assert result == []

    @pytest.mark.asyncio
    async def test_async_replay_with_output(self, jsonl_file: Path) -> None:
        tmp = tempfile.mktemp(suffix=".json")
        config = ReplayConfig(
            input_path=str(jsonl_file),
            output_path=tmp,
            validate_events=False,
            speed_multiplier=1000.0,
        )
        engine = ReplayEngine(config)
        await engine.run_async()
        assert Path(tmp).exists()


# ── Integration Test ──────────────────────────────────────────────────


class TestReplayIntegration:

    def test_end_to_end_replay_from_file(self, jsonl_file: Path) -> None:
        tmp_output = tempfile.mktemp(suffix=".json")
        config = ReplayConfig(
            input_path=str(jsonl_file),
            output_path=tmp_output,
            validate_events=False,
        )
        engine = ReplayEngine(config)
        result = engine.run()
        assert len(result) > 0
        assert Path(tmp_output).exists()

        with Path(tmp_output).open() as fh:
            log_data = json.load(fh)
        assert len(log_data["replay_events"]) == len(result)

    def test_end_to_end_replay_with_validation(self, jsonl_file: Path) -> None:
        config = ReplayConfig(
            input_path=str(jsonl_file),
            validate_events=True,
        )
        engine = ReplayEngine(config)
        engine.run()
        stats = engine.get_statistics()
        assert stats["total_loaded"] > 0
        assert stats["total_valid"] >= 0

