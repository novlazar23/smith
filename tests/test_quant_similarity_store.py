"""Unit-Tests für das Similarity-Storage (Phase 5, P5-2).

Die ``SimilarityEngine`` selbst läuft (deterministische
Stdlib-Berechnung); gemockt ist ausschließlich der ``InfluxDBStore``
(MVP-Store ohne I/O) und — für den Exception-Pfad — die Engine.
Es werden keine echten InfluxDB-Verbindungen aufgebaut.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from trading_harness.quant.similarity import SimilarityEngine, SimilarityResult
from trading_harness.quant.similarity_store import SimilarityStore, SimilarityStoreResult

SYMBOL = "BTCUSDT"
OTHER_SYMBOL = "ETHUSDT"
TIMEFRAME = "1m"
EXCHANGE = "binance"

BASE_TS = 1_767_225_600  # 2026-01-01T00:00:00Z (1-Minuten-Abstand)

PATTERN = [100.0, 102.0, 104.0, 103.0, 101.0]
MIDDLE = [90.0, 88.0, 87.0, 89.0, 91.0]
# 15 Kerzen: Pattern liegt exakt bei idx 0 und idx 10.
HISTORY = PATTERN + MIDDLE + PATTERN


def _candle_time(index: int) -> str:
    """UTC-ISO-Zeichenkette für die Kerze an Position *index*."""
    moment = datetime.fromtimestamp(BASE_TS + index * 60, tz=UTC)
    return moment.isoformat().replace("+00:00", "Z")


def _make_candles(prices: list[float]) -> list[dict]:
    return [
        {
            "time": _candle_time(i),
            "open": p,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "volume": 1000.0,
        }
        for i, p in enumerate(prices)
    ]


def make_mock_store() -> MagicMock:
    """InfluxDBStore-Mock: der MVP-Store führt kein I/O aus (Dasein genügt)."""
    store = MagicMock()
    store._bucket = "quant"
    store.is_available = True
    return store


def make_store(engine: SimilarityEngine | None = None) -> SimilarityStore:
    """Store mit deterministischer Engine (normalize=False → exakte Abstände)."""
    default = SimilarityEngine(window_size=5, top_k=3, normalize=False)
    return SimilarityStore(make_mock_store(), engine or default)


def run_default_store() -> SimilarityStore:
    """Store mit einem Standardlauf (Top-3: idx 0, 10 mit 0.0, dann idx 9)."""
    similarity_store = make_store()
    similarity_store.find_and_store(SYMBOL, TIMEFRAME, _make_candles(PATTERN), _make_candles(HISTORY))
    return similarity_store


# ----------------------------------------------------------------------
# find_and_store — Berechnung + Cache
# ----------------------------------------------------------------------


def test_find_and_store_calls_engine_and_returns_results() -> None:
    """Echte Engine: Ergebnis deckt sich mit dem direkten Engine-Aufruf."""
    engine = SimilarityEngine(window_size=5, top_k=3, normalize=False)
    similarity_store = SimilarityStore(make_mock_store(), engine)
    query = _make_candles(PATTERN)
    history = _make_candles(HISTORY)
    expected = engine.find_similar(query, history)

    result = similarity_store.find_and_store(SYMBOL, TIMEFRAME, query, history)

    assert isinstance(result, SimilarityStoreResult)
    assert result.query_length == len(PATTERN)
    assert result.matches_found == len(expected.matches)
    assert result.matches_found == 3
    assert result.best_distance == pytest.approx(expected.best_distance, abs=1e-9)
    assert result.best_correlation == pytest.approx(expected.best_correlation, abs=1e-9)
    # Pattern liegt exakt in der Historie → Distanz 0.0, Korrelation 1.0.
    assert result.best_distance == pytest.approx(0.0, abs=1e-9)
    assert result.best_correlation == pytest.approx(1.0)


def test_find_and_store_delegates_to_injected_engine() -> None:
    """Explizite Delegation: injizierte Engine wird mit query/history aufgerufen."""
    fake = SimilarityResult(
        query_length=2, matches=[], best_distance=None, best_correlation=None
    )
    engine = MagicMock()
    engine.find_similar_with_candles.return_value = fake
    similarity_store = SimilarityStore(make_mock_store(), engine)
    query = [{"close": 1.0}, {"close": 2.0}]
    history: list[dict] = []

    result = similarity_store.find_and_store(SYMBOL, TIMEFRAME, query, history)

    engine.find_similar_with_candles.assert_called_once_with(query, history)
    assert result.query_length == 2
    assert result.matches_found == 0
    assert result.best_distance is None
    assert result.best_correlation is None


def test_find_and_store_empty_history_returns_zero_matches() -> None:
    """Leere Historie → Engine-Leererfolg → 0 Matches, keine Exception."""
    similarity_store = make_store()

    result = similarity_store.find_and_store(SYMBOL, TIMEFRAME, _make_candles(PATTERN), [])

    assert result.query_length == len(PATTERN)
    assert result.matches_found == 0
    assert result.best_distance is None
    assert result.best_correlation is None
    assert similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME) == []


def test_find_and_store_engine_exception_is_handled_gracefully() -> None:
    """Engine-Exception → leeres Ergebnis, keine Exception, Cache unverändert."""
    engine = MagicMock()
    engine.find_similar_with_candles.side_effect = RuntimeError("boom")
    similarity_store = SimilarityStore(make_mock_store(), engine)
    query = _make_candles(PATTERN)

    result = similarity_store.find_and_store(SYMBOL, TIMEFRAME, query, _make_candles(HISTORY))

    assert result.query_length == len(PATTERN)
    assert result.matches_found == 0
    assert result.best_distance is None
    assert result.best_correlation is None
    # Der Store bleibt funktionsfähig, der Cache bleibt leer.
    assert similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME) == []


def test_find_and_store_cache_is_keyed_by_symbol_and_timeframe() -> None:
    """Cache-Keys (symbol, timeframe) sind unabhängig; Einträge tragen Key."""
    similarity_store = make_store()
    query = _make_candles(PATTERN)
    history = _make_candles(HISTORY)
    similarity_store.find_and_store(SYMBOL, "1m", query, history)
    similarity_store.find_and_store(SYMBOL, "5m", query, history)

    entries_1m = similarity_store.get_similar_patterns(SYMBOL, "1m")
    entries_5m = similarity_store.get_similar_patterns(SYMBOL, "5m")

    assert len(entries_1m) == 3
    assert len(entries_5m) == 3
    assert all(entry["timeframe"] == "1m" for entry in entries_1m)
    assert all(entry["timeframe"] == "5m" for entry in entries_5m)
    # Ein anderes Symbol hat keinen Cache-Eintrag.
    assert similarity_store.get_similar_patterns(OTHER_SYMBOL, "1m") == []


# ----------------------------------------------------------------------
# get_similar_patterns — Lesezugriff
# ----------------------------------------------------------------------


def test_get_similar_patterns_returns_results() -> None:
    """Cache-Einträge: Engine-Reihenfolge, korrekte Keys und Fensterzeiten."""
    similarity_store = run_default_store()

    entries = similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME)

    assert len(entries) == 3
    # Top-2 = Pattern bei idx 0 und 10 (Distanz 0.0), dann bestes Teilfenster idx 9.
    assert [entry["start_index"] for entry in entries] == [0, 10, 9]
    assert entries[0]["distance"] == pytest.approx(0.0, abs=1e-9)
    assert entries[0]["correlation"] == pytest.approx(1.0)
    for entry in entries:
        assert entry["symbol"] == SYMBOL
        assert entry["exchange"] == EXCHANGE
        assert entry["timeframe"] == TIMEFRAME
        assert entry["end_index"] == entry["start_index"] + 4
        assert entry["time"] == _candle_time(entry["start_index"])
    # Aufsteigende Distanz (Engine-Reihenfolge).
    assert entries[0]["distance"] <= entries[1]["distance"] <= entries[2]["distance"]


def test_get_similar_patterns_filters_by_time_range() -> None:
    """start/end filtern nach Fensterstartzeit (UTC-ISO)."""
    similarity_store = run_default_store()

    after = similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME, start="2026-01-01T00:05:00Z")
    assert [entry["start_index"] for entry in after] == [10, 9]

    only_first = similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME, end="2026-01-01T00:00:30Z")
    assert [entry["start_index"] for entry in only_first] == [0]


def test_get_similar_patterns_unknown_symbol_returns_empty() -> None:
    """Unbekanntes Symbol bzw. Zeitframe ohne Lauf → leere Liste."""
    similarity_store = run_default_store()

    assert similarity_store.get_similar_patterns(OTHER_SYMBOL, TIMEFRAME) == []
    assert similarity_store.get_similar_patterns(SYMBOL, "4h") == []


def test_get_similar_patterns_invalid_timestamp_raises_value_error() -> None:
    """Ungültiger start/end-Zeitstempel → ValueError."""
    similarity_store = run_default_store()

    with pytest.raises(ValueError):
        similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME, start="not-a-date")
    with pytest.raises(ValueError):
        similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME, end="not-a-date")
