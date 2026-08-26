"""Similarity-Integration (Quant-Plattform, Phase 5, P5-4).

End-to-End-Kette Matching → Storage mit ausschließlich Mocks:
``SimilarityEngine`` (P5-1) → ``SimilarityStore`` (P5-2) → ``InfluxDBStore``
(P1-4, gemockt). Kein Netzwerk, kein Docker, keine echte InfluxDB.

MVP-Semantik von ``SimilarityStore`` (P5-2): ``find_and_store`` berechnet
die Matches on-the-fly über die Engine und puffert sie pro
``(symbol, timeframe)`` im In-Memory-Cache; der übergebene
``InfluxDBStore`` wird im MVP weder gelesen noch geschrieben
(Schnittstellenkonsistenz, Persistierung = geplante Erweiterung).
Diese Tests verifizieren genau diesen Kontrakt gegen einen gemockten
InfluxDBStore: Engine-Output landet unverändert im Cache
(``get_similar_patterns``), der Store-Mock bleibt unangetastet.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant.similarity import SimilarityEngine
from trading_harness.quant.similarity_store import SimilarityStore

SYMBOL = "BTCUSDT"
EXCHANGE = "binance"
TIMEFRAME = "1m"


# ----------------------------------------------------------------------
# Test-Helfer
# ----------------------------------------------------------------------


def make_candles(prices: list[float]) -> list[dict]:
    """Erzeugt OHLCV-Kerzen aus einer Close-Sequenz (deterministische ISO-Zeiten)."""
    return [
        {
            "time": f"2026-01-01T00:{index:02d}:00Z",
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1000.0,
        }
        for index, price in enumerate(prices)
    ]


def make_mock_store(available: bool = True) -> MagicMock:
    """InfluxDBStore-Mock: Verfügbarkeits-Flag + async write/query, keine echte Verbindung."""
    store = MagicMock()
    store._bucket = "quant"
    store.is_available = available
    store.write_points = AsyncMock()
    store.query = AsyncMock(return_value=[])
    return store


def repeated_pattern_history() -> tuple[list[dict], list[dict]]:
    """Historie mit exakt wiederholtem Muster (Index 0 und 55) + (Query-Kerzen).

    Die Füll-Kerzen liegen strikt zwischen Muster-Min (100) und Muster-Max (104),
    damit die globale Min-Max-Normalisierung der Engine auf den Muster-Fenstern
    exakt der lokalen Query-Normalisierung entspricht → Distanz 0.
    """
    pattern = [100.0, 101.0, 102.0, 103.0, 104.0]
    filler = [101.5, 103.5] * 25  # oszillierend, immer im Muster-Wertebereich
    history = make_candles(pattern + filler + pattern)
    return make_candles(pattern), history


# ----------------------------------------------------------------------
# 1. Wiederholtes Muster wird mit niedriger Distanz gefunden
# ----------------------------------------------------------------------


def test_engine_finds_repeated_pattern() -> None:
    """Identisches Muster an zwei Stellen der Historie → Distanz 0 an genau diesen Stellen."""
    query, history = repeated_pattern_history()  # Muster bei Index 0 und 55

    engine = SimilarityEngine(window_size=5, top_k=5)
    result = engine.find_similar(query, history)

    assert result.matches, "Top-K-Suche muss Matches liefern"
    assert result.best_distance == pytest.approx(0.0)  # exakt wiederholtes Muster
    assert result.best_correlation == pytest.approx(1.0)  # identische Form → perfekte Korrelation
    exact_hits = {m.start_index for m in result.matches if m.distance == pytest.approx(0.0)}
    assert exact_hits == {0, 55}  # genau die beiden Muster-Positionen


# ----------------------------------------------------------------------
# 2. Normalisierung macht Muster über unterschiedliche Preisskalen vergleichbar
# ----------------------------------------------------------------------


def test_engine_normalization_makes_patterns_comparable() -> None:
    """Gleiche Form, 10-fache Preisskala → normalisierte Distanz ≈ 0, rohe Distanz riesig."""
    shape = [100.0, 105.0, 102.0, 108.0, 110.0, 107.0, 104.0, 109.0, 112.0, 115.0]
    query = make_candles(shape)
    history_low = make_candles(shape)  # Skala ~100
    history_high = make_candles([p * 10 for p in shape])  # Skala ~1000

    normalized = SimilarityEngine(window_size=10, top_k=3)
    result_low = normalized.find_similar(query, history_low)
    result_high = normalized.find_similar(query, history_high)

    # Normalisiert sind beide Skalen dasselbe Muster → identisch kleine Distanz.
    assert result_low.best_distance == pytest.approx(0.0)
    assert result_high.best_distance == pytest.approx(0.0)
    assert result_high.best_distance == pytest.approx(result_low.best_distance)

    # Ohne Normalisierung ist das Skalen-10-Muster für den Query praktisch unerkennbar.
    raw = SimilarityEngine(window_size=10, top_k=3, normalize=False)
    result_raw = raw.find_similar(query, history_high)
    assert result_raw.best_distance is not None
    assert result_raw.best_distance > 1000.0  # absolute Preisunterschiede dominieren
    assert (result_high.best_distance or 0.0) < result_raw.best_distance


# ----------------------------------------------------------------------
# 3. Distanz-Matrix: Diagonale null, symmetrisch, nicht-negativ
# ----------------------------------------------------------------------


def test_distance_matrix_diagonal_zero() -> None:
    """compute_distance_matrix: Diagonale 0, Matrix symmetrisch, Einträge ≥ 0 und endlich."""
    seq_shapes: list[list[float]] = [
        [100.0, 105.0, 102.0, 108.0, 110.0, 107.0, 104.0, 109.0],  # oszillierend
        [200.0, 210.0, 205.0, 215.0, 220.0, 218.0, 222.0, 225.0],  # anderer Verlauf
        [50.0, 51.0, 50.5, 52.0, 53.0, 52.5, 54.0, 53.5],  # klein, andere Skala
        [75.0] * 8,  # konstante Serie (Edge-Case: Normalisierung → [0.0] * n)
    ]
    sequences = [make_candles(shape) for shape in seq_shapes]

    engine = SimilarityEngine(window_size=8, top_k=3)
    matrix = engine.compute_distance_matrix(sequences)

    n = len(sequences)
    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)  # quadratisch

    for i in range(n):  # Diagonale ist exakt null (jeder Vergleich mit sich selbst)
        assert matrix[i][i] == 0.0

    for i in range(n):  # Symmetrie + Gültigkeit der Off-Diagonal-Einträge
        for j in range(n):
            if i == j:
                continue
            assert matrix[i][j] == pytest.approx(matrix[j][i])
            assert matrix[i][j] >= 0.0
            assert matrix[i][j] != float("inf")


# ----------------------------------------------------------------------
# 4. find_similar_with_candles liefert die Originalkerzen der Matches
# ----------------------------------------------------------------------


def test_find_similar_with_candles_returns_candle_data() -> None:
    """Jeder Match trägt die Originalkerzen history[start:end+1]; find_similar nicht."""
    closes = [100.0 + i for i in range(30)]
    history = make_candles(closes)
    query = make_candles(closes[10:15])

    engine = SimilarityEngine(window_size=5, top_k=4)
    result = engine.find_similar_with_candles(query, history)

    assert result.matches, "Suche muss Matches liefern"
    for match in result.matches:
        assert match.candles, "find_similar_with_candles muss Kerzen befüllen"
        assert len(match.candles) == engine.window_size
        # Die befüllten Kerzen sind exakt der beanspruchte Historie-Schnipsel.
        assert match.candles == history[match.start_index : match.end_index + 1]
        assert match.candles[0] is history[match.start_index]  # Original-Objekte, keine Kopien
        assert match.candles[0]["close"] == closes[match.start_index]
        assert match.candles[-1]["close"] == closes[match.end_index]

    # Kontrast: find_similar ohne Kerzen → candles bleibt leer.
    bare = engine.find_similar(query, history)
    assert bare.matches
    assert all(m.candles == [] for m in bare.matches)


# ----------------------------------------------------------------------
# 5. Integration: Engine → Store → Cache (gemockter InfluxDBStore)
# ----------------------------------------------------------------------


def test_engine_to_store_roundtrip() -> None:
    """Muster-Suche → Result-Felder korrekt, Cache-Lesezugriff liefert Engine-Matches 1:1.

    Im MVP berührt der Store weder InfluxDB-Write noch -Query.
    """
    query, history = repeated_pattern_history()
    store = make_mock_store()
    similarity_store = SimilarityStore(store, engine=SimilarityEngine(window_size=5, top_k=3))

    result = similarity_store.find_and_store(SYMBOL, TIMEFRAME, query, history, EXCHANGE)

    assert result.query_length == len(query)
    assert result.matches_found == 3  # top_k=3
    assert result.best_distance == pytest.approx(0.0)
    assert result.best_correlation == pytest.approx(1.0)

    # Cache-Lesezugriff: dieselben Matches in Engine-Reihenfolge (aufsteigende Distanz).
    entries = similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME)
    assert len(entries) == 3
    distances = [entry["distance"] for entry in entries]
    assert distances == sorted(distances)
    for entry in entries:
        assert entry["symbol"] == SYMBOL
        assert entry["exchange"] == EXCHANGE
        assert entry["timeframe"] == TIMEFRAME
        # Fenster-Zeit = Zeit der ersten Kerze des Matches (UTC-ISO).
        assert entry["time"] == history[entry["start_index"]]["time"]
        assert entry["end_index"] == entry["start_index"] + 4  # window_size=5
    assert entries[0]["distance"] == pytest.approx(0.0)
    assert (
        {entry["start_index"] for entry in entries if entry["distance"] == pytest.approx(0.0)}
        == {0, 55}
    )

    # MVP-Semantik: InfluxDB wird weder geschrieben noch gelesen.
    store.write_points.assert_not_awaited()
    store.query.assert_not_awaited()


def test_store_engine_error_graceful_and_cache_isolation() -> None:
    """Engine-Fehler → leeres Result ohne Exception, Cache bleibt leer; fremde Keys leer."""
    query, history = repeated_pattern_history()
    store = make_mock_store()
    broken_engine = MagicMock()
    broken_engine.find_similar_with_candles.side_effect = RuntimeError("defekte Kerzen")
    similarity_store = SimilarityStore(store, engine=broken_engine)

    result = similarity_store.find_and_store(SYMBOL, TIMEFRAME, query, history, EXCHANGE)

    assert result.matches_found == 0
    assert result.best_distance is None
    assert result.best_correlation is None
    # Cache wurde nicht berührt — kein Eintrag für das Symbol-Zeitfenster-Paar.
    assert similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME) == []
    # Isolation: ein anderes (Symbol, Timeframe)-Paar ist ebenfalls leer.
    assert similarity_store.get_similar_patterns("ETHUSDT", "1h") == []
    store.write_points.assert_not_awaited()
    store.query.assert_not_awaited()


def test_store_cache_readback_filtered_by_time_range() -> None:
    """get_similar_patterns: start/end filtern nach Fensterstartzeit; ungültig → ValueError."""
    query, history = repeated_pattern_history()
    store = make_mock_store()
    similarity_store = SimilarityStore(store, engine=SimilarityEngine(window_size=5, top_k=3))
    similarity_store.find_and_store(SYMBOL, TIMEFRAME, query, history, EXCHANGE)

    all_entries = similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME)
    assert len(all_entries) == 3

    # Nur Fenster, die exakt zu 00:00:00Z–00:00:59Z starten (das Muster bei Index 0).
    start_filtered = similarity_store.get_similar_patterns(
        SYMBOL, TIMEFRAME, start="2026-01-01T00:00:00Z", end="2026-01-01T00:00:59Z"
    )
    assert [entry["start_index"] for entry in start_filtered] == [0]

    # Ende vor allen Fensterzeiten → keine Treffer; Ende danach → alle drei.
    assert (
        similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME, end="2025-12-31T23:59:59Z") == []
    )
    assert (
        len(
            similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME, end="2026-01-02T00:00:00Z")
        )
        == 3
    )

    with pytest.raises(ValueError, match="invalid start timestamp"):
        similarity_store.get_similar_patterns(SYMBOL, TIMEFRAME, start="not-a-timestamp")
