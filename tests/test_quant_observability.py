"""Unit-Tests für die Quant-Observability (Quant-Plattform, P1-9).

Kein echtes InfluxDB: Stores werden ohne Verbindungsversuch erzeugt und
deren Zustand direkt gesetzt (``_ready`` / ``_client`` / ``_buffer``) —
gleiche Konvention wie ``test_quant_influxdb_client.py``.
"""

from __future__ import annotations

import pytest

from trading_harness.quant.influxdb_client import InfluxDBStore
from trading_harness.quant.observability import QuantHealthChecker, QuantMetrics

URL = "http://localhost:8086"


def make_store() -> InfluxDBStore:
    """Erzeugt einen Store ohne Verbindung (Fallback-Zustand)."""
    return InfluxDBStore(url=URL, token="test-token", org="smith", bucket="market_data")


def make_connected_store() -> InfluxDBStore:
    """Erzeugt einen Store im verbundenen Zustand (kein echter Ping)."""
    store = make_store()
    store._client = object()
    store._ready = True
    return store


def test_summary_initial_state_is_empty() -> None:
    """Frische Metriken liefern einen leeren, konsistenten Summary-Snapshot."""
    summary = QuantMetrics().get_summary()
    assert summary["total_writes"] == 0
    assert summary["total_downsampled"] == 0
    assert summary["total_queries"] == 0
    assert summary["total_errors"] == 0
    assert summary["candles_written"] == {}
    assert summary["query_duration_avg_ms"] == 0.0


def test_record_candles_written_aggregates_per_measurement() -> None:
    """Writes werden pro Measurement summiert und landen in total_writes."""
    metrics = QuantMetrics()
    metrics.record_candles_written("ohlcv", 10)
    metrics.record_candles_written("ohlcv", 5)
    metrics.record_candles_written("trades", 3)
    summary = metrics.get_summary()
    assert summary["total_writes"] == 18
    assert summary["candles_written"] == {"ohlcv": 15, "trades": 3}
    assert metrics.total_writes == 18


def test_record_candles_written_rejects_negative_count() -> None:
    """Negative Counts sind ein Programmierfehler und werden abgelehnt."""
    with pytest.raises(ValueError):
        QuantMetrics().record_candles_written("ohlcv", -1)


def test_record_candles_downsampled_tracks_timeframe_pairs() -> None:
    """Downsamples werden pro (source_tf, target_tf)-Paar getrennt gezählt."""
    metrics = QuantMetrics()
    metrics.record_candles_downsampled("1m", "1h", 24)
    metrics.record_candles_downsampled("1m", "1h", 1)
    metrics.record_candles_downsampled("5m", "1h", 12)
    summary = metrics.get_summary()
    assert summary["total_downsampled"] == 37
    assert summary["candles_downsampled"] == {"1m->1h": 25, "5m->1h": 12}


def test_record_query_tracks_duration_stats() -> None:
    """Queries zählen pro Measurement und erfassen total/avg/max-Dauer."""
    metrics = QuantMetrics()
    metrics.record_query("ohlcv", 10.0)
    metrics.record_query("ohlcv", 30.0)
    metrics.record_query("trades", 5.0)
    summary = metrics.get_summary()
    assert summary["total_queries"] == 3
    assert summary["queries"] == {"ohlcv": 2, "trades": 1}
    assert summary["query_duration_total_ms"] == 45.0
    assert summary["query_duration_avg_ms"] == 15.0
    assert summary["query_duration_max_ms"] == 30.0
    assert metrics.total_queries == 3


def test_record_query_rejects_negative_duration() -> None:
    """Negative Laufzeiten werden abgelehnt."""
    with pytest.raises(ValueError):
        QuantMetrics().record_query("ohlcv", -1.0)


def test_record_error_tracks_type_and_component() -> None:
    """Fehler werden nach Typ und Komponente getrennt gezählt."""
    metrics = QuantMetrics()
    metrics.record_error("write_failure", "influxdb")
    metrics.record_error("write_failure", "downsampler")
    metrics.record_error("query_timeout", "influxdb")
    summary = metrics.get_summary()
    assert summary["total_errors"] == 3
    assert summary["errors"] == {"write_failure": 2, "query_timeout": 1}
    assert summary["errors_by_component"] == {"influxdb": 2, "downsampler": 1}
    assert metrics.total_errors == 3


def test_health_check_with_connected_store() -> None:
    """Verbundener Store + injizierte Metriken → vollständiger Health-Snapshot."""
    store = make_connected_store()
    metrics = QuantMetrics()
    metrics.record_candles_written("ohlcv", 100)
    metrics.record_query("ohlcv", 42.0)
    metrics.record_error("write_failure", "influxdb")
    checker = QuantHealthChecker(store, metrics=metrics, url=URL, enabled=True)
    result = checker.check_all()
    assert result["influxdb"] == {"connected": True, "url": URL, "buffer_size": 0}
    assert result["metrics"] == {"total_writes": 100, "total_queries": 1, "total_errors": 1}
    assert result["enabled"] is True


def test_health_check_with_unavailable_store_reports_buffer() -> None:
    """Ausgefallener Store → connected=False, Buffer-Size sichtbar, URL-Fallback."""
    store = make_store()
    store._buffer.append({"measurement": "ohlcv", "tags": {}, "fields": {}, "timestamp": 1})
    store._buffer.append({"measurement": "ohlcv", "tags": {}, "fields": {}, "timestamp": 2})
    checker = QuantHealthChecker(store, enabled=False)
    result = checker.check_all()
    assert result["influxdb"]["connected"] is False
    assert result["influxdb"]["buffer_size"] == 2
    assert result["influxdb"]["url"] == URL
    assert result["enabled"] is False
    assert result["metrics"] == {"total_writes": 0, "total_queries": 0, "total_errors": 0}


def test_health_checker_defaults_to_fresh_shared_metrics() -> None:
    """Ohne injizierte Metriken erstellt der Checker eigene, wiederverwendbare Metriken."""
    checker = QuantHealthChecker(make_store(), url=URL)
    assert checker.check_all()["metrics"] == {
        "total_writes": 0,
        "total_queries": 0,
        "total_errors": 0,
    }
    checker.metrics.record_error("unexpected", "unknown")
    assert checker.check_all()["metrics"]["total_errors"] == 1
