"""Tests für die Quant-API-Endpunkte (P1-8).

Alle InfluxDB-Zugriffe sind gemockt — es wird nie eine echte InfluxDB-
Verbindung aufgebaut. Der Router wird in einen eigenständigen FastAPI-Test-App
gemountet (die Einhängung in die Haupt-App erfolgt erst in Phase 9).
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trading_harness.api import quant_routes, security
from trading_harness.config import Settings
from trading_harness.quant import schema as quant_schema
from trading_harness.quant.regime_detection import REGIME_NAMES

# Naiv (bewusst ohne tzinfo): die Route interpretiert naive Zeiten als UTC.
CANDLE_TIME = datetime.fromisoformat("2026-08-25T12:00:00")
EXPECTED_NS = int(CANDLE_TIME.replace(tzinfo=UTC).timestamp() * 1_000_000_000)
INGEST_URL = "/quant/ingest/ohlcv"
STATUS_URL = "/quant/status"
SCHEMA_URL = "/quant/schema"
FEATURES_COMPUTE_URL = "/quant/features/compute"
FEATURES_URL = "/quant/features/{symbol}"
ANOMALIES_DETECT_URL = "/quant/anomalies/detect"
ANOMALIES_URL = "/quant/anomalies/{symbol}"
REGIME_DETECT_URL = "/quant/regime/detect"
REGIME_URL = "/quant/regime/{symbol}"
SIMILARITY_FIND_URL = "/quant/similarity/find"
SIMILARITY_URL = "/quant/similarity/{symbol}"
OUTCOMES_COMPUTE_URL = "/quant/outcomes/compute"
OUTCOMES_URL = "/quant/outcomes/{symbol}"


def make_settings(**overrides: Any) -> Settings:
    """Deterministische Settings (Explizit-Kwargs schlagen .env/Evnt-Values)."""
    base: dict[str, object] = {
        "influxdb_url": "http://localhost:8086",
        "influxdb_token": "test-token",
        "influxdb_org": "smith",
        "influxdb_bucket": "market_data",
        "influxdb_enabled": True,
        "read_api_key": "",
        "trade_api_key": "",
    }
    base.update(overrides)
    return Settings(**base)


def make_store_mock(healthy: bool = True) -> MagicMock:
    """InfluxDBStore-Mock mit Async-Verhalten (kein echter Client)."""
    store = MagicMock()
    store.health_check = AsyncMock(return_value=healthy)
    store.write_batch = AsyncMock()
    store.write_points = AsyncMock()
    store.buffer_size = MagicMock(return_value=0)
    store.is_available = healthy
    return store


@pytest.fixture
def quant_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, MagicMock, Settings]:
    """Test-App mit Quant-Router; Settings und Store sind Mocks."""
    settings = make_settings()
    store = make_store_mock()
    monkeypatch.setattr(quant_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(quant_routes, "get_influx_store", lambda: store)
    # Auth-Dependencies (security.require_*_key) nutzen denselben Settings-Mock.
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(quant_routes.router)
    return TestClient(app), store, settings


@pytest.fixture
def feature_client(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, MagicMock, Settings, MagicMock]:
    """Quant-Test-App mit zusätzlichem FeatureStore-Mock (P2-2-Interface)."""
    client, store, settings = quant_client
    feature_store = MagicMock()
    feature_store.compute_and_store = AsyncMock(return_value=MagicMock())
    feature_store.get_features = AsyncMock(return_value=[])
    monkeypatch.setattr(quant_routes, "get_feature_store", lambda: feature_store)
    return client, store, settings, feature_store


@pytest.fixture
def anomaly_client(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, MagicMock, Settings, MagicMock]:
    """Quant-Test-App mit zusätzlichem AnomalyStore-Mock (P3-2-Interface)."""
    client, store, settings = quant_client
    anomaly_store = MagicMock()
    anomaly_store.detect_and_store = AsyncMock(
        return_value=MagicMock(anomalies_found=2, stored=True)
    )
    anomaly_store.get_anomalies = AsyncMock(return_value=[])
    monkeypatch.setattr(quant_routes, "get_anomaly_store", lambda: anomaly_store)
    return client, store, settings, anomaly_store


@pytest.fixture
def regime_client(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, MagicMock, Settings, MagicMock]:
    """Quant-Test-App mit zusätzlichem RegimeStore-Mock (P4-2-Interface)."""
    client, store, settings = quant_client
    regime_store = MagicMock()
    regime_store.detect_and_store = AsyncMock(
        return_value=MagicMock(regime="strong_bull", confidence=0.85, stored=True)
    )
    regime_store.get_regime = AsyncMock(return_value=[])
    monkeypatch.setattr(quant_routes, "get_regime_store", lambda: regime_store)
    return client, store, settings, regime_store


@pytest.fixture
def outcome_client(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, MagicMock, Settings, MagicMock]:
    """Quant-Test-App mit zusätzlichem ForwardOutcomeStore-Mock (P6-2-Interface)."""
    client, store, settings = quant_client
    outcome_store = MagicMock()
    outcome_store.compute_and_store = AsyncMock(return_value=MagicMock(stored=True))
    outcome_store.get_outcomes = AsyncMock(return_value=[])
    monkeypatch.setattr(quant_routes, "get_outcome_store", lambda: outcome_store)
    return client, store, settings, outcome_store


def make_candle(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "time": CANDLE_TIME.isoformat(),
        "open": 50000.0,
        "high": 51000.0,
        "low": 49000.0,
        "close": 50500.0,
        "volume": 12.5,
        "trade_count": 42,
    }
    base.update(overrides)
    return base


def make_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "candles": [make_candle()],
    }
    base.update(overrides)
    return base


def make_feature_candles(count: int = 42) -> list[dict[str, Any]]:
    """Deterministische, konsistente Kerzenreihe.

    42 Kerzen liefern bei Standard-Parametern Historie für alle sechs
    Indikatoren (MACD braucht 35) → ``feature_count`` = 6.
    """
    candles: list[dict[str, Any]] = []
    for i in range(count):
        open_price = 50000.0 + i * 10.0
        close_price = open_price + (15.0 if i % 2 == 0 else -8.0)
        candles.append(
            {
                "time": (CANDLE_TIME + timedelta(minutes=i)).isoformat(),
                "open": open_price,
                "high": max(open_price, close_price) + 5.0,
                "low": min(open_price, close_price) - 5.0,
                "close": close_price,
                "volume": 10.0 + i,
            }
        )
    return candles


def make_compute_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "exchange": "binance",
        "candles": make_feature_candles(),
    }
    base.update(overrides)
    return base


def make_anomaly_candles(count: int = 21) -> list[dict[str, Any]]:
    """Flache Baseline + Sprungkerze → deterministisch genau 2 Anomalien.

    20 flache Kerzen (close=50000, volume=10) bilden eine Null-Varianz-
    Baseline; die letzte Sprungkerze (close=60000, volume=100) triggert
    ``price_shock`` und ``volume_spike`` (flache Baseline → Z-Score
    2×Schwelle). Bei Default-``window_size=20`` wird mit 21 Kerzen nur die
    letzte Kerze bewertet.
    """
    candles: list[dict[str, Any]] = []
    for i in range(count):
        is_shock = i == count - 1
        price = 60000.0 if is_shock else 50000.0
        candles.append(
            {
                "time": (CANDLE_TIME + timedelta(minutes=i)).isoformat(),
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 100.0 if is_shock else 10.0,
            }
        )
    return candles


def make_flat_candles(count: int = 21) -> list[dict[str, Any]]:
    """Komplett flache Kerzenreihe → keine Anomalien."""
    return [
        {
            "time": (CANDLE_TIME + timedelta(minutes=i)).isoformat(),
            "open": 50000.0,
            "high": 50001.0,
            "low": 49999.0,
            "close": 50000.0,
            "volume": 10.0,
        }
        for i in range(count)
    ]


def make_detect_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "exchange": "binance",
        "candles": make_anomaly_candles(),
    }
    base.update(overrides)
    return base


def make_regime_candles(count: int = 60) -> list[dict[str, Any]]:
    """Deterministische stetige Uptrend-Reihe (60 Kerzen ≥ 51 Minimum).

    Lineare Trendfolge (close = 50000 + 100·i) macht das ``RegimeDetector``-
    Ergebnis vollständig deterministisch: SMA fast > SMA slow und ADX≈100 →
    ``strong_bull`` mit confidence 1.0.
    """
    candles: list[dict[str, Any]] = []
    for i in range(count):
        open_price = 50000.0 + i * 100.0 - 50.0
        close_price = 50000.0 + i * 100.0
        candles.append(
            {
                "time": (CANDLE_TIME + timedelta(minutes=i)).isoformat(),
                "open": open_price,
                "high": close_price + 10.0,
                "low": open_price - 10.0,
                "close": close_price,
                "volume": 10.0,
            }
        )
    return candles


def make_regime_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "exchange": "binance",
        "candles": make_regime_candles(),
    }
    base.update(overrides)
    return base


def make_similarity_candles(prices: list[float]) -> list[dict[str, Any]]:
    """Deterministische OHLC-konsistente Kerzenreihe aus einer Preisliste."""
    return [
        {
            "time": (CANDLE_TIME + timedelta(minutes=i)).isoformat(),
            "open": price,
            "high": price + 5.0,
            "low": price - 5.0,
            "close": price,
            "volume": 10.0,
        }
        for i, price in enumerate(prices)
    ]


def make_similarity_history(count: int = 20) -> list[dict[str, Any]]:
    """Deterministische Historie (Standard: exakt window_size Kerzen)."""
    return make_similarity_candles([100.0 + (i % 7) * 3.0 for i in range(count)])


def make_similarity_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "exchange": "binance",
        "query": make_similarity_candles([100.0, 110.0, 105.0, 115.0, 108.0]),
        "history": make_similarity_history(),
        "top_k": 5,
    }
    base.update(overrides)
    return base


def make_outcome_candles(count: int = 30) -> list[dict[str, Any]]:
    """Deterministische stetige Uptrend-Reihe (close = 100 + i).

    Alle Forward Returns sind positiv (hit_rate=1.0) und exakt bestimmbar:
    Return nach ``h`` Kerzen ab Index ``i`` = ``h / (100 + i)``.
    """
    return [
        {
            "time": (CANDLE_TIME + timedelta(minutes=i)).isoformat(),
            "open": 99.0 + i,
            "high": 102.0 + i,
            "low": 98.0 + i,
            "close": 100.0 + i,
            "volume": 10.0,
        }
        for i in range(count)
    ]


def make_outcome_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "exchange": "binance",
        "candles": make_outcome_candles(),
        "pattern_length": 10,
        "horizons": [5, 10],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# POST /quant/ingest/ohlcv
# ---------------------------------------------------------------------------


def test_ingest_ohlcv_success(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Gültige Kerze → 200, Punkt wird als Batch mit korrekten Tags/Fields geschrieben."""
    client, store, _ = quant_client
    resp = client.post(INGEST_URL, json=make_payload())
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "written": 1, "skipped": 0}

    store.write_batch.assert_called_once()
    kwargs = store.write_batch.call_args.kwargs
    assert kwargs["measurement"] == quant_schema.OHLCV_MEASUREMENT
    assert kwargs["tags"] == {"symbol": "BTCUSDT", "exchange": "api", "timeframe": "1m"}
    assert kwargs["points"] == [
        {
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 12.5,
            "trade_count": 42,
        }
    ]
    assert kwargs["timestamps"] == [EXPECTED_NS]


def test_ingest_ohlcv_multiple_symbols(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """2 Symbole × 2 Kerzen → 4 Punkte, ein Batch-Write pro Symbol."""
    client, store, _ = quant_client
    payload = make_payload(
        symbols=["BTCUSDT", "ETHUSDT"],
        candles=[make_candle(), make_candle(close=50600.0)],
    )
    resp = client.post(INGEST_URL, json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "written": 4, "skipped": 0}
    assert store.write_batch.call_count == 2
    first_tags = store.write_batch.call_args_list[0].kwargs["tags"]
    second_tags = store.write_batch.call_args_list[1].kwargs["tags"]
    assert first_tags["symbol"] == "BTCUSDT"
    assert second_tags["symbol"] == "ETHUSDT"
    assert len(store.write_batch.call_args_list[0].kwargs["points"]) == 2


def test_ingest_ohlcv_skips_inconsistent_candle(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Kerze mit high < close wird übersprungen, gültige Kerze wird geschrieben."""
    client, store, _ = quant_client
    payload = make_payload(
        candles=[
            make_candle(),  # konsistent
            make_candle(high=100.0, close=200.0),  # high umschließt close nicht
        ],
    )
    resp = client.post(INGEST_URL, json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "written": 1, "skipped": 1}
    store.write_batch.assert_called_once()
    assert len(store.write_batch.call_args.kwargs["points"]) == 1


def test_ingest_ohlcv_all_invalid_no_write(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Alle Kerzen inkonsistent → kein Write, written=0."""
    client, store, _ = quant_client
    payload = make_payload(candles=[make_candle(high=1.0, low=0.0, open=5.0, close=9.0)])
    resp = client.post(INGEST_URL, json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "written": 0, "skipped": 1}
    store.write_batch.assert_not_called()


def test_ingest_ohlcv_disabled_returns_403(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, kein Write."""
    client, store, settings = quant_client
    settings.influxdb_enabled = False
    resp = client.post(INGEST_URL, json=make_payload())
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    store.write_batch.assert_not_called()


def test_ingest_ohlcv_requires_trade_key(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Mit gesetztem Trade-Key: fehlender Key → 401, falscher Key → 403, richtiger → 200."""
    client, _, settings = quant_client
    settings.trade_api_key = "secret-trade"

    assert client.post(INGEST_URL, json=make_payload()).status_code == 401
    assert (
        client.post(INGEST_URL, json=make_payload(), headers={"X-Trade-API-Key": "wrong"})
        .status_code
        == 403
    )
    resp = client.post(INGEST_URL, json=make_payload(), headers={"X-Trade-API-Key": "secret-trade"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_ohlcv_invalid_timeframe_422(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Unbekannter Timeframe → 422 (Validierung gegen SUPPORTED_TIMEFRAMES)."""
    client, store, _ = quant_client
    resp = client.post(INGEST_URL, json=make_payload(timeframe="2m"))
    assert resp.status_code == 422
    store.write_batch.assert_not_called()


def test_ingest_ohlcv_empty_candles_422(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Leere Candle-Liste → 422."""
    client, _, _ = quant_client
    resp = client.post(INGEST_URL, json=make_payload(candles=[]))
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /quant/status
# ---------------------------------------------------------------------------


def test_status_connected(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """InfluxDB gesund → connected=true, URL und Feature-Flag aus den Settings."""
    client, store, settings = quant_client
    store.health_check = AsyncMock(return_value=True)
    store.buffer_size = MagicMock(return_value=3)
    resp = client.get(STATUS_URL)
    assert resp.status_code == 200
    data = resp.json()
    assert data["influxdb_connected"] is True
    assert data["influxdb_url"] == settings.influxdb_url
    assert data["quant_enabled"] is True
    assert data["buffered_points"] == 3
    assert data["feature_version"] == quant_schema.FEATURE_VERSION


def test_status_disconnected(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """InfluxDB down → connected=false (kein Fehler, nur Status)."""
    client, store, _ = quant_client
    store.health_check = AsyncMock(return_value=False)
    resp = client.get(STATUS_URL)
    assert resp.status_code == 200
    assert resp.json()["influxdb_connected"] is False


def test_status_requires_read_key(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Mit gesetztem Read-Key: fehlender Key → 401."""
    client, _, settings = quant_client
    settings.read_api_key = "secret-read"
    assert client.get(STATUS_URL).status_code == 401
    resp = client.get(STATUS_URL, headers={"X-Read-API-Key": "secret-read"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /quant/schema
# ---------------------------------------------------------------------------


def test_schema_response(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Schema-Doku: reale Konstanten aus quant/schema.py + trade_count als API-Field."""
    client, _, _ = quant_client
    resp = client.get(SCHEMA_URL)
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_version"] == quant_schema.FEATURE_VERSION
    assert quant_schema.OHLCV_MEASUREMENT in data["measurements"]
    assert data["timeframes"] == list(quant_schema.SUPPORTED_TIMEFRAMES)
    assert data["tags"] == list(quant_schema.OHLCV_TAGS)
    assert data["fields"] == list(quant_schema.OHLCV_FIELDS) + ["trade_count"]
    assert data["fields_by_measurement"][quant_schema.OHLCV_MEASUREMENT] == list(
        quant_schema.OHLCV_FIELDS
    )


def test_schema_requires_read_key(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """Mit gesetztem Read-Key: fehlender Key → 401, richtiger Key → 200."""
    client, _, settings = quant_client
    settings.read_api_key = "secret-read"
    assert client.get(SCHEMA_URL).status_code == 401
    resp = client.get(SCHEMA_URL, headers={"X-Read-API-Key": "secret-read"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /quant/features/compute (P2-3)
# ---------------------------------------------------------------------------


def test_compute_features_success(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """42 Kerzen → 200, feature_count=6 (alle Indikatoren), Store-Aufruf erfolgt."""
    client, _, _, feature_store = feature_client
    resp = client.post(FEATURES_COMPUTE_URL, json=make_compute_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "feature_count": 6,
        "stored": True,
    }

    feature_store.compute_and_store.assert_awaited_once()
    args = feature_store.compute_and_store.call_args.args
    kwargs = feature_store.compute_and_store.call_args.kwargs
    assert args[0] == "BTCUSDT"
    assert args[1] == "1m"
    assert len(args[2]) == 42
    assert args[2][0]["close"] == 50015.0  # erste Kerze: 50000 + 15
    assert "trade_count" not in args[2][0]
    assert kwargs == {"exchange": "binance"}


def test_compute_features_without_store(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FeatureStore nicht verfügbar (None) → Compute bleibt funktional, stored=false."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_feature_store", lambda: None)
    resp = client.post(FEATURES_COMPUTE_URL, json=make_compute_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["feature_count"] == 6
    assert data["stored"] is False


def test_compute_features_insufficient_history(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Nur 10 Kerzen → zu wenig Historie für alle Indikatoren, nur VWAP → count=1."""
    client, _, _, feature_store = feature_client
    resp = client.post(
        FEATURES_COMPUTE_URL, json=make_compute_payload(candles=make_feature_candles(10))
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_count"] == 1
    assert data["stored"] is True
    feature_store.compute_and_store.assert_awaited_once()


def test_compute_features_invalid_data_422(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Negativer Preis, leere Candle-Liste, ungültiger Timeframe → 422, kein Store-Write."""
    client, _, _, feature_store = feature_client

    bad_candles = make_feature_candles()
    bad_candles[0] = {**bad_candles[0], "open": -5.0}
    resp = client.post(FEATURES_COMPUTE_URL, json=make_compute_payload(candles=bad_candles))
    assert resp.status_code == 422

    assert client.post(FEATURES_COMPUTE_URL, json=make_compute_payload(candles=[])).status_code == 422
    assert (
        client.post(FEATURES_COMPUTE_URL, json=make_compute_payload(timeframe="2m")).status_code
        == 422
    )
    feature_store.compute_and_store.assert_not_awaited()


def test_compute_features_requires_trade_key(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Trade-Key: fehlender Key → 401, falscher Key → 403, richtiger → 200."""
    client, _, settings, _ = feature_client
    settings.trade_api_key = "secret-trade"

    assert client.post(FEATURES_COMPUTE_URL, json=make_compute_payload()).status_code == 401
    assert (
        client.post(
            FEATURES_COMPUTE_URL,
            json=make_compute_payload(),
            headers={"X-Trade-API-Key": "wrong"},
        ).status_code
        == 403
    )
    resp = client.post(
        FEATURES_COMPUTE_URL,
        json=make_compute_payload(),
        headers={"X-Trade-API-Key": "secret-trade"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_compute_features_disabled_returns_403(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, kein Compute/Write."""
    client, _, settings, feature_store = feature_client
    settings.influxdb_enabled = False
    resp = client.post(FEATURES_COMPUTE_URL, json=make_compute_payload())
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    feature_store.compute_and_store.assert_not_awaited()


# ---------------------------------------------------------------------------
# GET /quant/features/{symbol} (P2-3)
# ---------------------------------------------------------------------------


def test_get_features_returns_features(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Gespeicherte Features → 200 mit symbol, features und count; Filter-Parameter weitergereicht."""
    client, _, _, feature_store = feature_client
    feature_store.get_features = AsyncMock(
        return_value=[
            {"time": "2026-08-25T12:00:00Z", "rsi": 55.5},
            {"time": "2026-08-25T12:01:00Z", "rsi": 56.1},
        ]
    )
    resp = client.get(
        FEATURES_URL.format(symbol="BTCUSDT"),
        params={
            "timeframe": "1m",
            "features": "rsi,macd",
            "start": "2026-08-25T12:00:00Z",
            "end": "2026-08-25T12:05:00Z",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["count"] == 2
    assert data["features"][0]["rsi"] == 55.5

    # start/end als UTC-ISO-Strings an den FeatureStore weitergegeben.
    feature_store.get_features.assert_awaited_once_with(
        "BTCUSDT",
        "1m",
        ["rsi", "macd"],
        "2026-08-25T12:00:00+00:00",
        "2026-08-25T12:05:00+00:00",
    )


def test_get_features_unknown_symbol_empty(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekanntes Symbol ohne gespeicherte Features → 200 mit leerer Liste (kein 404-Fehlerpfad)."""
    client, _, _, feature_store = feature_client
    feature_store.get_features = AsyncMock(return_value=[])
    resp = client.get(FEATURES_URL.format(symbol="NOPE"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "NOPE", "features": [], "count": 0}


def test_get_features_without_store_empty(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FeatureStore nicht verfügbar → 200 mit leerer Feature-Liste."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_feature_store", lambda: None)
    resp = client.get(FEATURES_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "BTCUSDT", "features": [], "count": 0}


def test_get_features_requires_read_key(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Read-Key: fehlender Key → 401, richtiger Key → 200."""
    client, _, settings, _ = feature_client
    settings.read_api_key = "secret-read"
    assert client.get(FEATURES_URL.format(symbol="BTCUSDT")).status_code == 401
    resp = client.get(
        FEATURES_URL.format(symbol="BTCUSDT"),
        headers={"X-Read-API-Key": "secret-read"},
    )
    assert resp.status_code == 200


def test_get_features_invalid_timeframe_422(
    feature_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekannter Timeframe → 422, keine Store-Abfrage."""
    client, _, _, feature_store = feature_client
    resp = client.get(FEATURES_URL.format(symbol="BTCUSDT"), params={"timeframe": "2m"})
    assert resp.status_code == 422
    feature_store.get_features.assert_not_awaited()


def test_compute_features_real_store_wiring(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Echter FeatureStore (P2-2) auf mocktem InfluxDBStore: Getter + Write-Pipeline.

    Verifiziert die Integration der lazy ``get_feature_store``-Konstruktion
    (``FeatureStore(store=...)``) und des ``compute_and_store``-Aufrufs gegen
    das echte P2-2-Modul — nur der InfluxDB-Zugriff bleibt gemockt.
    """
    if importlib.util.find_spec("trading_harness.quant.feature_store") is None:
        pytest.skip("quant/feature_store.py nicht vorhanden (P2-2 parallel)")
    client, store, _ = quant_client
    # Globalen Cache leeren, damit der Getter in diesem Test neu konstruiert.
    monkeypatch.setattr(quant_routes, "_feature_store", None)
    resp = client.post(
        FEATURES_COMPUTE_URL, json=make_compute_payload(candles=make_feature_candles(5))
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "feature_count": 1,  # 5 Kerzen → nur VWAP verfügbar
        "stored": True,
    }
    # Echter FeatureStore: ein write_points-Aufruf pro Kerze.
    assert store.write_points.await_count == 5
    assert store.write_points.call_args.kwargs["measurement"] == quant_schema.FEATURE_MEASUREMENT


# ---------------------------------------------------------------------------
# POST /quant/anomalies/detect (P3-3)
# ---------------------------------------------------------------------------


def test_detect_anomalies_success(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Sprungkerzenreihe → 200, AnomalyResult-Felder weitergereicht, Store-Aufruf erfolgt."""
    client, _, _, anomaly_store = anomaly_client
    resp = client.post(ANOMALIES_DETECT_URL, json=make_detect_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "anomalies_found": 2,
        "stored": True,
    }

    anomaly_store.detect_and_store.assert_awaited_once()
    args = anomaly_store.detect_and_store.call_args.args
    kwargs = anomaly_store.detect_and_store.call_args.kwargs
    assert args[0] == "BTCUSDT"
    assert args[1] == "1m"
    assert len(args[2]) == 21
    assert args[2][-1]["close"] == 60000.0  # Sprungkerze
    assert args[2][-1]["volume"] == 100.0
    assert kwargs == {"exchange": "binance"}


def test_detect_anomalies_without_store(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnomalyStore nicht verfügbar (None) → echter Detector läuft, stored=false.

    Mit der Sprungkerzenreihe (20 flache Kerzen + Sprung) erkennt der
    ``AnomalyDetector`` deterministisch genau 2 Anomalien
    (``price_shock`` + ``volume_spike``).
    """
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_anomaly_store", lambda: None)
    resp = client.post(ANOMALIES_DETECT_URL, json=make_detect_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "anomalies_found": 2,
        "stored": False,
    }


def test_detect_anomalies_no_anomalies(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Flache Kerzenreihe ohne Sprünge → 200, Store-Result (0, stored=False) weitergereicht."""
    client, _, _, anomaly_store = anomaly_client
    anomaly_store.detect_and_store = AsyncMock(
        return_value=MagicMock(anomalies_found=0, stored=False)
    )
    resp = client.post(ANOMALIES_DETECT_URL, json=make_detect_payload(candles=make_flat_candles()))
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "anomalies_found": 0,
        "stored": False,
    }
    anomaly_store.detect_and_store.assert_awaited_once()
    assert len(anomaly_store.detect_and_store.call_args.args[2]) == 21


def test_detect_anomalies_insufficient_history(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nur 10 Kerzen (weniger als window_size+1) → 200, anomalies_found=0 (kein Fehler)."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_anomaly_store", lambda: None)
    resp = client.post(
        ANOMALIES_DETECT_URL, json=make_detect_payload(candles=make_flat_candles(10))
    )
    assert resp.status_code == 200
    assert resp.json()["anomalies_found"] == 0
    assert resp.json()["stored"] is False


def test_detect_anomalies_invalid_data_422(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Negativer Preis, leere Candle-Liste, ungültiger Timeframe → 422, kein Store-Write."""
    client, _, _, anomaly_store = anomaly_client

    bad_candles = make_anomaly_candles()
    bad_candles[0] = {**bad_candles[0], "open": -5.0}
    resp = client.post(ANOMALIES_DETECT_URL, json=make_detect_payload(candles=bad_candles))
    assert resp.status_code == 422

    assert (
        client.post(ANOMALIES_DETECT_URL, json=make_detect_payload(candles=[])).status_code == 422
    )
    assert (
        client.post(
            ANOMALIES_DETECT_URL, json=make_detect_payload(timeframe="2m")
        ).status_code
        == 422
    )
    assert client.post(ANOMALIES_DETECT_URL, json=make_detect_payload(symbol="")).status_code == 422
    anomaly_store.detect_and_store.assert_not_awaited()


def test_detect_anomalies_requires_trade_key(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Trade-Key: fehlender Key → 401, falscher Key → 403, richtiger → 200."""
    client, _, settings, _ = anomaly_client
    settings.trade_api_key = "secret-trade"

    assert client.post(ANOMALIES_DETECT_URL, json=make_detect_payload()).status_code == 401
    assert (
        client.post(
            ANOMALIES_DETECT_URL,
            json=make_detect_payload(),
            headers={"X-Trade-API-Key": "wrong"},
        ).status_code
        == 403
    )
    resp = client.post(
        ANOMALIES_DETECT_URL,
        json=make_detect_payload(),
        headers={"X-Trade-API-Key": "secret-trade"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_detect_anomalies_disabled_returns_403(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, kein Detect-Store-Aufruf."""
    client, _, settings, anomaly_store = anomaly_client
    settings.influxdb_enabled = False
    resp = client.post(ANOMALIES_DETECT_URL, json=make_detect_payload())
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    anomaly_store.detect_and_store.assert_not_awaited()


def test_detect_anomalies_real_store_wiring(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Echter AnomalyStore (P3-2) auf mocktem InfluxDBStore: Getter + Write-Pipeline.

    Verifiziert die Integration der lazy ``get_anomaly_store``-Konstruktion
    (``AnomalyStore(store=...)``) und des ``detect_and_store``-Aufrufs gegen
    das echte P3-2-Modul — nur der InfluxDB-Zugriff bleibt gemockt.
    """
    if importlib.util.find_spec("trading_harness.quant.anomaly_store") is None:
        pytest.skip("quant/anomaly_store.py nicht vorhanden (P3-2 parallel)")
    client, store, _ = quant_client
    # Globalen Cache leeren, damit der Getter in diesem Test neu konstruiert.
    monkeypatch.setattr(quant_routes, "_anomaly_store", None)
    resp = client.post(ANOMALIES_DETECT_URL, json=make_detect_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "anomalies_found": 2,
        "stored": True,
    }
    # Echter AnomalyStore: ein write_points-Aufruf pro erkannter Anomalie.
    assert store.write_points.await_count == 2
    assert store.write_points.call_args.kwargs["measurement"] == quant_schema.ANOMALY_MEASUREMENT


# ---------------------------------------------------------------------------
# GET /quant/anomalies/{symbol} (P3-3)
# ---------------------------------------------------------------------------


def test_get_anomalies_returns_anomalies(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Gespeicherte Anomalien → 200 mit symbol, anomalies und count; Filter weitergereicht."""
    client, _, _, anomaly_store = anomaly_client
    anomaly_store.get_anomalies = AsyncMock(
        return_value=[
            {
                "time": "2026-08-25T12:00:20Z",
                "anomaly_type": "price_shock",
                "severity": 1.0,
                "value": 0.182,
            },
            {
                "time": "2026-08-25T12:00:20Z",
                "anomaly_type": "volume_spike",
                "severity": 0.75,
                "value": 100.0,
            },
        ]
    )
    resp = client.get(
        ANOMALIES_URL.format(symbol="BTCUSDT"),
        params={
            "timeframe": "1m",
            "anomaly_type": "price_shock",
            "start": "2026-08-25T12:00:00Z",
            "end": "2026-08-25T12:05:00Z",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["count"] == 2
    assert data["anomalies"][0]["anomaly_type"] == "price_shock"

    # anomaly_type/start/end (UTC-ISO) an den AnomalyStore weitergegeben.
    anomaly_store.get_anomalies.assert_awaited_once_with(
        "BTCUSDT",
        "1m",
        "price_shock",
        "2026-08-25T12:00:00+00:00",
        "2026-08-25T12:05:00+00:00",
    )


def test_get_anomalies_without_store_empty(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnomalyStore nicht verfügbar → 200 mit leerer Anomalie-Liste."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_anomaly_store", lambda: None)
    resp = client.get(ANOMALIES_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "BTCUSDT", "anomalies": [], "count": 0}


def test_get_anomalies_unknown_symbol_empty(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekanntes Symbol ohne gespeicherte Anomalien → 200 mit leerer Liste (kein 404)."""
    client, _, _, _ = anomaly_client
    resp = client.get(ANOMALIES_URL.format(symbol="NOPE"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "NOPE", "anomalies": [], "count": 0}


def test_get_anomalies_requires_read_key(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Read-Key: fehlender Key → 401, richtiger Key → 200."""
    client, _, settings, _ = anomaly_client
    settings.read_api_key = "secret-read"
    assert client.get(ANOMALIES_URL.format(symbol="BTCUSDT")).status_code == 401
    resp = client.get(
        ANOMALIES_URL.format(symbol="BTCUSDT"),
        headers={"X-Read-API-Key": "secret-read"},
    )
    assert resp.status_code == 200


def test_get_anomalies_invalid_timeframe_422(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekannter Timeframe → 422, keine Store-Abfrage."""
    client, _, _, anomaly_store = anomaly_client
    resp = client.get(ANOMALIES_URL.format(symbol="BTCUSDT"), params={"timeframe": "2m"})
    assert resp.status_code == 422
    anomaly_store.get_anomalies.assert_not_awaited()


def test_get_anomalies_disabled_returns_403(
    anomaly_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, keine Store-Abfrage."""
    client, _, settings, anomaly_store = anomaly_client
    settings.influxdb_enabled = False
    resp = client.get(ANOMALIES_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    anomaly_store.get_anomalies.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /quant/regime/detect (P4-3)
# ---------------------------------------------------------------------------


def test_detect_regime_success(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Uptrend-Kerzen → 200, Store-Ergebnis (regime/confidence/stored) weitergereicht."""
    client, _, _, regime_store = regime_client
    resp = client.post(REGIME_DETECT_URL, json=make_regime_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "regime": "strong_bull",
        "confidence": 0.85,
        "stored": True,
    }

    regime_store.detect_and_store.assert_awaited_once()
    args = regime_store.detect_and_store.call_args.args
    kwargs = regime_store.detect_and_store.call_args.kwargs
    assert args[0] == "BTCUSDT"
    assert args[1] == "1m"
    assert len(args[2]) == 60
    assert args[2][-1]["close"] == 55900.0  # letzte Kerze des Uptrends
    assert kwargs == {"exchange": "binance"}


def test_detect_regime_without_store(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RegimeStore nicht verfügbar (None) → echter RegimeDetector läuft, stored=false.

    Die 60er-Uptrendreihe (≥ Minimum 51) liefert deterministisch
    ``strong_bull`` mit confidence 1.0 (ADX≈100 → min(ADX/50, 1.0)).
    """
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_regime_store", lambda: None)
    resp = client.post(REGIME_DETECT_URL, json=make_regime_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "regime": "strong_bull",
        "confidence": 1.0,
        "stored": False,
    }


def test_detect_regime_insufficient_history(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nur 10 Kerzen (< 51 Minimum) → 200, regime=range, confidence=0.5 (kein Fehler)."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_regime_store", lambda: None)
    resp = client.post(
        REGIME_DETECT_URL, json=make_regime_payload(candles=make_regime_candles(10))
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "symbol": "BTCUSDT",
        "regime": "range",
        "confidence": 0.5,
        "stored": False,
    }


def test_detect_regime_invalid_data_422(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Negativer Preis, leere Candle-Liste, ungültiger Timeframe, leerer Symbol → 422."""
    client, _, _, regime_store = regime_client

    bad_candles = make_regime_candles()
    bad_candles[0] = {**bad_candles[0], "open": -5.0}
    resp = client.post(REGIME_DETECT_URL, json=make_regime_payload(candles=bad_candles))
    assert resp.status_code == 422

    assert client.post(REGIME_DETECT_URL, json=make_regime_payload(candles=[])).status_code == 422
    assert (
        client.post(REGIME_DETECT_URL, json=make_regime_payload(timeframe="2m")).status_code == 422
    )
    assert client.post(REGIME_DETECT_URL, json=make_regime_payload(symbol="")).status_code == 422
    regime_store.detect_and_store.assert_not_awaited()


def test_detect_regime_requires_trade_key(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Trade-Key: fehlender Key → 401, falscher Key → 403, richtiger → 200."""
    client, _, settings, _ = regime_client
    settings.trade_api_key = "secret-trade"

    assert client.post(REGIME_DETECT_URL, json=make_regime_payload()).status_code == 401
    assert (
        client.post(
            REGIME_DETECT_URL,
            json=make_regime_payload(),
            headers={"X-Trade-API-Key": "wrong"},
        ).status_code
        == 403
    )
    resp = client.post(
        REGIME_DETECT_URL,
        json=make_regime_payload(),
        headers={"X-Trade-API-Key": "secret-trade"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_detect_regime_disabled_returns_403(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, kein Store-Aufruf."""
    client, _, settings, regime_store = regime_client
    settings.influxdb_enabled = False
    resp = client.post(REGIME_DETECT_URL, json=make_regime_payload())
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    regime_store.detect_and_store.assert_not_awaited()


def test_detect_regime_real_store_wiring(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Echter RegimeStore (P4-2) auf mocktem InfluxDBStore: Getter + Write-Pipeline.

    Verifiziert die Integration der lazy ``get_regime_store``-Konstruktion
    (``RegimeStore(store=...)``) und des ``detect_and_store``-Aufrufs gegen
    das echte P4-2-Modul — nur der InfluxDB-Zugriff bleibt gemockt.
    """
    if importlib.util.find_spec("trading_harness.quant.regime_store") is None:
        pytest.skip("quant/regime_store.py nicht vorhanden (P4-2 parallel)")
    client, store, _ = quant_client
    # Globalen Cache leeren, damit der Getter in diesem Test neu konstruiert.
    monkeypatch.setattr(quant_routes, "_regime_store", None)
    resp = client.post(REGIME_DETECT_URL, json=make_regime_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["stored"] is True
    assert data["regime"] in REGIME_NAMES
    # Echter RegimeStore: genau ein Write pro Detect-Lauf in das regime-Measurement.
    assert store.write_points.await_count == 1
    assert store.write_points.call_args.kwargs["measurement"] == quant_schema.REGIME_MEASUREMENT


# ---------------------------------------------------------------------------
# GET /quant/regime/{symbol} (P4-3)
# ---------------------------------------------------------------------------


def test_get_regimes_returns_regimes(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Gespeicherte Regimes → 200 mit symbol, regimes und count; Filter weitergereicht."""
    client, _, _, regime_store = regime_client
    regime_store.get_regime = AsyncMock(
        return_value=[
            {"time": "2026-08-25T12:00:00Z", "regime": "strong_bull", "confidence": 0.9},
            {"time": "2026-08-25T12:05:00Z", "regime": "range", "confidence": 0.6},
        ]
    )
    resp = client.get(
        REGIME_URL.format(symbol="BTCUSDT"),
        params={
            "timeframe": "1m",
            "start": "2026-08-25T12:00:00Z",
            "end": "2026-08-25T12:05:00Z",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["count"] == 2
    assert data["regimes"][0]["regime"] == "strong_bull"

    # start/end als UTC-ISO-Strings an den RegimeStore weitergegeben.
    regime_store.get_regime.assert_awaited_once_with(
        "BTCUSDT",
        "1m",
        "2026-08-25T12:00:00+00:00",
        "2026-08-25T12:05:00+00:00",
    )


def test_get_regimes_without_store_empty(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RegimeStore nicht verfügbar → 200 mit leerer Regime-Liste."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_regime_store", lambda: None)
    resp = client.get(REGIME_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "BTCUSDT", "regimes": [], "count": 0}


def test_get_regimes_unknown_symbol_empty(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekanntes Symbol ohne gespeicherte Regimes → 200 mit leerer Liste (kein 404)."""
    client, _, _, _ = regime_client
    resp = client.get(REGIME_URL.format(symbol="NOPE"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "NOPE", "regimes": [], "count": 0}


def test_get_regimes_requires_read_key(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Read-Key: fehlender Key → 401, richtiger Key → 200."""
    client, _, settings, _ = regime_client
    settings.read_api_key = "secret-read"
    assert client.get(REGIME_URL.format(symbol="BTCUSDT")).status_code == 401
    resp = client.get(
        REGIME_URL.format(symbol="BTCUSDT"),
        headers={"X-Read-API-Key": "secret-read"},
    )
    assert resp.status_code == 200


def test_get_regimes_invalid_timeframe_422(
    regime_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekannter Timeframe → 422, keine Store-Abfrage."""
    client, _, _, regime_store = regime_client
    resp = client.get(REGIME_URL.format(symbol="BTCUSDT"), params={"timeframe": "2m"})
    assert resp.status_code == 422
    regime_store.get_regime.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /quant/similarity/find (P5-3)
# ---------------------------------------------------------------------------


def test_find_similarity_success(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """20er-Historie (exakt ein Fenster) → 1 Match, Bestwerte = Match-Werte."""
    client, _, _ = quant_client
    resp = client.post(SIMILARITY_FIND_URL, json=make_similarity_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["query_length"] == 5
    assert len(data["matches"]) == 1
    match = data["matches"][0]
    assert match["start_index"] == 0
    assert match["end_index"] == 19
    assert match["distance"] >= 0.0
    assert -1.0 <= match["correlation"] <= 1.0
    assert data["best_distance"] == match["distance"]
    assert data["best_correlation"] == match["correlation"]


def test_find_similarity_top_k_sorted(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """40er-Historie → mehrere Kandidaten, top_k begrenzt, Distanzen aufsteigend."""
    client, _, _ = quant_client
    resp = client.post(
        SIMILARITY_FIND_URL,
        json=make_similarity_payload(history=make_similarity_history(40), top_k=3),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 1 <= len(data["matches"]) <= 3
    distances = [m["distance"] for m in data["matches"]]
    assert distances == sorted(distances)
    assert data["best_distance"] == distances[0]
    assert data["best_correlation"] == data["matches"][0]["correlation"]


def test_find_similarity_insufficient_history(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """Historie kürzer als window_size → 200, leere Matches, Bestwerte None."""
    client, _, _ = quant_client
    resp = client.post(
        SIMILARITY_FIND_URL,
        json=make_similarity_payload(history=make_similarity_history(5)),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_length"] == 5
    assert data["matches"] == []
    assert data["best_distance"] is None
    assert data["best_correlation"] is None


def test_find_similarity_invalid_data_422(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """Negativer Preis, zu kurze Query, leere History, 2m, leerer Symbol, top_k=0 → 422."""
    client, _, _ = quant_client

    bad_query = make_similarity_candles([100.0, 110.0])
    bad_query[0] = {**bad_query[0], "open": -5.0}
    assert (
        client.post(SIMILARITY_FIND_URL, json=make_similarity_payload(query=bad_query)).status_code
        == 422
    )
    # query=min_length=2 → 1 Kerze ist zu kurz
    assert (
        client.post(
            SIMILARITY_FIND_URL, json=make_similarity_payload(query=make_similarity_candles([100.0]))
        ).status_code
        == 422
    )
    assert client.post(SIMILARITY_FIND_URL, json=make_similarity_payload(history=[])).status_code == 422
    assert (
        client.post(SIMILARITY_FIND_URL, json=make_similarity_payload(timeframe="2m")).status_code
        == 422
    )
    assert (
        client.post(SIMILARITY_FIND_URL, json=make_similarity_payload(symbol="")).status_code == 422
    )
    assert (
        client.post(SIMILARITY_FIND_URL, json=make_similarity_payload(top_k=0)).status_code == 422
    )


def test_find_similarity_requires_trade_key(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """Mit gesetztem Trade-Key: fehlender Key → 401, falscher Key → 403, richtiger → 200."""
    client, _, settings = quant_client
    settings.trade_api_key = "secret-trade"

    assert client.post(SIMILARITY_FIND_URL, json=make_similarity_payload()).status_code == 401
    assert (
        client.post(
            SIMILARITY_FIND_URL,
            json=make_similarity_payload(),
            headers={"X-Trade-API-Key": "wrong"},
        ).status_code
        == 403
    )
    resp = client.post(
        SIMILARITY_FIND_URL,
        json=make_similarity_payload(),
        headers={"X-Trade-API-Key": "secret-trade"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_find_similarity_disabled_returns_403(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, keine Engine-Berechnung."""
    client, _, settings = quant_client
    settings.influxdb_enabled = False
    resp = client.post(SIMILARITY_FIND_URL, json=make_similarity_payload())
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"


# ---------------------------------------------------------------------------
# GET /quant/similarity/{symbol} (P5-3)
# ---------------------------------------------------------------------------


def test_get_similarity_info(quant_client: tuple[TestClient, MagicMock, Settings]) -> None:
    """InfluxDB verbunden → 200 mit symbol, available=true, Standard-Fenstergröße 20."""
    client, store, _ = quant_client
    store.health_check = AsyncMock(return_value=True)
    resp = client.get(SIMILARITY_URL.format(symbol="BTCUSDT"), params={"timeframe": "1m"})
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "BTCUSDT", "available": True, "window_size": 20}
    store.health_check.assert_awaited_once()


def test_get_similarity_info_unavailable(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """InfluxDB down → 200 mit available=false (kein Fehler, nur Status)."""
    client, store, _ = quant_client
    store.health_check = AsyncMock(return_value=False)
    resp = client.get(SIMILARITY_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["available"] is False
    assert data["window_size"] == 20


def test_get_similarity_info_requires_read_key(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """Mit gesetztem Read-Key: fehlender Key → 401, richtiger Key → 200."""
    client, _, settings = quant_client
    settings.read_api_key = "secret-read"
    assert client.get(SIMILARITY_URL.format(symbol="BTCUSDT")).status_code == 401
    resp = client.get(
        SIMILARITY_URL.format(symbol="BTCUSDT"),
        headers={"X-Read-API-Key": "secret-read"},
    )
    assert resp.status_code == 200


def test_get_similarity_info_invalid_timeframe_422(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """Unbekannter Timeframe → 422, keine Health-Check-Abfrage."""
    client, store, _ = quant_client
    resp = client.get(SIMILARITY_URL.format(symbol="BTCUSDT"), params={"timeframe": "2m"})
    assert resp.status_code == 422
    store.health_check.assert_not_awaited()


def test_get_similarity_info_disabled_returns_403(
    quant_client: tuple[TestClient, MagicMock, Settings]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, keine Health-Check-Abfrage."""
    client, store, settings = quant_client
    settings.influxdb_enabled = False
    resp = client.get(SIMILARITY_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    store.health_check.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /quant/outcomes/compute (P6-3)
# ---------------------------------------------------------------------------


def test_compute_outcomes_success(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Uptrend-Kerzen → 200, Outcomes pro Horizont, Store-Aufruf mit Parametern."""
    client, _, _, outcome_store = outcome_client
    resp = client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["symbol"] == "BTCUSDT"
    assert data["stored"] is True
    assert set(data["outcomes"].keys()) == {"5", "10"}

    # 30 Kerzen, max_horizon=10 → 20 Startpunkte pro Horizont, alle positiv.
    outcome_5 = data["outcomes"]["5"]
    assert outcome_5["horizon"] == 5
    assert outcome_5["sample_size"] == 20
    assert outcome_5["hit_rate"] == 1.0
    assert outcome_5["mean_return"] > 0.0
    assert outcome_5["max_gain"] == pytest.approx(0.05)  # 5/(100+0)
    assert outcome_5["max_loss"] > 0.0  # kleinster positiver Return (Uptrend)
    assert outcome_5["std_return"] > 0.0
    assert data["outcomes"]["10"]["max_gain"] == pytest.approx(0.1)  # 10/(100+0)

    outcome_store.compute_and_store.assert_awaited_once()
    args = outcome_store.compute_and_store.call_args.args
    kwargs = outcome_store.compute_and_store.call_args.kwargs
    assert args[0] == "BTCUSDT"
    assert args[1] == "1m"
    assert len(args[2]) == 30
    assert kwargs == {"pattern_length": 10, "exchange": "binance"}


def test_compute_outcomes_without_store(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store nicht verfügbar (None) → echter Engine läuft, stored=false, Stats voll."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_outcome_store", lambda: None)
    resp = client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["stored"] is False
    assert data["outcomes"]["5"]["sample_size"] == 20
    assert data["outcomes"]["5"]["hit_rate"] == 1.0
    assert data["outcomes"]["5"]["mean_return"] > 0.0
    assert data["outcomes"]["10"]["sample_size"] == 20


def test_compute_outcomes_insufficient_history(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10 Kerzen mit pattern_length=10 (< pattern_length+2) → 200, nullgefüllte Outcomes."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_outcome_store", lambda: None)
    resp = client.post(
        OUTCOMES_COMPUTE_URL,
        json=make_outcome_payload(candles=make_outcome_candles(10)),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stored"] is False
    for horizon in ("5", "10"):
        assert data["outcomes"][horizon]["sample_size"] == 0
        assert data["outcomes"][horizon]["mean_return"] == 0.0
        assert data["outcomes"][horizon]["hit_rate"] == 0.0


def test_compute_outcomes_invalid_data_422(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Negativer Preis, leere Kerzen, 2m, leerer Symbol, pattern_length=0,
    horizons=[0] → 422, kein Store-Write."""
    client, _, _, outcome_store = outcome_client

    bad_candles = make_outcome_candles()
    bad_candles[0] = {**bad_candles[0], "open": -5.0}
    assert (
        client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload(candles=bad_candles))
        .status_code
        == 422
    )
    assert client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload(candles=[])).status_code == 422
    assert (
        client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload(timeframe="2m")).status_code
        == 422
    )
    assert client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload(symbol="")).status_code == 422
    assert (
        client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload(pattern_length=0)).status_code
        == 422
    )
    assert (
        client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload(horizons=[0])).status_code
        == 422
    )
    outcome_store.compute_and_store.assert_not_awaited()


def test_compute_outcomes_requires_trade_key(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Trade-Key: fehlender Key → 401, falscher Key → 403, richtiger → 200."""
    client, _, settings, _ = outcome_client
    settings.trade_api_key = "secret-trade"

    assert client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload()).status_code == 401
    assert (
        client.post(
            OUTCOMES_COMPUTE_URL,
            json=make_outcome_payload(),
            headers={"X-Trade-API-Key": "wrong"},
        ).status_code
        == 403
    )
    resp = client.post(
        OUTCOMES_COMPUTE_URL,
        json=make_outcome_payload(),
        headers={"X-Trade-API-Key": "secret-trade"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_compute_outcomes_disabled_returns_403(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, kein Compute/Write."""
    client, _, settings, outcome_store = outcome_client
    settings.influxdb_enabled = False
    resp = client.post(OUTCOMES_COMPUTE_URL, json=make_outcome_payload())
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    outcome_store.compute_and_store.assert_not_awaited()


def test_compute_outcomes_real_store_wiring(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Echter ForwardOutcomeStore (P6-2) auf mocktem InfluxDBStore: Getter + Write-Pipeline.

    Verifiziert die Integration der lazy ``get_outcome_store``-Konstruktion
    (``ForwardOutcomeStore(store=...)``) und des ``compute_and_store``-Aufrufs
    gegen das echte P6-2-Modul — nur der InfluxDB-Zugriff bleibt gemockt.
    """
    if importlib.util.find_spec("trading_harness.quant.forward_outcomes_store") is None:
        pytest.skip("quant/forward_outcomes_store.py nicht vorhanden (P6-2 parallel)")
    client, store, _ = quant_client
    # Globalen Cache leeren, damit der Getter in diesem Test neu konstruiert.
    monkeypatch.setattr(quant_routes, "_outcome_store", None)
    # 60 Kerzen: der Store nutzt Default-Horizonte (max 50) → alle vier
    # Horizonte liefern sample_size>0 → ein write_points-Aufruf pro Horizont.
    resp = client.post(
        OUTCOMES_COMPUTE_URL,
        json=make_outcome_payload(candles=make_outcome_candles(60)),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["stored"] is True
    assert data["outcomes"]["5"]["sample_size"] == 50  # 60 Kerzen - max_horizon 10
    assert store.write_points.await_count == 4
    assert store.write_points.call_args.kwargs["measurement"] == "forward_outcomes"


# ---------------------------------------------------------------------------
# GET /quant/outcomes/{symbol} (P6-3)
# ---------------------------------------------------------------------------


def test_get_outcomes_returns_outcomes(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Gespeicherte Outcomes → 200 mit symbol, outcomes und count; Filter weitergereicht."""
    client, _, _, outcome_store = outcome_client
    outcome_store.get_outcomes = AsyncMock(
        return_value=[
            {"time": "2026-08-25T12:00:00Z", "horizon": 5, "mean_return": 0.01, "hit_rate": 0.6},
            {"time": "2026-08-25T12:05:00Z", "horizon": 10, "mean_return": 0.02, "hit_rate": 0.7},
            {"time": "2026-08-25T12:10:00Z", "horizon": 20, "mean_return": 0.03, "hit_rate": 0.8},
        ]
    )
    resp = client.get(
        OUTCOMES_URL.format(symbol="BTCUSDT"),
        params={
            "timeframe": "1m",
            "start": "2026-08-25T12:00:00Z",
            "end": "2026-08-25T12:05:00Z",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["count"] == 3
    assert data["outcomes"][0]["horizon"] == 5

    # start/end als UTC-ISO-Strings an den OutcomeStore weitergegeben.
    outcome_store.get_outcomes.assert_awaited_once_with(
        "BTCUSDT",
        "1m",
        "2026-08-25T12:00:00+00:00",
        "2026-08-25T12:05:00+00:00",
    )


def test_get_outcomes_without_store_empty(
    quant_client: tuple[TestClient, MagicMock, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store nicht verfügbar (None) → 200 mit leerer Outcome-Liste."""
    client, _, _ = quant_client
    monkeypatch.setattr(quant_routes, "get_outcome_store", lambda: None)
    resp = client.get(OUTCOMES_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "BTCUSDT", "outcomes": [], "count": 0}


def test_get_outcomes_unknown_symbol_empty(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekanntes Symbol ohne gespeicherte Outcomes → 200 mit leerer Liste (kein 404)."""
    client, _, _, _ = outcome_client
    resp = client.get(OUTCOMES_URL.format(symbol="NOPE"))
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "NOPE", "outcomes": [], "count": 0}


def test_get_outcomes_requires_read_key(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Mit gesetztem Read-Key: fehlender Key → 401, richtiger Key → 200."""
    client, _, settings, _ = outcome_client
    settings.read_api_key = "secret-read"
    assert client.get(OUTCOMES_URL.format(symbol="BTCUSDT")).status_code == 401
    resp = client.get(
        OUTCOMES_URL.format(symbol="BTCUSDT"),
        headers={"X-Read-API-Key": "secret-read"},
    )
    assert resp.status_code == 200


def test_get_outcomes_invalid_timeframe_422(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """Unbekannter Timeframe → 422, keine Store-Abfrage."""
    client, _, _, outcome_store = outcome_client
    resp = client.get(OUTCOMES_URL.format(symbol="BTCUSDT"), params={"timeframe": "2m"})
    assert resp.status_code == 422
    outcome_store.get_outcomes.assert_not_awaited()


def test_get_outcomes_disabled_returns_403(
    outcome_client: tuple[TestClient, MagicMock, Settings, MagicMock]
) -> None:
    """influxdb_enabled=False → 403 QUANT_DISABLED, keine Store-Abfrage."""
    client, _, settings, outcome_store = outcome_client
    settings.influxdb_enabled = False
    resp = client.get(OUTCOMES_URL.format(symbol="BTCUSDT"))
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "QUANT_DISABLED"
    outcome_store.get_outcomes.assert_not_awaited()
