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

# Naiv (bewusst ohne tzinfo): die Route interpretiert naive Zeiten als UTC.
CANDLE_TIME = datetime.fromisoformat("2026-08-25T12:00:00")
EXPECTED_NS = int(CANDLE_TIME.replace(tzinfo=UTC).timestamp() * 1_000_000_000)
INGEST_URL = "/quant/ingest/ohlcv"
STATUS_URL = "/quant/status"
SCHEMA_URL = "/quant/schema"
FEATURES_COMPUTE_URL = "/quant/features/compute"
FEATURES_URL = "/quant/features/{symbol}"


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
