"""Unit-Tests für den InfluxDB-Client-Wrapper (Quant-Plattform, P1-4).

Alle Tests nutzen einen gemockten InfluxDB-Client — es werden keine echten
InfluxDB-Verbindungen aufgebaut. ``influxdb_client.InfluxDBClient`` wird
per ``unittest.mock.patch`` umgebunden, damit der lazy Import im Store
die Mock-Klasse liefert.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading_harness.quant.influxdb_client import InfluxDBStore

pytestmark = pytest.mark.asyncio

URL = "http://localhost:8086"
TOKEN = "test-token"
ORG = "smith"
BUCKET = "market_data"
TS_NS = 1_700_000_000_000_000_000  # Nanosekunden seit Epoch


def make_store() -> InfluxDBStore:
    """Erzeugt einen Store mit Config-Defaults (kein Connection-Versuch)."""
    return InfluxDBStore(url=URL, token=TOKEN, org=ORG, bucket=BUCKET)


def mock_client(ping: bool | Exception) -> MagicMock:
    """Erzeugt einen Mock-Client mit konfigurierbarem Ping-Verhalten."""
    client = MagicMock()
    if isinstance(ping, Exception):
        client.ping.side_effect = ping
    else:
        client.ping.return_value = ping
    return client


async def test_init_defaults() -> None:
    """Store mit Defaults: lazy init — nicht verbunden, leerer Buffer."""
    store = make_store()
    assert store.is_available is False
    assert store.buffer_size() == 0
    assert store._client is None
    assert store._url == URL
    assert store._bucket == BUCKET


async def test_health_check_success() -> None:
    """Ping erfolgreich → health_check True, APIs initialisiert."""
    client = mock_client(ping=True)
    with patch("influxdb_client.InfluxDBClient", return_value=client) as client_cls:
        store = make_store()
        assert await store.health_check() is True
    assert store.is_available is True
    client_cls.assert_called_once_with(url=URL, token=TOKEN, org=ORG)
    assert store._write_api is client.write_api.return_value
    assert store._query_api is client.query_api.return_value


async def test_health_check_failure() -> None:
    """Client wirft Exception → health_check False, Client wird geschlossen."""
    client = mock_client(ping=Exception("connection refused"))
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = make_store()
        assert await store.health_check() is False
    assert store.is_available is False
    client.close.assert_called_once()


async def test_write_points_success() -> None:
    """Punkt wird als Line-Protocol-Point in den Bucket geschrieben."""
    client = mock_client(ping=True)
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = make_store()
        await store.write_points("ohlcv", {"symbol": "BTC"}, {"close": 42000.5}, TS_NS)
    write = client.write_api.return_value.write
    write.assert_called_once()
    kwargs = write.call_args.kwargs
    assert kwargs["bucket"] == BUCKET
    line = str(kwargs["record"])
    assert "ohlcv,symbol=BTC" in line
    assert "close=42000.5" in line
    assert line.endswith(str(TS_NS))
    assert store.buffer_size() == 0


async def test_write_points_offline_buffer() -> None:
    """InfluxDB down (Ping fehlschlägt) → Punkt landet im In-Memory-Buffer."""
    client = mock_client(ping=False)
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = make_store()
        await store.write_points("ohlcv", {"symbol": "BTC"}, {"close": 42000.5}, TS_NS)
    assert store.is_available is False
    assert store.buffer_size() == 1
    client.write_api.return_value.write.assert_not_called()
    buffered = store._buffer[0]
    assert buffered["measurement"] == "ohlcv"
    assert buffered["tags"] == {"symbol": "BTC"}
    assert buffered["fields"] == {"close": 42000.5}
    assert buffered["timestamp"] == TS_NS


async def test_write_batch_efficient() -> None:
    """Batch-Write nutzt einen einzigen write_api.write-Aufruf für alle Punkte."""
    client = mock_client(ping=True)
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = make_store()
        await store.write_batch(
            "ohlcv",
            {"symbol": "BTC"},
            [{"close": 1.0}, {"close": 2.0}, {"close": 3.0}],
            [TS_NS, TS_NS + 1, TS_NS + 2],
        )
    write = client.write_api.return_value.write
    write.assert_called_once()  # ein Aufruf statt drei
    kwargs = write.call_args.kwargs
    assert kwargs["bucket"] == BUCKET
    assert len(kwargs["record"]) == 3
    assert store.buffer_size() == 0


async def test_query_returns_data() -> None:
    """Query gibt die Table-Records als Liste von Dicts zurück."""
    client = mock_client(ping=True)
    record1 = MagicMock()
    record1.values = {"_measurement": "ohlcv", "_field": "close", "_value": 42000.5}
    record2 = MagicMock()
    record2.values = {"_measurement": "ohlcv", "_field": "volume", "_value": 12.0}
    table = MagicMock()
    table.records = [record1, record2]
    client.query_api.return_value.query.return_value = [table]
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = make_store()
        result = await store.query('from(bucket:"market_data")')
    assert result == [
        {"_measurement": "ohlcv", "_field": "close", "_value": 42000.5},
        {"_measurement": "ohlcv", "_field": "volume", "_value": 12.0},
    ]


async def test_query_offline_returns_empty() -> None:
    """InfluxDB down → Query liefert leere Liste statt Exception."""
    client = mock_client(ping=False)
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = make_store()
        assert await store.query('from(bucket:"market_data")') == []
    client.query_api.return_value.query.assert_not_called()


async def test_is_available_property() -> None:
    """is_available folgt dem Verbindungszyklus: False → True / False."""
    store = make_store()
    assert store.is_available is False  # lazy: noch nie verbunden

    client = mock_client(ping=True)
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        assert await store.health_check() is True
    assert store.is_available is True

    offline_store = make_store()
    failing_client = mock_client(ping=False)
    with patch("influxdb_client.InfluxDBClient", return_value=failing_client):
        assert await offline_store.health_check() is False
    assert offline_store.is_available is False


async def test_buffer_size() -> None:
    """buffer_size zählt alle offline gepufferten Punkte (Punkt + Batch)."""
    client = mock_client(ping=False)
    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = make_store()
        assert store.buffer_size() == 0
        await store.write_points("ohlcv", {"symbol": "BTC"}, {"close": 1.0}, TS_NS)
        assert store.buffer_size() == 1
        await store.write_batch(
            "ohlcv",
            {"symbol": "ETH"},
            [{"close": 2.0}, {"close": 3.0}],
            [TS_NS + 1, TS_NS + 2],
        )
        assert store.buffer_size() == 3
