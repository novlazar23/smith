"""Unit-Tests für die OHLCV-Ingestion (Quant-Plattform, P1-6).

Alle Tests nutzen einen gemockten ``InfluxDBStore`` (``spec=InfluxDBStore`` —
die asynchronen Store-Methoden werden automatisch zu ``AsyncMock``s); es wird
nie eine echte InfluxDB-Verbindung aufgebaut. Geprüft wird, dass die Ingestion
in das einzige ``ohlcv``-Measurement schreibt (Timeframe als Tag) und die
Flux-Queries die korrekten Filter enthalten.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_harness.quant.influxdb_client import InfluxDBStore
from trading_harness.quant.ohlcv_ingestion import IngestResult, OHLCVIngestion

pytestmark = pytest.mark.asyncio

BUCKET = "market_data"
NS = 1_000_000_000
T0 = "2026-01-01T00:00:00Z"


def make_store() -> MagicMock:
    """Gemockter Store: async API als AsyncMock, keine echte Verbindung."""
    return MagicMock(spec=InfluxDBStore)


def make_ingestion(store: MagicMock, **kwargs: Any) -> OHLCVIngestion:
    return OHLCVIngestion(store, bucket=BUCKET, **kwargs)


def candle(
    time: str,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 99.0,
    close: float = 103.0,
    volume: float = 5000.0,
) -> dict[str, Any]:
    return {"time": time, "open": open_, "high": high, "low": low, "close": close, "volume": volume}


def candle_series(step: timedelta, count: int) -> list[dict[str, Any]]:
    """Erzeugt ``count`` aufeinanderfolgende Kerzen ab 2026-01-01T00:00:00Z (UTC)."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[dict[str, Any]] = []
    for i in range(count):
        moment = base + step * i
        candles.append(
            candle(
                moment.isoformat().replace("+00:00", "Z"),
                open_=100.0 + i,
                high=105.0 + i,
                low=99.0 + i,
                close=103.0 + i,
                volume=1000.0 * (i + 1),
            )
        )
    return candles


# ---------------------------------------------------------------------------
# ingest_candles
# ---------------------------------------------------------------------------


async def test_ingest_candles_writes_ohlcv_with_correct_tags() -> None:
    """Kerzen landen im einzigen ``ohlcv``-Measurement mit symbol/exchange/timeframe-Tags."""
    store = make_store()
    ing = make_ingestion(store)
    candles = [candle(T0), candle("2026-01-01T00:01:00Z")]

    result = await ing.ingest_candles(["BTCUSDT"], "1m", candles)

    assert result == IngestResult(written=2, skipped=0, measurement="ohlcv")
    store.write_batch.assert_awaited_once()
    kwargs = store.write_batch.await_args.kwargs
    assert kwargs["measurement"] == "ohlcv"
    assert kwargs["tags"] == {"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1m"}
    assert kwargs["points"] == [
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 5000.0},
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 5000.0},
    ]
    ts0 = int(datetime.fromisoformat(T0).timestamp())
    assert kwargs["timestamps"] == [ts0 * NS, (ts0 + 60) * NS]


async def test_ingest_candles_writes_once_per_symbol() -> None:
    """Derselben Kerzenreihe wird pro Symbol geschrieben: written = Symbole × Kerzen."""
    store = make_store()
    ing = make_ingestion(store)

    result = await ing.ingest_candles(["BTCUSDT", "ETHUSDT"], "1m", candle_series(timedelta(minutes=1), 2))

    assert result.written == 4
    assert result.skipped == 0
    assert store.write_batch.await_count == 2
    assert store.write_batch.await_args_list[0].kwargs["tags"]["symbol"] == "BTCUSDT"
    assert store.write_batch.await_args_list[1].kwargs["tags"]["symbol"] == "ETHUSDT"


async def test_ingest_candles_empty_returns_zero() -> None:
    """Leere Kerzenliste → written=0, kein Schreibaufruf."""
    store = make_store()
    ing = make_ingestion(store)

    result = await ing.ingest_candles(["BTCUSDT"], "1m", [])

    assert result == IngestResult(written=0, skipped=0, measurement="ohlcv")
    store.write_batch.assert_not_awaited()


async def test_ingest_candles_exchange_defaults_to_constructor() -> None:
    """Ohne explizite Exchange gilt der Konstruktor-Default."""
    store = make_store()
    ing = make_ingestion(store, default_exchange="kraken")

    await ing.ingest_candles(["BTCUSDT"], "1m", [candle(T0)])

    assert store.write_batch.await_args.kwargs["tags"]["exchange"] == "kraken"


async def test_ingest_candles_explicit_exchange_overrides_default() -> None:
    """Expliziter Exchange-Parameter übersteuert den Konstruktor-Default."""
    store = make_store()
    ing = make_ingestion(store, default_exchange="kraken")

    await ing.ingest_candles(["BTCUSDT"], "1m", [candle(T0)], exchange="bybit")

    assert store.write_batch.await_args.kwargs["tags"]["exchange"] == "bybit"


async def test_ingest_candles_invalid_timeframe_raises_valueerror() -> None:
    """Unbekannter Timeframe → ValueError, nichts wird geschrieben."""
    store = make_store()
    ing = make_ingestion(store)

    with pytest.raises(ValueError, match="unsupported timeframe '2m'"):
        await ing.ingest_candles(["BTCUSDT"], "2m", [candle(T0)])
    store.write_batch.assert_not_awaited()


async def test_ingest_candles_skips_structurally_invalid_candles() -> None:
    """Kerze ohne ``volume`` wird übersprungen, gültige Kerze wird geschrieben."""
    store = make_store()
    ing = make_ingestion(store)
    invalid = {"time": T0, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}

    result = await ing.ingest_candles(["BTCUSDT"], "1m", [candle(T0), invalid])

    assert result.written == 1
    assert result.skipped == 1
    assert len(store.write_batch.await_args.kwargs["points"]) == 1


# ---------------------------------------------------------------------------
# downsample_candles (rein)
# ---------------------------------------------------------------------------


async def test_downsample_1m_to_5m() -> None:
    """10×1m → 2×5m mit open=erste, high=max, low=min, close=letzte, volume=Summe."""
    ing = make_ingestion(make_store())
    out = ing.downsample_candles(candle_series(timedelta(minutes=1), 10), "1m", "5m")

    assert len(out) == 2
    first = out[0]
    assert first["time"] == "2026-01-01T00:00:00Z"
    assert first["open"] == 100.0
    assert first["high"] == 109.0
    assert first["low"] == 99.0
    assert first["close"] == 107.0  # close der i=4-Kerze: 103+4
    assert first["volume"] == 15000.0  # 1000+2000+3000+4000+5000
    second = out[1]
    assert second["time"] == "2026-01-01T00:05:00Z"
    assert second["open"] == 105.0
    assert second["high"] == 114.0
    assert second["low"] == 104.0
    assert second["close"] == 112.0  # close der i=9-Kerze: 103+9
    assert second["volume"] == 40000.0  # 6000+...+10000


async def test_downsample_1m_to_1h() -> None:
    """60×1m → 1×1h, Kerzenzeit = Stundenbeginn."""
    ing = make_ingestion(make_store())
    out = ing.downsample_candles(candle_series(timedelta(minutes=1), 60), "1m", "1h")

    assert len(out) == 1
    hour = out[0]
    assert hour["time"] == "2026-01-01T00:00:00Z"
    assert hour["open"] == 100.0
    assert hour["high"] == 164.0
    assert hour["low"] == 99.0
    assert hour["close"] == 162.0  # close der i=59-Kerze: 103+59
    assert hour["volume"] == 1830000.0  # 1000 × (1+...+60)


async def test_downsample_5m_to_1h() -> None:
    """12×5m → 1×1h."""
    ing = make_ingestion(make_store())
    out = ing.downsample_candles(candle_series(timedelta(minutes=5), 12), "5m", "1h")

    assert len(out) == 1
    hour = out[0]
    assert hour["time"] == "2026-01-01T00:00:00Z"
    assert hour["open"] == 100.0
    assert hour["high"] == 116.0  # 105+11
    assert hour["low"] == 99.0
    assert hour["close"] == 114.0  # 103+11
    assert hour["volume"] == 78000.0  # 1000 × (1+...+12)


async def test_downsample_drops_incomplete_trailing_group() -> None:
    """7×1m → nur 1 vollständige 5m-Kerze; die offenen 2 Minuten werden verworfen."""
    ing = make_ingestion(make_store())
    out = ing.downsample_candles(candle_series(timedelta(minutes=1), 7), "1m", "5m")

    assert len(out) == 1
    assert out[0]["time"] == "2026-01-01T00:00:00Z"
    assert out[0]["close"] == 107.0  # close der i=4-Kerze: 103+4


async def test_downsample_target_smaller_than_source_raises_valueerror() -> None:
    """Upsampling (Ziel < Quelle) ist keine Downsample-Mapping → ValueError."""
    ing = make_ingestion(make_store())
    with pytest.raises(ValueError, match="must be larger"):
        ing.downsample_candles(candle_series(timedelta(minutes=1), 10), "1h", "15m")


# ---------------------------------------------------------------------------
# ingest_and_downsample
# ---------------------------------------------------------------------------


async def test_ingest_and_downsample_writes_both_timeframes() -> None:
    """Quell- (1m) UND Zielkerzen (5m) werden im selben Measurement geschrieben."""
    store = make_store()
    ing = make_ingestion(store)

    result = await ing.ingest_and_downsample(["BTCUSDT"], "1m", "5m", candle_series(timedelta(minutes=1), 10))

    assert result == IngestResult(written=12, skipped=0, measurement="ohlcv")
    assert store.write_batch.await_count == 2
    source = store.write_batch.await_args_list[0].kwargs
    assert source["tags"]["timeframe"] == "1m"
    assert len(source["points"]) == 10
    target = store.write_batch.await_args_list[1].kwargs
    assert target["tags"]["timeframe"] == "5m"
    assert len(target["points"]) == 2
    assert target["points"][0]["close"] == 107.0  # close der ersten 5m-Kerze


async def test_ingest_and_downsample_invalid_target_raises() -> None:
    """Ungültige Ziel-Timeframe → ValueError, nichts wird geschrieben."""
    store = make_store()
    ing = make_ingestion(store)

    with pytest.raises(ValueError, match="must be larger"):
        await ing.ingest_and_downsample(["BTCUSDT"], "1h", "15m", [candle(T0)])
    store.write_batch.assert_not_awaited()


# ---------------------------------------------------------------------------
# Lesezugriffe (Flux-Queries)
# ---------------------------------------------------------------------------


def _record(time: datetime, close: float = 112.0) -> dict[str, Any]:
    return {
        "_time": time,
        "open": 110.0,
        "high": 115.0,
        "low": 108.0,
        "close": close,
        "volume": 42.0,
    }


async def test_get_latest_candle_queries_correctly() -> None:
    """Flux-Query filtert auf ohlcv + Tags, pivotet Fields, sortiert absteigend, limit 1."""
    store = make_store()
    store.query.return_value = [_record(datetime(2026, 1, 1, 0, 10, tzinfo=UTC))]
    ing = make_ingestion(store)

    result = await ing.get_latest_candle("BTCUSDT", "1m")

    assert result == {
        "time": "2026-01-01T00:10:00Z",
        "open": 110.0,
        "high": 115.0,
        "low": 108.0,
        "close": 112.0,
        "volume": 42.0,
    }
    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert f'from(bucket: "{BUCKET}")' in flux
    assert 'r._measurement == "ohlcv"' in flux
    assert 'r["symbol"] == "BTCUSDT"' in flux
    assert 'r["exchange"] == "binance"' in flux
    assert 'r["timeframe"] == "1m"' in flux
    assert "pivot(" in flux
    assert "desc: true" in flux
    assert "limit(n: 1)" in flux


async def test_get_latest_candle_empty_result_returns_none() -> None:
    """Keine Daten → None statt Exception."""
    store = make_store()
    store.query.return_value = []
    ing = make_ingestion(store)

    assert await ing.get_latest_candle("BTCUSDT", "1m") is None


async def test_get_latest_candle_invalid_timeframe_raises_valueerror() -> None:
    store = make_store()
    ing = make_ingestion(store)

    with pytest.raises(ValueError, match="unsupported timeframe"):
        await ing.get_latest_candle("BTCUSDT", "2m")
    store.query.assert_not_awaited()


async def test_get_candle_range_queries_correctly() -> None:
    """Flux-Query mit [start, stop], aufsteigend sortiert, ohne limit."""
    store = make_store()
    store.query.return_value = [
        _record(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), close=101.0),
        _record(datetime(2026, 1, 1, 0, 1, tzinfo=UTC), close=102.0),
    ]
    ing = make_ingestion(store)

    result = await ing.get_candle_range("BTCUSDT", "1m", "2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z")

    assert [c["time"] for c in result] == ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"]
    assert [c["close"] for c in result] == [101.0, 102.0]
    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'range(start: "2026-01-01T00:00:00Z", stop: "2026-01-01T00:30:00Z")' in flux
    assert 'r["timeframe"] == "1m"' in flux
    assert "desc: true" not in flux
    assert "limit(" not in flux


async def test_get_candle_range_invalid_bounds_raise_valueerror() -> None:
    store = make_store()
    ing = make_ingestion(store)

    with pytest.raises(ValueError, match="invalid start timestamp"):
        await ing.get_candle_range("BTCUSDT", "1m", "not-a-time", "2026-01-01T00:30:00Z")
    store.query.assert_not_awaited()
