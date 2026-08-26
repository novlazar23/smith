"""Hardening-Integration (Quant-Plattform, Phase 11, P11-4).

End-to-End-Ketten aus ``Validator`` und ``ErrorRecovery``:
Validierung -> Fehler-Erfassung -> Recovery per Default, Retry, Fallback
und Batch-Verarbeitung mit Skip. Kein Netzwerk, keine echte InfluxDB —
der InfluxDB-Ausfall wird mit einem gemockten Client simuliert
(Muster aus ``test_quant_influxdb_client.py``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trading_harness.quant.error_recovery import ErrorRecovery, RecoveryResult, RetryConfig
from trading_harness.quant.influxdb_client import InfluxDBStore
from trading_harness.quant.validation import Validator

URL = "http://localhost:8086"
TOKEN = "test-token"
ORG = "smith"
BUCKET = "market_data"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
TS_NS = 1_700_000_000_000_000_000  # Nanosekunden seit Epoch

CANDLE_SCHEMA: dict[str, dict[str, Any]] = {
    "time": {"type": str},
    "open": {"type": float},
    "high": {"type": float},
    "low": {"type": float},
    "close": {"type": float},
    "volume": {"type": float, "default": 0.0},
}


def make_candle(
    index: int, close: float = 100.0, volume: float = 10.0, high: float | None = None
) -> dict[str, Any]:
    """Gültige Kerze (High/Low konsistent mit Open/Close, positives Volume)."""
    return {
        "time": f"2026-01-01T00:{index:02d}:00Z",
        "open": close,
        "high": high if high is not None else close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
    }


def flat_default(candle: dict[str, Any]) -> dict[str, Any]:
    """Recovery-Default: flache Kerze zum Close-Preis (valid, Volume 0)."""
    return {
        **candle,
        "open": candle["close"],
        "high": candle["close"],
        "low": candle["close"],
        "volume": 0.0,
    }


def test_validate_and_recover_pipeline() -> None:
    """Ungültige Kerze -> Validierung schlägt mit konkreten Fehlern fehl ->
    Recovery (with_fallback) liefert die Default-Kerze, die wieder valid ist."""
    validator = Validator()
    recovery = ErrorRecovery()

    bad = make_candle(0)
    bad["high"] = 90.0  # High < max(open, close)
    bad["volume"] = -5.0  # negatives Volume

    result = validator.validate_candle(bad)
    assert result.valid is False
    assert any("High" in e for e in result.errors)
    assert any("Negative volume" in e for e in result.errors)

    default_candle = make_candle(0)

    def pipeline() -> dict[str, Any]:
        res = validator.validate_candle(bad)
        if not res.valid:
            raise ValueError("invalid candle: " + "; ".join(res.errors))
        return bad

    recovered = recovery.with_fallback(pipeline, default_candle, name="candle_pipeline")

    assert recovered == default_candle
    assert recovered is not bad
    # Die erholte Kerze besteht die Validierung wieder.
    assert validator.validate_candle(recovered).valid is True


def test_retry_with_validation() -> None:
    """Erster Load schlägt fehl (simulierter InfluxDB-Timeout) -> Retry mit
    korrigierten, validierten Daten -> Erfolg im zweiten Versuch; dauerhaft
    ungültige Operationen verbrauchen alle Retries und melden Failure."""
    validator = Validator()
    recovery = ErrorRecovery(RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.01))
    corrected = make_candle(1, close=105.0)
    assert validator.validate_candle(corrected).valid is True

    attempts = 0

    def load_candle() -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("InfluxDB read timeout")
        candle = dict(corrected)
        res = validator.validate_candle(candle)
        if not res.valid:
            raise ValueError("; ".join(res.errors))
        return candle

    result = recovery.with_retry(load_candle, operation_name="load_candle")

    assert isinstance(result, RecoveryResult)
    assert result.success is True
    assert result.retries == 1
    assert attempts == 2
    assert result.value == corrected
    assert validator.validate_candle(result.value).valid is True

    # Dauerhaft ungültige Kerze: alle Retries aufgebraucht -> success=False.
    bad = make_candle(2, volume=-1.0)

    def load_bad() -> dict[str, Any]:
        res = validator.validate_candle(bad)
        if not res.valid:
            raise ValueError("; ".join(res.errors))
        return bad

    failed = recovery.with_retry(load_bad, operation_name="load_candle")
    assert failed.success is False
    assert failed.retries == 3
    assert failed.error is not None
    assert "Negative volume" in failed.error


@pytest.mark.asyncio
async def test_fallback_on_influx_failure() -> None:
    """InfluxDB-Ausfall -> der echte InfluxDBStore degradiert (Query -> leer,
    Write -> Buffer); die Recovery-Ebene verwendet die registrierten
    Fallback-Werte und liefert bei Erfolg die echten Daten."""
    client = MagicMock(name="InfluxDBClient")
    # Erster Connect OK; Reconnect nach mark_unavailable schlägt fehl.
    client.ping.side_effect = [True, False]
    query_api = MagicMock()
    query_api.query.side_effect = ConnectionError("InfluxDB connection refused")
    client.query_api.return_value = query_api

    with patch("influxdb_client.InfluxDBClient", return_value=client):
        store = InfluxDBStore(url=URL, token=TOKEN, org=ORG, bucket=BUCKET)
        assert await store.health_check() is True  # initial verbunden
        rows = await store.query('from(bucket: "market_data")')
        assert rows == []  # Query-Fehler -> leeres Ergebnis, kein Re-Raise
        assert store.is_available is False
        await store.write_points("ohlcv", {"symbol": SYMBOL}, {"close": 100.0}, TS_NS)
        assert store.buffer_size() == 1  # Write fällt auf In-Memory-Buffer zurück

    recovery = ErrorRecovery()
    fallback_candles = [make_candle(i) for i in range(3)]
    key = f"candles:{SYMBOL}:{TIMEFRAME}"
    recovery.register_fallback(key, fallback_candles)
    default = recovery.get_fallback(key)

    def failing_read() -> list[dict[str, Any]]:
        raise ConnectionError("InfluxDB connection refused")

    got = recovery.with_fallback(failing_read, default, name="influx_read")
    assert got == fallback_candles  # Ausfall -> registrierte Fallback-Werte

    real_candles = [make_candle(9, close=111.0)]
    got_ok = recovery.with_fallback(lambda: list(real_candles), default, name="influx_read")
    assert got_ok == real_candles  # Erfolg -> echte Daten, nicht der Fallback


def test_batch_validation_with_recovery() -> None:
    """Batch mit zwei ungültigen Kerzen -> Batch insgesamt ungültig, aber die
    gültigen Kerzen werden verarbeitet und die ungültigen per flachem Default
    erholt; abschließend sichert validate_and_fix die Schemavollständigkeit."""
    validator = Validator()
    recovery = ErrorRecovery()

    candles = [
        make_candle(0, close=100.0),
        make_candle(1, close=105.0, volume=-1.0),  # ungültig: negatives Volume
        make_candle(2, close=110.0),
        make_candle(3, close=95.0, high=90.0),  # ungültig: High < max(open, close)
    ]

    batch = validator.validate_ohlcv_batch(candles)
    assert batch.valid is False
    assert any(e.startswith("Candle 1:") for e in batch.errors)
    assert any(e.startswith("Candle 3:") for e in batch.errors)
    assert not any(e.startswith(("Candle 0:", "Candle 2:")) for e in batch.errors)

    processed: list[dict[str, Any]] = []
    skipped: list[int] = []
    for index, candle in enumerate(candles):
        if validator.validate_candle(candle).valid:
            processed.append(candle)
        else:
            skipped.append(index)

    assert skipped == [1, 3]
    assert [c["time"] for c in processed] == [candles[0]["time"], candles[2]["time"]]
    assert all(validator.validate_candle(c).valid for c in processed)

    # Recovery: ungültige Kerzen durch flache Defaults ersetzen, keine Datenlücke.
    recovered_series = [
        flat_default(c) if index in skipped else c for index, c in enumerate(candles)
    ]
    assert len(recovered_series) == len(candles)
    final = validator.validate_candles(recovered_series)
    assert final.valid is True
    assert final.errors == []

    # Abschließende Hardening-Stufe: Schemavollständigkeit pro Kerze.
    final_series = [recovery.validate_and_fix(c, CANDLE_SCHEMA) for c in recovered_series]
    assert all(validator.validate_candle(c).valid for c in final_series)
