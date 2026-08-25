"""Quant-Plattform API-Endpunkte (Phase 1, P1-8).

Stellt die InfluxDB-basierte OHLCV-Ingestion unter ``/quant/*`` als eigenen
``APIRouter`` bereit. Die Einhängung in die Haupt-App erfolgt erst in Phase 9
(siehe ``docs/quant-platform-phase01-plan.md``, P1-8) — deshalb
trägt der Router selbst seinen Prefix und ``main.py`` bleibt unangetastet.

Auth-Trennung nach dem bestehenden Read/Trade-Muster (R5.21):

- ``POST /quant/ingest/ohlcv`` → Trade-Key
- ``GET  /quant/status``       → Read-Key
- ``GET  /quant/schema``       → Read-Key

Testbarkeit: Abhängigkeiten (Settings, ``InfluxDBStore``) werden über die
modul-scope-Getter ``get_settings`` (re-export aus ``config``) und
``get_influx_store`` aufgelöst — Unit-Tests binden beide per Monkeypatch auf
Mocks, es wird nie eine echte InfluxDB-Verbindung aufgebaut.

Die Candle-Validierung (OHLC-Konsistenz) und die Point-Konvertierung sind die
eigentliche Endpunkt-Logik; die Ticker-basierte Ingestion aus dem
Shadow-Loop (``quant/ohlcv_ingestion.py``, P1-6) ist bewusst keine
Abhängigkeit dieses Moduls.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from trading_harness.api.security import require_read_key, require_trade_key
from trading_harness.config import get_settings
from trading_harness.quant import schema as quant_schema
from trading_harness.quant.influxdb_client import InfluxDBStore

# Exchange-Tag für manuell per API ingestete Daten (keine konkrete Exchange).
MANUAL_INGEST_EXCHANGE: str = "api"

# Alle im Quant-Schema definierten Measurements (Dokumentation für /quant/schema).
_ALL_MEASUREMENTS: tuple[str, ...] = (
    quant_schema.OHLCV_MEASUREMENT,
    quant_schema.TRADES_MEASUREMENT,
    quant_schema.ORDERBOOK_MEASUREMENT,
    quant_schema.DERIVATIVES_MEASUREMENT,
    quant_schema.ANOMALY_MEASUREMENT,
    quant_schema.FEATURE_MEASUREMENT,
    quant_schema.REGIME_MEASUREMENT,
)

# OHLCV-Fields des API-Contracts: Schema-Fields + trade_count (API-Erweiterung).
QUANT_OHLCV_FIELDS: tuple[str, ...] = quant_schema.OHLCV_FIELDS + ("trade_count",)

router = APIRouter(prefix="/quant", tags=["quant"])


# ---------------------------------------------------------------------------
# Abhängigkeits-Auflösung (Tests binden diese Getter auf Mocks um)
# ---------------------------------------------------------------------------

_influx_store: InfluxDBStore | None = None


def get_influx_store() -> InfluxDBStore:
    """Liefert den geteilten ``InfluxDBStore`` (lazy, ein Instanz pro Prozess)."""
    global _influx_store
    if _influx_store is None:
        settings = get_settings()
        _influx_store = InfluxDBStore(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
            bucket=settings.influxdb_bucket,
        )
    return _influx_store


# ---------------------------------------------------------------------------
# Request/Response-Modelle
# ---------------------------------------------------------------------------


class OHLCVCandle(BaseModel):
    """Einzelne OHLCV-Kerze (UTC-Zeitstempel, ``trade_count`` optional)."""

    time: datetime
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: float = Field(ge=0)
    trade_count: int = Field(default=0, ge=0)


class OHLCVIngestRequest(BaseModel):
    """Ingest-Anfrage: dieselbe Candle-Reihe für alle gelisteten Symbole."""

    symbols: list[str] = Field(min_length=1)
    timeframe: str
    candles: list[OHLCVCandle] = Field(min_length=1)

    @field_validator("timeframe")
    @classmethod
    def _check_timeframe(cls, value: str) -> str:
        if value not in quant_schema.SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"unsupported timeframe '{value}', "
                f"expected one of: {', '.join(quant_schema.SUPPORTED_TIMEFRAMES)}"
            )
        return value

    @field_validator("symbols")
    @classmethod
    def _check_symbols(cls, value: list[str]) -> list[str]:
        for symbol in value:
            if not symbol.strip():
                raise ValueError("symbols must be non-empty")
        return value


class OHLCVIngestResponse(BaseModel):
    """Ergebnis eines Ingest-Laufs: geschriebene Punkte + übersprungene Kerzen."""

    status: str = "ok"
    written: int = Field(ge=0)
    skipped: int = Field(ge=0)


class QuantStatusResponse(BaseModel):
    """InfluxDB-Health + Quant-Feature-Status."""

    influxdb_connected: bool
    influxdb_url: str
    quant_enabled: bool
    buffered_points: int = Field(ge=0)
    feature_version: str


class QuantSchemaResponse(BaseModel):
    """Maschinenlesbare Schema-Dokumentation der Quant-Plattform."""

    feature_version: str
    measurements: list[str]
    timeframes: list[str]
    tags: list[str]
    fields: list[str]
    fields_by_measurement: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Hilfsfunktionen (deterministische Ingest-Logik)
# ---------------------------------------------------------------------------


def _candle_is_consistent(candle: OHLCVCandle) -> bool:
    """OHLC-Konsistenz: high/low müssen open/close umschließen."""
    return (
        candle.low <= min(candle.open, candle.close)
        and max(candle.open, candle.close) <= candle.high
    )


def _to_nanoseconds(moment: datetime) -> int:
    """UTC-Nanosekunden-Timestamp für InfluxDB (naive Zeiten = UTC)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1_000_000_000)


def _candle_to_fields(candle: OHLCVCandle) -> dict[str, float | int | str]:
    """Kerze → InfluxDB-Field-Set (Schema-Fields + trade_count)."""
    return {
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "trade_count": candle.trade_count,
    }


def _require_quant_enabled() -> None:
    """Fail-closed Guard: Ingest nur wenn das Quant-Feature aktiviert ist."""
    settings = get_settings()
    if not settings.influxdb_enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "QUANT_DISABLED",
                "message": "Quant feature is disabled (influxdb_enabled=false)",
            },
        )


# ---------------------------------------------------------------------------
# Endpunkte
# ---------------------------------------------------------------------------


@router.post(
    "/ingest/ohlcv",
    response_model=OHLCVIngestResponse,
    dependencies=[Depends(require_trade_key)],
)
async def ingest_ohlcv(payload: OHLCVIngestRequest) -> OHLCVIngestResponse:
    """Nimmt OHLCV-Kerzen entgegen und schreibt sie in InfluxDB.

    Inkonsistente Kerzen (high/low umschließen open/close nicht) werden
    deterministisch übersprungen und in ``skipped`` gezählt; ``written``
    zählt die tatsächlich geschriebenen Punkte (Symbole × gültige Kerzen).
    """
    _require_quant_enabled()
    store = get_influx_store()

    valid_candles = [c for c in payload.candles if _candle_is_consistent(c)]
    skipped = len(payload.candles) - len(valid_candles)
    written = 0
    if valid_candles:
        points = [_candle_to_fields(c) for c in valid_candles]
        timestamps = [_to_nanoseconds(c.time) for c in valid_candles]
        for symbol in payload.symbols:
            tags = {
                "symbol": symbol,
                "exchange": MANUAL_INGEST_EXCHANGE,
                "timeframe": payload.timeframe,
            }
            await store.write_batch(
                measurement=quant_schema.OHLCV_MEASUREMENT,
                tags=tags,
                points=points,
                timestamps=timestamps,
            )
        written = len(valid_candles) * len(payload.symbols)
    return OHLCVIngestResponse(status="ok", written=written, skipped=skipped)


@router.get(
    "/status",
    response_model=QuantStatusResponse,
    dependencies=[Depends(require_read_key)],
)
async def quant_status() -> QuantStatusResponse:
    """InfluxDB-Health + Quant-Feature-Status."""
    settings = get_settings()
    store = get_influx_store()
    connected = await store.health_check()
    return QuantStatusResponse(
        influxdb_connected=connected,
        influxdb_url=settings.influxdb_url,
        quant_enabled=settings.influxdb_enabled,
        buffered_points=store.buffer_size(),
        feature_version=quant_schema.FEATURE_VERSION,
    )


@router.get(
    "/schema",
    response_model=QuantSchemaResponse,
    dependencies=[Depends(require_read_key)],
)
def quant_schema_doc() -> QuantSchemaResponse:
    """Liefert die Schema-Dokumentation (Measurements, Timeframes, Tags, Fields)."""
    fields_by_measurement: dict[str, list[str]] = {}
    for measurement in _ALL_MEASUREMENTS:
        info = quant_schema.get_measurement_info(measurement)
        if info is not None:
            fields_by_measurement[measurement] = list(info["fields"])
    return QuantSchemaResponse(
        feature_version=quant_schema.FEATURE_VERSION,
        measurements=list(_ALL_MEASUREMENTS),
        timeframes=list(quant_schema.SUPPORTED_TIMEFRAMES),
        tags=list(quant_schema.OHLCV_TAGS),
        fields=list(QUANT_OHLCV_FIELDS),
        fields_by_measurement=fields_by_measurement,
    )
