"""Quant-Plattform API-Endpunkte (Phase 1, P1-8; Phase 2, P2-3; Phase 3, P3-3).

Stellt die InfluxDB-basierte OHLCV-Ingestion, die Feature-Endpunkte und die
Anomaly-Endpunkte unter ``/quant/*`` als eigenen ``APIRouter`` bereit. Die
Einhängung in die Haupt-App erfolgt erst in Phase 9 (siehe
``docs/quant-platform-phase01-plan.md``, P1-8) — deshalb trägt der Router
selbst seinen Prefix und ``main.py`` bleibt unangetastet.

Auth-Trennung nach dem bestehenden Read/Trade-Muster (R5.21):

- ``POST /quant/ingest/ohlcv``        → Trade-Key
- ``GET  /quant/status``              → Read-Key
- ``GET  /quant/schema``              → Read-Key
- ``POST /quant/features/compute``    → Trade-Key (Schreiboperation)
- ``GET  /quant/features/{symbol}``   → Read-Key
- ``POST /quant/anomalies/detect``    → Trade-Key (Schreiboperation)
- ``GET  /quant/anomalies/{symbol}``  → Read-Key

Testbarkeit: Abhängigkeiten (Settings, ``InfluxDBStore``, ``FeatureStore``,
``AnomalyStore``) werden über die modul-scope-Getter ``get_settings``
(re-export aus ``config``), ``get_influx_store``, ``get_feature_store`` und
``get_anomaly_store`` aufgelöst — Unit-Tests binden alle per Monkeypatch auf
Mocks, es wird nie eine echte InfluxDB-Verbindung aufgebaut.

Die Candle-Validierung (OHLC-Konsistenz) und die Point-Konvertierung sind die
eigentliche Endpunkt-Logik; die Ticker-basierte Ingestion aus dem
Shadow-Loop (``quant/ohlcv_ingestion.py``, P1-6) ist bewusst keine
Abhängigkeit dieses Moduls.

Feature-Endpunkte (P2-3): ``POST /quant/features/compute`` berechnet mit dem
deterministischen ``FeatureEngine`` und persistiert über ``FeatureStore``
(falls verfügbar); ``GET /quant/features/{symbol}`` liefert gespeicherte
Features. ``quant/feature_store.py`` entsteht parallel (P2-2) — der Getter
liefert daher ``None``, bis das Modul vorhanden ist, und die Endpunkte
bleiben ohne Persistenz funktional (``stored=false`` / leere Feature-Listen).

Anomaly-Endpunkte (P3-3): ``POST /quant/anomalies/detect`` erkennt mit dem
deterministischen ``AnomalyDetector`` Anomalien in einer Kerzenreihe und
persistiert sie über ``AnomalyStore`` (falls verfügbar);
``GET /quant/anomalies/{symbol}`` liefert gespeicherte Anomalien.
``quant/anomaly_store.py`` entsteht parallel — der Getter
``get_anomaly_store`` liefert daher ``None``, bis das Modul vorhanden ist,
und die Endpunkte bleiben ohne Persistenz funktional
(``stored=false`` / leere Anomalie-Listen).
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from trading_harness.api.security import require_read_key, require_trade_key
from trading_harness.config import get_settings
from trading_harness.quant import schema as quant_schema
from trading_harness.quant.anomaly_detection import Anomaly, AnomalyDetector
from trading_harness.quant.features import FeatureEngine
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


_feature_store: Any | None = None


def get_feature_store() -> Any | None:
    """Liefert den geteilten ``FeatureStore`` (lazy) — oder ``None``.

    ``quant/feature_store.py`` entsteht parallel (P2-2) und wird deshalb
    lazy per ``importlib`` geladen: solange das Modul fehlt (oder der
    Konstruktor noch nicht kompatibel ist), liefert der Getter ``None``
    und die Feature-Endpoints bleiben funktional ohne Persistenz.
    """
    global _feature_store
    if _feature_store is None:
        try:
            module = importlib.import_module("trading_harness.quant.feature_store")
        except ImportError:
            return None
        feature_store_cls = getattr(module, "FeatureStore", None)
        if feature_store_cls is None:
            return None
        try:
            _feature_store = feature_store_cls(store=get_influx_store())
        except (TypeError, ValueError):
            return None
    return _feature_store


_anomaly_store: Any | None = None


def get_anomaly_store() -> Any | None:
    """Liefert den geteilten ``AnomalyStore`` (lazy) — oder ``None``.

    ``quant/anomaly_store.py`` entsteht parallel (P3-2) und wird deshalb
    lazy per ``importlib`` geladen: solange das Modul fehlt (oder der
    Konstruktor noch nicht kompatibel ist), liefert der Getter ``None``
    und die Anomaly-Endpoints bleiben funktional ohne Persistenz.
    Endpunkt-Vertrag: ``detect_and_store(symbol, timeframe, candles,
    exchange=...)`` → ``AnomalyResult(anomalies_found, stored)`` und
    ``get_anomalies(symbol, timeframe, anomaly_type, start, end)``.
    """
    global _anomaly_store
    if _anomaly_store is None:
        try:
            module = importlib.import_module("trading_harness.quant.anomaly_store")
        except ImportError:
            return None
        anomaly_store_cls = getattr(module, "AnomalyStore", None)
        if anomaly_store_cls is None:
            return None
        try:
            _anomaly_store = anomaly_store_cls(store=get_influx_store())
        except (TypeError, ValueError):
            return None
    return _anomaly_store


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


class FeatureComputeRequest(BaseModel):
    """Compute-Anfrage: Kerzenreihe eines Symbols → Features berechnen + speichern."""

    symbol: str = Field(min_length=1)
    timeframe: str
    exchange: str = Field(min_length=1)
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

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("symbol must be non-empty")
        return value


class FeatureComputeResponse(BaseModel):
    """Ergebnis eines Compute-Laufs: verfügbare Features + Persistenz-Status."""

    status: str = "ok"
    symbol: str
    feature_count: int = Field(ge=0)
    stored: bool


class FeatureQueryResponse(BaseModel):
    """Gespeicherte Features für ein Symbol (leere Liste, falls keine vorhanden)."""

    symbol: str
    features: list[dict[str, Any]]
    count: int = Field(ge=0)


class AnomalyDetectRequest(BaseModel):
    """Detect-Anfrage: Kerzenreihe eines Symbols → Anomalien erkennen + speichern."""

    symbol: str = Field(min_length=1)
    timeframe: str
    exchange: str = Field(min_length=1)
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

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("symbol must be non-empty")
        return value


class AnomalyDetectResponse(BaseModel):
    """Ergebnis eines Detect-Laufs: gefundene Anomalien + Persistenz-Status."""

    status: str = "ok"
    symbol: str
    anomalies_found: int = Field(ge=0)
    stored: bool


class AnomalyQueryResponse(BaseModel):
    """Gespeicherte Anomalien für ein Symbol (leere Liste, falls keine vorhanden)."""

    symbol: str
    anomalies: list[dict[str, Any]]
    count: int = Field(ge=0)


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


def _candles_to_engine_input(candles: list[OHLCVCandle]) -> list[dict[str, Any]]:
    """API-Kerzen → FeatureEngine-Kerzen (numerische OHLCV-Fields + time)."""
    return [
        {
            "time": candle.time,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]


def _candles_to_detector_input(candles: list[OHLCVCandle]) -> list[dict[str, Any]]:
    """API-Kerzen → AnomalyDetector-Kerzen (ISO-Zeitstempel + numerische Fields)."""
    return [
        {
            "time": candle.time.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]


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


@router.post(
    "/features/compute",
    response_model=FeatureComputeResponse,
    dependencies=[Depends(require_trade_key)],
)
async def compute_features(payload: FeatureComputeRequest) -> FeatureComputeResponse:
    """Berechnet Features mit dem ``FeatureEngine`` und persistiert sie.

    ``feature_count`` zählt die verfügbaren Indikatoren (FeatureEngine-
    Semantik: ``None`` bei unzureichender Historie zählt nicht). ``stored``
    ist True, wenn ein ``FeatureStore`` die Ergebnisse übernommen hat;
    ohne Store bleibt der Endpunkt funktional (``stored=false``).
    """
    _require_quant_enabled()
    engine = FeatureEngine()
    candle_dicts = _candles_to_engine_input(payload.candles)
    result = engine.compute(candle_dicts)
    feature_count = int(result["feature_count"])

    store = get_feature_store()
    stored = False
    if store is not None:
        await store.compute_and_store(
            payload.symbol,
            payload.timeframe,
            candle_dicts,
            exchange=payload.exchange,
        )
        stored = True
    return FeatureComputeResponse(
        status="ok",
        symbol=payload.symbol,
        feature_count=feature_count,
        stored=stored,
    )


@router.get(
    "/features/{symbol}",
    response_model=FeatureQueryResponse,
    dependencies=[Depends(require_read_key)],
)
async def get_features(
    symbol: str,
    timeframe: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
    features: str | None = None,
) -> FeatureQueryResponse:
    """Liefert gespeicherte Features für ein Symbol.

    ``features`` ist eine Komma-getrennte Feature-Namensliste; ohne Angabe
    werden alle gespeicherten Features zurückgegeben. Unbekannte Symbole
    liefern eine leere Liste (kein Fehler).
    """
    _require_quant_enabled()
    if timeframe not in quant_schema.SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported timeframe '{timeframe}'",
        )
    if features:
        feature_names = [name.strip() for name in features.split(",") if name.strip()]
    else:
        feature_names = []
    # FeatureStore erwartet UTC-ISO-Strings (naive API-Zeiten sind UTC).
    start_iso = start.isoformat() if start is not None else None
    end_iso = end.isoformat() if end is not None else None
    store = get_feature_store()
    rows: list[dict[str, Any]] = []
    if store is not None:
        rows = await store.get_features(symbol, timeframe, feature_names, start_iso, end_iso)
    return FeatureQueryResponse(symbol=symbol, features=rows, count=len(rows))


@router.post(
    "/anomalies/detect",
    response_model=AnomalyDetectResponse,
    dependencies=[Depends(require_trade_key)],
)
async def detect_anomalies(payload: AnomalyDetectRequest) -> AnomalyDetectResponse:
    """Erkennt Anomalien und persistiert sie über ``AnomalyStore``.

    ``anomalies_found`` zählt die erkannten Anomalien über die gesamte
    Kerzenreihe (Detector-Semantik: rolling Fenster, deterministisch).
    ``stored`` ist True, wenn mindestens eine Anomalie in einen
    verfügbaren ``AnomalyStore`` geschrieben wurde; ohne Store bleibt
    der Endpunkt funktional (``stored=false``).
    """
    _require_quant_enabled()
    candle_dicts = _candles_to_detector_input(payload.candles)
    store = get_anomaly_store()
    if store is not None:
        result = await store.detect_and_store(
            payload.symbol,
            payload.timeframe,
            candle_dicts,
            exchange=payload.exchange,
        )
        anomalies_found = int(result.anomalies_found)
        stored = bool(result.stored)
    else:
        anomalies: list[Anomaly] = AnomalyDetector().detect(candle_dicts)
        anomalies_found = len(anomalies)
        stored = False
    return AnomalyDetectResponse(
        status="ok",
        symbol=payload.symbol,
        anomalies_found=anomalies_found,
        stored=stored,
    )


@router.get(
    "/anomalies/{symbol}",
    response_model=AnomalyQueryResponse,
    dependencies=[Depends(require_read_key)],
)
async def get_anomalies(
    symbol: str,
    timeframe: str = "1m",
    anomaly_type: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> AnomalyQueryResponse:
    """Liefert gespeicherte Anomalien für ein Symbol.

    ``anomaly_type`` filtert nach Anomalietyp (``price_shock``,
    ``volume_spike``, ``volatility_outlier``); ``start``/``end`` sind
    optionale UTC-Zeitgrenzen. Unbekannte Symbole liefern eine leere
    Liste (kein Fehler).
    """
    _require_quant_enabled()
    if timeframe not in quant_schema.SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported timeframe '{timeframe}'",
        )
    # AnomalyStore erwartet UTC-ISO-Strings (naive API-Zeiten sind UTC).
    start_iso = start.isoformat() if start is not None else None
    end_iso = end.isoformat() if end is not None else None
    store = get_anomaly_store()
    rows: list[dict[str, Any]] = []
    if store is not None:
        rows = await store.get_anomalies(symbol, timeframe, anomaly_type, start_iso, end_iso)
    return AnomalyQueryResponse(symbol=symbol, anomalies=rows, count=len(rows))
