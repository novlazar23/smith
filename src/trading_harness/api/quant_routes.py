"""Quant-Plattform API-Endpunkte (Phase 1, P1-8; Phase 2, P2-3; Phase 3, P3-3;
Phase 4, P4-3; Phase 5, P5-3; Phase 6, P6-3; Phase 7, P7-3; Phase 8, P8-3;
Phase 9, P9-3; Phase 10, P10-3; Phase 11, P11-3).

Stellt die InfluxDB-basierte OHLCV-Ingestion, die Feature-Endpunkte, die
Anomaly-Endpunkte und die Similarity-Endpunkte unter ``/quant/*`` als eigenen
``APIRouter`` bereit. Die Einhängung in die Haupt-App erfolgt erst in Phase 9
(siehe ``docs/quant-platform-phase01-plan.md``, P1-8) — deshalb trägt der
Router selbst seinen Prefix und ``main.py`` bleibt unangetastet.

Auth-Trennung nach dem bestehenden Read/Trade-Muster (R5.21):

- ``POST /quant/ingest/ohlcv``        → Trade-Key
- ``GET  /quant/status``              → Read-Key
- ``GET  /quant/schema``              → Read-Key
- ``POST /quant/features/compute``    → Trade-Key (Schreiboperation)
- ``GET  /quant/features/{symbol}``   → Read-Key
- ``POST /quant/anomalies/detect``    → Trade-Key (Schreiboperation)
- ``GET  /quant/anomalies/{symbol}``  → Read-Key
- ``POST /quant/regime/detect``       → Trade-Key (Schreiboperation)
- ``GET  /quant/regime/{symbol}``     → Read-Key
- ``POST /quant/similarity/find``     → Trade-Key (Schreiboperation)
- ``GET  /quant/similarity/{symbol}`` → Read-Key
- ``POST /quant/outcomes/compute``    → Trade-Key (Schreiboperation)
- ``GET  /quant/outcomes/{symbol}``   → Read-Key
- ``POST /quant/ml/features``         → Trade-Key (reine Berechnung)
- ``POST /quant/ml/importance``       → Trade-Key (reine Berechnung)
- ``POST /quant/backtest/run``        → Trade-Key (reine Berechnung)
- ``GET  /quant/backtest/{symbol}``   → Read-Key
- ``GET  /quant/shadow/status``       → Read-Key
- ``GET  /quant/perf/cache-stats``    → Read-Key
- ``GET  /quant/perf/batch-status``   → Read-Key
- ``POST /quant/validate``            → Trade-Key (reine Berechnung)

 Testbarkeit: Abhängigkeiten (Settings, ``InfluxDBStore``, ``FeatureStore``,
``AnomalyStore``, ``RegimeStore``, ``ForwardOutcomeStore``, ``BacktestStore``)
werden über die modul-scope-Getter ``get_settings`` (re-export aus ``config``),
``get_influx_store``, ``get_feature_store``, ``get_anomaly_store``,
``get_regime_store``, ``get_outcome_store`` und ``get_backtest_store``
aufgelöst — Unit-Tests binden alle per Monkeypatch auf Mocks, es wird nie eine
echte InfluxDB-Verbindung aufgebaut.

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

Regime-Endpunkte (P4-3): ``POST /quant/regime/detect`` erkennt mit dem
deterministischen ``RegimeDetector`` die Marktphase einer Kerzenreihe und
persistiert sie über ``RegimeStore`` (falls verfügbar);
``GET /quant/regime/{symbol}`` liefert gespeicherte Regimes.
 ``quant/regime_store.py`` entsteht parallel (P4-2) — der Getter
 ``get_regime_store`` liefert daher ``None``, bis das Modul vorhanden ist,
 und die Endpunkte bleiben ohne Persistenz funktional
 (``stored=false`` / leere Regime-Listen).

 Similarity-Endpunkte (P5-3): ``POST /quant/similarity/find`` sucht mit dem
 deterministischen ``SimilarityEngine`` (``quant/similarity.py``) die
 top_k ähnlichsten historischen Fenster für eine Query-Kerzenreihe — reine
 In-Memory-Berechnung ohne Persistenz (``best_distance``/``best_correlation``
 sind ``None``, wenn zu wenig Historie vorhanden ist).
  ``GET /quant/similarity/{symbol}`` liefert Similitäts-Infos:
  ``available`` spiegelt die InfluxDB-Verfügbarkeit (Datenquelle für
  Historie) und ``window_size`` die Standard-Fenstergröße der Engine.

  Forward-Outcome-Endpunkte (P6-3): ``POST /quant/outcomes/compute``
  berechnet mit dem deterministischen ``ForwardOutcomeEngine``
  (``quant/forward_outcomes.py``, P6-1) die Forward-Return-Statistik
  (Mean/Median, Hit Rate, Profit Factor, Expectancy, Std) für die
  angefragten Horizonte und persistiert über ``ForwardOutcomeStore``
  (falls verfügbar); ``GET /quant/outcomes/{symbol}`` liefert gespeicherte
   Outcomes. ``quant/forward_outcomes_store.py`` entsteht parallel (P6-2) —
   der Getter ``get_outcome_store`` liefert daher ``None``, bis das Modul
   vorhanden ist, und die Endpunkte bleiben ohne Persistenz funktional
   (``stored=false`` / leere Outcome-Listen).

   ML-Endpunkte (P7-3): ``POST /quant/ml/features`` baut mit dem
   deterministischen ``MLFeatureBuilder`` (``quant/ml_features.py``, P7-1)
   einen ML-Feature-Vektor aus übergebenen Roh-Features — Z-Score-
   Normalisierung (optional) und NaN-/Inf-Sanitisierung, reine
   In-Memory-Berechnung ohne Persistenz. ``POST /quant/ml/importance``
   berechnet mit der deterministischen ``FeatureImportanceEngine``
    (``quant/feature_importance.py``, P7-2) die Feature-Importance
    (Pearson-Korrelation gegen ein Ziel, Ranking, Threshold-Filter und
    Feature-Gruppen-Aggregation) — ebenfalls reine In-Memory-Berechnung.

    Backtest-Endpunkte (P8-3): ``POST /quant/backtest/run`` führt mit dem
    deterministischen ``BacktestEngine`` (``quant/backtesting.py``, P8-1)
    einen Backtest für die übergebene Kerzenreihe aus — die Strategie
    (``sma`` = SMA-Crossover, ``rsi`` = RSI-Overbought/Oversold) und ihre
    Parameter werden über ``strategy``/``params`` übergeben; PnL, Win Rate,
    Drawdown, Sharpe Ratio und Profit Factor werden pro Lauf berechnet.
    Persistenz erfolgt über ``BacktestStore`` (falls verfügbar);
     ``GET /quant/backtest/{symbol}`` liefert gespeicherte Backtest-Ergebnisse.
     ``quant/backtest_store.py`` entsteht parallel (P8-2) — der Getter
     ``get_backtest_store`` liefert daher ``None``, bis das Modul vorhanden
     ist, und die Endpunkte bleiben ohne Persistenz funktional
     (``stored=false`` / leere Ergebnis-Listen).

     Shadow-Loop-Status (P9-3): ``GET /quant/shadow/status`` liefert den
     Integrationsstatus der Quant-Plattform im Shadow Trading Loop —
     ``integration_active`` spiegelt das Quant-Feature-Flag
     (``influxdb_enabled``), ``quant_engines`` listet die Quant-Module,
     deren Ergebnisse der ``EvidenceAggregator`` (``quant/evidence_aggregator.py``,
      P9-2) als Evidence aufnehmen kann, und ``last_evidence`` enthält die
      jeweils letzten pro Engine aggregierten Einträge (leeres Dict, solange
      keine Evidence aggregiert wurde). Reiner Status-Endpunkt wie
      ``GET /quant/status``: kein InfluxDB-Zugriff, kein Fail-closed-Guard.

      Performance-Endpunkte (P10-3): ``GET /quant/perf/cache-stats``
      liefert die Statistiken des geteilten ``FeatureCache``
      (``quant/feature_cache.py``) — Hits, Misses, aktuelle Größe und
      Hit-Rate; ``GET /quant/perf/batch-status`` liefert den Gesamtstatus
      des geteilten ``BatchProcessor`` (``quant/batch_processor.py``) —
      Job-Zählungen nach Status. Beide sind reine In-Memory-Status-
       Endpunkte wie ``GET /quant/shadow/status``: kein InfluxDB-Zugriff,
       kein Fail-closed-Guard.

       Validierungs-Endpunkte (P11-3): ``POST /quant/validate`` validiert
       Eingabedaten über den deterministischen ``Validator``
       (``quant/validation.py``, Phase 11) ohne Persistenz:
       ``type='candle'`` prüft ein Kerzen-Dict (OHLC-Konsistenz, positive
       Preise, nicht-negatives Volume), ``type='features'`` prüft eine
       Feature-Map (NaN/Inf/Range, leere Map → Warning). Ungültige Daten
       werden mit ``valid=false`` + Fehlerliste gemeldet (Status 200) —
       der Endpunkt dient als reine Validierungsschnittstelle und
       verändert nichts (wie ``POST /quant/ml/*``).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from trading_harness.api.security import require_read_key, require_trade_key
from trading_harness.config import get_settings
from trading_harness.quant import schema as quant_schema
from trading_harness.quant.anomaly_detection import Anomaly, AnomalyDetector
from trading_harness.quant.backtesting import BacktestEngine, BacktestResult, Signal
from trading_harness.quant.batch_processor import BatchProcessor
from trading_harness.quant.evidence_aggregator import EvidenceAggregator
from trading_harness.quant.feature_cache import FeatureCache
from trading_harness.quant.feature_importance import FeatureImportanceEngine
from trading_harness.quant.features import FeatureEngine
from trading_harness.quant.forward_outcomes import ForwardOutcomeEngine
from trading_harness.quant.influxdb_client import InfluxDBStore
from trading_harness.quant.ml_features import MLFeatureBuilder
from trading_harness.quant.regime_detection import REGIME_NAMES, RegimeDetector, RegimeResult
from trading_harness.quant.similarity import SimilarityEngine
from trading_harness.quant.validation import ValidationResult, Validator

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


_regime_store: Any | None = None


def get_regime_store() -> Any | None:
    """Liefert den geteilten ``RegimeStore`` (lazy) — oder ``None``.

    ``quant/regime_store.py`` entsteht parallel (P4-2) und wird deshalb
    lazy per ``importlib`` geladen: solange das Modul fehlt (oder der
    Konstruktor noch nicht kompatibel ist), liefert der Getter ``None``
    und die Regime-Endpoints bleiben funktional ohne Persistenz.
    Endpunkt-Vertrag: ``detect_and_store(symbol, timeframe, candles,
    exchange=...)`` → Ergebnis mit ``regime``/``confidence``/``stored`` und
    ``get_regime(symbol, timeframe, start, end)``.
    """
    global _regime_store
    if _regime_store is None:
        try:
            module = importlib.import_module("trading_harness.quant.regime_store")
        except ImportError:
            return None
        regime_store_cls = getattr(module, "RegimeStore", None)
        if regime_store_cls is None:
            return None
        try:
            _regime_store = regime_store_cls(store=get_influx_store())
        except (TypeError, ValueError):
            return None
    return _regime_store


_outcome_store: Any | None = None


def get_outcome_store() -> Any | None:
    """Liefert den geteilten ``ForwardOutcomeStore`` (lazy) — oder ``None``.

    ``quant/forward_outcomes_store.py`` entsteht parallel (P6-2) und wird
    deshalb lazy per ``importlib`` geladen: solange das Modul fehlt (oder
    der Konstruktor noch nicht kompatibel ist), liefert der Getter ``None``
    und die Outcome-Endpoints bleiben funktional ohne Persistenz.
    Endpunkt-Vertrag: ``compute_and_store(symbol, timeframe, candles,
    pattern_length=..., exchange=...)`` → Ergebnis mit ``stored`` und
    ``get_outcomes(symbol, timeframe, start, end)``.
    """
    global _outcome_store
    if _outcome_store is None:
        try:
            module = importlib.import_module("trading_harness.quant.forward_outcomes_store")
        except ImportError:
            return None
        outcome_store_cls = getattr(module, "ForwardOutcomeStore", None)
        if outcome_store_cls is None:
            return None
        try:
            _outcome_store = outcome_store_cls(store=get_influx_store())
        except (TypeError, ValueError):
            return None
    return _outcome_store


_backtest_store: Any | None = None


def get_backtest_store() -> Any | None:
    """Liefert den geteilten ``BacktestStore`` (lazy) — oder ``None``.

    ``quant/backtest_store.py`` entsteht parallel (P8-2) und wird deshalb
    lazy per ``importlib`` geladen: solange das Modul fehlt (oder der
    Konstruktor noch nicht kompatibel ist), liefert der Getter ``None``
    und die Backtest-Endpoints bleiben funktional ohne Persistenz.
    Endpunkt-Vertrag: ``run_and_store(symbol, timeframe, result,
    exchange=...)`` → Ergebnis mit ``stored`` und
    ``get_backtests(symbol, timeframe, start, end)``.
    """
    global _backtest_store
    if _backtest_store is None:
        try:
            module = importlib.import_module("trading_harness.quant.backtest_store")
        except ImportError:
            return None
        backtest_store_cls = getattr(module, "BacktestStore", None)
        if backtest_store_cls is None:
            return None
        try:
            _backtest_store = backtest_store_cls(store=get_influx_store())
        except (TypeError, ValueError):
            return None
    return _backtest_store


_evidence_aggregator: EvidenceAggregator | None = None


def get_evidence_aggregator() -> EvidenceAggregator:
    """Liefert den geteilten ``EvidenceAggregator`` (lazy, eine Instanz pro Prozess).

    Der Aggregator (P9-2) sammelt die Quant-Evidence des Shadow-Loops;
    ``GET /quant/shadow/status`` (P9-3) liest seinen aktuellen Stand.
    """
    global _evidence_aggregator
    if _evidence_aggregator is None:
        _evidence_aggregator = EvidenceAggregator()
    return _evidence_aggregator


_feature_cache: FeatureCache | None = None


def get_feature_cache() -> FeatureCache:
    """Liefert den geteilten ``FeatureCache`` (lazy, eine Instanz pro Prozess).

    Der LRU/TTL-Cache merkt berechnete Features;
    ``GET /quant/perf/cache-stats`` (P10-3) liert seine Statistiken.
    """
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCache()
    return _feature_cache


_batch_processor: BatchProcessor | None = None


def get_batch_processor() -> BatchProcessor:
    """Liefert den geteilten ``BatchProcessor`` (lazy, eine Instanz pro Prozess).

    Der Processor verarbeitet mehrere Symbole in Batches;
    ``GET /quant/perf/batch-status`` (P10-3) liest seinen Gesamtstatus.
    """
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor


_validator: Validator | None = None


def get_validator() -> Validator:
    """Liefert den geteilten ``Validator`` (lazy, eine Instanz pro Prozess).

    Der ``Validator`` (``quant/validation.py``, Phase 11) ist stateless und
    deterministisch; ``POST /quant/validate`` (P11-3) prüft damit
    Candle-/Feature-Daten ohne Persistenz.
    """
    global _validator
    if _validator is None:
        _validator = Validator()
    return _validator


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


class RegimeDetectRequest(BaseModel):
    """Detect-Anfrage: Kerzenreihe eines Symbols → Regime erkennen + speichern."""

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


class RegimeDetectResponse(BaseModel):
    """Ergebnis eines Detect-Laufs: erkannte Marktphase + Persistenz-Status."""

    status: str = "ok"
    symbol: str
    regime: str
    confidence: float = Field(ge=0.0, le=1.0)
    stored: bool

    @field_validator("regime")
    @classmethod
    def _check_regime(cls, value: str) -> str:
        if value not in REGIME_NAMES:
            raise ValueError(
                f"unknown regime '{value}', expected one of: {', '.join(REGIME_NAMES)}"
            )
        return value


class RegimeQueryResponse(BaseModel):
    """Gespeicherte Regimes für ein Symbol (leere Liste, falls keine vorhanden)."""

    symbol: str
    regimes: list[dict[str, Any]]
    count: int = Field(ge=0)


class SimilarityFindRequest(BaseModel):
    """Find-Anfrage: Query-Kerzenreihe → ähnliche historische Fenster suchen.

    ``query`` und ``history`` sind beide Kerzenreihen; die Ähnlichkeit wird
    rein in-memory (ohne Persistenz) mit dem ``SimilarityEngine`` berechnet.
    """

    symbol: str = Field(min_length=1)
    timeframe: str
    exchange: str = Field(min_length=1)
    query: list[OHLCVCandle] = Field(min_length=2)
    history: list[OHLCVCandle] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)

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


class SimilarMatchModel(BaseModel):
    """Ein ähnlicher historischer Ausschnitt (API-Darstellung)."""

    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    distance: float = Field(ge=0.0)
    correlation: float = Field(ge=-1.0, le=1.0)


class SimilarityFindResponse(BaseModel):
    """Ergebnis der Similarity-Suche: Top-K-Matches + Bestwerte.

    ``best_distance``/``best_correlation`` sind ``None``, wenn keine Matches
    gefunden wurden (z. B. unzureichende Historie).
    """

    status: str = "ok"
    query_length: int = Field(ge=0)
    matches: list[SimilarMatchModel]
    best_distance: float | None = None
    best_correlation: float | None = None


class SimilarityInfoResponse(BaseModel):
    """Similitäts-Infos für ein Symbol (Datenverfügbarkeit + Fenstergröße)."""

    symbol: str
    available: bool
    window_size: int = Field(ge=1)


class OutcomeComputeRequest(BaseModel):
    """Compute-Anfrage: Kerzenreihe eines Symbols → Forward Outcomes berechnen.

    ``pattern_length`` ist die Referenzmuster-Länge in Kerzen; ``horizons``
    die Forward-Rücklauf-Horizonte in Kerzen (jeder ≥ 1).
    """

    symbol: str = Field(min_length=1)
    timeframe: str
    exchange: str = Field(min_length=1)
    candles: list[OHLCVCandle] = Field(min_length=1)
    pattern_length: int = Field(default=10, ge=1, le=1000)
    horizons: list[int] = Field(default_factory=lambda: [5, 10, 20], min_length=1)

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

    @field_validator("horizons")
    @classmethod
    def _check_horizons(cls, value: list[int]) -> list[int]:
        if any(horizon < 1 for horizon in value):
            raise ValueError("horizons must be positive integers")
        return value


class OutcomeStatsModel(BaseModel):
    """Forward-Outcome-Statistik für einen einzelnen Horizont (Kerzen)."""

    horizon: int = Field(ge=1)
    mean_return: float
    median_return: float
    hit_rate: float = Field(ge=0.0, le=1.0)
    profit_factor: float = Field(ge=0.0)
    expectancy: float
    std_return: float = Field(ge=0.0)
    sample_size: int = Field(ge=0)
    max_gain: float
    max_loss: float


class OutcomeComputeResponse(BaseModel):
    """Ergebnis eines Compute-Laufs: Outcomes pro Horizont + Persistenz-Status.

    ``outcomes`` ist eine Map ``str(horizon)`` → Statistik. ``stored`` ist
    True, wenn ein ``ForwardOutcomeStore`` die Ergebnisse übernommen hat;
    ohne Store bleibt der Endpunkt funktional (``stored=false``).
    """

    status: str = "ok"
    symbol: str
    outcomes: dict[str, OutcomeStatsModel]
    stored: bool


class OutcomeQueryResponse(BaseModel):
    """Gespeicherte Outcomes für ein Symbol (leere Liste, falls keine vorhanden)."""

    symbol: str
    outcomes: list[dict[str, Any]]
    count: int = Field(ge=0)


class MLFeaturesRequest(BaseModel):
    """Build-Anfrage: Roh-Features eines Symbols → ML-Feature-Vektor (P7-3).

    ``features`` ist die Roh-Feature-Map (Key → Wert); NaN/Inf-Werte werden
    vom ``MLFeatureBuilder`` deterministisch durch ``fill_nan`` ersetzt.
    ``normalize`` steuert die Z-Score-Normalisierung (Default: aktiv).
    """

    symbol: str = Field(min_length=1)
    timeframe: str
    features: dict[str, float] = Field(min_length=1)
    normalize: bool = True

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


class MLFeaturesResponse(BaseModel):
    """Ergebnis des ML-Feature-Vector-Builds: bereinigte Features + Namen."""

    status: str = "ok"
    symbol: str
    features: dict[str, float]
    feature_names: list[str]


class MLImportanceRequest(BaseModel):
    """Importance-Anfrage: Feature-Zeitreihen gegen ein Ziel (P7-3).

    ``features`` ist eine Map Feature-Name → Werteliste; jede Liste muss
    exakt die Länge von ``target`` haben. ``threshold`` ist die
    Mindest-Importance für ``top_features`` (Default: 0.1).
    """

    features: dict[str, list[float]] = Field(min_length=1)
    target: list[float] = Field(min_length=2)
    threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_lengths(self) -> MLImportanceRequest:
        for name, values in self.features.items():
            if len(values) != len(self.target):
                raise ValueError(
                    f"feature '{name}' length {len(values)} does not match "
                    f"target length {len(self.target)}"
                )
        return self


class MLImportanceFeatureModel(BaseModel):
    """Einzelnes Feature mit Importance, Korrelation und Rang (API-Darstellung)."""

    name: str
    importance: float = Field(ge=0.0)
    correlation: float
    rank: int = Field(ge=1)


class MLImportanceResponse(BaseModel):
    """Ergebnis der Feature-Importance: Ranking, Top-Features, Gruppen."""

    status: str = "ok"
    features: list[MLImportanceFeatureModel]
    top_features: list[str]
    feature_groups: dict[str, float]


# Erlaubte Strategien (P8-1-Factory) und deren zulässige Parameter-Keys.
BACKTEST_STRATEGY_PARAMS: dict[str, tuple[str, ...]] = {
    "sma": ("fast_period", "slow_period"),
    "rsi": ("period", "oversold", "overbought"),
}


class BacktestRunRequest(BaseModel):
    """Run-Anfrage: Kerzenreihe eines Symbols → Backtest ausführen (P8-3).

    ``strategy`` ist der Strategie-Name (``sma``/``rsi``), ``params`` deren
    Parameter (Default-Werte = ``BacktestEngine``-Standards). Die
    Engine-Parameter (Kapital, Risiko, Stop-Loss, Take-Profit) sind optional
    und übernehmen sonst die Engine-Defaults.
    """

    symbol: str = Field(min_length=1)
    timeframe: str
    exchange: str = Field(min_length=1)
    candles: list[OHLCVCandle] = Field(min_length=1)
    strategy: str
    params: dict[str, float] = Field(default_factory=dict)
    initial_capital: float = Field(default=10000.0, gt=0)
    risk_per_trade: float = Field(default=0.02, gt=0.0, le=1.0)
    stop_loss_pct: float = Field(default=0.02, gt=0.0, le=1.0)
    take_profit_pct: float = Field(default=0.04, gt=0.0, le=1.0)

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

    @model_validator(mode="after")
    def _check_strategy_params(self) -> BacktestRunRequest:
        allowed = BACKTEST_STRATEGY_PARAMS.get(self.strategy)
        if allowed is None:
            raise ValueError(
                f"unknown strategy '{self.strategy}', "
                f"expected one of: {', '.join(sorted(BACKTEST_STRATEGY_PARAMS))}"
            )
        unknown = sorted(set(self.params) - set(allowed))
        if unknown:
            raise ValueError(
                f"unknown params for strategy '{self.strategy}': {', '.join(unknown)}"
            )
        for key, value in self.params.items():
            if key.endswith("_period") and (value < 1 or not float(value).is_integer()):
                raise ValueError(f"param '{key}' must be a positive integer")
        return self


class BacktestResultModel(BaseModel):
    """Statistiken eines Backtests (API-Darstellung von ``BacktestResult``)."""

    strategy: str
    timeframe: str
    candle_count: int = Field(ge=0)
    total_trades: int = Field(ge=0)
    winning_trades: int = Field(ge=0)
    losing_trades: int = Field(ge=0)
    win_rate: float = Field(ge=0.0, le=1.0)
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float = Field(ge=0.0)
    sharpe_ratio: float
    avg_trade_pnl: float
    profit_factor: float = Field(ge=0.0)


class BacktestRunResponse(BaseModel):
    """Ergebnis eines Backtest-Laufs: Statistiken + Persistenz-Status.

    ``stored`` ist True, wenn ein ``BacktestStore`` das Ergebnis übernommen
    hat; ohne Store bleibt der Endpunkt funktional (``stored=false``).
    """

    status: str = "ok"
    symbol: str
    result: BacktestResultModel
    stored: bool


class BacktestQueryResponse(BaseModel):
    """Gespeicherte Backtests für ein Symbol (leere Liste, falls keine vorhanden)."""

    symbol: str
    results: list[dict[str, Any]]
    count: int = Field(ge=0)


# Quant-Module, deren Ergebnisse der Shadow-Loop als Evidence nutzt (P9-3);
# entspricht den bekannten Quellen des ``EvidenceAggregator`` (P9-2).
SHADOW_QUANT_ENGINES: tuple[str, ...] = (
    "features",
    "anomalies",
    "regime",
    "similarity",
    "forward_outcomes",
    "ml_features",
    "backtest",
)


class ShadowStatusResponse(BaseModel):
    """Shadow-Loop-Integrationsstatus der Quant-Plattform (P9-3).

    ``integration_active`` ist True, wenn die Quant-Plattform als
    Evidence-Quelle des Shadow-Loops aktiviert ist (``influxdb_enabled``);
    ``quant_engines`` listet alle Quant-Module des ``EvidenceAggregator``;
    ``last_evidence`` enthält die jeweils letzten pro Engine aggregierten
    Einträge (leeres Dict, solange keine Evidence aggregiert wurde).
    """

    status: str = "ok"
    integration_active: bool
    quant_engines: list[str]
    last_evidence: dict[str, Any]


class PerfCacheStatsModel(BaseModel):
    """Cache-Statistiken des ``FeatureCache`` (API-Darstellung)."""

    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    size: int = Field(ge=0)
    hit_rate: float = Field(ge=0.0, le=1.0)


class PerfCacheStatsResponse(BaseModel):
    """Cache-Statistiken (P10-3): Treffer, Verfehlungen, Größe, Hit-Rate."""

    status: str = "ok"
    cache: PerfCacheStatsModel


class PerfBatchStatusModel(BaseModel):
    """Batch-Processor-Gesamtstatus (API-Darstellung von ``BatchStatus``).

    ``completed``/``running``/``pending`` zählen die Jobs nach Status;
    ``failed``-Jobs sind in ``total_jobs`` enthalten, aber in keiner der
    drei Status-Zählungen.
    """

    total_jobs: int = Field(ge=0)
    completed: int = Field(ge=0)
    running: int = Field(ge=0)
    pending: int = Field(ge=0)


class PerfBatchStatusResponse(BaseModel):
    """Batch-Processor-Status (P10-3): Job-Zählungen nach Status."""

    status: str = "ok"
    batch: PerfBatchStatusModel


class QuantValidateRequest(BaseModel):
    """Validate-Anfrage: Daten eines unterstützten Typs prüfen (P11-3).

    ``type`` wählt die Validierungs-Regel: ``candle`` = ein Kerzen-Dict
    (``time``/``open``/``high``/``low``/``close``/``volume``),
    ``features`` = Feature-Map (Name → Zahl). ``data`` ist das zu
    prüfende Datenobjekt; inhaltlich fehlerhafte Daten werden nicht mit
    422 abgewiesen, sondern als ``valid=false`` gemeldet (Validierungs-
    Semantik des Endpunkts).
    """

    type: Literal["candle", "features"]
    data: dict[str, Any]


class QuantValidateResponse(BaseModel):
    """Ergebnis der Validierung: Gültigkeits-Flag + Fehler-/Warnliste."""

    status: str = "ok"
    valid: bool
    errors: list[str]
    warnings: list[str]


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


def _candles_to_regime_input(candles: list[OHLCVCandle]) -> list[dict[str, Any]]:
    """API-Kerzen → RegimeDetector-Kerzen (ISO-Zeitstempel + numerische Fields)."""
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


def _candles_to_similarity_input(candles: list[OHLCVCandle]) -> list[dict[str, Any]]:
    """API-Kerzen → SimilarityEngine-Kerzen (ISO-Zeitstempel + numerische Fields).

    Die Engine nutzt ausschließlich das ``close``-Field für die Distanz-/
    Korrelationsberechnung; die übrigen Fields bleiben zur Vollständigkeit.
    """
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


def _candles_to_outcome_input(candles: list[OHLCVCandle]) -> list[dict[str, Any]]:
    """API-Kerzen → ForwardOutcomeEngine-Kerzen (ISO-Zeitstempel + Fields).

    Die Engine nutzt ausschließlich das ``close``-Field für die
    Forward-Return-Berechnung; die übrigen Fields bleiben zur Vollständigkeit.
    """
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


def _outcome_stats_to_model(stats: Any) -> OutcomeStatsModel:
    """Engine-``ForwardOutcome`` → API-Modell (defensive Typisierung)."""
    return OutcomeStatsModel(
        horizon=int(stats.horizon),
        mean_return=float(stats.mean_return),
        median_return=float(stats.median_return),
        hit_rate=float(stats.hit_rate),
        profit_factor=float(stats.profit_factor),
        expectancy=float(stats.expectancy),
        std_return=float(stats.std_return),
        sample_size=int(stats.sample_size),
        max_gain=float(stats.max_gain),
        max_loss=float(stats.max_loss),
    )


def _regime_result_from_store(result: Any) -> tuple[str, float, bool]:
    """Store-Ergebnis → (regime, confidence, stored) mit defensiver Typisierung."""
    return str(result.regime), float(result.confidence), bool(result.stored)


def _candles_to_backtest_input(candles: list[OHLCVCandle]) -> list[dict[str, Any]]:
    """API-Kerzen → BacktestEngine-Kerzen (ISO-Zeitstempel + numerische Fields).

    Die Engine nutzt ausschließlich das ``close``-Field für die Trade-
    Simulation; die übrigen Fields bleiben zur Vollständigkeit.
    """
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


def _build_backtest_strategy(
    engine: BacktestEngine, strategy: str, params: dict[str, float]
) -> Callable[[list[dict[str, Any]], int], Signal]:
    """Strategie-Factory: Name + Parameter → Engine-Strategie-Callable.

    Perioden-Parameter werden als Ganzzahlen übergeben (Slicing der Engine);
    fehlende Parameter übernehmen die Engine-Standardwerte.
    """
    if strategy == "sma":
        return engine.simple_moving_average_strategy(
            fast_period=int(params.get("fast_period", 10)),
            slow_period=int(params.get("slow_period", 30)),
        )
    return engine.rsi_strategy(
        period=int(params.get("period", 14)),
        oversold=float(params.get("oversold", 30.0)),
        overbought=float(params.get("overbought", 70.0)),
    )


def _backtest_result_to_model(
    result: BacktestResult, strategy: str, candle_count: int
) -> BacktestResultModel:
    """Engine-``BacktestResult`` → API-Modell (defensive Typisierung)."""
    return BacktestResultModel(
        strategy=strategy,
        timeframe=str(result.timeframe),
        candle_count=candle_count,
        total_trades=int(result.total_trades),
        winning_trades=int(result.winning_trades),
        losing_trades=int(result.losing_trades),
        win_rate=float(result.win_rate),
        total_pnl=float(result.total_pnl),
        total_pnl_pct=float(result.total_pnl_pct),
        max_drawdown=float(result.max_drawdown),
        sharpe_ratio=float(result.sharpe_ratio),
        avg_trade_pnl=float(result.avg_trade_pnl),
        profit_factor=float(result.profit_factor),
    )


# Candle-Fields, die vom ``Validator`` numerisch verglichen werden
# (OHLC-Konsistenz, Nicht-Negativität) — defensive Vorabprüfung.
_VALIDATE_CANDLE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _numeric_value_errors(data: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    """Defensive Prüfung: gelistete ``data``-Keys müssen numerisch sein.

    Schützt den ``Validator`` vor ``TypeError`` (z. B. ``high < max(open,
    close)`` auf String-Werten) — nicht-numerische Werte werden als
    Validierungsfehler gemeldet (``valid=false``), nicht als 500-Fehler.
    Fehlende Keys werden ignoriert (``Validator`` meldet sie selbst).
    """
    errors: list[str] = []
    for name in fields:
        if name not in data:
            continue
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"Field '{name}' must be a number, got {type(value).__name__}")
    return errors


def _last_evidence_from_aggregator(aggregator: EvidenceAggregator) -> dict[str, Any]:
    """Aggregator-Einträge → API-Dict (die jeweils letzten Einträge pro Engine)."""
    last_evidence: dict[str, Any] = {}
    for source in aggregator.sources:
        entry = aggregator.get_entry(source)
        if entry is None:
            continue
        last_evidence[source] = {
            "timestamp": str(entry.timestamp),
            "confidence": float(entry.confidence),
            "priority": int(entry.priority),
            "data": dict(entry.data),
        }
    return last_evidence


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


@router.post(
    "/regime/detect",
    response_model=RegimeDetectResponse,
    dependencies=[Depends(require_trade_key)],
)
async def detect_regime(payload: RegimeDetectRequest) -> RegimeDetectResponse:
    """Erkennt die Marktphase mit dem ``RegimeDetector`` und persistiert sie.

    ``regime``/``confidence`` kommen aus der deterministischen Erkennung
    (``quant/regime_detection.py``); ``stored`` ist True, wenn ein
    verfügbarer ``RegimeStore`` das Ergebnis übernommen hat; ohne Store
    bleibt der Endpunkt funktional (``stored=false``).
    """
    _require_quant_enabled()
    candle_dicts = _candles_to_regime_input(payload.candles)
    store = get_regime_store()
    if store is not None:
        result = await store.detect_and_store(
            payload.symbol,
            payload.timeframe,
            candle_dicts,
            exchange=payload.exchange,
        )
        regime, confidence, stored = _regime_result_from_store(result)
    else:
        regime_result: RegimeResult = RegimeDetector().detect(candle_dicts)
        regime, confidence, stored = (
            regime_result.regime,
            regime_result.confidence,
            False,
        )
    return RegimeDetectResponse(
        status="ok",
        symbol=payload.symbol,
        regime=regime,
        confidence=confidence,
        stored=stored,
    )


@router.get(
    "/regime/{symbol}",
    response_model=RegimeQueryResponse,
    dependencies=[Depends(require_read_key)],
)
async def get_regimes(
    symbol: str,
    timeframe: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
) -> RegimeQueryResponse:
    """Liefert gespeicherte Regimes für ein Symbol.

    ``timeframe`` wird gegen die unterstützten Werte validiert;
    ``start``/``end`` sind optionale UTC-Zeitgrenzen. Unbekannte Symbole
    liefern eine leere Liste (kein Fehler).
    """
    _require_quant_enabled()
    if timeframe not in quant_schema.SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported timeframe '{timeframe}'",
        )
    # RegimeStore erwartet UTC-ISO-Strings (naive API-Zeiten sind UTC).
    start_iso = start.isoformat() if start is not None else None
    end_iso = end.isoformat() if end is not None else None
    store = get_regime_store()
    rows: list[dict[str, Any]] = []
    if store is not None:
        rows = await store.get_regime(symbol, timeframe, start_iso, end_iso)
    return RegimeQueryResponse(symbol=symbol, regimes=rows, count=len(rows))


# ---------------------------------------------------------------------------
# Similarity-Endpunkte (P5-3)
# ---------------------------------------------------------------------------

# Standard-Fenstergröße der SimilarityEngine (Sliding-Window-Länge in Kerzen).
SIMILARITY_DEFAULT_WINDOW_SIZE: int = 20


@router.post(
    "/similarity/find",
    response_model=SimilarityFindResponse,
    dependencies=[Depends(require_trade_key)],
)
async def find_similarity(payload: SimilarityFindRequest) -> SimilarityFindResponse:
    """Findet die top_k ähnlichsten historischen Fenster für eine Query-Reihe.

    Die Ähnlichkeit wird mit dem deterministischen ``SimilarityEngine``
    (Euclidean Distance + Pearson-Korrelation auf normalisierten Close-
    Reihen) rein in-memory berechnet — es findet keine Persistenz statt.
    ``best_distance``/``best_correlation`` referenzieren den besten Match
    bzw. sind ``None``, wenn die Historie zu kurz für ein Fenster ist.
    """
    _require_quant_enabled()
    engine = SimilarityEngine(window_size=SIMILARITY_DEFAULT_WINDOW_SIZE, top_k=payload.top_k)
    query = _candles_to_similarity_input(payload.query)
    history = _candles_to_similarity_input(payload.history)
    result = engine.find_similar(query, history)
    matches = [
        SimilarMatchModel(
            start_index=m.start_index,
            end_index=m.end_index,
            distance=m.distance,
            correlation=m.correlation,
        )
        for m in result.matches
    ]
    return SimilarityFindResponse(
        status="ok",
        query_length=result.query_length,
        matches=matches,
        best_distance=result.best_distance,
        best_correlation=result.best_correlation,
    )


@router.get(
    "/similarity/{symbol}",
    response_model=SimilarityInfoResponse,
    dependencies=[Depends(require_read_key)],
)
async def get_similarity_info(symbol: str, timeframe: str = "1m") -> SimilarityInfoResponse:
    """Liefert Similitäts-Infos für ein Symbol.

    ``available`` spiegelt die InfluxDB-Verfügbarkeit (Datenquelle für die
    Historie); ``window_size`` ist die Standard-Fenstergröße der Engine.
    ``timeframe`` wird gegen die unterstützten Werte validiert.
    """
    _require_quant_enabled()
    if timeframe not in quant_schema.SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported timeframe '{timeframe}'",
        )
    store = get_influx_store()
    available = await store.health_check()
    return SimilarityInfoResponse(
        symbol=symbol,
        available=available,
        window_size=SIMILARITY_DEFAULT_WINDOW_SIZE,
    )


# ---------------------------------------------------------------------------
# Forward-Outcome-Endpunkte (P6-3)
# ---------------------------------------------------------------------------


@router.post(
    "/outcomes/compute",
    response_model=OutcomeComputeResponse,
    dependencies=[Depends(require_trade_key)],
)
async def compute_outcomes(payload: OutcomeComputeRequest) -> OutcomeComputeResponse:
    """Berechnet Forward Outcomes mit dem ``ForwardOutcomeEngine``.

    Die Forward-Return-Statistik (Mean/Median, Hit Rate, Profit Factor,
    Expectancy, Std, Max-Gain/Loss) wird pro angefragtem Horizont mit dem
    deterministischen Engine (``quant/forward_outcomes.py``) berechnet.
    ``outcomes`` ist eine Map ``str(horizon)`` → Statistik. ``stored`` ist
    True, wenn ein verfügbarer ``ForwardOutcomeStore`` die Ergebnisse
    übernommen hat; ohne Store bleibt der Endpunkt funktional
    (``stored=false``).
    """
    _require_quant_enabled()
    engine = ForwardOutcomeEngine(horizons=payload.horizons)
    candle_dicts = _candles_to_outcome_input(payload.candles)
    result = engine.compute(
        candle_dicts,
        pattern_length=payload.pattern_length,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
    )

    store = get_outcome_store()
    stored = False
    if store is not None:
        # Der Store nutzt seine eigenen (Default-)Horizonte für die
        # Persistenz; die API-Antwort spiegelt die angefragten Horizonte.
        store_result = await store.compute_and_store(
            payload.symbol,
            payload.timeframe,
            candle_dicts,
            pattern_length=payload.pattern_length,
            exchange=payload.exchange,
        )
        stored = bool(store_result.stored)

    outcomes = {
        str(horizon): _outcome_stats_to_model(stats)
        for horizon, stats in result.outcomes.items()
    }
    return OutcomeComputeResponse(
        status="ok",
        symbol=payload.symbol,
        outcomes=outcomes,
        stored=stored,
    )


@router.get(
    "/outcomes/{symbol}",
    response_model=OutcomeQueryResponse,
    dependencies=[Depends(require_read_key)],
)
async def get_outcomes(
    symbol: str,
    timeframe: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
) -> OutcomeQueryResponse:
    """Liefert gespeicherte Outcomes für ein Symbol.

    ``timeframe`` wird gegen die unterstützten Werte validiert;
    ``start``/``end`` sind optionale UTC-Zeitgrenzen. Unbekannte Symbole
    liefern eine leere Liste (kein Fehler). Ohne ``ForwardOutcomeStore``
    (P6-2) ist die Liste immer leer.
    """
    _require_quant_enabled()
    if timeframe not in quant_schema.SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported timeframe '{timeframe}'",
        )
    # ForwardOutcomeStore erwartet UTC-ISO-Strings (naive API-Zeiten sind UTC).
    start_iso = start.isoformat() if start is not None else None
    end_iso = end.isoformat() if end is not None else None
    store = get_outcome_store()
    rows: list[dict[str, Any]] = []
    if store is not None:
        rows = await store.get_outcomes(symbol, timeframe, start_iso, end_iso)
    return OutcomeQueryResponse(symbol=symbol, outcomes=rows, count=len(rows))


# ---------------------------------------------------------------------------
# ML-Endpunkte (P7-3)
# ---------------------------------------------------------------------------


@router.post(
    "/ml/features",
    response_model=MLFeaturesResponse,
    dependencies=[Depends(require_trade_key)],
)
async def build_ml_features(payload: MLFeaturesRequest) -> MLFeaturesResponse:
    """Baut einen ML-Feature-Vektor mit dem ``MLFeatureBuilder``.

    Die Roh-Features werden deterministisch bereinigt (NaN/Inf →
    ``fill_nan``) und optional Z-Score-normalisiert. Reine In-Memory-
    Berechnung — es findet keine Persistenz und kein InfluxDB-Zugriff statt.
    """
    _require_quant_enabled()
    builder = MLFeatureBuilder(normalize=payload.normalize)
    result = builder.build(payload.symbol, payload.timeframe, payload.features)
    return MLFeaturesResponse(
        status="ok",
        symbol=result.symbol,
        features=result.features,
        feature_names=result.feature_names,
    )


@router.post(
    "/ml/importance",
    response_model=MLImportanceResponse,
    dependencies=[Depends(require_trade_key)],
)
async def compute_ml_importance(payload: MLImportanceRequest) -> MLImportanceResponse:
    """Berechnet Feature-Importance mit der ``FeatureImportanceEngine``.

    Importance = |Pearson-Korrelation| jedes Features gegen ``target``;
    Features werden absteigend gerankt, ``top_features`` enthält alle mit
    Importance ≥ ``threshold``, ``feature_groups`` die durchschnittliche
    Importance pro Namensgruppe (Präfix vor dem ersten ``_``). Reine
    In-Memory-Berechnung ohne Persistenz und InfluxDB-Zugriff.
    """
    _require_quant_enabled()
    engine = FeatureImportanceEngine(threshold=payload.threshold)
    result = engine.compute(payload.features, payload.target)
    return MLImportanceResponse(
        status="ok",
        features=[
            MLImportanceFeatureModel(
                name=f.name,
                importance=f.importance,
                correlation=f.correlation,
                rank=f.rank,
            )
            for f in result.features
        ],
        top_features=result.top_features,
        feature_groups=result.feature_groups,
    )


# ---------------------------------------------------------------------------
# Backtest-Endpunkte (P8-3)
# ---------------------------------------------------------------------------


@router.post(
    "/backtest/run",
    response_model=BacktestRunResponse,
    dependencies=[Depends(require_trade_key)],
)
async def run_backtest(payload: BacktestRunRequest) -> BacktestRunResponse:
    """Führt einen Backtest mit dem ``BacktestEngine`` aus und persistiert ihn.

    Die Strategie (``sma``/``rsi``) wird mit den übergebenen ``params``
    aus der Engine-Factory erzeugt; PnL, Win Rate, Drawdown, Sharpe Ratio
    und Profit Factor kommen aus dem deterministischen ``BacktestResult``.
    ``stored`` ist True, wenn ein verfügbarer ``BacktestStore`` (P8-2) das
    Ergebnis übernommen hat; ohne Store bleibt der Endpunkt funktional
    (``stored=false``).
    """
    _require_quant_enabled()
    engine = BacktestEngine(
        initial_capital=payload.initial_capital,
        risk_per_trade=payload.risk_per_trade,
        stop_loss_pct=payload.stop_loss_pct,
        take_profit_pct=payload.take_profit_pct,
    )
    strategy = _build_backtest_strategy(engine, payload.strategy, payload.params)
    candle_dicts = _candles_to_backtest_input(payload.candles)
    result = engine.run(
        candle_dicts,
        strategy,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
    )

    store = get_backtest_store()
    stored = False
    if store is not None:
        store_result = await store.run_and_store(
            payload.symbol,
            payload.timeframe,
            result,
            exchange=payload.exchange,
        )
        stored = bool(store_result.stored)

    return BacktestRunResponse(
        status="ok",
        symbol=payload.symbol,
        result=_backtest_result_to_model(result, payload.strategy, len(payload.candles)),
        stored=stored,
    )


@router.get(
    "/backtest/{symbol}",
    response_model=BacktestQueryResponse,
    dependencies=[Depends(require_read_key)],
)
async def get_backtests(
    symbol: str,
    timeframe: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
) -> BacktestQueryResponse:
    """Liefert gespeicherte Backtests für ein Symbol.

    ``timeframe`` wird gegen die unterstützten Werte validiert;
    ``start``/``end`` sind optionale UTC-Zeitgrenzen. Unbekannte Symbole
    liefern eine leere Liste (kein Fehler). Ohne ``BacktestStore`` (P8-2)
    ist die Liste immer leer.
    """
    _require_quant_enabled()
    if timeframe not in quant_schema.SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported timeframe '{timeframe}'",
        )
    # BacktestStore erwartet UTC-ISO-Strings (naive API-Zeiten sind UTC).
    start_iso = start.isoformat() if start is not None else None
    end_iso = end.isoformat() if end is not None else None
    store = get_backtest_store()
    rows: list[dict[str, Any]] = []
    if store is not None:
        rows = await store.get_backtests(symbol, timeframe, start_iso, end_iso)
    return BacktestQueryResponse(symbol=symbol, results=rows, count=len(rows))


# ---------------------------------------------------------------------------
# Shadow-Loop-Status (P9-3)
# ---------------------------------------------------------------------------


@router.get(
    "/shadow/status",
    response_model=ShadowStatusResponse,
    dependencies=[Depends(require_read_key)],
)
def shadow_status() -> ShadowStatusResponse:
    """Liefert den Shadow-Loop-Integrationsstatus der Quant-Plattform.

    ``integration_active`` spiegelt das Quant-Feature-Flag
    (``influxdb_enabled``), ``quant_engines`` listet die Quant-Module,
    deren Ergebnisse der ``EvidenceAggregator`` als Evidence aufnehmen
    kann, und ``last_evidence`` die jeweils letzten pro Engine
    aggregierten Einträge. Reiner Status-Endpunkt wie
    ``GET /quant/status``: kein InfluxDB-Zugriff, kein Fail-closed-Guard.
    """
    settings = get_settings()
    aggregator = get_evidence_aggregator()
    return ShadowStatusResponse(
        status="ok",
        integration_active=settings.influxdb_enabled,
        quant_engines=list(SHADOW_QUANT_ENGINES),
        last_evidence=_last_evidence_from_aggregator(aggregator),
    )


# ---------------------------------------------------------------------------
# Performance-Endpunkte (P10-3)
# ---------------------------------------------------------------------------


@router.get(
    "/perf/cache-stats",
    response_model=PerfCacheStatsResponse,
    dependencies=[Depends(require_read_key)],
)
def perf_cache_stats() -> PerfCacheStatsResponse:
    """Liefert die Statistiken des geteilten ``FeatureCache`` (P10-3).

    ``hits``/``misses`` zählen die Cache-Zugriffe, ``size`` die aktuell
    gespeicherten Einträge und ``hit_rate`` den Trefferquotienten
    (``0.0`` ohne Zugriffe). Reiner In-Memory-Status-Endpunkt wie
    ``GET /quant/shadow/status``: kein InfluxDB-Zugriff, kein
    Fail-closed-Guard.
    """
    stats = get_feature_cache().stats()
    return PerfCacheStatsResponse(
        status="ok",
        cache=PerfCacheStatsModel(
            hits=int(stats.hits),
            misses=int(stats.misses),
            size=int(stats.size),
            hit_rate=float(stats.hit_rate),
        ),
    )


@router.get(
    "/perf/batch-status",
    response_model=PerfBatchStatusResponse,
    dependencies=[Depends(require_read_key)],
)
def perf_batch_status() -> PerfBatchStatusResponse:
    """Liefert den Gesamtstatus des geteilten ``BatchProcessor`` (P10-3).

    ``total_jobs`` zählt alle bekannten Jobs; ``completed``/``running``/
    ``pending`` die Jobs nach Status (``failed`` zählt nur in
    ``total_jobs``). Reiner In-Memory-Status-Endpunkt wie
    ``GET /quant/shadow/status``: kein InfluxDB-Zugriff, kein
    Fail-closed-Guard.
    """
    status = get_batch_processor().get_status()
    return PerfBatchStatusResponse(
        status="ok",
        batch=PerfBatchStatusModel(
            total_jobs=int(status.total_jobs),
            completed=int(status.completed_jobs),
            running=int(status.running_jobs),
            pending=int(status.pending_jobs),
        ),
    )


@router.post(
    "/validate",
    response_model=QuantValidateResponse,
    dependencies=[Depends(require_trade_key)],
)
def validate_input(payload: QuantValidateRequest) -> QuantValidateResponse:
    """Validiert Eingabedaten mit dem deterministischen ``Validator`` (P11-3).

    ``type='candle'`` prüft ein Kerzen-Dict (OHLC-Konsistenz, positive
    Preise, nicht-negatives Volume); ``type='features'`` prüft eine
    Feature-Map (NaN/Inf/Range, leere Map → Warning). Ungültige Daten
    werden mit ``valid=false`` + Fehlerliste gemeldet (Status 200) —
    der Endpunkt dient als reine Validierungsschnittstelle und
    verändert nichts (kein Store-Zugriff, kein Fail-closed-Guard).
    """
    validator = get_validator()
    if payload.type == "candle":
        type_errors = _numeric_value_errors(payload.data, _VALIDATE_CANDLE_FIELDS)
        result = (
            ValidationResult(valid=False, errors=type_errors)
            if type_errors
            else validator.validate_candle(payload.data)
        )
    else:
        type_errors = _numeric_value_errors(payload.data, tuple(payload.data))
        result = (
            ValidationResult(valid=False, errors=type_errors)
            if type_errors
            else validator.validate_features(payload.data)
        )
    return QuantValidateResponse(
        status="ok", valid=result.valid, errors=result.errors, warnings=result.warnings
    )
