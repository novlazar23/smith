"""Shadow-Loop-Integrationstests der Quant-Pipeline (P9-4).

Verifiziert die Zusammensetzung

    FeatureEngine / AnomalyDetector / RegimeDetector → EvidenceAggregator

ausschließlich mit Mocks: InfluxDB wird durch ``unittest.mock`` ersetzt
(``MagicMock(spec=InfluxDBStore)``) — kein Netzwerk, kein Docker, keine
echte InfluxDB. Die Quant-Engines selbst sind reine Standardbibliothek
(deterministisch), daher laufen sie echt; gemockt ist nur die
Persistenzebene.

Ketten je Test:

1. ``test_feature_engine_to_evidence`` — Features → Aggregator → Summary.
2. ``test_anomaly_detection_to_evidence`` — Anomalien → Aggregator →
   ``summary["anomaly_count"]``.
3. ``test_regime_detection_to_evidence`` — Regime → Aggregator →
   ``summary["current_regime"]``.
4. ``test_full_pipeline_aggregation`` — alle drei Quellen → vollständige
   Summary inkl. Prioritäten und Konfidenz.
5. ``test_full_pipeline_with_mocked_influxdb`` — Stores (Feature/Anomaly/
   Regime) auf gemockter InfluxDB → Aggregator → vollständige Summary.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.anomaly_detection import AnomalyDetector
from trading_harness.quant.anomaly_store import AnomalyStore
from trading_harness.quant.evidence_aggregator import AggregatedEvidence, EvidenceAggregator
from trading_harness.quant.feature_store import FeatureStore
from trading_harness.quant.features import FeatureEngine
from trading_harness.quant.influxdb_client import InfluxDBStore
from trading_harness.quant.regime_detection import RegimeDetector
from trading_harness.quant.regime_store import RegimeStore

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
BASE = datetime(2026, 8, 25, tzinfo=UTC)

# FeatureEngine.compute liefert genau diese sechs Indikatoren (Phase 2).
_FEATURE_KEYS: tuple[str, ...] = ("rsi", "macd", "bollinger", "atr", "volatility", "vwap")


# ----------------------------------------------------------------------
# Hilfsfunktionen (deterministische, synthetische Kerzenreihen)
# ----------------------------------------------------------------------


def _ts(index: int) -> str:
    """ISO-8601-Zeitstempel (UTC, 'Z') für Kerze ``index``."""
    return (BASE + timedelta(minutes=index)).isoformat().replace("+00:00", "Z")


def make_uptrend_candles(count: int = 60) -> list[dict[str, Any]]:
    """Linearer Aufwärtstrend (60 Kerzen > sma_slow=50) → ``strong_bull``."""
    candles: list[dict[str, Any]] = []
    for index in range(count):
        close = 100.0 + 2.0 * index
        candles.append(
            {
                "time": _ts(index),
                "open": close - 2.0,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 100.0,
            }
        )
    return candles


def make_spike_candles() -> list[dict[str, Any]]:
    """20 flache Kerzen + 1 Kerze mit Preis- und Volumensprung.

    Die flache Baseline (Varianz 0) erzeugt bewusst keine Anomalien; die
    letzte Kerze (close 100 → 120, volume 100 → 1000) erzeugt
    deterministisch genau ``price_shock`` + ``volume_spike``.
    """
    flat = [
        {
            "time": _ts(index),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 100.0,
        }
        for index in range(20)
    ]
    spike = {
        "time": _ts(20),
        "open": 100.0,
        "high": 121.0,
        "low": 99.5,
        "close": 120.0,
        "volume": 1000.0,
    }
    return [*flat, spike]


def make_influx_store() -> MagicMock:
    """Gemockter InfluxDBStore (unittest.mock) — keine echte Verbindung."""
    store = MagicMock(spec=InfluxDBStore)
    store.is_available = True
    return store


# ----------------------------------------------------------------------
# 1. FeatureEngine → EvidenceAggregator
# ----------------------------------------------------------------------


def test_feature_engine_to_evidence() -> None:
    """Features berechnen, in den Aggregator geben, Summary verifizieren."""
    engine = FeatureEngine()
    aggregator = EvidenceAggregator()
    candles = make_uptrend_candles(60)

    features = engine.compute(candles)
    # 60 Kerzen → alle sechs Indikatoren berechnet (keine None).
    for key in _FEATURE_KEYS:
        assert features[key] is not None, key
    assert features["feature_count"] == len(_FEATURE_KEYS)

    aggregator.add_entry("features", dict(features), confidence=1.0)

    result = aggregator.aggregate(SYMBOL, TIMEFRAME)
    assert isinstance(result, AggregatedEvidence)
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert aggregator.has_source("features")
    # Der Eintrag trägt die Engine-Daten und die Quell-Priorität (4).
    entry = result.entries["features"]
    assert entry.data["rsi"] == features["rsi"]
    assert entry.data["macd"] == features["macd"]
    assert entry.priority == EvidenceAggregator.SOURCE_PRIORITIES["features"]
    # Die Summary zählt die übergebenen Feature-Felder (Indikatoren + Meta).
    assert set(result.summary["sources"]) == {"features"}
    assert result.summary["total_entries"] == 1
    assert result.summary["feature_count"] == len(features)
    assert result.total_confidence == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 2. AnomalyDetector → EvidenceAggregator
# ----------------------------------------------------------------------


def test_anomaly_detection_to_evidence() -> None:
    """Anomalien erkennen, in den Aggregator geben, anomaly_count verifizieren."""
    detector = AnomalyDetector()
    aggregator = EvidenceAggregator()
    candles = make_spike_candles()

    anomalies = detector.detect(candles)
    assert len(anomalies) == 2, "genau price_shock + volume_spike"
    assert {a.anomaly_type for a in anomalies} == {"price_shock", "volume_spike"}

    aggregator.add_entry(
        "anomalies",
        {"count": len(anomalies), "anomalies": [asdict(a) for a in anomalies]},
    )

    result = aggregator.aggregate(SYMBOL, TIMEFRAME)
    assert result.summary["anomaly_count"] == 2
    assert result.summary["total_entries"] == 1
    assert set(result.summary["sources"]) == {"anomalies"}
    # Anomalien haben die höchste Quell-Priorität (10) → high-priority.
    entry = result.entries["anomalies"]
    assert entry.priority == EvidenceAggregator.SOURCE_PRIORITIES["anomalies"]
    assert result.high_priority_count == 1
    assert result.total_confidence == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 3. RegimeDetector → EvidenceAggregator
# ----------------------------------------------------------------------


def test_regime_detection_to_evidence() -> None:
    """Regime erkennen, in den Aggregator geben, current_regime verifizieren."""
    detector = RegimeDetector()
    aggregator = EvidenceAggregator()
    candles = make_uptrend_candles(60)

    regime = detector.detect(candles)
    assert regime.regime == "strong_bull"
    assert 0.0 < regime.confidence <= 1.0

    aggregator.add_entry(
        "regime",
        {
            "regime": regime.regime,
            "confidence": regime.confidence,
            "duration": regime.duration,
        },
        confidence=regime.confidence,
    )

    result = aggregator.aggregate(SYMBOL, TIMEFRAME)
    assert result.summary["current_regime"] == "strong_bull"
    assert result.summary["total_entries"] == 1
    assert set(result.summary["sources"]) == {"regime"}
    entry = result.entries["regime"]
    assert entry.priority == EvidenceAggregator.SOURCE_PRIORITIES["regime"]
    assert entry.data["regime"] == "strong_bull"
    assert result.total_confidence == pytest.approx(regime.confidence)


# ----------------------------------------------------------------------
# 4. Vollständige Pipeline: alle Quellen → Aggregator
# ----------------------------------------------------------------------


def test_full_pipeline_aggregation() -> None:
    """Alle drei Quellen zusammenführen und die vollständige Summary prüfen."""
    engine = FeatureEngine()
    anomaly_detector = AnomalyDetector()
    regime_detector = RegimeDetector()
    aggregator = EvidenceAggregator()

    features = engine.compute(make_uptrend_candles(60))
    aggregator.add_entry("features", dict(features), confidence=1.0)

    anomalies = anomaly_detector.detect(make_spike_candles())
    aggregator.add_entry(
        "anomalies",
        {"count": len(anomalies), "anomalies": [asdict(a) for a in anomalies]},
    )

    regime = regime_detector.detect(make_uptrend_candles(60))
    aggregator.add_entry(
        "regime",
        {"regime": regime.regime, "confidence": regime.confidence, "duration": regime.duration},
        confidence=regime.confidence,
    )

    result = aggregator.aggregate(SYMBOL, TIMEFRAME)
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert set(result.entries) == {"features", "anomalies", "regime"}
    assert len(result.entries) == 3
    assert result.summary["total_entries"] == 3
    assert result.summary["current_regime"] == "strong_bull"
    assert result.summary["anomaly_count"] == 2
    assert result.summary["feature_count"] == len(features)
    # High-priority: anomalies (10) und regime (8) — features (4) nicht.
    assert result.high_priority_count == 2
    expected_confidence = (1.0 + 1.0 + regime.confidence) / 3
    assert result.total_confidence == pytest.approx(expected_confidence)


# ----------------------------------------------------------------------
# 5. Vollständige Pipeline mit gemockter InfluxDB (Store-Ebene)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_influxdb() -> None:
    """Stores auf gemockter InfluxDB → Aggregator → vollständige Summary."""
    influx_store = make_influx_store()
    feature_store = FeatureStore(influx_store, FeatureEngine())
    anomaly_store = AnomalyStore(influx_store, AnomalyDetector())
    regime_store = RegimeStore(influx_store, RegimeDetector())

    trend_candles = make_uptrend_candles(60)
    spike_candles = make_spike_candles()

    feature_result = await feature_store.compute_and_store(SYMBOL, TIMEFRAME, trend_candles)
    anomaly_result = await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, spike_candles)
    regime_result = await regime_store.detect_and_store(SYMBOL, TIMEFRAME, trend_candles)

    # Alle Stores haben in die (gemockte) InfluxDB geschrieben.
    assert feature_result.stored is True
    assert anomaly_result.stored is True
    assert anomaly_result.anomalies_found == 2
    assert regime_result.stored is True
    assert regime_result.regime == "strong_bull"
    measurements = {
        call.kwargs["measurement"] for call in influx_store.write_points.await_args_list
    }
    assert measurements == {
        quant_schema.FEATURE_MEASUREMENT,
        quant_schema.ANOMALY_MEASUREMENT,
        quant_schema.REGIME_MEASUREMENT,
    }

    # Store-Ergebnisse in den Aggregator führen.
    aggregator = EvidenceAggregator()
    features = FeatureEngine().compute(trend_candles)
    aggregator.add_entry("features", dict(features), confidence=1.0)
    aggregator.add_entry("anomalies", {"count": anomaly_result.anomalies_found})
    aggregator.add_entry(
        "regime",
        {"regime": regime_result.regime, "confidence": regime_result.confidence},
        confidence=regime_result.confidence,
    )

    result = aggregator.aggregate(SYMBOL, TIMEFRAME)
    assert set(result.entries) == {"features", "anomalies", "regime"}
    assert result.summary["total_entries"] == 3
    assert result.summary["current_regime"] == "strong_bull"
    assert result.summary["anomaly_count"] == 2
    assert result.summary["feature_count"] == len(features)
