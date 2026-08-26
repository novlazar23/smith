"""Shadow-Loop Quant Integration (Quant-Plattform, Phase 9, P9-1).

Integrationssuite für das P9-1-Kontrakt (siehe
``docs/quant-platform-phase09-plan.md``): jede Shadow-Loop-Iteration erfasst
pro Symbol/Tick Quant-Evidence — Feature-Extraktion (``FeatureEngine``),
Anomalie-Erkennung (``AnomalyDetector``) und Regime-Erkennung
(``RegimeDetector``) — und speist alle Quellen in den gemeinsamen
``EvidenceAggregator``.

``shadow_trading_loop.py`` bleibt in dieser Suite bewusst unverändert
(P9-1-Aufgabenrestriktion). Der Loop-Hook wird deshalb als funktionaler
Double (``ShadowQuantEvidenceHook``) modelliert — analog zum etablierten
Muster der Phasen-4-Integrationstests (``test_quant_regime_integration.py``):
Die Quant-Engines werden durch ``unittest.mock``-Mocks ersetzt (``spec``-gebunden,
damit nur der echte Kontrakt adressierbar ist); der ``EvidenceAggregator``
ist die echte P9-2-Implementierung.

Semantik des Hooks (analog zum bestehenden Quant-Hook des Loops, P1-7
``ohlcv_ingestion``): best-effort — ein Fehler in einer einzelnen
Quant-Quelle überspringt nur diese Quelle und unterbricht die
Iteration niemals.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from trading_harness.quant.anomaly_detection import Anomaly, AnomalyDetector
from trading_harness.quant.evidence_aggregator import (
    AggregatedEvidence,
    EvidenceAggregator,
)
from trading_harness.quant.features import FeatureEngine
from trading_harness.quant.regime_detection import RegimeDetector, RegimeResult

logger = logging.getLogger(__name__)

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"

FEATURE_DATA: dict[str, Any] = {
    "rsi": 62.5,
    "macd": {"macd": 1.2, "signal": 0.8, "histogram": 0.4},
    "bollinger": {
        "middle": 101.0,
        "upper": 103.0,
        "lower": 99.0,
        "std_dev": 1.0,
        "bandwidth": 0.04,
    },
    "atr": 1.25,
    "volatility": 0.012,
    "vwap": 100.8,
    "feature_count": 6,
    "computation_time_ms": 0.42,
}

REGIME_RESULT = RegimeResult(
    regime="strong_bull",
    confidence=0.9,
    duration=12,
    indicators={"adx": 34.0, "sma_fast": 101.5, "sma_slow": 100.2},
)


# ----------------------------------------------------------------------
# P9-1-Integrationskontrakt (Shadow-Loop-Hook)
# ----------------------------------------------------------------------


class ShadowQuantEvidenceHook:
    """P9-1-Integrationskontrakt: Quant-Evidence pro Shadow-Loop-Tick.

    Modelliert den Hook, den Phase 9 in den Shadow-Trading-Loop einführt
    (``docs/quant-platform-phase09-plan.md``, P9-1): pro Symbol und
    Iteration laufen Feature-Extraktion, Anomalie-Erkennung und
    Regime-Erkennung; ihre Ergebnisse werden als Evidence-Einträge in den
    gemeinsamen ``EvidenceAggregator`` übernommen, der sie für die
    Schatten-Iteration als strukturiertes Dict bereitstellt.

    Best-effort-Semantik (analog zum P1-7-``ohlcv_ingestion``-Hook des
    Loops): ein Fehler in einer einzelnen Quant-Quelle überspringt nur
    diese Quelle und unterbricht die Iteration niemals.
    """

    def __init__(
        self,
        *,
        feature_engine: Any | None = None,
        anomaly_detector: Any | None = None,
        regime_detector: Any | None = None,
        aggregator: EvidenceAggregator | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._feature_engine = feature_engine if feature_engine is not None else FeatureEngine()
        self._anomaly_detector = anomaly_detector if anomaly_detector is not None else AnomalyDetector()
        self._regime_detector = regime_detector if regime_detector is not None else RegimeDetector()
        self._aggregator = aggregator if aggregator is not None else EvidenceAggregator()
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))

    @property
    def aggregator(self) -> EvidenceAggregator:
        return self._aggregator

    def process_tick(
        self, symbol: str, timeframe: str, candles: list[dict[str, Any]]
    ) -> AggregatedEvidence:
        """Eine Shadow-Loop-Iteration: alle Quant-Quellen -> Aggregator."""
        self._aggregator.clear()
        timestamp = self._clock().isoformat()

        try:
            features = self._feature_engine.compute(candles)
        except Exception:  # noqa: BLE001 — P9-1: Quant-Quellen sind best-effort, nie iterationskritisch
            logger.warning("SHADOW_QUANT_FEATURES_FAILED symbol=%s", symbol)
        else:
            self._aggregator.add_entry("features", features, timestamp=timestamp)

        try:
            anomalies = self._anomaly_detector.detect(candles)
        except Exception:  # noqa: BLE001 — P9-1: Quant-Quellen sind best-effort, nie iterationskritisch
            logger.warning("SHADOW_QUANT_ANOMALIES_FAILED symbol=%s", symbol)
        else:
            self._aggregator.add_entry(
                "anomalies",
                {"count": len(anomalies), "anomalies": [asdict(item) for item in anomalies]},
                timestamp=timestamp,
            )

        try:
            regime = self._regime_detector.detect(candles)
        except Exception:  # noqa: BLE001 — P9-1: Quant-Quellen sind best-effort, nie iterationskritisch
            logger.warning("SHADOW_QUANT_REGIME_FAILED symbol=%s", symbol)
        else:
            self._aggregator.add_entry(
                "regime",
                {
                    "regime": regime.regime,
                    "confidence": regime.confidence,
                    "duration": regime.duration,
                },
                timestamp=timestamp,
            )

        return self._aggregator.aggregate(symbol, timeframe)


class AdvancingClock:
    """Deterministische Uhr: liefert pro Aufruf start, start+step, start+2*step, ..."""

    def __init__(self, start: datetime, step: timedelta) -> None:
        self._current = start
        self._step = step

    def __call__(self) -> datetime:
        value = self._current
        self._current = value + self._step
        return value


# ----------------------------------------------------------------------
# Test-Helfer
# ----------------------------------------------------------------------


def make_anomalies(count: int = 3) -> list[Anomaly]:
    """Deterministische Anomalie-Liste (Mock-Rückgabewert des Detektors)."""
    return [
        Anomaly(
            timestamp=f"2026-01-01T00:{index:02d}:00+00:00",
            symbol=SYMBOL,
            anomaly_type="price_shock",
            severity=0.8,
            feature="close",
            value=0.12,
            zscore=6.1,
            threshold=3.0,
        )
        for index in range(count)
    ]


def sample_candles() -> list[dict[str, Any]]:
    """Kleine deterministische Kerzenreihe (3 Kerzen)."""
    return [
        {
            "time": "2026-01-01T00:00:00+00:00",
            "open": 100.0,
            "high": 102.0,
            "low": 99.5,
            "close": 101.0,
            "volume": 1000.0,
        },
        {
            "time": "2026-01-01T00:01:00+00:00",
            "open": 101.0,
            "high": 103.5,
            "low": 100.8,
            "close": 103.0,
            "volume": 1200.0,
        },
        {
            "time": "2026-01-01T00:02:00+00:00",
            "open": 103.0,
            "high": 104.0,
            "low": 102.5,
            "close": 103.8,
            "volume": 900.0,
        },
    ]


def make_mock_engines(
    feature_data: dict[str, Any] | None = None,
    anomalies: list[Anomaly] | None = None,
    regime: RegimeResult | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Alle drei Quant-Engines als Mocks (``spec`` hält den echten Kontrakt fest)."""
    feature_engine = MagicMock(spec=FeatureEngine)
    feature_engine.compute.return_value = FEATURE_DATA if feature_data is None else feature_data
    anomaly_detector = MagicMock(spec=AnomalyDetector)
    anomaly_detector.detect.return_value = make_anomalies() if anomalies is None else anomalies
    regime_detector = MagicMock(spec=RegimeDetector)
    regime_detector.detect.return_value = REGIME_RESULT if regime is None else regime
    return feature_engine, anomaly_detector, regime_detector


def make_hook(
    *,
    engines: tuple[MagicMock, MagicMock, MagicMock] | None = None,
    aggregator: EvidenceAggregator | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ShadowQuantEvidenceHook:
    """Baut den Hook; ohne explizite Engines laufen alle drei als Mocks."""
    feature_engine, anomaly_detector, regime_detector = (
        engines if engines is not None else make_mock_engines()
    )
    return ShadowQuantEvidenceHook(
        feature_engine=feature_engine,
        anomaly_detector=anomaly_detector,
        regime_detector=regime_detector,
        aggregator=aggregator if aggregator is not None else EvidenceAggregator(),
        clock=clock if clock is not None else (lambda: datetime.now(UTC)),
    )


# ----------------------------------------------------------------------
# P9-1: Shadow-Loop-Tick -> Quant-Evidence
# ----------------------------------------------------------------------


def test_evidence_aggregator_collects_all_sources() -> None:
    """Alle drei (gemockten) Quant-Engines liefern ihre Evidence in den echten Aggregator."""
    engines = make_mock_engines()
    hook = make_hook(engines=engines)
    candles = sample_candles()

    result = hook.process_tick(SYMBOL, TIMEFRAME, candles)

    assert hook.aggregator.entry_count == 3
    assert set(hook.aggregator.sources) == {"features", "anomalies", "regime"}
    assert isinstance(result, AggregatedEvidence)
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert set(result.entries) == {"features", "anomalies", "regime"}
    # Die Aggregator-Einträge tragen die (gemockten) Engine-Outputs 1:1.
    assert result.entries["features"].data == FEATURE_DATA
    assert result.entries["anomalies"].data["count"] == 3
    assert result.entries["anomalies"].data["anomalies"] == [asdict(a) for a in make_anomalies()]
    assert result.entries["regime"].data == {
        "regime": REGIME_RESULT.regime,
        "confidence": REGIME_RESULT.confidence,
        "duration": REGIME_RESULT.duration,
    }
    assert set(result.summary["sources"]) == {"features", "anomalies", "regime"}
    assert result.summary["total_entries"] == 3


def test_shadow_loop_calls_feature_engine() -> None:
    """Der Shadow-Loop-Tick triggert Feature-Extraktion auf die Kerzenreihe."""
    feature_engine, anomaly_detector, regime_detector = make_mock_engines()
    hook = make_hook(engines=(feature_engine, anomaly_detector, regime_detector))
    candles = sample_candles()

    hook.process_tick(SYMBOL, TIMEFRAME, candles)

    feature_engine.compute.assert_called_once_with(candles)


def test_shadow_loop_calls_anomaly_detection() -> None:
    """Der Shadow-Loop-Tick triggert Anomalie-Erkennung auf die Kerzenreihe."""
    feature_engine, anomaly_detector, regime_detector = make_mock_engines()
    hook = make_hook(engines=(feature_engine, anomaly_detector, regime_detector))
    candles = sample_candles()

    hook.process_tick(SYMBOL, TIMEFRAME, candles)

    anomaly_detector.detect.assert_called_once_with(candles)


def test_shadow_loop_calls_regime_detection() -> None:
    """Der Shadow-Loop-Tick triggert Regime-Erkennung auf die Kerzenreihe."""
    feature_engine, anomaly_detector, regime_detector = make_mock_engines()
    hook = make_hook(engines=(feature_engine, anomaly_detector, regime_detector))
    candles = sample_candles()

    hook.process_tick(SYMBOL, TIMEFRAME, candles)

    regime_detector.detect.assert_called_once_with(candles)


def test_evidence_summary_includes_regime() -> None:
    """Das erkannte Regime erscheint im Aggregator-Summary (``current_regime``)."""
    engines = make_mock_engines()
    hook = make_hook(engines=engines)

    result = hook.process_tick(SYMBOL, TIMEFRAME, sample_candles())

    assert result.summary["current_regime"] == "strong_bull"
    # Die "regime"-Quelle ist dafür verantwortlich (kein Zufallstreffer aus anderem Data).
    assert "regime" in result.entries
    assert result.entries["regime"].data["regime"] == "strong_bull"


def test_evidence_summary_includes_anomaly_count() -> None:
    """Die Anomalie-Anzahl erscheint im Aggregator-Summary (``anomaly_count``)."""
    engines = make_mock_engines(anomalies=make_anomalies(5))
    hook = make_hook(engines=engines)

    result = hook.process_tick(SYMBOL, TIMEFRAME, sample_candles())

    assert result.summary["anomaly_count"] == 5
    assert result.entries["anomalies"].data["count"] == 5


def test_empty_evidence_handled() -> None:
    """Leere Evidence (keine Einträge bzw. leere Engine-Outputs) führt nicht zu einem Crash."""
    # (a) Frischer Aggregator ohne Einträge: leeres, vollständiges Aggregat.
    empty = EvidenceAggregator().aggregate(SYMBOL, TIMEFRAME)
    assert isinstance(empty, AggregatedEvidence)
    assert empty.entries == {}
    assert empty.summary == {}
    assert empty.total_confidence == 0.0
    assert empty.high_priority_count == 0

    # (b) Tick ohne verwertbare Engine-Outputs (leere Features, keine Anomalien,
    #     unzureichende Historie für das Regime) — ebenfalls kein Fehler.
    feature_engine, anomaly_detector, regime_detector = make_mock_engines(
        feature_data={},
        anomalies=[],
        regime=RegimeResult(
            regime="range",
            confidence=0.5,
            duration=0,
            indicators={"reason": "insufficient_data"},
        ),
    )
    hook = make_hook(engines=(feature_engine, anomaly_detector, regime_detector))

    result = hook.process_tick(SYMBOL, TIMEFRAME, [])

    assert result.summary["anomaly_count"] == 0
    assert result.summary["feature_count"] == 0
    assert result.summary["current_regime"] == "range"
    assert set(result.entries) == {"features", "anomalies", "regime"}


def test_evidence_timestamps_updated() -> None:
    """Evidence-Timestamps folgen dem Tick-Zeitpunkt und werden pro Tick aktualisiert."""
    start = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)
    step = timedelta(minutes=1)
    clock = AdvancingClock(start, step)
    engines = make_mock_engines()
    hook = make_hook(engines=engines, clock=clock)
    candles = sample_candles()

    first = hook.process_tick(SYMBOL, TIMEFRAME, candles)
    expected_first = start.isoformat()
    for entry in first.entries.values():
        assert entry.timestamp == expected_first
    parsed = datetime.fromisoformat(first.entries["features"].timestamp)
    assert parsed.tzinfo is not None  # UTC-ISO-Zeitstempel, nicht naiv

    second = hook.process_tick(SYMBOL, TIMEFRAME, candles)
    expected_second = (start + step).isoformat()
    assert expected_second != expected_first
    for entry in second.entries.values():
        # Der zweite Tick ersetzt den Stempel — nichts Altbestandenes bleibt liegen.
        assert entry.timestamp == expected_second

    # Der Aggregat-Zeitstempel ist ebenfalls ein gültiger (UTC-)ISO-Zeitstempel.
    datetime.fromisoformat(second.timestamp)


def test_engine_failure_does_not_break_tick() -> None:
    """Ein fehlerhafter Quant-Engine unterbricht den Tick nicht (best-effort-Semantik)."""
    feature_engine, anomaly_detector, regime_detector = make_mock_engines()
    feature_engine.compute.side_effect = RuntimeError("feature backend down")
    hook = make_hook(engines=(feature_engine, anomaly_detector, regime_detector))

    result = hook.process_tick(SYMBOL, TIMEFRAME, sample_candles())

    assert isinstance(result, AggregatedEvidence)
    assert set(result.entries) == {"anomalies", "regime"}
    assert not hook.aggregator.has_source("features")
    # Die übrigen Quellen liefern ihre Evidence unverändert.
    assert result.summary["current_regime"] == "strong_bull"
    assert result.summary["anomaly_count"] == 3


def test_evidence_priorities_follow_source_policy() -> None:
    """Quellen-Prioritäten und High-Priority-Zählung folgen ``SOURCE_PRIORITIES``."""
    hook = make_hook()  # alle drei Engines als Mocks (Default)

    result = hook.process_tick(SYMBOL, TIMEFRAME, sample_candles())

    priorities = {name: entry.priority for name, entry in result.entries.items()}
    assert priorities["anomalies"] == EvidenceAggregator.SOURCE_PRIORITIES["anomalies"]
    assert priorities["regime"] == EvidenceAggregator.SOURCE_PRIORITIES["regime"]
    assert priorities["features"] == EvidenceAggregator.SOURCE_PRIORITIES["features"]
    assert priorities["anomalies"] > priorities["regime"] > priorities["features"]
    # anomalies (10) und regime (8) zählen als High-Priority (>= 7), features (4) nicht.
    assert result.high_priority_count == 2
