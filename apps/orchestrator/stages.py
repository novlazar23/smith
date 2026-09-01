"""Stage-Implementierungen — die 18 Graphknoten (§11)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from apps.orchestrator.graph import StageManager, TradingGraphState
from apps.orchestrator.stages_enum import AnalysisStage


class MarketDataProvider(Protocol):
    """Protokoll für den Datenprovider in Stage 11.2.

    Liefert Kerzen, Trades und Orderbook als Snapshot bis zum
    angegebenen Zeitpunkt (Availability Time).
    """

    def fetch_candles(self, instrument: str, as_of: datetime) -> list[Any]:
        """Kerzen bis as_of."""
        ...

    def fetch_trades(self, instrument: str, as_of: datetime) -> list[Any]:
        """Trades bis as_of."""
        ...

    def fetch_orderbook(self, instrument: str, as_of: datetime) -> dict[str, Any]:
        """Orderbook-Snapshot bis as_of."""
        ...


class FeatureProvider(Protocol):
    """Protokoll für den Feature-Provider in Stage 11.4."""

    def compute(self, state: TradingGraphState) -> dict[str, Any]:
        """Berechnet die Features für den aktuellen Graphzustand."""
        ...


class RegimeProvider(Protocol):
    """Protokoll für den Regime-Provider in Stage 11.5."""

    def classify(self, state: TradingGraphState) -> dict[str, Any]:
        """Klassifiziert das Markt-Regime für den aktuellen Graphzustand."""
        ...


def create_run(
    request: dict[str, Any],
) -> tuple[TradingGraphState, StageManager]:
    """11.1 create_run — Run-ID, Konfiguration, Modell/Prompt-Versionen.

    Args:
        request: Analyse-Anfrage mit instrument, timeframe, analysis_time, etc.

    Returns:
        (TradingGraphState, StageManager) — initialer Zustand und Manager.
    """
    run_id = request.get("run_id", str(uuid.uuid4()))
    instrument = request.get("instrument", "UNKNOWN")
    analysis_time = request.get("analysis_time", datetime.now(UTC))
    model_version = request.get("model_version", "1.0.0")
    prompt_version = request.get("prompt_version", "1.0.0")

    state = TradingGraphState(
        run_id=run_id,
        instrument=instrument,
        request=request,
        analysis_time=analysis_time,
        model_version=model_version,
        prompt_version=prompt_version,
        started_at=datetime.now(UTC),
        status="running",
        current_stage=AnalysisStage.CREATE_RUN.value,
    )

    manager = StageManager()
    manager.transition(
        AnalysisStage.CREATE_RUN,
        inputs={"request_keys": list(request.keys())},
        outputs={"run_id": run_id},
    )

    return state, manager


def build_market_snapshot(
    state: TradingGraphState,
    manager: StageManager,
    data_provider: MarketDataProvider | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.2 build_market_snapshot — Daten bis analysis_time laden.

    Lädt alle Marktdaten bis analysis_time unter Berücksichtigung
    von Availability Time. Snapshot wird hash-gesichert.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager für Transition.
        data_provider: Datenprovider für Abfrage (optional, Mock in Tests).

    Returns:
        (TradingGraphState, StageManager) — aktualisierter Zustand.
    """
    analysis_time = state.analysis_time
    instrument = state.instrument

    # Daten laden (Mock wenn kein Provider)
    if data_provider is not None:
        candles = data_provider.fetch_candles(instrument, analysis_time)
        trades = data_provider.fetch_trades(instrument, analysis_time)
        orderbook = data_provider.fetch_orderbook(instrument, analysis_time)
    else:
        candles = []
        trades = []
        orderbook = {}

    snapshot_data = {
        "instrument": instrument,
        "analysis_time": analysis_time.isoformat(),
        "candle_count": len(candles),
        "trade_count": len(trades),
        "orderbook_depth": len(orderbook.get("bids", [])) + len(orderbook.get("asks", [])),
    }

    # Hash sichern
    snapshot_hash = hashlib.sha256(
        f"{instrument}:{analysis_time.isoformat()}:{len(candles)}".encode()
    ).hexdigest()

    state = state.model_copy(update={
        "market_snapshot_id": snapshot_hash,
        "market_snapshot": snapshot_data,
        "current_stage": AnalysisStage.BUILD_MARKET_SNAPSHOT.value,
    })

    manager.transition(
        AnalysisStage.BUILD_MARKET_SNAPSHOT,
        inputs={"instrument": instrument, "analysis_time": analysis_time.isoformat()},
        outputs={"snapshot_hash": snapshot_hash},
    )

    return state, manager


def validate_data(
    state: TradingGraphState,
    manager: StageManager,
    quality_threshold: float = 0.95,
) -> tuple[TradingGraphState, StageManager]:
    """11.3 validate_data — Qualität prüfen, kritische Sperrbedingungen.

    Prüft Datenqualität. Bei kritischen Fehlern → NO_TRADE_DATA_QUALITY.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        quality_threshold: Minimaler Qualitäts-Schwellwert.

    Returns:
        (TradingGraphState, StageManager) — mit validated_snapshot oder NO_TRADE.
    """
    quality = state.market_snapshot.get("quality", 1.0) if state.market_snapshot else 1.0
    critical_fields = state.market_snapshot.get("critical_fields", {}) if state.market_snapshot else {}

    missing_critical = [
        k for k, v in critical_fields.items() if not v
    ]

    if missing_critical or quality < quality_threshold:
        state = state.model_copy(update={
            "current_stage": AnalysisStage.VALIDATE_DATA.value,
            "data_quality": {
                "overall_quality": quality,
                "below_threshold": quality < quality_threshold,
                "missing_critical_fields": missing_critical,
                "valid": False,
            },
            "validation_status": "NO_TRADE_DATA_QUALITY",
            "status": "completed",
        })

        manager.transition(
            AnalysisStage.VALIDATE_DATA,
            inputs={"quality": quality},
            outputs={"valid": False, "reason": "data_quality_below_threshold"},
        )
        return state, manager

    state = state.model_copy(update={
        "current_stage": AnalysisStage.VALIDATE_DATA.value,
        "data_quality": {
            "overall_quality": quality,
            "below_threshold": False,
            "missing_critical_fields": [],
            "valid": True,
        },
        "validation_status": "VALID",
    })

    manager.transition(
        AnalysisStage.VALIDATE_DATA,
        inputs={"quality": quality},
        outputs={"valid": True},
    )

    return state, manager


def compute_features(
    state: TradingGraphState,
    manager: StageManager,
    feature_provider: FeatureProvider | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.4 compute_features — Indikatoren, Struktur, Orderflow.

    Berechnet alle Feature-Kategorien: technische Indikatoren,
    Chart-Struktur, Orderflow, Volatilität, Regime, Cross-Market.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        feature_provider: Feature-Berechnungsprovider.

    Returns:
        (TradingGraphState, StageManager) — mit features und feature_snapshot_id.
    """
    if feature_provider is not None:
        features = feature_provider.compute(state)
    else:
        features = {
            "atr": 0.0,
            "volatility": 0.0,
            "indicators": {},
            "structure": {},
            "orderflow": {},
            "regime": {},
            "cross_market": {},
        }

    feature_hash = hashlib.sha256(
        f"{state.run_id}:{state.instrument}:features".encode()
    ).hexdigest()

    state = state.model_copy(update={
        "current_stage": AnalysisStage.COMPUTE_FEATURES.value,
        "features": features,
        "feature_snapshot_id": feature_hash,
    })

    manager.transition(
        AnalysisStage.COMPUTE_FEATURES,
        inputs={"instrument": state.instrument},
        outputs={"feature_snapshot_id": feature_hash},
    )

    return state, manager


def classify_regime(
    state: TradingGraphState,
    manager: StageManager,
    regime_provider: RegimeProvider | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.5 classify_regime — Regime-Wahrscheinlichkeiten.

    Klassifiziert Markt-Regime: Trending/Bounding/Volatile/Calm.
    Berücksichtigt Stabilität, Wechselrisiko, Multi-Timeframe.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        regime_provider: Regime-Klassifikationsprovider.

    Returns:
        (TradingGraphState, StageManager) — mit regime_report.
    """
    if regime_provider is not None:
        regime_report = regime_provider.classify(state)
    else:
        regime_report = {
            "primary_regime": "UNKNOWN",
            "probabilities": {
                "trending": 0.0,
                "bounding": 0.0,
                "volatile": 0.0,
                "calm": 0.0,
            },
            "stability": 0.5,
            "change_risk": 0.5,
            "timeframes": {},
        }

    state = state.model_copy(update={
        "current_stage": AnalysisStage.CLASSIFY_REGIME.value,
        "regime_report": regime_report,
    })

    manager.transition(
        AnalysisStage.CLASSIFY_REGIME,
        inputs={"instrument": state.instrument},
        outputs={"regime": regime_report.get("primary_regime", "UNKNOWN")},
    )

    return state, manager
