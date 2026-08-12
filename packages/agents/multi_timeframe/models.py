"""Data models for the multi-timeframe agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MultiTimeframeConfig:
    """Konfiguration fuer den Multi-Timeframe-Agenten."""

    agent_id: str = "multi_timeframe"
    agent_version: str = "0.1.0"
    timeframes: tuple[str, ...] = (
        "1m", "5m", "15m", "1h", "4h", "1d",
    )
    timeframe_weights: dict[str, float] = field(
        default_factory=lambda: {
            "1m": 0.05,
            "5m": 0.1,
            "15m": 0.15,
            "1h": 0.25,
            "4h": 0.25,
            "1d": 0.2,
        },
    )


@dataclass(frozen=True, slots=True)
class TimeframeSignal:
    """Signal aus einem einzelnen Zeitrahmen."""

    timeframe: str
    direction: str  # LONG, SHORT, RANGE
    probability: float
    weight: float
    confidence: float


@dataclass(frozen=True, slots=True)
class MultiTimeframeReport:
    """Aggregierter Bericht ueber alle Zeitrahmen."""

    direction: str
    confidence: float
    timeframe_signals: list[TimeframeSignal]
    conflicts: list[str]
    overall_agreement: float  # 0.0-1.0
