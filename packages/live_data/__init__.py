"""Live Data Pipeline — health, reconnect, quality, failover, gap recovery.

This package hardens the live-data ingestion pipeline for EPIC-16.  It provides:

- Per-venue health monitoring (WebSocket, REST, REST backup) with latency
  tracking, timeout detection, and connection-pool management.
- Auto-reconnect with exponential backoff + jitter, configurable max attempts,
  and state recovery after reconnect.
- Data-quality gates: freshness, gap detection, price sanity, volume sanity.
- Primary/backup venue failover with automatic switching and consistency checks.
- Gap detection → gap report, historical-data replay, and orderbook reset
  when gaps exceed a configurable threshold.

Public API
----------
- HealthMonitor — per-venue health checks (``health_monitor``)
- AutoReconnector — reconnect with exponential backoff + jitter (``reconnect``)
- QualityGateEvaluator — data quality gates (``quality_gate``)
- FailoverManager — primary/backup failover (``failover``)
- GapRecoveryEngine — gap detection and recovery (``gap_recovery``)
"""

from __future__ import annotations

from packages.live_data.failover import (
    FailoverDecision,
    FailoverManager,
    FailoverState,
    VenueConfig,
)
from packages.live_data.gap_recovery import (
    GapDetector,
    GapRecoveryEngine,
    GapReport,
    GapType,
    RecoveryState,
)
from packages.live_data.health_monitor import (
    ConnectionPool,
    ConnectionType,
    HealthMonitor,
    VenueHealthState,
)
from packages.live_data.quality_gate import (
    FreshnessGate,
    GapDetectionGate,
    GateViolation,
    PriceSanityGate,
    QualityGateEvaluator,
    QualityGateResult,
    VolumeSanityGate,
)
from packages.live_data.reconnect import (
    AutoReconnector,
    ReconnectConfig,
    ReconnectEvent,
    ReconnectState,
)

__all__ = [
    "AutoReconnector",
    "ConnectionPool",
    "ConnectionType",
    "FailoverDecision",
    "FailoverManager",
    "FailoverState",
    "FreshnessGate",
    "GapDetectionGate",
    "GapDetector",
    "GapRecoveryEngine",
    "GapReport",
    "GapType",
    "GateViolation",
    "HealthMonitor",
    "PriceSanityGate",
    "QualityGateEvaluator",
    "QualityGateResult",
    "ReconnectConfig",
    "ReconnectEvent",
    "ReconnectState",
    "RecoveryState",
    "VenueConfig",
    "VenueHealthState",
    "VolumeSanityGate",
]
