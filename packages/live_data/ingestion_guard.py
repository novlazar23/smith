"""Producer-seitiger Guard für den aktiven Live-Market-Data-Pfad.

Der Guard verbindet die bestehenden `packages.live_data`-Bausteine mit dem
`apps.market_producer`-Tick:

- ``HealthMonitor`` erfasst REST-Latenz und Health pro Venue.
- ``FailoverManager`` wechselt bei ungesundem Primär-Venue auf das Backup.
- ``QualityGateEvaluator`` prüft Freshness, Preis- und Volumen-Sanity.
- ``GapDetector`` meldet fehlende 1m-Kerzen aus der Open-Time-Reihenfolge.
- ``AutoReconnector`` läuft im Hintergrund, wenn ein Register-Hook existiert.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from packages.live_data.failover import FailoverDecision, FailoverManager, VenueConfig
from packages.live_data.gap_recovery import GapDetector, GapType
from packages.live_data.health_monitor import ConnectionType, HealthMonitor
from packages.live_data.quality_gate import (
    GateViolation,
    QualityGateEvaluator,
    QualityGateResult,
)
from packages.live_data.reconnect import AutoReconnector, ReconnectConfig

logger = logging.getLogger(__name__)

_MISSING_CANDLE_WARN_THRESHOLD = 5


@dataclass
class _IngestionState:
    """Pro-Venue/Instrument-Zustand für Quality und Gap-Tracking."""

    last_success_at: float | None = None
    prev_close: float | None = None
    avg_volume: float = 0.0
    consecutive_failures: int = 0
    last_candle_open_time: datetime | None = None


class LiveIngestionGuard:
    """Härtet einen Producer-Tick für Live-/Fallback-Venues.

    Der Guard ist bewusst producer-nah gehalten: Er entscheidet nicht selbst,
    welche Kerze gepublished wird, sondern liefert Health/Quality-Entscheidungen,
    die der Producer als Fallback- und Failover-Steuerung nutzt.
    """

    def __init__(
        self,
        primary_venue: str,
        backup_venue: str,
        *,
        interval_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_max_attempts: int = 10,
        health_threshold: float = 0.5,
        failover_after_failures: int = 3,
    ) -> None:
        if primary_venue == backup_venue:
            msg = "primary_venue und backup_venue müssen sich unterscheiden"
            raise ValueError(msg)
        if failover_after_failures < 1:
            msg = "failover_after_failures muss >= 1 sein"
            raise ValueError(msg)

        self._primary_venue = primary_venue
        self._backup_venue = backup_venue
        self._clock = clock or time.monotonic
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_max_attempts = reconnect_max_attempts
        self._failover_after_failures = failover_after_failures

        self._health = HealthMonitor(venues=[primary_venue, backup_venue])
        self._failover = FailoverManager(
            venues=[
                VenueConfig(name=primary_venue, role="primary", priority=1),
                VenueConfig(name=backup_venue, role="backup", priority=2),
            ],
            health_threshold=health_threshold,
            failover_cooldown=0.0,
        )
        self._quality = QualityGateEvaluator(
            freshness_window=max(30.0, interval_seconds * 3.0),
            price_deviation_pct=25.0,
            volume_spike_factor=10.0,
        )
        self._gap_detector = GapDetector(gap_type=GapType.CANDLE)

        self._states: dict[tuple[str, str], _IngestionState] = {}
        self._connect_hooks: dict[str, Callable[[], Any]] = {}
        self._reconnectors: dict[str, AutoReconnector] = {}
        self._tasks: dict[str, asyncio.Task[bool]] = {}

    @property
    def primary_venue(self) -> str:
        return self._primary_venue

    @property
    def backup_venue(self) -> str:
        return self._backup_venue

    @property
    def active_venue(self) -> str:
        return self._failover.active_venue

    def register_reconnect_hook(self, venue: str, hook: Callable[[], Any]) -> None:
        """Registriert den Connect-Hook für einen Hintergrund-Reconnector."""
        self._connect_hooks[venue] = hook

    async def evaluate_failover(self) -> FailoverDecision:
        """Bewertet Health-States und vollzieht genehmigte Venue-Wechsel."""
        decision = await self._failover.evaluate()
        if decision.new_venue is not None:
            self._failover.complete_switch(decision.new_venue)
            logger.info(
                "Live-Data-Failover: aktiv=%s neu=%s (%s)",
                decision.active_venue,
                decision.new_venue,
                decision.reason,
            )
        return decision

    async def record_success(
        self,
        venue: str,
        instrument: str,
        candle: dict[str, Any],
        latency_ms: float,
    ) -> QualityGateResult:
        """Verzeichnet einen erfolgreichen Fetch und prüft die Kerzen-Qualität."""
        state = self._state_for(venue, instrument)
        state.consecutive_failures = 0

        await self._mark_healthy(venue, latency_ms)

        now = self._clock()
        open_time: datetime = candle["open_time"]  # typeignore[reportAssignmentType]
        close = float(candle["close"])
        volume = float(candle.get("volume", 0.0))

        quality = await self._quality.evaluate(
            instrument=instrument,
            venue=venue,
            current_time=now,
            last_event_time=state.last_success_at if state.last_success_at is not None else now,
            price=close,
            prev_price=state.prev_close,
            volume=volume,
            avg_volume=state.avg_volume if state.avg_volume > 0 else volume,
        )

        if state.last_candle_open_time is not None:
            missing = self._missing_candles(state.last_candle_open_time, open_time)
            if missing > _MISSING_CANDLE_WARN_THRESHOLD:
                quality.violations.append(
                    GateViolation(
                        gate_name="candle_gap",
                        severity="fail",
                        message=f"{missing} 1m-Kerzen fehlen",
                        details={"missing_candles": missing},
                    )
                )
                quality.passed = False
            elif missing:
                logger.debug(
                    "Kleine Candle-Gap bei %s/%s: %d Kerzen",
                    venue,
                    instrument,
                    missing,
                )

        if quality.passed:
            state.last_success_at = now
            state.prev_close = close
            state.avg_volume = (
                volume
                if state.avg_volume <= 0
                else 0.9 * state.avg_volume + 0.1 * volume
            )
            state.last_candle_open_time = open_time
        else:
            logger.warning(
                "Live-Data-Quality fehlgeschlagen (%s/%s): %s",
                venue,
                instrument,
                [v.message for v in quality.violations],
            )

        return quality

    async def record_failure(
        self,
        venue: str,
        instrument: str,
        reason: str,
        *,
        trigger_reconnect: bool = False,
        critical: bool = False,
    ) -> None:
        """Verzeichnet einen Fehlzug und markiert bei Bedarf die Venue ungesund.

        Ein einzelner transienter Fetch-Fehler fällt nur auf das Backup
        zurück; erst ``critical`` oder ``failover_after_failures``
        aufeinanderfolgende Fehler machen die Venue für den Failover
        ungesund.
        """
        state = self._state_for(venue, instrument)
        state.consecutive_failures += 1
        degraded = critical or state.consecutive_failures >= self._failover_after_failures

        if degraded:
            await self._mark_unhealthy(venue)
        logger.warning("Live-Data-Fehler (%s/%s): %s", venue, instrument, reason)

        if trigger_reconnect or degraded:
            self._ensure_reconnector(venue)

    async def record_reconnect_success(self, venue: str) -> None:
        """Setzt die Fehlerzähler einer Venue zurück und markiert sie gesund."""
        for (known_venue, _instrument), state in self._states.items():
            if known_venue == venue:
                state.consecutive_failures = 0
        await self._mark_healthy(venue, 0.0)

    async def close(self) -> None:
        """Stoppt alle Hintergrund-Reconnect-Tasks."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _state_for(self, venue: str, instrument: str) -> _IngestionState:
        key = (venue, instrument)
        if key not in self._states:
            self._states[key] = _IngestionState()
        return self._states[key]

    def _missing_candles(self, previous: datetime, current: datetime) -> int:
        if current <= previous:
            return 0
        previous_minute = int(previous.timestamp() // 60)
        current_minute = int(current.timestamp() // 60)
        reports = self._gap_detector.detect(
            "ingestion",
            "guard",
            previous_minute,
            current_minute,
        )
        return sum(report.total_missing for report in reports)

    async def _mark_healthy(self, venue: str, latency_ms: float) -> None:
        state = await self._health.check_venue(
            venue,
            ConnectionType.REST,
            latency_ms=latency_ms,
        )
        self._failover.update_health(venue, 1.0 if state.overall_healthy else 0.0)

    async def _mark_unhealthy(self, venue: str) -> None:
        await self._health.check_venue(venue, ConnectionType.REST)
        self._failover.update_health(venue, 0.0)

    def _ensure_reconnector(self, venue: str) -> None:
        existing = self._tasks.get(venue)
        if existing is not None and not existing.done():
            return
        hook = self._connect_hooks.get(venue)
        if hook is None:
            return

        reconnector = AutoReconnector(
            venue=venue,
            config=ReconnectConfig(
                max_attempts=self._reconnect_max_attempts,
                base_delay=self._reconnect_base_delay,
                max_delay=self._reconnect_max_delay,
                jitter_factor=0.2,
            ),
            connect_hook=hook,
        )
        self._reconnectors[venue] = reconnector
        task = asyncio.create_task(
            reconnector.run(),
            name=f"live-data-reconnect-{venue}",
        )
        self._tasks[venue] = task

        def _cleanup(finished: asyncio.Task[bool], key: str = venue) -> None:
            self._tasks.pop(key, None)

        task.add_done_callback(_cleanup)
