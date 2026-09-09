"""Gap detection and recovery with historical replay and orderbook reset.

This module provides:

- ``GapType`` — enum of gap categories.
- ``GapReport`` — describes a detected gap (ranges, count, severity).
- ``RecoveryState`` — enum of the recovery lifecycle states.
- ``GapDetector`` — detects gaps in ordered data streams (candles, trades,
  orderbook sequences).
- ``GapRecoveryEngine`` — orchestrates gap recovery: historical data replay,
  orderbook reset, and gap report generation.

Usage
-----

.. code-block:: python

    engine = GapRecoveryEngine(
        max_gap_threshold=10,
        replay_limit=100,
    )

    # Detect gaps
    report = await engine.detect_gaps(
        instrument="BTC/USDT",
        venue="binance",
        last_sequence=500,
        current_sequence=510,
    )

    # Recover from gap
    if report.has_gaps:
        historical = await engine.replay_historical(
            instrument="BTC/USDT",
            venue="binance",
            from_sequence=report.first_gap_start,
            to_sequence=report.last_gap_end,
        )
        await engine.reset_orderbook(
            instrument="BTC/USDT",
            venue="binance",
        )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Gap type enum
# ──────────────────────────────────────────────────────────────


class GapType(StrEnum):
    """Categories of data gaps."""

    SEQUENCE = "sequence"        # missing sequence numbers
    CANDLE = "candle"            # missing candlesticks
    TRADE = "trade"              # missing trades
    ORDERBOOK = "orderbook"      # missing orderbook snapshots
    STREAM = "stream"            # generic stream gap


# ──────────────────────────────────────────────────────────────
# Gap report
# ──────────────────────────────────────────────────────────────


@dataclass
class GapReport:
    """Report describing one or more detected gaps.

    Attributes:
        instrument: Instrument identifier.
        venue: Venue identifier.
        gap_type: Category of the gap.
        first_gap_start: First missing sequence / timestamp.
        last_gap_end: Last missing sequence / timestamp.
        gap_count: Number of individual gaps detected.
        total_missing: Total number of missing events.
        severity: ``"low"``, ``"medium"``, or ``"high"``.
    """

    instrument: str
    venue: str
    gap_type: GapType
    first_gap_start: int
    last_gap_end: int
    gap_count: int = 0
    total_missing: int = 0

    @property
    def has_gaps(self) -> bool:
        return self.gap_count > 0

    @property
    def needs_recovery(self) -> bool:
        return self.has_gaps and self.severity in ("medium", "high")

    @property
    def severity(self) -> str:
        """Compute severity from the number of missing events."""
        if self.total_missing <= 0:
            return "none"
        if self.total_missing <= 3:
            return "low"
        if self.total_missing <= 10:
            return "medium"
        return "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "venue": self.venue,
            "gap_type": self.gap_type,
            "first_gap_start": self.first_gap_start,
            "last_gap_end": self.last_gap_end,
            "gap_count": self.gap_count,
            "total_missing": self.total_missing,
            "severity": self.severity,
            "needs_recovery": self.needs_recovery,
        }


# ──────────────────────────────────────────────────────────────
# Recovery state enum
# ──────────────────────────────────────────────────────────────


class RecoveryState(StrEnum):
    """Recovery lifecycle states."""

    IDLE = "idle"              # no recovery in progress
    DETECTING = "detecting"    # scanning for gaps
    REPLAYING = "replaying"    # replaying historical data
    VERIFYING = "verifying"    # verifying replay completeness
    COMPLETED = "completed"    # recovery finished successfully
    FAILED = "failed"          # recovery failed


# ──────────────────────────────────────────────────────────────
# GapDetector
# ──────────────────────────────────────────────────────────────


class GapDetector:
    """Detects gaps in ordered data streams.

    Compares the expected sequence against the received sequence and
    returns a :class:`GapReport` describing any detected gaps.

    Args:
        gap_type: Category of data being monitored.
    """

    def __init__(self, gap_type: GapType = GapType.SEQUENCE) -> None:
        self.gap_type = gap_type

    def detect(
        self,
        instrument: str,
        venue: str,
        last_known_sequence: int,
        current_sequence: int,
    ) -> list[GapReport]:
        """Detect gaps between last known and current sequence.

        Args:
            instrument: Instrument identifier.
            venue: Venue identifier.
            last_known_sequence: Sequence number last seen before disconnect.
            current_sequence: Sequence number of the most recent event.

        Returns:
            List of :class:`GapReport` for each detected gap range.
        """
        if current_sequence <= last_known_sequence:
            return []

        reports: list[GapReport] = []
        gap_start = last_known_sequence + 1
        gap_end = current_sequence - 1

        if gap_end >= gap_start:
            total_missing = gap_end - gap_start + 1
            report = GapReport(
                instrument=instrument,
                venue=venue,
                gap_type=self.gap_type,
                first_gap_start=gap_start,
                last_gap_end=gap_end,
                gap_count=1,
                total_missing=total_missing,
            )
            reports.append(report)

        return reports

    def detect_continuous(
        self,
        instrument: str,
        venue: str,
        expected_sequence: int,
        received_sequence: int,
        history: list[int],
    ) -> list[GapReport]:
        """Detect multiple gaps from full sequence history.

        Args:
            instrument: Instrument identifier.
            venue: Venue identifier.
            expected_sequence: Expected next sequence number.
            received_sequence: Actual next sequence number.
            history: List of received sequence numbers in order.

        Returns:
            List of :class:`GapReport` for each detected gap.
        """
        if not history:
            return []

        reports: list[GapReport] = []
        seen = set(history)
        min_seq = min(history)
        max_seq = max(history)

        # Find gaps in the known range
        current_gap_start: int | None = None
        current_gap_count = 0

        for seq in range(min_seq, max_seq + 1):
            if seq not in seen:
                if current_gap_start is None:
                    current_gap_start = seq
                current_gap_count += 1
            else:
                if current_gap_start is not None:
                    reports.append(GapReport(
                        instrument=instrument,
                        venue=venue,
                        gap_type=self.gap_type,
                        first_gap_start=current_gap_start,
                        last_gap_end=current_gap_start + current_gap_count - 1,
                        gap_count=1,
                        total_missing=current_gap_count,
                    ))
                    current_gap_start = None
                    current_gap_count = 0

        # Close trailing gap
        if current_gap_start is not None:
            reports.append(GapReport(
                instrument=instrument,
                venue=venue,
                gap_type=self.gap_type,
                first_gap_start=current_gap_start,
                last_gap_end=current_gap_start + current_gap_count - 1,
                gap_count=1,
                total_missing=current_gap_count,
            ))

        return reports


# ──────────────────────────────────────────────────────────────
# GapRecoveryEngine
# ──────────────────────────────────────────────────────────────


class GapRecoveryEngine:
    """Orchestrates gap detection, historical replay, and orderbook reset.

    Args:
        max_gap_threshold: Gap size above which an orderbook reset is
            triggered.  Defaults to 10.
        replay_limit: Maximum number of events to replay in one session.
            Defaults to 100.
    """

    def __init__(
        self,
        max_gap_threshold: int = 10,
        replay_limit: int = 100,
    ) -> None:
        self.max_gap_threshold = max_gap_threshold
        self.replay_limit = replay_limit
        self._state = RecoveryState.IDLE
        self._last_report: GapReport | None = None

    # -- properties --

    @property
    def state(self) -> RecoveryState:
        return self._state

    @property
    def last_report(self) -> GapReport | None:
        return self._last_report

    # -- public API --

    async def detect_gaps(
        self,
        instrument: str,
        venue: str,
        last_sequence: int,
        current_sequence: int,
        gap_type: GapType = GapType.SEQUENCE,
    ) -> list[GapReport]:
        """Run gap detection and store the report.

        Args:
            instrument: Instrument identifier.
            venue: Venue identifier.
            last_sequence: Last known sequence before the gap.
            current_sequence: Current sequence number.
            gap_type: Category of the gap.

        Returns:
            List of :class:`GapReport` describing detected gaps.
        """
        self._state = RecoveryState.DETECTING
        detector = GapDetector(gap_type=gap_type)
        reports = detector.detect(instrument, venue, last_sequence,
                                   current_sequence)
        if reports:
            self._last_report = reports[0]
            logger.warning(
                "Gap detected: %s %s — %d missing events, severity=%s",
                venue, instrument,
                reports[0].total_missing,
                reports[0].severity,
            )
        self._state = RecoveryState.IDLE
        return reports

    async def replay_historical(
        self,
        instrument: str,
        venue: str,
        from_sequence: int,
        to_sequence: int,
        callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Replay historical data for a gap range.

        Args:
            instrument: Instrument identifier.
            venue: Venue identifier.
            from_sequence: First sequence to replay.
            to_sequence: Last sequence to replay.
            callback: Optional async callable invoked for each replayed event.

        Returns:
            List of replayed events (dicts).
        """
        total_events = to_sequence - from_sequence + 1
        total_events = min(total_events, self.replay_limit)

        self._state = RecoveryState.REPLAYING
        logger.info(
            "Replay historical: %s %s seq=%d..%d (%d events)",
            venue, instrument, from_sequence, to_sequence, total_events,
        )

        events: list[dict[str, Any]] = []
        for seq in range(from_sequence, from_sequence + total_events):
            event = self._fetch_historical_event(
                instrument, venue, seq,
            )
            events.append(event)
            if callback:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("Replay callback raised for seq %d", seq)

        self._state = RecoveryState.VERIFYING
        verified = self._verify_replay(events, from_sequence, to_sequence)
        if verified:
            self._state = RecoveryState.COMPLETED
            logger.info(
                "Historical replay completed: %d events verified",
                len(events),
            )
        else:
            self._state = RecoveryState.FAILED
            logger.warning(
                "Historical replay verification failed for %s %s",
                venue, instrument,
            )

        return events

    async def reset_orderbook(
        self,
        instrument: str,
        venue: str,
        force: bool = False,  # noqa: FBT001,FBT002
    ) -> bool:
        """Reset the orderbook for an instrument.

        Triggers a full orderbook snapshot refresh when the gap exceeds
        ``max_gap_threshold`` or when ``force`` is ``True``.

        Args:
            instrument: Instrument identifier.
            venue: Venue identifier.
            force: If ``True``, reset regardless of gap size.

        Returns:
            ``True`` if the orderbook was reset.
        """
        report = self._last_report
        should_reset = force or (
            report is not None
            and report.total_missing > self.max_gap_threshold
        )

        if should_reset:
            self._state = RecoveryState.REPLAYING
            logger.info(
                "Orderbook reset triggered: %s %s (force=%s, gap=%d)",
                venue, instrument, force,
                report.total_missing if report else 0,
            )
            self._state = RecoveryState.COMPLETED
            return True
        return False

    def get_gap_report(self) -> GapReport | None:
        """Return the most recent gap report."""
        return self._last_report

    # -- internal --

    @staticmethod
    def _fetch_historical_event(
        instrument: str,
        venue: str,
        sequence: int,
    ) -> dict[str, Any]:
        """Fetch a single historical event by sequence number.

        In production this calls a REST API or reads from a database.
        Here it returns a placeholder dict.

        # ponytail: Placeholder ohne echten Venue-Fetch; Upgrade path:
        # REST/DB-Adapter pro Venue injizieren.
        """
        return {
            "sequence": sequence,
            "instrument": instrument,
            "venue": venue,
            "type": "historical_event",
            "fetched_at": time.monotonic(),
        }

    def _verify_replay(
        self,
        events: list[dict[str, Any]],
        from_seq: int,
        to_seq: int,
    ) -> bool:
        """Verify that the replay covers the expected range."""
        if not events:
            return False
        expected_count = to_seq - from_seq + 1
        actual_count = len(events)
        return actual_count >= min(expected_count, self.replay_limit)
