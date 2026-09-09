"""Data quality gates: freshness, gap detection, price sanity, volume sanity.

This module provides:

- ``GateViolation`` — describes a single quality gate failure.
- ``QualityGateResult`` — aggregate result from evaluating all gates.
- ``FreshnessGate`` — rejects data older than a configurable threshold.
- ``GapDetectionGate`` — detects missing sequences in ordered streams.
- ``PriceSanityGate`` — flags unrealistic price movements.
- ``VolumeSanityGate`` — flags unrealistic volume spikes.
- ``QualityGateEvaluator`` — runs all configured gates and collects results.

Usage
-----

.. code-block:: python

    evaluator = QualityGateEvaluator(
        freshness_window=30.0,
        price_deviation_pct=25.0,
        volume_spike_factor=10.0,
        max_gap_size=5,
    )

    result = await evaluator.evaluate(
        instrument="BTC/USDT",
        venue="binance",
        current_time=datetime.now(UTC),
        last_event_time=datetime.now(UTC) - timedelta(seconds=60),
        sequence=100,
        expected_sequence=105,
        price=50000.0,
        prev_price=45000.0,
        volume=100.0,
        avg_volume=50.0,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Violation / result
# ──────────────────────────────────────────────────────────────


@dataclass
class GateViolation:
    """Describes a single quality gate failure.

    Attributes:
        gate_name: Name of the gate that failed.
        severity: ``"warn"`` or ``"fail"``.
        message: Human-readable description.
        details: Optional extra context.
    """

    gate_name: str
    severity: str = "warn"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """Aggregate result from evaluating all quality gates.

    Attributes:
        passed: ``True`` if no gate violated at ``"fail"`` severity.
        violations: List of all violations detected.
        instrument: Instrument identifier (e.g. ``"BTC/USDT"``).
        venue: Venue identifier.
    """

    passed: bool = True
    violations: list[GateViolation] = field(default_factory=list)
    instrument: str = ""
    venue: str = ""

    def add_violation(self, violation: GateViolation) -> None:
        self.violations.append(violation)
        if violation.severity == "fail":
            self.passed = False

    @property
    def warn_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warn")

    @property
    def fail_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "violations": [
                {
                    "gate_name": v.gate_name,
                    "severity": v.severity,
                    "message": v.message,
                    "details": v.details,
                }
                for v in self.violations
            ],
            "instrument": self.instrument,
            "venue": self.venue,
        }


# ──────────────────────────────────────────────────────────────
# Freshness gate
# ──────────────────────────────────────────────────────────────


class FreshnessGate:
    """Rejects data older than a configurable window.

    Args:
        max_age_seconds: Maximum acceptable age of data in seconds.
            Defaults to 30.0.
    """

    def __init__(self, max_age_seconds: float = 30.0) -> None:
        self.max_age_seconds = max_age_seconds

    def check(self, current_time: float, last_event_time: float,
              instrument: str, venue: str) -> list[GateViolation]:
        """Check if the data is fresh enough.

        Args:
            current_time: Current monotonic time in seconds.
            last_event_time: Timestamp of the last received event.
            instrument: Instrument identifier.
            venue: Venue identifier.

        Returns:
            List of violations (empty if fresh).
        """
        age = current_time - last_event_time
        if age > self.max_age_seconds:
            return [
                GateViolation(
                    gate_name="freshness",
                    severity="fail" if age > self.max_age_seconds * 2 else "warn",
                    message=(
                        f"Data stale: age={age:.1f}s exceeds "
                        f"threshold={self.max_age_seconds:.1f}s"
                    ),
                    details={
                        "instrument": instrument,
                        "venue": venue,
                        "age_seconds": age,
                        "threshold_seconds": self.max_age_seconds,
                    },
                )
            ]
        return []


# ──────────────────────────────────────────────────────────────
# Gap detection gate
# ──────────────────────────────────────────────────────────────


class GapDetectionGate:
    """Detects missing sequences in ordered data streams.

    Compares the received sequence against the expected sequence and flags
    gaps larger than ``max_gap_size``.

    Args:
        max_gap_size: Maximum acceptable gap between sequences.
            Defaults to 5.
    """

    def __init__(self, max_gap_size: int = 5) -> None:
        self.max_gap_size = max_gap_size

    def check(
        self,
        sequence: int,
        expected_sequence: int,
        instrument: str,
        venue: str,
    ) -> list[GateViolation]:
        """Check for gaps in the sequence.

        Args:
            sequence: Sequence number of the received event.
            expected_sequence: Expected next sequence number.
            instrument: Instrument identifier.
            venue: Venue identifier.

        Returns:
            List of violations (empty if no gap detected).
        """
        # ponytail: Nur fehlende Sequenzen; Stream-Overshoot/Restart wird
        # erst mit echten Venue-Adaptoren modelliert.
        if expected_sequence > 0 and sequence < expected_sequence:
            gap_size = expected_sequence - sequence
            if gap_size > self.max_gap_size:
                return [
                    GateViolation(
                        gate_name="gap_detection",
                        severity="fail",
                        message=(
                            f"Large gap detected: expected seq {expected_sequence}, "
                            f"got {sequence} (gap={gap_size})"
                        ),
                        details={
                            "instrument": instrument,
                            "venue": venue,
                            "received_sequence": sequence,
                            "expected_sequence": expected_sequence,
                            "gap_size": gap_size,
                        },
                    )
                ]
            elif gap_size > 0:
                return [
                    GateViolation(
                        gate_name="gap_detection",
                        severity="warn",
                        message=(
                            f"Small gap detected: expected seq {expected_sequence}, "
                            f"got {sequence} (gap={gap_size})"
                        ),
                        details={
                            "instrument": instrument,
                            "venue": venue,
                            "received_sequence": sequence,
                            "expected_sequence": expected_sequence,
                            "gap_size": gap_size,
                        },
                    )
                ]
        return []


# ──────────────────────────────────────────────────────────────
# Price sanity gate
# ──────────────────────────────────────────────────────────────


class PriceSanityGate:
    """Flags unrealistic price movements.

    A price movement is flagged if the percentage deviation from the previous
    price exceeds ``deviation_pct``.

    Args:
        deviation_pct: Maximum acceptable price deviation in percent.
            Defaults to 25.0 (i.e. 25 %).
        min_price: Minimum acceptable price (defaults to 0.01).
    """

    def __init__(
        self,
        deviation_pct: float = 25.0,
        min_price: float = 0.01,
    ) -> None:
        self.deviation_pct = deviation_pct
        self.min_price = min_price

    def check(
        self,
        price: float,
        prev_price: float | None,
        instrument: str,
        venue: str,
    ) -> list[GateViolation]:
        """Check if the price is within acceptable bounds.

        Args:
            price: Current price.
            prev_price: Previous price (``None`` for first data point).
            instrument: Instrument identifier.
            venue: Venue identifier.

        Returns:
            List of violations (empty if price is sane).
        """
        violations: list[GateViolation] = []

        # Check minimum price
        if price < self.min_price:
            violations.append(
                GateViolation(
                    gate_name="price_sanity",
                    severity="fail",
                    message=f"Price below minimum: {price}",
                    details={
                        "instrument": instrument,
                        "venue": venue,
                        "price": price,
                        "min_price": self.min_price,
                    },
                )
            )

        # Check deviation from previous price
        if prev_price is not None and prev_price > 0:
            pct_change = abs(price - prev_price) / prev_price * 100.0
            if pct_change > self.deviation_pct:
                violations.append(
                    GateViolation(
                        gate_name="price_sanity",
                        severity="fail" if pct_change > self.deviation_pct * 2
                                     else "warn",
                        message=(
                            f"Unrealistic price movement: {pct_change:.1f}% "
                            f"(threshold={self.deviation_pct}%)"
                        ),
                        details={
                            "instrument": instrument,
                            "venue": venue,
                            "price": price,
                            "prev_price": prev_price,
                            "deviation_pct": pct_change,
                            "threshold_pct": self.deviation_pct,
                        },
                    )
                )

        return violations


# ──────────────────────────────────────────────────────────────
# Volume sanity gate
# ──────────────────────────────────────────────────────────────


class VolumeSanityGate:
    """Flags unrealistic volume spikes.

    A volume spike is flagged if the current volume exceeds
    ``avg_volume * spike_factor``.

    Args:
        spike_factor: Multiplier above average volume considered a spike.
            Defaults to 10.0.
        min_volume: Minimum acceptable volume (defaults to 0.0).
    """

    def __init__(
        self,
        spike_factor: float = 10.0,
        min_volume: float = 0.0,
    ) -> None:
        self.spike_factor = spike_factor
        self.min_volume = min_volume

    def check(
        self,
        volume: float,
        avg_volume: float,
        instrument: str,
        venue: str,
    ) -> list[GateViolation]:
        """Check if the volume is within acceptable bounds.

        Args:
            volume: Current volume.
            avg_volume: Average / baseline volume.
            instrument: Instrument identifier.
            venue: Venue identifier.

        Returns:
            List of violations (empty if volume is sane).
        """
        violations: list[GateViolation] = []

        if volume < self.min_volume:
            violations.append(
                GateViolation(
                    gate_name="volume_sanity",
                    severity="warn",
                    message=f"Volume below minimum: {volume}",
                    details={
                        "instrument": instrument,
                        "venue": venue,
                        "volume": volume,
                        "min_volume": self.min_volume,
                    },
                )
            )

        if avg_volume > 0:
            ratio = volume / avg_volume
            if ratio > self.spike_factor:
                violations.append(
                    GateViolation(
                        gate_name="volume_sanity",
                        severity="fail" if ratio > self.spike_factor * 2
                                     else "warn",
                        message=(
                            f"Volume spike: {ratio:.1f}x average "
                            f"(threshold={self.spike_factor}x)"
                        ),
                        details={
                            "instrument": instrument,
                            "venue": venue,
                            "volume": volume,
                            "avg_volume": avg_volume,
                            "ratio": ratio,
                            "threshold_factor": self.spike_factor,
                        },
                    )
                )

        return violations


# ──────────────────────────────────────────────────────────────
# QualityGateEvaluator
# ──────────────────────────────────────────────────────────────


class QualityGateEvaluator:
    """Runs all configured quality gates and collects results.

    Args:
        freshness_window: Passed to :class:`FreshnessGate`.
        max_gap_size: Passed to :class:`GapDetectionGate`.
        price_deviation_pct: Passed to :class:`PriceSanityGate`.
        volume_spike_factor: Passed to :class:`VolumeSanityGate`.
    """

    def __init__(
        self,
        freshness_window: float = 30.0,
        max_gap_size: int = 5,
        price_deviation_pct: float = 25.0,
        volume_spike_factor: float = 10.0,
    ) -> None:
        self.freshness = FreshnessGate(max_age_seconds=freshness_window)
        self.gap_detection = GapDetectionGate(max_gap_size=max_gap_size)
        self.price_sanity = PriceSanityGate(deviation_pct=price_deviation_pct)
        self.volume_sanity = VolumeSanityGate(
            spike_factor=volume_spike_factor,
        )

    async def evaluate(
        self,
        instrument: str,
        venue: str,
        current_time: float | None = None,
        last_event_time: float | None = None,
        sequence: int = 0,
        expected_sequence: int = 0,
        price: float | None = None,
        prev_price: float | None = None,
        volume: float | None = None,
        avg_volume: float = 0.0,
    ) -> QualityGateResult:
        """Evaluate all configured gates and return results.

        Only the gates for which the corresponding input is non-None are
        evaluated (e.g. if ``price`` is ``None``, the price gate is skipped).

        Args:
            instrument: Instrument identifier.
            venue: Venue identifier.
            current_time: Current monotonic time.
            last_event_time: Timestamp of the last event.
            sequence: Received sequence number.
            expected_sequence: Expected next sequence number.
            price: Current price.
            prev_price: Previous price.
            volume: Current volume.
            avg_volume: Average / baseline volume.

        Returns:
            :class:`QualityGateResult` with all violations collected.
        """
        now = current_time or time.monotonic()
        last = last_event_time if last_event_time is not None else now
        result = QualityGateResult(
            instrument=instrument,
            venue=venue,
        )

        # Freshness
        result.violations.extend(
            self.freshness.check(now, last, instrument, venue)
        )

        # Gap detection
        if expected_sequence > 0 and sequence != expected_sequence:
            result.violations.extend(
                self.gap_detection.check(
                    sequence, expected_sequence, instrument, venue,
                )
            )

        # Price sanity
        if price is not None:
            result.violations.extend(
                self.price_sanity.check(price, prev_price, instrument, venue)
            )

        # Volume sanity
        if volume is not None and avg_volume > 0:
            result.violations.extend(
                self.volume_sanity.check(volume, avg_volume, instrument, venue)
            )

        # Set passed based on any fail-level violations
        result.passed = all(
            v.severity != "fail" for v in result.violations
        )

        return result
