"""Market Data Validator & Gap Detector.

Validiert alle Markt-Ereignistypen und erkennt Sequenzlücken:
- Candle: OHLC-Integrität, Zeitreihen-Konsistenz
- Trade: Preis/Quantity-Validität
- OrderBook: Bid-Ask-Spread, Sequence-Kontinuität
- Derivatives: Funding Rate, Open Interest, Liquidation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """Schweregrad einer Validierungsmeldung."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ValidationIssue:
    """Ein einzelnes Validierungsproblem."""

    field: str
    message: str
    severity: Severity
    value: Any | None = None


@dataclass
class ValidationResult:
    """Ergebnis der Validierung eines Events."""

    event_type: str
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    quality_score: float = 1.0  # 0.0 (schlecht) bis 1.0 (perfekt)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Fügt ein Problem hinzu und berechnet neuen Score."""
        self.issues.append(issue)
        # Score reduzieren je nach Schweregrad
        penalties = {
            Severity.INFO: 0.0,
            Severity.WARNING: 0.05,
            Severity.ERROR: 0.2,
            Severity.CRITICAL: 0.5,
        }
        self.quality_score = max(0.0, self.quality_score - penalties.get(issue.severity, 0.0))
        self.is_valid = not any(
            i.severity in (Severity.WARNING, Severity.ERROR, Severity.CRITICAL) for i in self.issues
        )


class MarketDataValidator:
    """Zentraler Validator für alle Markt-Ereignistypen.

    Validiert:
    - Candle: high >= low, low > 0, open_time < close_time
    - Trade: price > 0, quantity > 0
    - OrderBook: bids sorted desc, asks sorted asc, bid[0] < ask[0]
    - FundingRate: rate in [-1, 1], mark_price > 0
    - OpenInterest: open_interest >= 0
    - Liquidation: quantity > 0, price > 0
    """

    # Kritische Qualitätsschwellen
    QUALITY_THRESHOLD_HIGH = 0.9
    QUALITY_THRESHOLD_LOW = 0.5

    def validate(self, event: dict[str, Any]) -> ValidationResult:
        """Validiert ein einzelnes Event.

        Args:
            event: Event-Dict mit 'type', 'instrument', 'venue' und payload

        Returns:
            ValidationResult mit Issues und Quality-Score
        """
        event_type = event.get("type", event.get("event_type", "unknown"))
        result = ValidationResult(
            event_type=event_type,
            is_valid=True,
            quality_score=1.0,
        )

        validators = {
            "candle": self._validate_candle,
            "trade": self._validate_trade,
            "orderbook_snapshot": self._validate_orderbook,
            "orderbook_delta": self._validate_orderbook_delta,
            "funding_rate": self._validate_funding_rate,
            "open_interest": self._validate_open_interest,
            "liquidation": self._validate_liquidation,
        }

        validator = validators.get(event_type)
        if validator is None:
            result.add_issue(ValidationIssue(
                field="event_type",
                message=f"Unknown event type: {event_type}",
                severity=Severity.WARNING,
            ))
            return result

        validator(event, result)
        return result

    def _validate_candle(self, event: dict, result: ValidationResult) -> None:
        """Validiert eine Candle."""
        # Prüfe OHLC-Integrität
        high = event.get("high", 0)
        low = event.get("low", 0)
        open_price = event.get("open", 0)
        close = event.get("close", 0)
        volume = event.get("volume", 0)

        if high < low:
            result.add_issue(ValidationIssue(
                field="high_low",
                message=f"high ({high}) < low ({low})",
                severity=Severity.CRITICAL,
            ))

        if low <= 0:
            result.add_issue(ValidationIssue(
                field="low",
                message=f"low must be > 0, got {low}",
                severity=Severity.CRITICAL,
            ))

        if open_price <= 0 or close <= 0:
            result.add_issue(ValidationIssue(
                field="price",
                message="OHLC values must be > 0",
                severity=Severity.ERROR,
            ))

        if volume < 0:
            result.add_issue(ValidationIssue(
                field="volume",
                message=f"Volume must be >= 0, got {volume}",
                severity=Severity.ERROR,
            ))

        # Zeitreihen-Konsistenz
        open_time = event.get("open_time")
        close_time = event.get("close_time")
        if open_time and close_time and open_time >= close_time:
                result.add_issue(ValidationIssue(
                    field="timestamps",
                    message="open_time must be < close_time",
                    severity=Severity.WARNING,
                ))

        # Venue/Instrument vorhanden?
        if not event.get("instrument"):
            result.add_issue(ValidationIssue(
                field="instrument",
                message="Missing instrument field",
                severity=Severity.ERROR,
            ))
        if not event.get("venue"):
            result.add_issue(ValidationIssue(
                field="venue",
                message="Missing venue field",
                severity=Severity.ERROR,
            ))

    def _validate_trade(self, event: dict, result: ValidationResult) -> None:
        """Validiert einen Trade."""
        price = event.get("price", 0)
        quantity = event.get("quantity", 0)

        if price <= 0:
            result.add_issue(ValidationIssue(
                field="price",
                message=f"Price must be > 0, got {price}",
                severity=Severity.CRITICAL,
            ))

        if quantity <= 0:
            result.add_issue(ValidationIssue(
                field="quantity",
                message=f"Quantity must be > 0, got {quantity}",
                severity=Severity.CRITICAL,
            ))

        if not event.get("trade_id"):
            result.add_issue(ValidationIssue(
                field="trade_id",
                message="Missing trade_id",
                severity=Severity.ERROR,
            ))

    def _validate_orderbook(self, event: dict, result: ValidationResult) -> None:
        """Validiert einen Orderbook-Snapshot."""
        bids = event.get("bids", [])
        asks = event.get("asks", [])

        # Bids müssen absteigend sortiert sein
        if len(bids) > 1:
            for i in range(1, len(bids)):
                bid_price = bids[i].get("price", 0) if isinstance(bids[i], dict) else bids[i][0]
                prev_price = bids[i - 1].get("price", 0) if isinstance(bids[i - 1], dict) else bids[i - 1][0]
                if bid_price >= prev_price:
                    result.add_issue(ValidationIssue(
                        field="bids_sorting",
                        message=f"Bad bid sorting at index {i}",
                        severity=Severity.WARNING,
                    ))
                    break

        # Asks müssen aufsteigend sortiert sein
        if len(asks) > 1:
            for i in range(1, len(asks)):
                ask_price = asks[i].get("price", 0) if isinstance(asks[i], dict) else asks[i][0]
                prev_price = asks[i - 1].get("price", 0) if isinstance(asks[i - 1], dict) else asks[i - 1][0]
                if ask_price <= prev_price:
                    result.add_issue(ValidationIssue(
                        field="asks_sorting",
                        message=f"Bad ask sorting at index {i}",
                        severity=Severity.WARNING,
                    ))
                    break

        # Kein交叉
        if bids and asks:
            best_bid = bids[0].get("price", 0) if isinstance(bids[0], dict) else bids[0][0]
            best_ask = asks[0].get("price", 0) if isinstance(asks[0], dict) else asks[0][0]
            if best_bid >= best_ask:
                result.add_issue(ValidationIssue(
                    field="spread",
                    message=f"Best bid ({best_bid}) >= best ask ({best_ask})",
                    severity=Severity.CRITICAL,
                ))

    def _validate_orderbook_delta(self, event: dict, result: ValidationResult) -> None:
        """Validiert ein Orderbook-Delta."""
        # Deltas haben ähnliche Struktur wie Snapshots
        self._validate_orderbook(event, result)

    def _validate_funding_rate(self, event: dict, result: ValidationResult) -> None:
        """Validiert einen Funding Rate Event."""
        rate = event.get("funding_rate", 0)
        mark_price = event.get("mark_price", 0)

        if mark_price <= 0:
            result.add_issue(ValidationIssue(
                field="mark_price",
                message=f"Mark price must be > 0, got {mark_price}",
                severity=Severity.CRITICAL,
            ))

        if rate < -1.0 or rate > 1.0:
            result.add_issue(ValidationIssue(
                field="funding_rate",
                message=f"Funding rate {rate} outside [-1, 1]",
                severity=Severity.ERROR,
            ))

    def _validate_open_interest(self, event: dict, result: ValidationResult) -> None:
        """Validiert einen Open Interest Event."""
        oi = event.get("open_interest", 0)
        if oi < 0:
            result.add_issue(ValidationIssue(
                field="open_interest",
                message=f"Open interest must be >= 0, got {oi}",
                severity=Severity.ERROR,
            ))

    def _validate_liquidation(self, event: dict, result: ValidationResult) -> None:
        """Validiert einen Liquidation Event."""
        quantity = event.get("quantity", 0)
        price = event.get("price", 0)

        if quantity <= 0:
            result.add_issue(ValidationIssue(
                field="quantity",
                message=f"Quantity must be > 0, got {quantity}",
                severity=Severity.CRITICAL,
            ))
        if price <= 0:
            result.add_issue(ValidationIssue(
                field="price",
                message=f"Price must be > 0, got {price}",
                severity=Severity.CRITICAL,
            ))

    def batch_validate(
        self,
        events: list[dict[str, Any]],
    ) -> list[ValidationResult]:
        """Validiert eine Batch von Events.

        Returns:
            Liste von ValidationResult-Objekten
        """
        return [self.validate(event) for event in events]

    def classify_quality(self, result: ValidationResult) -> str:
        """Klassifiziert die Event-Qualität.

        Returns:
            'pass' (≥0.9), 'degraded' (0.5-0.9), 'quarantine' (<0.5)
        """
        if result.quality_score >= self.QUALITY_THRESHOLD_HIGH:
            return "pass"
        elif result.quality_score >= self.QUALITY_THRESHOLD_LOW:
            return "degraded"
        return "quarantine"


class GapDetector:
    """Erkennt Sequenzlücken in Event-Strömen.

    Trackt die letzte bekannte Sequence pro (instrument, venue, event_type)
    und meldet Lücken wenn die aktuelle Sequence nicht nahtlos fortfährt.
    """

    def __init__(self) -> None:
        self._last_sequence: dict[str, int] = {}

    def check_gap(
        self,
        sequence: int,
        key_fields: dict[str, str],
    ) -> ValidationResult:
        """Prüft auf Sequenzlücken.

        Args:
            sequence: Aktuelle Sequence-Nummer
            key_fields: Eindeutige Kombination (instrument, venue, event_type)

        Returns:
            ValidationResult mit Gap-Info falls vorhanden
        """
        composite_key = ":".join(str(v) for v in sorted(key_fields.values()))
        result = ValidationResult(
            event_type=key_fields.get("event_type", "unknown"),
            is_valid=True,
            quality_score=1.0,
        )

        last_seq = self._last_sequence.get(composite_key, 0)

        if last_seq > 0 and sequence > last_seq + 1:
            gap_size = sequence - last_seq - 1
            result.add_issue(ValidationIssue(
                field="sequence",
                message=f"Gap of {gap_size} events detected (last: {last_seq}, current: {sequence})",
                severity=Severity.WARNING,
            ))
            result.quality_score = max(0.0, result.quality_score - 0.3 * gap_size)

        if sequence < last_seq:
            result.add_issue(ValidationIssue(
                field="sequence",
                message=f"Sequence regression: last={last_seq}, current={sequence}",
                severity=Severity.ERROR,
            ))
            result.quality_score = 0.0

        self._last_sequence[composite_key] = sequence
        return result

    def reset(self, composite_key: str) -> None:
        """Setzt die Gap-Tracking für einen Schlüssel zurück."""
        self._last_sequence.pop(composite_key, None)
