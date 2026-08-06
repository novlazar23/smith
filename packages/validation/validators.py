"""Validation — Cross-field und Data Quality Validierung.

Stellt sicher, dass alle Eingabedaten die DoD-Anforderungen erfüllen:
  - AT-004: Mindestens eine Evidenz pro Agenten-Aussage
  - AT-005: Wahrscheinlichkeitssumme = 1.0 ± 0.0001
  - AT-011: Point-in-Time-Korrektheit (availability <= analysis_time)
  - AT-012: Data Quality Score >= 0.8
  - AT-013: max_position_size >= 0 und <= 1.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class ValidationResult(BaseModel):
    """Ergebnis einer einzelnen Validierungsprüfung."""

    model_config = ConfigDict(frozen=True)

    check: str
    passed: bool
    message: str = ""


class Validator(ABC):
    """Basis-Protocol für alle Validatoren."""

    @abstractmethod
    def validate(self, **kwargs: Any) -> list[ValidationResult]:  # noqa: ANN401
        """Führt alle Prüfungen durch und gibt die Ergebnisse zurück."""


class DataQualityValidator(Validator):
    """Prüft Data Quality-Anforderungen (AT-012).

    Anforderungen:
      - Mindestens eine Evidenz pro Report (AT-004)
      - Wahrscheinlichkeitssumme = 1.0 ± 0.0001 (AT-005)
      - Data Quality Score >= 0.8
    """

    PROBABILITY_TOLERANCE = 0.0001
    MIN_DATA_QUALITY = 0.8

    def validate(  # type: ignore[override]
        self,
        probabilities: dict[str, float],
        evidence_count: int,
        counter_evidence_count: int = 0,
        data_quality: float = 1.0,
        raw_confidence: float | None = None,
        calibrated_confidence: float | None = None,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []

        # AT-004: Mindestens eine Evidenz
        results.append(
            ValidationResult(
                check="evidence_minimum",
                passed=evidence_count >= 1,
                message="Mindestens eine Evidenz erforderlich (AT-004)",
            )
        )

        # AT-005: Wahrscheinlichkeitssumme
        if probabilities:
            total = sum(probabilities.values())
            prob_ok = abs(total - 1.0) <= self.PROBABILITY_TOLERANCE
            results.append(
                ValidationResult(
                    check="probability_sum",
                    passed=prob_ok,
                    message=f"Wahrscheinlichkeitssumme={total:.6f}, "
                    f"erlaubt 1.0 ± {self.PROBABILITY_TOLERANCE} (AT-005)",
                )
            )

        # AT-012: Data Quality Score
        dq_ok = data_quality >= self.MIN_DATA_QUALITY
        results.append(
            ValidationResult(
                check="data_quality_score",
                passed=dq_ok,
                message=f"Data Quality {data_quality:.2f} >= {self.MIN_DATA_QUALITY} (AT-012)",
            )
        )

        # Confidence must be calibrated
        if raw_confidence is not None and calibrated_confidence is not None:
            results.append(
                ValidationResult(
                    check="confidence_calibration",
                    passed=calibrated_confidence <= raw_confidence,
                    message="Calibrated confidence <= raw confidence",
                )
            )

        # Counter-evidence sollte vorhanden sein für starke Aussagen
        if evidence_count > 0 and counter_evidence_count == 0 and data_quality >= 0.9:
            results.append(
                ValidationResult(
                    check="counter_evidence_missing",
                    passed=False,
                    message="Keine Gegenhypothesen-Evidenz bei hoher Qualität",
                )
            )

        return results


class PointInTimeValidator(Validator):
    """Verhindert Look-Ahead-Bias durch Point-in-Time-Prüfung (AT-011).

    Anforderungen:
      - availability_time <= analysis_time (keine Zukunftsinformationen)
      - ingestion_time >= event_time (Ereignis vor Aufnahme)
      - Quellenqualität >= Schwellenwert
    """

    MIN_SOURCE_QUALITY = 0.5

    def validate(  # type: ignore[override]
        self,
        analysis_time: datetime,
        availability_time: datetime,
        ingestion_time: datetime,
        event_time: datetime,
        source_quality: float = 1.0,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []

        # AT-011: availability_time <= analysis_time
        results.append(
            ValidationResult(
                check="no_lookahead_availability",
                passed=availability_time <= analysis_time,
                message=f"availability_time ({availability_time.isoformat()}) "
                f"<= analysis_time ({analysis_time.isoformat()})",
            )
        )

        # ingestion_time >= event_time  # noqa: ERA001
        results.append(
            ValidationResult(
                check="ingestion_after_event",
                passed=ingestion_time >= event_time,
                message="Ingestion muss nach dem Ereignis erfolgen",
            )
        )

        # event_time <= analysis_time (Ereignis darf nicht in der Zukunft liegen)
        results.append(
            ValidationResult(
                check="event_not_future",
                passed=event_time <= analysis_time,
                message="Ereignis darf nicht in der Zukunft liegen",
            )
        )

        # Source quality threshold
        results.append(
            ValidationResult(
                check="source_quality_threshold",
                passed=source_quality >= self.MIN_SOURCE_QUALITY,
                message=f"Source quality {source_quality} >= {self.MIN_SOURCE_QUALITY}",
            )
        )

        return results


class CrossFieldValidator(Validator):
    """Kreuzfeld-Validierung über mehrere Domänen hinweg.

    Prüft:
      - RiskGate: max_position_size >= 0 und <= 1.0 (AT-013)
      - RiskDecision: veto consistency mit hard gates
      - FinalDecision: reason nicht leer wenn NO_TRADE
      - Portfolio: exposure limits eingehalten
    """

    def validate(  # type: ignore[override]
        self,
        risk_approved: bool = True,
        has_hard_block: bool = False,
        decision_type: str = "LONG_BIAS",
        reason: str = "",
        blocking_reasons: list[str] | None = None,
        max_position_size: float | None = None,
        reduction_factor: float = 1.0,
        portfolio_exposure_ratio: float = 0.0,
        portfolio_max_exposure: float = 1.0,
        is_long: bool = False,
        is_short: bool = False,
        is_range: bool = False,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        blocking_reasons = blocking_reasons or []

        # AT-013: max_position_size Range
        if max_position_size is not None:
            results.append(
                ValidationResult(
                    check="max_position_size_range",
                    passed=0.0 <= max_position_size <= 1.0,
                    message=f"max_position_size {max_position_size} in [0, 1] (AT-013)",
                )
            )

        # Risk veto consistency
        if has_hard_block and risk_approved:
            results.append(
                ValidationResult(
                    check="veto_consistency",
                    passed=False,
                    message="Hard gate blockiert, aber risk_approved=True",
                )
            )

        # NO_TRADE decision requires reason
        if decision_type.startswith("NO_TRADE"):
            results.append(
                ValidationResult(
                    check="no_trade_reason",
                    passed=bool(reason) and len(reason.strip()) > 0,
                    message="NO_TRADE decision erfordert Begründung",
                )
            )
            results.append(
                ValidationResult(
                    check="no_trade_blocking_reasons",
                    passed=len(blocking_reasons) > 0,
                    message="NO_TRADE decision erfordert blocking_reasons",
                )
            )

        # Risk veto overrides decision
        if has_hard_block:
            results.append(
                ValidationResult(
                    check="veto_overrides_decision",
                    passed=decision_type.startswith("NO_TRADE"),
                    message="Hard gate veto sollte NO_TRADE erzwingen",
                )
            )

        # Portfolio exposure limits
        results.append(
            ValidationResult(
                check="portfolio_exposure_limit",
                passed=portfolio_exposure_ratio <= portfolio_max_exposure,
                message=f"Portfolio-Exposure {portfolio_exposure_ratio:.2%} "
                f"<= {portfolio_max_exposure:.0%} Limit",
            )
        )

        # Single direction: not long AND short simultaneously
        results.append(
            ValidationResult(
                check="single_direction",
                passed=not (is_long and is_short),
                message="Nicht gleichzeitig LONG und SHORT",
            )
        )

        # Reduction factor range
        results.append(
            ValidationResult(
                check="reduction_factor_range",
                passed=0.0 <= reduction_factor <= 1.0,
                message=f"reduction_factor {reduction_factor} in [0, 1]",
            )
        )

        return results


class MarketEventValidator(Validator):
    """Validiert Marktdaten-Ereignisse auf Integrität.

    Prüft:
      - Candle: high >= low > 0, is_closed consistency
      - Trade: price > 0, quantity > 0
      - OrderBook: bids/asks non-empty
    """

    def validate_candle(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        is_closed: bool = True,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []

        results.append(
            ValidationResult(
                check="candle_low_positive",
                passed=low_price > 0,
                message="low > 0",
            )
        )
        results.append(
            ValidationResult(
                check="candle_high_gte_low",
                passed=high_price >= low_price,
                message="high >= low",
            )
        )
        results.append(
            ValidationResult(
                check="candle_open_positive",
                passed=open_price > 0,
                message="open > 0",
            )
        )
        results.append(
            ValidationResult(
                check="candle_close_positive",
                passed=close_price > 0,
                message="close > 0",
            )
        )

        return results

    def validate_trade(
        self,
        price: float,
        quantity: float,
    ) -> list[ValidationResult]:
        return [
            ValidationResult(
                check="trade_price_positive",
                passed=price > 0,
                message="price > 0",
            ),
            ValidationResult(
                check="trade_quantity_positive",
                passed=quantity > 0,
                message="quantity > 0",
            ),
        ]

    def validate_orderbook(
        self,
        bid_count: int,
        ask_count: int,
    ) -> list[ValidationResult]:
        return [
            ValidationResult(
                check="orderbook_bids",
                passed=bid_count > 0,
                message=f"bids: {bid_count} > 0",
            ),
            ValidationResult(
                check="orderbook_asks",
                passed=ask_count > 0,
                message=f"asks: {ask_count} > 0",
            ),
        ]

    def validate(
        self,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[ValidationResult]:
        """Dispatch-basierte Validierung je nach event_type."""
        event_type = kwargs.get("event_type", "")
        if event_type == "candle":
            return self.validate_candle(
                kwargs.get("open", 0),
                kwargs.get("high", 0),
                kwargs.get("low", 0),
                kwargs.get("close", 0),
                kwargs.get("is_closed", True),
            )
        if event_type == "trade":
            return self.validate_trade(
                kwargs.get("price", 0),
                kwargs.get("quantity", 0),
            )
        if event_type == "orderbook":
            return self.validate_orderbook(
                kwargs.get("bid_count", 0),
                kwargs.get("ask_count", 0),
            )
        return [
            ValidationResult(
                check="unknown_event_type",
                passed=False,
                message=f"Unbekannter event_type: {event_type}",
            )
        ]


__all__ = [
    "CrossFieldValidator",
    "DataQualityValidator",
    "MarketEventValidator",
    "PointInTimeValidator",
    "ValidationResult",
    "Validator",
]
