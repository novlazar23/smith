"""Shadow Quality Metrics.

Tracks predictive quality of shadow-mode decisions:
- Brier Score for probabilistic predictions
- Calibration curves and statistics
- Virtual PnL tracking (no real capital)

All metrics are purely observational — shadow decisions are
never executed, only compared against later outcomes.

Public API:
    - ShadowMetrics — accumulates and queries quality metrics
    - ShadowBrierScore — accumulates Brier Score components
    - ShadowCalibration — tracks calibration data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ProbabilityOutcome:
    """Ein einzelner Wahrscheinlichkeits-Outcome-Paar.

    Used for Brier Score and calibration tracking.
    """

    predicted_up: float
    predicted_down: float
    predicted_range: float
    actual_direction: str  # "up", "down", "range"
    actual_return: float | None = None


@dataclass
class ShadowBrierScore:
    """Bewertung probabilistischer Vorhersagen via Brier Score.

    Brier Score = mean((p_actual - y_actual)^2) über alle Vorhersagen.

    For three-class (up/down/range), uses the multi-class variant:
    Brier = mean(sum over k of (f_ik - y_ik)^2)
    """

    _scores: list[float] = field(default_factory=list)

    def add(self, predicted_up: float, predicted_down: float, predicted_range: float, actual: str) -> float:
        """Fügt ein neues Vorhersage-Outcome hinzu.

        Computes the individual Brier Score for this prediction
        and appends it to the running total.

        Args:
            predicted_up: Vorhergesagte Wahrscheinlichkeit für "up".
            predicted_down: Vorhergesagte Wahrscheinlichkeit für "down".
            predicted_range: Vorhergesagte Wahrscheinlichkeit für "range".
            actual: Tatsächliche Richtung ("up", "down", "range").

        Returns:
            Der Brier Score für diese einzelne Vorhersage.
        """
        actual_vector: list[float] = [0.0, 0.0, 0.0]
        actual_idx = {"up": 0, "down": 1, "range": 2}.get(actual)
        if actual_idx is None:
            msg = f"Unknown actual direction: {actual!r}"
            raise ValueError(msg)
        actual_vector[actual_idx] = 1.0

        # Predicted vector
        predicted = [predicted_up, predicted_down, predicted_range]

        # Individual Brier Score: sum of squared errors
        score = sum((p - a) ** 2 for p, a in zip(predicted, actual_vector, strict=True))
        self._scores.append(score)

        return score

    @property
    def mean_score(self) -> float:
        """Durchschnittlicher Brier Score über alle Vorhersagen."""
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    @property
    def count(self) -> int:
        """Anzahl der ausgewerteten Vorhersagen."""
        return len(self._scores)

    @property
    def is_valid(self) -> bool:
        """True wenn mindestens eine Bewertung vorliegt."""
        return len(self._scores) > 0


@dataclass
class ShadowCalibration:
    """Kalibrierungs-Tracking für Shadow-Entscheidungen.

    Misst, wie gut vorhergesagte Konfidenz与实际结果 übereinstimmt.
    E.g., if predictions have 80% confidence, actual outcomes
    should be correct ~80% of the time.
    """

    _bins: dict[str, list[tuple[float, bool]]] = field(default_factory=dict)
    _bin_boundaries: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    def __post_init__(self) -> None:
        """Initialisiert leere Bins."""
        if not self._bins:
            for i in range(len(self._bin_boundaries) - 1):
                low = self._bin_boundaries[i]
                high = self._bin_boundaries[i + 1]
                label = f"{low:.1f}-{high:.1f}"
                self._bins[label] = []

    def add(
        self,
        confidence: float,
        correct: bool,  # noqa: FBT001
        direction: str,
    ) -> None:
        """Fügt eine neue Konfidenz-Wahrheit-Beobachtung hinzu.

        Args:
            confidence: Vorhergesagte Konfidenz (0.0-1.0).
            correct: True wenn die Vorhersage korrekt war.
            direction: Richtung der Vorhersage.
        """
        # Find bin
        bin_label = self._find_bin(confidence)
        self._bins[bin_label].append((confidence, correct))

    def _find_bin(self, confidence: float) -> str:
        """Findet das passende Bin für einen Konfidenzwert."""
        for i in range(len(self._bin_boundaries) - 1):
            low = self._bin_boundaries[i]
            high = self._bin_boundaries[i + 1]
            if low <= confidence < high:
                return f"{low:.1f}-{high:.1f}"
        return f"{self._bin_boundaries[-2]:.1f}-{self._bin_boundaries[-1]:.1f}"

    @property
    def calibration_data(self) -> dict[str, dict[str, float]]:
        """Berechnet aggregierte Kalibrierungsdaten pro Bin."""
        result: dict[str, dict[str, float]] = {}

        for label, observations in self._bins.items():
            if not observations:
                result[label] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                }
                continue

            avg_conf = sum(c for c, _ in observations) / len(observations)
            accuracy = sum(1 for _, c in observations if c) / len(observations)

            result[label] = {
                "count": len(observations),
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(accuracy, 4),
            }

        return result

    @property
    def reliability_diagram_data(self) -> list[dict[str, float]]:
        """Returns data for a reliability (calibration) diagram.

        Returns list of dicts with avg_confidence and accuracy for
        non-empty bins only.
        """
        return [
            data for data in self.calibration_data.values()
            if data["count"] > 0
        ]

    @property
    def total_observations(self) -> int:
        """Gesamtanzahl der Kalibrierungsbeobachtungen."""
        return sum(len(v) for v in self._bins.values())

    @property
    def overall_accuracy(self) -> float:
        """Gesamte Genauigkeit über alle Beobachtungen."""
        total_correct = 0
        total_count = 0
        for observations in self._bins.values():
            for _, correct in observations:
                if correct:
                    total_correct += 1
                total_count += 1
        return total_correct / total_count if total_count > 0 else 0.0


@dataclass
class ShadowPnLRecord:
    """Ein einzelner Shadow PnL-Eintrag (virtuell, keine Ausführung)."""

    run_id: str
    direction: str
    predicted_return: float | None
    actual_return: float | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def virtual_pnl(self) -> float | None:
        """Virtueller PnL: return * direction_sign.

        If direction is LONG or RANGE (neutral), PnL is the raw return.
        If direction is SHORT, PnL is the negated return.

        Returns None if actual_return is not yet known.
        """
        if self.actual_return is None:
            return None
        if self.direction in ("LONG", "LONG_BIAS"):
            return self.actual_return
        if self.direction in ("SHORT", "SHORT_BIAS"):
            return -self.actual_return
        # RANGE or NO_TRADE — no PnL
        return 0.0


@dataclass
class ShadowPnL:
    """Tracking des virtuellen PnL über alle Shadow-Entscheidungen."""

    _records: list[ShadowPnLRecord] = field(default_factory=list)

    def add(
        self,
        run_id: str,
        direction: str,
        predicted_return: float | None,
        actual_return: float | None,
    ) -> ShadowPnLRecord:
        """Fügt einen neuen PnL-Eintrag hinzu.

        Args:
            run_id: Run-ID der Shadow-Entscheidung.
            direction: Entscheidung Richtung.
            predicted_return: Vorhergesagte Rendite.
            actual_return: Tatsächliche Rendite (wenn bekannt).

        Returns:
            ShadowPnLRecord für diesen Eintrag.
        """
        record = ShadowPnLRecord(
            run_id=run_id,
            direction=direction,
            predicted_return=predicted_return,
            actual_return=actual_return,
        )
        self._records.append(record)
        return record

    @property
    def total_virtual_pnl(self) -> float:
        """Gesamt-PnL aus allen abgeschlossenen Entscheidungen."""
        return sum(
            r.virtual_pnl or 0.0
            for r in self._records
            if r.virtual_pnl is not None
        )

    @property
    def realized_count(self) -> int:
        """Anzahl Entscheidungen mit bekanntem tatsächlichen Return."""
        return sum(
            1 for r in self._records if r.virtual_pnl is not None
        )

    @property
    def count(self) -> int:
        """Gesamtanzahl aller PnL-Einträge."""
        return len(self._records)

    @property
    def records(self) -> list[ShadowPnLRecord]:
        """Alle PnL-Einträge."""
        return list(self._records)


@dataclass
class ShadowMetrics:
    """Zentrale Sammlung aller Shadow-Qualitätsmetriken.

    Combines Brier Score, calibration, and PnL tracking.

    Example:
        metrics = ShadowMetrics()
        metrics.record_prediction(
            predicted_up=0.7, predicted_down=0.2, predicted_range=0.1,
            actual="up", confidence=0.75, correct=True,
        )
        metrics.record_pnl(
            run_id="run-001", direction="LONG",
            predicted_return=0.05, actual_return=0.03,
        )
        print(metrics.brier_score.mean_score)
        print(metrics.calibration.overall_accuracy)
        print(metrics.pnl.total_virtual_pnl)
    """

    brier_score: ShadowBrierScore = field(default_factory=ShadowBrierScore)
    calibration: ShadowCalibration = field(default_factory=ShadowCalibration)
    pnl: ShadowPnL = field(default_factory=ShadowPnL)

    def record_prediction(
        self,
        predicted_up: float,
        predicted_down: float,
        predicted_range: float,
        actual: str,
        confidence: float | None = None,
        correct: bool | None = None,  # noqa: FBT001
    ) -> None:
        """Recordet eine vollständige Shadow-Vorhersage.

        Updates Brier Score and calibration simultaneously.

        Args:
            predicted_up: Vorhergesagte Wahrscheinlichkeit "up".
            predicted_down: Vorhergesagte Wahrscheinlichkeit "down".
            predicted_range: Vorhergesagte Wahrscheinlichkeit "range".
            actual: Tatsächliche Richtung.
            confidence: Vorhergesagte Konfidenz (optional).
            correct: Ob korrekt (optional).
        """
        self.brier_score.add(
            predicted_up, predicted_down, predicted_range, actual
        )

        if confidence is not None and correct is not None:
            self.calibration.add(
                confidence=confidence,
                correct=correct,
                direction=actual,
            )

    def record_pnl(
        self,
        run_id: str,
        direction: str,
        predicted_return: float | None,
        actual_return: float | None,
    ) -> None:
        """Recordet einen PnL-Eintrag.

        Args:
            run_id: Run-ID.
            direction: Entscheidung Richtung.
            predicted_return: Vorhergesagte Rendite.
            actual_return: Tatsächliche Rendite (None wenn noch offen).
        """
        self.pnl.add(
            run_id=run_id,
            direction=direction,
            predicted_return=predicted_return,
            actual_return=actual_return,
        )

    @property
    def summary(self) -> dict[str, Any]:
        """Kompakte Zusammenfassung aller Metriken."""
        return {
            "brier_score": {
                "mean": round(self.brier_score.mean_score, 6),
                "count": self.brier_score.count,
                "is_valid": self.brier_score.is_valid,
            },
            "calibration": {
                "overall_accuracy": round(self.calibration.overall_accuracy, 4),
                "total_observations": self.calibration.total_observations,
                "bins": self.calibration.calibration_data,
            },
            "pnl": {
                "total_virtual_pnl": round(self.pnl.total_virtual_pnl, 6),
                "realized_count": self.pnl.realized_count,
                "total_count": self.pnl.count,
            },
        }
