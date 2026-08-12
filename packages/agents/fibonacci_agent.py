"""Fibonacci Analysis Agent — detects swing pivots, fib zones, and confluence."""
from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray
from packages.fibonacci import (
    ConfluenceResult,
    ConfluenceScanner,
    FibonacciArea,
    FibonacciPivot,
    FibonacciRetracement,
    PivotDetector,
)
from packages.schemas.agent_report import AgentReport, InvalidationCondition

from .base import AgentConfig, AgentType, BaseAgent

if TYPE_CHECKING:
    from packages.chart_structure.base import SupportResistanceLevel


class FibonacciAgent(BaseAgent):
    """Fibonacci-Analyse-Agent — erkennt Swing-Pivots, Fib-Zonen und Konfluenz.

    Der Agent durchlauft mehrere Stufen:
      1. Pivot-Erkennung aus OHLC-Daten (algorithmisch)
      2. Fibonacci-Retracement- und Extension-Zonen berechnen
      3. Konfluenz-Scoring über mehrere Timeframes / SR-Level
      4. Wahrscheinlichkeiten aus Zonen-Konfluenz (keine LLM-Schätzung)
      5. Gegenhypothese und Invalidierungen bilden
    """

    DEFAULT_ZONE_BAND = 0.005
    CONFLUENCE_THRESHOLD = 0.6
    SUPPORTED_KEYS: ClassVar[frozenset[str]] = frozenset({"high", "low", "close"})

    def __init__(
        self,
        config: AgentConfig | None = None,
        high_window: int = 10,
        low_window: int = 10,
        zone_band: float = 0.005,
        price_proximity: float = 0.002,
    ) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="fibonacci",
                agent_type=AgentType.FIBONACCI,
            )
        super().__init__(config)
        self._pivot_detector = PivotDetector(
            high_window=high_window, low_window=low_window
        )
        self._retracement = FibonacciRetracement(zone_band=zone_band)
        self._confluence_scanner = ConfluenceScanner(
            price_proximity=price_proximity
        )

    # ── public API ───────────────────────────────────────────────────────

    def analyze(
        self, data: dict[str, NDArray[np.float64]]
    ) -> AgentReport:
        """Analysiert OHLCV-Daten auf Fibonacci-Zonen und Konfluenz.

        Required keys:
            close (NDArray) - Schliessungspreise
            high  (NDArray) - Hoechste Preise
            low   (NDArray) - Tiefste Preise

        Optional keys:
            sr_levels (list[SupportResistanceLevel]) - SR-Level fuer Konfluenz

        Returns:
            AgentReport mit Fib-Zonen, Wahrscheinlichkeiten, Evidenz.

        Raises:
            ValueError: Wenn erforderliche Schlüssel fehlen.
        """
        missing = self.SUPPORTED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required data keys: {sorted(missing)}"
            )

        pivots: list[FibonacciPivot] = self._pivot_detector.detect_pivots(
            {"high": data["high"], "low": data["low"], "close": data["close"]}
        )
        areas: list[FibonacciArea] = self._retracement.calculate_retracements(
            pivots
        )
        sr_levels: list[SupportResistanceLevel] | None = data.get(
            "sr_levels"
        )  # type: ignore[assignment]
        confluence_results: list[ConfluenceResult] = (
            self._confluence_scanner.find_confluence(areas, sr_levels)
            if sr_levels is not None
            else []
        )
        areas = self._enrich_areas_with_confluence(areas, confluence_results)

        current_price: float = float(data["close"][-1])
        close: NDArray[np.float64] = data["close"]
        hypothesis, probabilities, evidence, counter_evidence, invalidations = (
            self._build_report(
                pivots=pivots,
                areas=areas,
                current_price=current_price,
                confluence_results=confluence_results,
                close=close,
            )
        )

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(),
            hypothesis=hypothesis,
            probabilities=probabilities,
            evidence=evidence,
            counter_evidence=counter_evidence,
            invalidations=invalidations,
            status=self.config.status,
            raw_confidence=self._compute_confidence(areas),
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _enrich_areas_with_confluence(
        self,
        areas: list[FibonacciArea],
        confluence_results: list[ConfluenceResult],
    ) -> list[FibonacciArea]:
        """Reichert FibonacciArea-Objekte mit Konfluenz-Scores auf."""
        if not confluence_results:
            return areas

        level_to_score: dict[str, float] = {}
        for cr in confluence_results:
            level_to_score[cr.level] = cr.score

        for area in areas:
            if area.level_types:
                key = str(area.level_types[0])
                score = level_to_score.get(key, 0.0)
                area.confluence_score = max(area.confluence_score, score)

        return areas

    def _build_report(
        self,
        pivots: list[FibonacciPivot],
        areas: list[FibonacciArea],
        current_price: float,
        confluence_results: list[ConfluenceResult],
        close: NDArray[np.float64],
    ) -> tuple[
        str,
        dict[str, float],
        list,  # EvidenceReference
        list,  # EvidenceReference
        list,  # InvalidationCondition
    ]:
        """Erstellt Hypothese, Wahrscheinlichkeiten, Evidenz, Gegenhypothese, Invalidierungen."""
        high_pivots = [p for p in pivots if p.type == "swing_high"]
        low_pivots = [p for p in pivots if p.type == "swing_low"]

        # ── Hypothese formulieren ──
        zone_descriptions: list[str] = []
        for area in areas:
            if area.confluence_score >= self.CONFLUENCE_THRESHOLD:
                types_str = ", ".join(area.level_types)
                zone_descriptions.append(
                    f"{types_str} @ {area.lower:.2f}-{area.upper:.2f} "
                    f"(confluence={area.confluence_score:.2f})"
                )

        if zone_descriptions:
            hypothesis = (
                f"Fibonacci-Zonen erkannt: {len(areas)} Zonen, "
                f"{len(confluence_results)} mit hoher Konfluenz. "
                f"Pivots: {len(high_pivots)} High, {len(low_pivots)} Low. "
                f"Starke Zonen: {', '.join(zone_descriptions[:3])}"
            )
        elif areas:
            hypothesis = (
                f"Fibonacci-Zonen erkannt: {len(areas)} Zonen "
                f"({len(high_pivots)} High-Pivots, {len(low_pivots)} Low-Pivots), "
                f"keine hohe Konfluenz bei {current_price:.2f}"
            )
        else:
            hypothesis = (
                f"Nicht genug Pivots ({len(pivots)}): "
                f"benötigt mindestens 2 für Retracement-Berechnung"
            )

        # ── Wahrscheinlichkeiten aus Zonen-Konfluenz ──
        probabilities = self._compute_probabilities(
            areas, current_price, confluence_results, close
        )

        # ── Evidenz ──
        evidence: list = []
        high_confluence_areas = [
            a for a in areas if a.confluence_score >= self.CONFLUENCE_THRESHOLD
        ]
        for i, area in enumerate(high_confluence_areas):
            types_str = ", ".join(area.level_types)
            evidence.append(
                self._make_evidence(
                    f"fib_zone_{i}",
                    f"{types_str} @ {area.lower:.2f}-{area.upper:.2f}",
                    "positive",
                    area.confluence_score,
                )
            )

        # Evidenz aus Pivots
        if high_pivots:
            latest_high = high_pivots[-1]
            evidence.append(
                self._make_evidence(
                    "pivot_high",
                    f"{latest_high.price:.2f} at t={latest_high.time}",
                    "positive",
                    0.5,
                )
            )
        if low_pivots:
            latest_low = low_pivots[-1]
            evidence.append(
                self._make_evidence(
                    "pivot_low",
                    f"{latest_low.price:.2f} at t={latest_low.time}",
                    "positive",
                    0.5,
                )
            )

        # Konfluenz als Evidenz
        for cr in confluence_results:
            evidence.append(
                self._make_evidence(
                    f"confluence_{cr.level}",
                    f"score={cr.score:.2f}, matches={len(cr.matching_prices)}",
                    "positive",
                    min(cr.score, 0.9),
                )
            )

        # Sicherstellen: min 1 evidence
        if not evidence:
            evidence.append(
                self._make_evidence(
                    "fib_levels",
                    f"{len(areas)} zones computed from {len(pivots)} pivots",
                    "positive",
                    0.3,
                )
            )

        # ── Gegenhypothese (counter_evidence) ──
        counter_evidence: list = []
        counter_evidence = self._build_counter_evidence(
            areas, current_price, high_pivots, low_pivots
        )

        # ── Invalidierungen ──
        invalidations: list[InvalidationCondition] = self._build_invalidations(
            areas, high_pivots, low_pivots
        )

        return hypothesis, probabilities, evidence, counter_evidence, invalidations

    def _compute_probabilities(
        self,
        areas: list[FibonacciArea],
        current_price: float,
        confluence_results: list[ConfluenceResult],
        close: NDArray[np.float64],
    ) -> dict[str, float]:
        """Berechnet up/down/range-Wahrscheinlichkeiten aus Zonen-Konfluenz.

        Keine LLM-Schaetzung — rein algorithmisch basierend auf:
          - Trend-Richtung aus Close-Preisen (momentum)
          - Anzahl Zonen mit hoher Konfluenz oberhalb / unterhalb des Preises
          - Distanz des aktuellen Preises zu den naechsten Fib-Zonen
        """
        if not areas:
            return self._probabilities_from_trend(close, trend_weight=0.1)

        # Trend-Momentum aus Close-Preisen
        close_series = close[-30:] if len(close) >= 30 else close
        mid = len(close_series) // 2
        first_half_avg = float(np.mean(close_series[:mid]))
        second_half_avg = float(np.mean(close_series[mid:]))
        if first_half_avg > 0:
            trend_pct = (second_half_avg - first_half_avg) / first_half_avg
        else:
            trend_pct = 0.0

        # Zonen-Konfluenz
        above_high_confluence = 0
        below_high_confluence = 0
        above_any = 0
        below_any = 0

        for area in areas:
            center = (area.lower + area.upper) / 2
            if center > current_price:
                above_any += 1
                if area.confluence_score >= self.CONFLUENCE_THRESHOLD:
                    above_high_confluence += 1
            else:
                below_any += 1
                if area.confluence_score >= self.CONFLUENCE_THRESHOLD:
                    below_high_confluence += 1

        # Gewichtete Score-Summe
        above_score = sum(
            a.confluence_score
            for a in areas
            if (a.lower + a.upper) / 2 > current_price
        )
        below_score = sum(
            a.confluence_score
            for a in areas
            if (a.lower + a.upper) / 2 <= current_price
        )
        total_score = above_score + below_score

        # Konfluenz-Ergebnisse gewichten zusätzlich
        confluence_boost = min(0.15, len(confluence_results) * 0.03)

        # Combine trend and fib-zone weights
        if total_score > 0:
            up_weight = (above_score + confluence_boost) / total_score
            down_weight = (below_score + confluence_boost) / total_score
        else:
            up_weight = 0.5
            down_weight = 0.5

        trend_factor = min(0.4, max(-0.4, trend_pct * 100))
        up_weight += trend_factor
        down_weight -= trend_factor

        up_weight = max(0.05, min(0.95, up_weight))
        down_weight = max(0.05, min(0.95, down_weight))

        base_up = 0.2 + 0.3 * up_weight + confluence_boost * 2
        base_down = 0.2 + 0.3 * down_weight + confluence_boost * 2
        strength = abs(up_weight - down_weight)
        base_range = 0.2 + 0.1 * (1 - strength)

        total = base_up + base_down + base_range
        up_prob = round(base_up / total, 4)
        down_prob = round(base_down / total, 4)
        range_prob = round(1.0 - up_prob - down_prob, 4)

        return {"up": up_prob, "down": down_prob, "range": range_prob}

    def _probabilities_from_trend(
        self, close: NDArray[np.float64], *, trend_weight: float = 0.2
    ) -> dict[str, float]:
        close_series = close[-30:] if len(close) >= 30 else close
        mid = len(close_series) // 2
        first_half_avg = float(np.mean(close_series[:mid]))
        second_half_avg = float(np.mean(close_series[mid:]))
        if first_half_avg > 0:
            trend_pct = (second_half_avg - first_half_avg) / first_half_avg
        else:
            trend_pct = 0.0

        trend_factor = min(trend_weight, max(-trend_weight, trend_pct * 100))
        base_up = 0.33 + trend_factor
        base_down = 0.33 - trend_factor
        base_range = 1.0 - base_up - base_down
        if base_range < 0.05:
            base_range = 0.05
            base_up = 0.475
            base_down = 0.475

        return {"up": round(base_up, 4), "down": round(base_down, 4), "range": round(1.0 - base_up - base_down, 4)}

    def _build_counter_evidence(
        self,
        areas: list[FibonacciArea],
        current_price: float,
        high_pivots: list[FibonacciPivot],
        low_pivots: list[FibonacciPivot],
    ) -> list:
        """Erstellt Gegenhypothesen-Evidenz.

        Die Gegenhypothese ist die entgegengesetzte Richtung zur aktuellen
        Fib-Zonen-Struktur. Wenn viele Zonen oberhalb konfluieren, ist die
        Gegenhypothese down.
        """
        counter: list = []

        if not areas:
            counter.append(
                self._make_evidence(
                    "no_zones",
                    "keine Fibonacci-Zonen erkannt",
                    "negative",
                    0.3,
                )
            )
            return counter

        above_center = sum(
            1
            for a in areas
            if (a.lower + a.upper) / 2 > current_price
        )
        below_center = sum(
            1
            for a in areas
            if (a.lower + a.upper) / 2 <= current_price
        )

        if above_center >= below_center:
            # Hauptthese up → Gegenhypothese down
            if low_pivots:
                counter.append(
                    self._make_evidence(
                        "lower_pivot_opposing",
                        f"niedrigster Low-Pivot {low_pivots[-1].price:.2f}",
                        "negative",
                        0.4,
                    )
                )
            # Strongest opposition level (lowest fib level below price)
            below_areas = [
                a
                for a in areas
                if (a.lower + a.upper) / 2 <= current_price
            ]
            if below_areas:
                weakest = min(below_areas, key=lambda a: a.confluence_score)
                counter.append(
                    self._make_evidence(
                        "fib_support_break",
                        f"{weakest.level_types[0] if weakest.level_types else 'fib'} zone at {weakest.lower:.2f}-{weakest.upper:.2f}",
                        "negative",
                        max(0.2, 1.0 - weakest.confluence_score),
                    )
                )
        else:
            # Hauptthese down → Gegenhypothese up
            if high_pivots:
                counter.append(
                    self._make_evidence(
                        "higher_pivot_opposing",
                        f"höchster High-Pivot {high_pivots[-1].price:.2f}",
                        "negative",
                        0.4,
                    )
                )
            above_areas = [
                a
                for a in areas
                if (a.lower + a.upper) / 2 > current_price
            ]
            if above_areas:
                weakest = min(above_areas, key=lambda a: a.confluence_score)
                counter.append(
                    self._make_evidence(
                        "fib_resistance_break",
                        f"{weakest.level_types[0] if weakest.level_types else 'fib'} zone at {weakest.lower:.2f}-{weakest.upper:.2f}",
                        "negative",
                        max(0.2, 1.0 - weakest.confluence_score),
                    )
                )

        return counter

    def _build_invalidations(
        self,
        areas: list[FibonacciArea],
        high_pivots: list[FibonacciPivot],
        low_pivots: list[FibonacciPivot],
    ) -> list[InvalidationCondition]:
        """Erstellt Invalidierungsbedingungen für die Fib-Hypothese."""
        invalidations: list[InvalidationCondition] = []

        if not areas:
            invalidations.append(
                self._make_invalidations(
                    condition="Zu wenige Pivots für Fib-Berechnung",
                    indicator="pivot_count",
                    threshold=2.0,
                    direction="below",
                )
            )
            return invalidations

        # Invalidation: Preis durchbricht die stärkste Zone
        if areas:
            strongest = max(areas, key=lambda a: a.confluence_score)
            invalidations.append(
                self._make_invalidations(
                    condition=(
                        f"Preis durchbricht stärkste Fib-Zone "
                        f"{strongest.level_types[0] if strongest.level_types else 'fib'} "
                        f"bei {strongest.lower:.2f}"
                    ),
                    indicator=f"fib_zone_{strongest.level_types[0] if strongest.level_types else 'unknown'}",
                    threshold=strongest.lower,
                    direction="below"
                    if (strongest.lower + strongest.upper) / 2 > 0
                    else "above",
                )
            )
            invalidations.append(
                self._make_invalidations(
                    condition=(
                        f"Preis bricht oberhalb stärkste Fib-Zone "
                        f"{strongest.level_types[0] if strongest.level_types else 'fib'} "
                        f"bei {strongest.upper:.2f}"
                    ),
                    indicator=f"fib_zone_{strongest.level_types[0] if strongest.level_types else 'unknown'}",
                    threshold=strongest.upper,
                    direction="above",
                )
            )

        # Invalidation: neuer Swing-Pivot im Widerspruch zur Fib-Struktur
        if high_pivots:
            invalidations.append(
                self._make_invalidations(
                    condition="Neuer Swing-High widerlegt Fib-Retracement-Struktur",
                    indicator="pivot_high",
                    threshold=high_pivots[-1].price,
                    direction="above",
                )
            )
        if low_pivots:
            invalidations.append(
                self._make_invalidations(
                    condition="Neuer Swing-Low widerlegt Fib-Retracement-Struktur",
                    indicator="pivot_low",
                    threshold=low_pivots[-1].price,
                    direction="below",
                )
            )

        return invalidations

    def _compute_confidence(self, areas: list[FibonacciArea]) -> float:
        """Berechnet raw_confidence basierend auf Zonen-Anzahl und Konfluenz."""
        if not areas:
            return 0.0

        avg_confluence = sum(a.confluence_score for a in areas) / len(areas)
        # Confidence: Anzahl Zonen (logarithmisch gedämpft) + durchschnittliche Konfluenz
        num_factor = min(0.3, 0.1 * np.log1p(len(areas)))
        confidence = num_factor + 0.4 * avg_confluence
        return round(min(0.95, confidence), 4)
