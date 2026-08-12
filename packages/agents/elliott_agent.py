"""Elliott Wave Agent — multi-scenario wave counting with rule validation."""

from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.elliott_wave.core import (
    ElliottScenario,
    ElliottWaveDetector,
    WaveCount,
    WaveType,
)
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent

SUPPORTED_KEYS = frozenset({"close", "high", "low"})


class ElliottAgent(BaseAgent):
    """Elliott-Wave-Agent — erzeugt Multi-Szenario-Analyse.

    Produziert mindestens 2 Regel-gepruefte Szenarios:
      1. Primaires Impuls-Szenario
      2. Alternatives Korrektur-Szenario
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        pivot_window: int = 5,
    ) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="elliott",
                agent_type=AgentType.ELLIOTT,
            )
        super().__init__(config)
        self._detector = ElliottWaveDetector(pivot_window=pivot_window)

    def analyze(
        self, data: dict[str, NDArray[np.float64]]
    ) -> AgentReport:
        """Analysiert Daten auf Elliott-Wave-Szenarios.

        Required keys:
            close (NDArray)
            high  (NDArray)
            low   (NDArray)

        Returns:
            AgentReport mit mindestens 2 Szenarios.

        Raises:
            ValueError: Wenn erforderliche Fehlt.
        """
        missing = SUPPORTED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required data keys: {sorted(missing)}"
            )

        scenarios = self._detector.detect(data)
        scenarios = self._ensure_min_two_scenarios(scenarios)

        current_price = float(data["close"][-1])

        hypothesis, probabilities, evidence, counter_evidence, invalidations = (
            self._build_report(scenarios, current_price)
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
            raw_confidence=self._compute_confidence(scenarios),
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _ensure_min_two_scenarios(
        self, scenarios: list[ElliottScenario]
    ) -> list[ElliottScenario]:
        """Stellt sicher, dass mindestens 2 Szenarios zurueckgegeben werden."""
        if len(scenarios) >= 2:
            return scenarios

        if len(scenarios) == 1:
            s = scenarios[0]
            opposite = ElliottScenario(
                scenario_id=f"fallback_{s.scenario_id}",
                wave_count=s.wave_count,
                direction="down" if s.direction == "up" else "up",
                rule_valid=True,
                guideline_score=0.3,
                invalidation_price=s.invalidation_price * 0.98,
                description=f"Fallback alternative to {s.description}",
            )
            return [s, opposite]

        # No scenarios at all — create fallback impulse/correction pair
        return [
            ElliottScenario(
                scenario_id="no_wave_up",
                wave_count=WaveCount(wave_type=WaveType.IMPULSE, directions=["up"] * 5),
                direction="up",
                rule_valid=False,
                guideline_score=0.1,
                invalidation_price=0.0,
                description="No wave structure detected, default impulse up",
            ),
            ElliottScenario(
                scenario_id="no_wave_down",
                wave_count=WaveCount(wave_type=WaveType.CORRECTIVE, directions=["down"] * 3),
                direction="down",
                rule_valid=False,
                guideline_score=0.1,
                invalidation_price=0.0,
                description="No wave structure detected, default corrective down",
            ),
        ]

    def _build_report(
        self,
        scenarios: list[ElliottScenario],
        current_price: float,
    ) -> tuple[
        str,
        dict[str, float],
        list,  # EvidenceReference
        list,  # EvidenceReference
        list,  # InvalidationCondition
    ]:
        """Erstellt Hypothese, Wahrscheinlichkeiten, Evidenz, Gegenhypothese, Invalidierungen."""

        valid_scenarios = [s for s in scenarios if s.rule_valid]
        primary = valid_scenarios[0] if valid_scenarios else scenarios[0]
        alternatives = [s for s in scenarios if s != primary]

        # ── Hypothese ──
        hypothesis = (
            f"Elliott wave analysis: {len(scenarios)} scenario(s), "
            f"{len(valid_scenarios)} rule-valid. "
            f"Primary: {primary.scenario_id} ({primary.direction}), "
            f"alt: {', '.join(s.scenario_id for s in alternatives[:2])}"
        )

        # ── Wahrscheinlichkeiten ──
        probabilities = self._compute_probabilities(scenarios)

        # ── Evidenz ──
        evidence = self._build_evidence(scenarios)

        # ── Gegenhypothese ──
        counter_evidence = self._build_counter_evidence(scenarios, primary)

        # ── Invalidierungen ──
        invalidations = self._build_invalidations(scenarios)

        return hypothesis, probabilities, evidence, counter_evidence, invalidations

    def _compute_probabilities(
        self, scenarios: list[ElliottScenario]
    ) -> dict[str, float]:
        """Berechnet up/down/range aus Szenario-Regelvaliditaet und Score."""
        up_weight = 0.0
        down_weight = 0.0

        for s in scenarios:
            if s.rule_valid:
                if s.direction == "up":
                    up_weight += s.guideline_score + 0.3
                elif s.direction == "down":
                    down_weight += s.guideline_score + 0.3
                else:
                    down_weight += s.guideline_score * 0.5
                    up_weight += s.guideline_score * 0.5
            else:
                down_weight += 0.05
                up_weight += 0.05

        if up_weight == 0 and down_weight == 0:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        strength = abs(up_weight - down_weight)
        base_up = 0.2 + up_weight / (up_weight + down_weight + 1)
        base_down = 0.2 + down_weight / (up_weight + down_weight + 1)
        base_range = 0.2 + 0.1 * (1 - min(1, strength))

        total = base_up + base_down + base_range
        up_prob = round(base_up / total, 4)
        down_prob = round(base_down / total, 4)
        range_prob = round(1.0 - up_prob - down_prob, 4)

        return {"up": up_prob, "down": down_prob, "range": range_prob}

    def _build_evidence(self, scenarios: list[ElliottScenario]) -> list:
        """Erstellt Evidenz aus Wave-Strukturen."""
        evidence: list = []

        for i, s in enumerate(scenarios):
            evidence.append(
                self._make_evidence(
                    f"scenario_{s.scenario_id}",
                    f"{s.direction} {s.description}",
                    "positive" if s.rule_valid else "neutral",
                    s.guideline_score if s.rule_valid else 0.2,
                )
            )

            if s.wave_count and len(s.wave_count.pivot_prices) > 1:
                evidence.append(
                        self._make_evidence(
                            f"pivot_count_{i}",
                            f"{len(s.wave_count.pivot_prices)} pivots detected",
                            "positive",
                            min(0.5, 0.1 * len(s.wave_count.pivot_prices)),
                        )
                    )

        if not evidence:
            evidence.append(
                self._make_evidence(
                    "no_structure",
                    "no wave structure found",
                    "neutral",
                    0.1,
                )
            )

        return evidence

    def _build_counter_evidence(
        self, scenarios: list[ElliottScenario], primary: ElliottScenario
    ) -> list:
        """Erstellt Gegenhypothese-Evidenz.

        Die Gegenhypothese ist das Szenario mit der entgegengesetzten Richtung
        und der hoechsten guideline_score unter den Alternativen.
        """
        counter: list = []

        # Finde das beste alternative Szenario mit entgegengesetzter Richtung
        opposing = [
            s for s in scenarios
            if s.direction != primary.direction and s.rule_valid
        ]
        if opposing:
            # Nimm das mit der hoechsten guideline_score
            best_opposing = max(opposing, key=lambda s: s.guideline_score)
            counter.append(
                self._make_evidence(
                    f"counter_{best_opposing.scenario_id}",
                    f"opposing {best_opposing.direction} scenario valid "
                    f"(score={best_opposing.guideline_score})",
                    "negative",
                    best_opposing.guideline_score * 0.8,
                )
            )

        # Falls keine regel-gueltige Gegenrichtung, nehme das beste rule-violated Szenario
        if not counter:
            invalid = [s for s in scenarios if not s.rule_valid]
            if invalid:
                best_invalid = max(invalid, key=lambda s: s.guideline_score)
                counter.append(
                    self._make_evidence(
                        f"invalid_{best_invalid.scenario_id}",
                        f"scenario violates Elliott rules (score={best_invalid.guideline_score})",
                        "negative",
                        0.3,
                    )
                )

        # Fallback: allgemeine Gegenhypothese
        if not counter:
            counter.append(
                self._make_evidence(
                    "no_opposing",
                    f"no rule-valid opposing scenario to {primary.scenario_id}",
                    "negative",
                    0.1,
                )
            )

        return counter

    def _build_invalidations(
        self, scenarios: list[ElliottScenario]
    ) -> list:
        """Erstellt Invalidierungsbedingungen pro Szenario."""
        invalidations: list = []

        for i, s in enumerate(scenarios):
            if s.wave_count and s.wave_count.pivot_prices:
                if s.direction == "up":
                    invalidations.append(
                        self._make_invalidations(
                            condition=f"{s.scenario_id} invalidated if price breaks below pivot low",
                            indicator=f"wave_{i}_low",
                            threshold=min(s.wave_count.pivot_prices),
                            direction="below",
                        )
                    )
                else:
                    invalidations.append(
                        self._make_invalidations(
                            condition=f"{s.scenario_id} invalidated if price breaks above pivot high",
                            indicator=f"wave_{i}_high",
                            threshold=max(s.wave_count.pivot_prices),
                            direction="above",
                        )
                    )

        if not invalidations:
            invalidations.append(
                self._make_invalidations(
                    condition="No wave structure to invalidate",
                    indicator="wave_structure",
                    threshold=0.0,
                    direction="below",
                )
            )

        return invalidations

    def _compute_confidence(
        self, scenarios: list[ElliottScenario]
    ) -> float:
        """Berechnet raw_confidence aus Regelvaliditaet und Scores."""
        valid = [s for s in scenarios if s.rule_valid]
        if not valid:
            return 0.1
        avg_score = sum(s.guideline_score for s in valid) / len(valid)
        return round(min(0.9, 0.3 + 0.3 * avg_score + 0.1 * min(1, len(valid) - 1)), 4)
