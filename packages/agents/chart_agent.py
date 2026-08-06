from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.chart_structure import (
    PatternDetector,
    SupportResistanceDetector,
    SwingDetector,
)
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent


class ChartAgent(BaseAgent):
    """Chart-Struktur-Agent — detektiert Swing-Pivots, S/R-Level und Chart-Muster."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="chart",
                agent_type=AgentType.CHART,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert Chart-Struktur aus OHLC-Daten.

        Required keys:
            close (NDArray) - required
            high (NDArray) - required
            low (NDArray) - required

        Returns:
            AgentReport mit chart-struktur-basierten Wahrscheinlichkeiten.
        """
        if "close" not in data or "high" not in data or "low" not in data:
            raise ValueError("Missing required data keys: ['close', 'high', 'low']")

        close = data["close"]

        # --- Swing Pivots ---
        swing_detector = SwingDetector(lookback=3)
        pivots = swing_detector.detect_swings(data)

        # --- Support/Resistance Levels ---
        sr_detector = SupportResistanceDetector()
        sr_levels = sr_detector.detect_levels(data)

        # --- Pattern Detection ---
        pattern_detector = PatternDetector()
        result = pattern_detector.detect_all_patterns(data)
        bos_patterns = result.patterns
        metadata = result.metadata

        # --- Wahrscheinlichkeiten ---
        bull_signals = 0
        bear_signals = 0

        # Richte Signale anhand der letzten Pivots aus
        recent_pivots = pivots[-5:] if len(pivots) >= 5 else pivots
        for p in recent_pivots:
            if p.direction == "high":
                bear_signals += 1
            else:
                bull_signals += 1

        # Wenn CHoCH erkannt -> Indecision / Range
        if metadata.get("choch_detected", False):
            bull_signals += 1
            bear_signals += 1

        # Wenn Failed Breakout -> Range Bias
        if metadata.get("failed_breakout_count", 0) > 0:
            bull_signals += 1
            bear_signals += 1

        total = bull_signals + bear_signals + 1
        up_prob = round(bull_signals / total, 4)
        down_prob = round(bear_signals / total, 4)
        range_prob = round(1.0 - up_prob - down_prob, 4)

        # --- Evidence ---
        evidence: list = []

        # Always include at least one piece of evidence from S/R levels or patterns
        high_pivots = [p for p in pivots if p.direction == "high"]
        low_pivots = [p for p in pivots if p.direction == "low"]

        if high_pivots:
            last_high = high_pivots[-1]
            evidence.append(
                self._make_evidence(
                    "Swing_High",
                    f"price={last_high.price:.2f} quality={last_high.quality_score:.2f}",
                    "negative",
                    0.6,
                )
            )

        if low_pivots:
            last_low = low_pivots[-1]
            evidence.append(
                self._make_evidence(
                    "Swing_Low",
                    f"price={last_low.price:.2f} quality={last_low.quality_score:.2f}",
                    "positive",
                    0.6,
                )
            )

        # S/R Levels
        for level in sr_levels[:3]:
            direction = "positive" if level.level_type == "support" else "negative"
            evidence.append(
                self._make_evidence(
                    "SupportResistance",
                    f"{level.level_type} price={level.price:.2f} touches={level.touch_count}",
                    direction,
                    0.5,
                )
            )

        # Patterns
        if bos_patterns:
            evidence.append(
                self._make_evidence(
                    "BOS", f"count={metadata.get('bos_count', 0)}", "neutral", 0.5,
                )
            )

        if metadata.get("choch_detected", False):
            evidence.append(
                self._make_evidence(
                    "CHoCH", "detected", "neutral", 0.7,
                )
            )

        # Fallback: always include at least one evidence if list is empty
        if not evidence:
            evidence.append(
                self._make_evidence(
                    "Price_Range",
                    f"min={float(close.min()):.2f} max={float(close.max()):.2f}",
                    "neutral",
                    0.3,
                )
            )

        # --- Invalidations ---
        invalidations = [
            self._make_invalidations(
                condition="Neuer Swing-Pivot in Gegenrichtung",
                indicator="Swing_Pivot",
                threshold=0.0,
                direction="above" if high_pivots else "below",
            ),
            self._make_invalidations(
                condition="S/R-Level durchbrochen",
                indicator="SupportResistance",
                threshold=sr_levels[-1].price if sr_levels else 0.0,
                direction="above" if sr_levels and sr_levels[-1].level_type == "resistance" else "below",
            ),
        ]

        hypothesis = (
            f"Pivots: {len(pivots)}, BOS: {metadata.get('bos_count', 0)}, "
            f"CHoCH: {metadata.get('choch_detected', False)}, "
            f"S/R Levels: {len(sr_levels)}"
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
            probabilities={"up": up_prob, "down": down_prob, "range": range_prob},
            evidence=evidence,
            invalidations=invalidations,
            status=self.config.status,
            raw_confidence=round(min(0.9, 0.3 + 0.1 * len(pivots)), 4),
        )
