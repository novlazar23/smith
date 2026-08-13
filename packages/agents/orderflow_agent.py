from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.orderflow import (
    AbsorptionDetector,
    FootprintAnalyzer,
    IcebergDetector,
    OrderFlowSignal,
)
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent


class OrderFlowAgent(BaseAgent):
    """Order-Flow-Agent — analysiert Footprint, Absorption und Iceberg-Muster."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="orderflow",
                agent_type=AgentType.ORDERFLOW,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert Order-Flow aus OHLCV-Daten.

        Required keys:
            open (NDArray) - required
            high (NDArray) - required
            low (NDArray) - required
            close (NDArray) - required
            volume (NDArray) - required

        Returns:
            AgentReport mit Order-Flow-basierten Wahrscheinlichkeiten.
        """
        required = ("open", "high", "low", "close", "volume")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

        # --- Footprint Analysis ---
        footprint = FootprintAnalyzer()
        footprint_result = footprint.analyze(data)
        cumulative_delta = footprint_result.cumulative_delta
        footprint_signals = footprint_result.signals

        # --- Absorption Detection ---
        absorption = AbsorptionDetector()
        absorption_result = absorption.detect_absorption(data)
        absorption_signals = absorption_result.signals
        absorption_levels = absorption_result.metadata.get("contact_levels", [])

        # --- Iceberg Detection ---
        iceberg = IcebergDetector()
        iceberg_result = iceberg.detect_iceberg(data)
        iceberg_signals = iceberg_result.signals
        iceberg_direction = iceberg_result.metadata.get("direction", None)

        # --- Wahrscheinlichkeiten ---
        up_score = 0.0
        down_score = 0.0
        range_score = 0.0

        # Cumulative Delta Signal
        if cumulative_delta > 0:
            up_score += 0.4
        elif cumulative_delta < 0:
            down_score += 0.4

        # Absorption -> Range Bias
        has_absorption = OrderFlowSignal.ABSORPTION in absorption_signals
        if has_absorption:
            range_score += 0.3

        # Iceberg Direction Bias
        if iceberg_signals:
            if iceberg_direction == "buying":
                up_score += 0.2
            elif iceberg_direction == "selling":
                down_score += 0.2

        # Aggressive Buy/Sell signals
        if OrderFlowSignal.AGGRESSIVE_BUY in footprint_signals:
            up_score += 0.15
        if OrderFlowSignal.AGGRESSIVE_SELL in footprint_signals:
            down_score += 0.15

        # Normalize to sum to 1.0
        total = up_score + down_score + range_score
        if total > 0:
            up_prob = round(up_score / total, 4)
            down_prob = round(down_score / total, 4)
            range_prob = round(range_score / total, 4)
        else:
            up_prob = 0.3333
            down_prob = 0.3333
            range_prob = 0.3334

        # --- Evidence ---
        evidence: list = []

        delta_direction = "positive" if cumulative_delta > 0 else ("negative" if cumulative_delta < 0 else "neutral")
        evidence.append(
            self._make_evidence(
                "Cumulative_Delta",
                f"{cumulative_delta:.2f}",
                delta_direction,
                0.7,
            )
        )

        if absorption_signals:
            for level in absorption_levels[:2]:
                evidence.append(
                    self._make_evidence(
                        "Absorption_Level",
                        f"price={level['price_level']:.2f} touches={level['touch_count']}",
                        "negative",
                        0.5,
                    )
                )

        if iceberg_signals:
            evidence.append(
                self._make_evidence(
                    "Iceberg",
                    f"direction={iceberg_direction or 'unknown'}",
                    "positive" if iceberg_direction == "buying" else ("negative" if iceberg_direction == "selling" else "neutral"),
                    0.6,
                )
            )

        # --- Counter-evidence ---
        counter_evidence: list = []
        if has_absorption:
            counter_evidence.append(
                self._make_evidence(
                    "Absorption",
                    "detected",
                    "negative" if cumulative_delta > 0 else "positive",
                    0.6,
                )
            )

        # --- Invalidations ---
        invalidations = [
            self._make_invalidations(
                condition="Cumulatives Delta dreht um",
                indicator="Cumulative_Delta",
                threshold=0.0,
                direction="below" if cumulative_delta > 0 else "above",
            ),
            self._make_invalidations(
                condition="Absorption-Schwelle überschritten",
                indicator="Absorption_Touches",
                threshold=3.0,
                direction="above",
            ),
        ]

        hypothesis = (
            f"Delta: {cumulative_delta:.2f}, "
            f"Absorption: {len(absorption_signals)} signals, "
            f"Iceberg: {len(iceberg_signals)} signals dir={iceberg_direction}"
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
            counter_evidence=counter_evidence,
            invalidations=invalidations,
            status=self.config.status,
            raw_confidence=round(min(0.9, 0.3 + 0.1 * len(evidence)), 4),
            expected_return=None,
            calibrated_confidence=0.0,
        )
