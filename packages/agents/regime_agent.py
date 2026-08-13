from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.regime import MarketRegime, RuleBasedRegimeDetector
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent


class RegimeAgent(BaseAgent):
    """Regime-Detection-Agent — klassifiziert Marktregime und leitet Wahrscheinlichkeiten ab."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="regime",
                agent_type=AgentType.REGIME,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert Marktregime aus OHLCV-Daten.

        Required keys:
            close (NDArray) - required
            high (NDArray) - required
            low (NDArray) - required
            volume (NDArray) - optional

        Returns:
            AgentReport mit regime-basierten Wahrscheinlichkeiten.
        """
        if "close" not in data or "high" not in data or "low" not in data:
            raise ValueError("Missing required data keys: ['close', 'high', 'low']")

        detector = RuleBasedRegimeDetector()
        regime_result = detector.detect(data)
        regime = regime_result.regime
        confidence = regime_result.confidence
        metadata = regime_result.metadata

        # --- Regime -> Wahrscheinlichkeiten ---
        if regime == MarketRegime.BULL:
            up_prob = round(0.7 * confidence + 0.3, 4)
            down_prob = round(0.1 * (1.0 - confidence), 4)
            range_prob = round(1.0 - up_prob - down_prob, 4)
        elif regime == MarketRegime.BEAR:
            down_prob = round(0.7 * confidence + 0.3, 4)
            up_prob = round(0.1 * (1.0 - confidence), 4)
            range_prob = round(1.0 - up_prob - down_prob, 4)
        else:  # CHOPPY
            range_prob = round(0.6 * confidence + 0.4, 4)
            up_prob = round(0.2 * (1.0 - confidence), 4)
            down_prob = round(1.0 - up_prob - range_prob, 4)

        # --- Evidence ---
        evidence: list = []
        adx_val = metadata.get("adx", 0.0)
        rsi_val = metadata.get("rsi", 50.0)
        is_uptrend = metadata.get("is_uptrend", False)

        evidence.append(
            self._make_evidence(
                "Regime", str(regime.value),
                "positive" if regime != MarketRegime.CHOPPY else "neutral",
                0.8,
            )
        )
        evidence.append(
            self._make_evidence(
                "ADX", f"{adx_val:.1f}",
                "positive" if adx_val > 25 else "neutral",
                0.5,
            )
        )
        evidence.append(
            self._make_evidence(
                "RSI", f"{rsi_val:.1f}",
                "positive" if rsi_val > 50 else "negative",
                0.5,
            )
        )
        evidence.append(
            self._make_evidence(
                "SMA_Cross",
                "uptrend" if is_uptrend else "downtrend",
                "positive" if is_uptrend else "negative",
                0.6,
            )
        )

        # --- Invalidations ---
        invalidations = [
            self._make_invalidations(
                condition="Regime wechselt",
                indicator="Regime_Score",
                threshold=0.5,
                direction="below" if regime != MarketRegime.CHOPPY else "above",
            ),
            self._make_invalidations(
                condition="ADX fällt unter Trend-Schwelle",
                indicator="ADX",
                threshold=20.0,
                direction="below",
            ),
        ]

        hypothesis = (
            f"Regime: {regime.value}, Confidence: {confidence:.2f}, "
            f"ADX: {adx_val:.1f}, RSI: {rsi_val:.1f}"
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
            raw_confidence=round(confidence, 4),
            expected_return=None,
            calibrated_confidence=0.0,
        )
