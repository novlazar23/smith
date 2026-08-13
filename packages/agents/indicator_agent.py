from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.indicators import MACD, RSI, SMA
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent


class IndicatorAgent(BaseAgent):
    """Technische Indikator-Agent — berechnet RSI, MACD, SMA und kombiniert Signale."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="indicator",
                agent_type=AgentType.INDICATOR,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert Marktdaten mit technischen Indikatoren.

        Required keys:
            close (NDArray) - required
            volume (NDArray) - optional
            high (NDArray) - optional
            low (NDArray) - optional

        Returns:
            AgentReport mit kombinierten Indikator-Signalen.
        """
        if "close" not in data:
            raise ValueError("Missing required data keys: ['close']")

        close = data["close"]

        # --- RSI ---
        rsi_result = RSI(period=14).compute({"close": close})
        rsi_values = rsi_result.values
        valid_rsi = rsi_values[~np.isnan(rsi_values)]
        rsi_latest = float(valid_rsi[-1]) if len(valid_rsi) > 0 else 50.0

        # --- MACD ---
        macd_result = MACD().compute({"close": close})
        macd_values = macd_result.values
        valid_macd = macd_values[~np.isnan(macd_values)]
        macd_latest = float(valid_macd[-1]) if len(valid_macd) > 0 else 0.0

        # --- SMA(20) & SMA(50) ---
        sma20 = SMA(period=20).compute({"close": close})
        sma50 = SMA(period=50).compute({"close": close})
        sma20_vals = sma20.values[~np.isnan(sma20.values)]
        sma50_vals = sma50.values[~np.isnan(sma50.values)]

        sma20_latest = float(sma20_vals[-1]) if len(sma20_vals) > 0 else float(close[-1])
        sma50_latest = float(sma50_vals[-1]) if len(sma50_vals) > 0 else float(close[-1])
        cross_uptrend = sma20_latest > sma50_latest

        # --- Signal-Kombination ---
        bull_signals: list[str] = []
        bear_signals: list[str] = []

        if rsi_latest < 30:
            bull_signals.append("rsi_oversold")
        elif rsi_latest > 70:
            bear_signals.append("rsi_overbought")

        if macd_latest > 0:
            bull_signals.append("macd_positive")
        elif macd_latest < 0:
            bear_signals.append("macd_negative")

        if cross_uptrend:
            bull_signals.append("sma_cross_uptrend")
        else:
            bear_signals.append("sma_cross_downtrend")

        bull_count = len(bull_signals)
        bear_count = len(bear_signals)
        total = bull_count + bear_count + 1  # +1 for potential range tie

        up_prob = round(bull_count / total, 4)
        down_prob = round(bear_count / total, 4)
        range_prob = round(1.0 - up_prob - down_prob, 4)

        # --- Evidence ---
        evidence: list = []

        if rsi_latest < 30:
            evidence.append(
                self._make_evidence(
                    "RSI", f"{rsi_latest:.1f}", "positive", 0.7,
                )
            )
        elif rsi_latest > 70:
            evidence.append(
                self._make_evidence(
                    "RSI", f"{rsi_latest:.1f}", "negative", 0.7,
                )
            )

        if macd_latest > 0:
            evidence.append(
                self._make_evidence(
                    "MACD_Histogram", f"{macd_latest:.4f}", "positive", 0.6,
                )
            )
        else:
            evidence.append(
                self._make_evidence(
                    "MACD_Histogram", f"{macd_latest:.4f}", "negative", 0.6,
                )
            )

        if cross_uptrend:
            evidence.append(
                self._make_evidence(
                    "SMA_Cross", "SMA20>SMA50", "positive", 0.5,
                )
            )

        # --- Counter-evidence ---
        counter_evidence: list = []
        if bull_count > bear_count:
            for sig in bear_signals:
                counter_evidence.append(
                    self._make_evidence(
                        sig, "present", "negative", 0.4,
                    )
                )
        elif bear_count > bull_count:
            for sig in bull_signals:
                counter_evidence.append(
                    self._make_evidence(
                        sig, "present", "negative", 0.4,
                    )
                )

        # --- Invalidations ---
        invalidations = [
            self._make_invalidations(
                condition="RSI überkauft/überverkauft ändert sich",
                indicator="RSI",
                threshold=70.0 if rsi_latest < 50 else 30.0,
                direction="above" if rsi_latest < 50 else "below",
            ),
            self._make_invalidations(
                condition="MACD Histogram kehrt um",
                indicator="MACD_Histogram",
                threshold=0.0,
                direction="below" if macd_latest > 0 else "above",
            ),
        ]

        hypothesis = (
            f"Bullish signals: {bull_count}, Bearish signals: {bear_count}. "
            f"RSI={rsi_latest:.1f}, MACD_hist={macd_latest:.4f}, "
            f"SMA_cross={'uptrend' if cross_uptrend else 'downtrend'}"
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
            raw_confidence=round(min(0.9, 0.4 + 0.2 * abs(bull_count - bear_count) / max(total, 1)), 4),
            expected_return=None,
            calibrated_confidence=0.0,
        )
