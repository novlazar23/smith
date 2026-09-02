# allow: SIZE_OK — Ein-Datei-Agent-Modul nach Repo-Konvention (ein Modul pro
# Agent; Referenz anomaly_agent.py ist 640 Zeilen). Agent-Klasse, Indikator-
# Berechnung und Berichtskomponenten bilden hier eine kohärente Einheit.

"""Trend-Agent — EMA-Ausrichtung, ROC, ATR-normierte Trennung.

Perspektive: „Gibt es einen nachhaltigen Trend und in welche Richtung?"
Richtung und Stärke aus EMA12/EMA26-Ausrichtung, 5-Bar-EMA-Steigung,
10-Bar-Veränderungsrate und ATR14-normierter EMA-Trennung. Kalibrierung
auf die Konsens-Vote-Schwellen: starker Trend → „up"/„down" 0,65 - 0,85,
ohne Trend bleibt „range" dominant (>= 0,5). Reines NumPy, kein Lookahead.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent

REQUIRED_KEYS = frozenset({"open", "high", "low", "close", "volume"})

type TrendDirection = Literal["up", "down", "range"]


@dataclass(frozen=True, slots=True)
class TrendState:
    """Momentaufnahme der Trend-Indikatoren (nur aktuelle und vergangene Kerzen)."""

    ema_fast: float  # EMA12 auf close
    ema_slow: float  # EMA26 auf close
    separation: float  # (EMA12 - EMA26) / ATR14, in ATR-Einheiten
    slope: float  # 5-Bar-Steigung von EMA12, durch ATR14 normiert
    roc10: float  # close[-1] / close[-11] - 1
    atr14: float  # Mittel der letzten 14 True Ranges


def _direction_label(value: float) -> str:
    """Mapt einen signierten Wert auf die Evidenz-Richtung."""
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "neutral"


class TrendAgent(BaseAgent):
    """Trend-Agent — Richtung und Stärke aus EMA-Ausrichtung, ROC, ATR-normierter Trennung."""

    MIN_BARS: int = 50
    EMA_FAST: int = 12
    EMA_SLOW: int = 26
    ROC_PERIOD: int = 10
    ATR_PERIOD: int = 14

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="trend",
                agent_type=AgentType.INDICATOR,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert OHLCV-Daten auf nachhaltige Trendrichtung und -stärke.

        Accepts a dict with OHLCV arrays (aufsteigend, älteste → neueste,
        Index -1 = aktuelle Kerze):
            open, high, low, close, volume: NDArray[float64]

        Returns:
            AgentReport mit auf die Konsens-Vote-Schwellen kalibrierten
            up/down/range-Wahrscheinlichkeiten.

        Raises:
            ValueError: Wenn erforderliche Schlüssel fehlen.
        """
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required OHLCV keys: {sorted(missing)}"
            )

        close = np.asarray(data["close"], dtype=np.float64)
        high = np.asarray(data["high"], dtype=np.float64)
        low = np.asarray(data["low"], dtype=np.float64)

        if len(close) < self.MIN_BARS:
            return self._short_data_report(len(close))

        state = self._compute_state(close, high, low)
        direction = self._classify(state)
        probabilities, confidence = self._calibrate(state, direction)

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(datetime.UTC),
            hypothesis=self._build_hypothesis(state, direction),
            probabilities=probabilities,
            evidence=self._build_evidence(state),
            counter_evidence=self._build_counter_evidence(state, direction),
            invalidations=self._build_invalidations(state, direction),
            raw_confidence=confidence,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )

    # ── Indikatoren ─────────────────────────────────────────────────────────

    @staticmethod
    def _ema(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        """Rekursive EMA, gezeichnet auf den SMA der ersten `period` Kerzen.

        Kein Lookahead: Kerze i verwendet nur Kerzen <= i. Die ersten
        `period - 1` Ausgabewerte werden mit dem Seed befüllt (Warmup).
        """
        alpha = 2.0 / (period + 1.0)
        out = np.empty_like(values)
        seed = float(values[:period].mean())
        out[:period] = seed
        prev = seed
        for i in range(period, len(values)):
            prev = alpha * float(values[i]) + (1.0 - alpha) * prev
            out[i] = prev
        return out

    def _atr(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
    ) -> float:
        """ATR14: Mittel der letzten 14 True Ranges.

        TR_i = max(high - low, |high - prev_close|, |low - prev_close|).
        """
        prev_close = close[:-1]
        true_range = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - prev_close),
                np.abs(low[1:] - prev_close),
            ),
        )
        return float(true_range[-self.ATR_PERIOD :].mean())

    def _compute_state(
        self,
        close: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
    ) -> TrendState:
        """Berechnet alle Trend-Indikatoren für die aktuelle Kerze."""
        ema_fast = self._ema(close, self.EMA_FAST)
        ema_slow = self._ema(close, self.EMA_SLOW)
        atr14 = self._atr(high, low, close)
        atr_safe = atr14 if atr14 > 1e-12 else 1e-12
        separation = (ema_fast[-1] - ema_slow[-1]) / atr_safe
        slope = (ema_fast[-1] - ema_fast[-6]) / (5.0 * atr_safe)
        roc_base = close[-1 - self.ROC_PERIOD]
        roc10 = float(close[-1] / roc_base - 1.0) if abs(roc_base) > 1e-12 else 0.0
        return TrendState(
            ema_fast=float(ema_fast[-1]),
            ema_slow=float(ema_slow[-1]),
            separation=float(separation),
            slope=float(slope),
            roc10=roc10,
            atr14=atr14,
        )

    # ── Klassifikation & Kalibrierung ───────────────────────────────────────

    def _classify(self, state: TrendState) -> TrendDirection:
        """Klassifiziert den Trendzustand: starker Auf-, starker Ab- oder Seitwärtstrend."""
        if (
            state.ema_fast > state.ema_slow
            and state.slope > 0.0
            and state.roc10 > 0.0
            and state.separation > 0.5
        ):
            return "up"
        if (
            state.ema_fast < state.ema_slow
            and state.slope < 0.0
            and state.roc10 < 0.0
            and state.separation < -0.5
        ):
            return "down"
        return "range"

    def _calibrate(
        self, state: TrendState, direction: TrendDirection
    ) -> tuple[dict[str, float], float]:
        """Kalibrierte up/down/range-Verteilung plus rohe Konfidenz.

        Monoton: je stärker der Trend, desto höher die richtungsweisende
        Wahrscheinlichkeit (Deckel bei 0,85); ohne Trend bleibt „range"
        mit >= 0,5 dominant. Summe ist exakt 1,0 (± 1e-6).
        """
        match direction:
            case "up":
                return self._directional_probabilities(
                    "up",
                    self._trend_strength(state.separation, state.slope, state.roc10),
                )
            case "down":
                return self._directional_probabilities(
                    "down",
                    self._trend_strength(-state.separation, -state.slope, -state.roc10),
                )
            case "range":
                return self._range_probabilities(state)

    @staticmethod
    def _trend_strength(separation: float, slope: float, roc: float) -> float:
        """Richtungs-Wahrscheinlichkeit eines starken Trends (monoton, [0,65, 0,85]).

        Eingaben sind die Trend-seitigen (d. h. für Aufwärtstrends positive,
        für Abwärtstrends negierte) Indikatoren.
        """
        sep_score = min(max(separation, 0.0), 3.0) / 3.0
        slope_score = min(max(slope, 0.0), 1.0)
        roc_score = min(max(roc / 0.05, 0.0), 1.0)
        raw = 0.62 + 0.15 * sep_score + 0.10 * slope_score + 0.10 * roc_score
        return float(min(0.85, max(0.65, raw)))

    def _directional_probabilities(
        self, winner: Literal["up", "down"], p_winner: float
    ) -> tuple[dict[str, float], float]:
        """Verteilung für einen starken Trend: Gewinner in [0,65, 0,85]."""
        p_winner = round(p_winner, 4)
        loser = "down" if winner == "up" else "up"
        probabilities = {
            winner: p_winner,
            loser: 0.05,
            "range": round(1.0 - p_winner - 0.05, 4),
        }
        confidence = 0.35 + 0.5 * (p_winner - 0.65) / 0.20
        return probabilities, float(round(min(0.9, max(0.1, confidence)), 4))

    def _range_probabilities(self, state: TrendState) -> tuple[dict[str, float], float]:
        """Verteilung für schwache/mischende Zustände: „range" mit >= 0,5."""
        sep_score = min(abs(state.separation), 3.0) / 3.0
        slope_score = min(abs(state.slope), 1.0)
        strength = 0.5 * sep_score + 0.5 * slope_score
        p_directional = 0.20 + 0.25 * strength  # Rest nach Range: 0,80 .. 0,55
        bull = 0.6 * min(max(state.separation, 0.0), 3.0) / 3.0 + 0.4 * min(
            max(state.slope, 0.0), 1.0
        )
        bear = 0.6 * min(max(-state.separation, 0.0), 3.0) / 3.0 + 0.4 * min(
            max(-state.slope, 0.0), 1.0
        )
        if bull + bear > 0.0:
            p_up = p_directional * bull / (bull + bear)
            p_down = p_directional * bear / (bull + bear)
        else:
            p_up = p_directional / 2.0
            p_down = p_up
        p_up_r = round(p_up, 4)
        p_down_r = round(p_down, 4)
        probabilities = {
            "up": p_up_r,
            "down": p_down_r,
            "range": round(1.0 - p_up_r - p_down_r, 4),
        }
        confidence = 0.10 + 0.20 * strength
        return probabilities, float(round(min(0.9, max(0.1, confidence)), 4))

    # ── Berichtskomponenten ─────────────────────────────────────────────────

    def _build_hypothesis(self, state: TrendState, direction: TrendDirection) -> str:
        """Einzeilige, menschenlesbare Zusammenfassung des Trendzustands."""
        details = (
            f"EMA12/EMA26 separation {state.separation:+.2f} ATR, "
            f"slope {state.slope:+.2f} ATR/bar, ROC10 {state.roc10:+.2%}, "
            f"ATR14 {state.atr14:.3f}"
        )
        match direction:
            case "up":
                return f"Sustained uptrend: {details}"
            case "down":
                return f"Sustained downtrend: {details}"
            case "range":
                return f"No sustained trend (range-bound): {details}"

    def _build_evidence(self, state: TrendState) -> list:
        """Evidenz — Indikatorsnapshot der aktuellen Kerze, nur Scores, keine Behauptungen."""
        return [
            self._make_evidence(
                "ema_separation",
                f"EMA12-EMA26 separation {state.separation:+.3f} ATR units",
                _direction_label(state.separation),
                min(abs(state.separation) / 3.0, 1.0),
            ),
            self._make_evidence(
                "ema_slope",
                f"EMA12 5-bar slope {state.slope:+.3f} ATR/bar",
                _direction_label(state.slope),
                min(abs(state.slope), 1.0),
            ),
            self._make_evidence(
                "roc10",
                f"10-bar rate of change {state.roc10:+.2%}",
                _direction_label(state.roc10),
                min(abs(state.roc10) / 0.05, 1.0),
            ),
            self._make_evidence(
                "atr14",
                f"ATR14 {state.atr14:.4f}",
                "neutral",
                0.3,
            ),
        ]

    _COUNTER_EXPLANATIONS: ClassVar[dict[TrendDirection, tuple[str, str, float]]] = {
        "up": (
            "counter_exhaustion",
            "extended EMA separation; mean reversion / pullback to EMA26 is plausible",
            0.4,
        ),
        "down": (
            "counter_bounce",
            "extended EMA separation; oversold bounce back to EMA26 is plausible",
            0.4,
        ),
        "range": (
            "counter_breakout",
            "range-bound market can break in either direction on volatility expansion",
            0.3,
        ),
    }

    def _build_counter_evidence(
        self, state: TrendState, direction: TrendDirection
    ) -> list:
        """Gegenhypothesen: alternative Erklärungen für das Trendbild."""
        feature, explanation, relevance = self._COUNTER_EXPLANATIONS[direction]
        if direction == "range":
            value = explanation
        else:
            value = f"{explanation} (separation {state.separation:.2f} ATR)"
        return [self._make_evidence(feature, value, "negative", relevance)]

    _CROSS_INVALIDATIONS: ClassVar[dict[TrendDirection, tuple[str, float, str]]] = {
        "up": ("EMA26 crosses above EMA12 — uptrend invalidated", 0.0, "below"),
        "down": ("EMA12 crosses below EMA26 — downtrend invalidated", 0.0, "above"),
        "range": ("trend breakout: |EMA12-EMA26| exceeds 0.5 ATR", 0.5, "above"),
    }

    def _build_invalidations(
        self, state: TrendState, direction: TrendDirection
    ) -> list:
        """Invalidierungsbedingungen für die Trend-Hypothese."""
        condition, threshold, inv_direction = self._CROSS_INVALIDATIONS[direction]
        return [
            self._make_invalidations(
                condition=condition,
                indicator="ema_separation",
                threshold=threshold,
                direction=inv_direction,
            ),
            self._make_invalidations(
                condition="volatility regime shift — ATR14 more than doubles",
                indicator="atr14",
                threshold=round(2.0 * state.atr14, 4),
                direction="above",
            ),
            self._make_invalidations(
                condition="sample size too small for stable EMA/ATR estimates",
                indicator="sample_size",
                threshold=float(self.MIN_BARS),
                direction="below",
            ),
        ]

    def _short_data_report(self, n: int) -> AgentReport:
        """Bericht bei unzureichenden Daten (weniger als MIN_BARS Kerzen)."""
        hypothesis = (
            f"Insufficient data for trend analysis: {n} bars available, "
            f"minimum {self.MIN_BARS} required. Probabilities stay neutral."
        )
        probabilities = {"up": 0.33, "down": 0.33, "range": 0.34}
        evidence = [
            self._make_evidence(
                "insufficient_data",
                f"only {n} bars available, need at least {self.MIN_BARS}",
                "neutral",
                0.0,
            )
        ]
        counter_evidence = [
            self._make_evidence(
                "counter_insufficient",
                "no trend claim possible with insufficient data — by design",
                "negative",
                0.0,
            )
        ]
        invalidations = [
            self._make_invalidations(
                condition="sample size too small for stable EMA/ATR estimates",
                indicator="sample_size",
                threshold=float(self.MIN_BARS),
                direction="below",
            ),
            self._make_invalidations(
                condition="data quality below threshold (gaps, out-of-range values)",
                indicator="data_quality",
                threshold=0.9,
                direction="below",
            ),
        ]
        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(datetime.UTC),
            hypothesis=hypothesis,
            probabilities=probabilities,
            evidence=evidence,
            counter_evidence=counter_evidence,
            invalidations=invalidations,
            raw_confidence=0.08,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )
