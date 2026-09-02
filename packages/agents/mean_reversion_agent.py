# allow: SIZE_OK — Ein-Datei-Agent-Modul nach Repo-Konvention (ein Modul pro
# Agent; Referenz anomaly_agent.py ist 640 Zeilen). Agent-Klasse, Indikator-
# Berechnung und Berichtskomponenten bilden hier eine kohärente Einheit.

"""Mean-Reversion-Agent — z-Score gegen SMA50, RSI14, Prozentabstand.

Perspektive: „Ist die Bewegung überdehnt und reverts sie?" Oversold
(z <= -2, RSI < 45) → Reversions-Setup nach oben („up" 0,65 - 0,85);
overbought (z >= +2, RSI > 55) → Reversions-Setup nach unten („down"
0,65 - 0,85). Widersprüchliche Signale (z extrem, RSI dagegen) und die
Mittellage bleiben „range"-dominant (>= 0,5). Reines NumPy, kein
Lookahead, keine Modell-Gewichte.
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

type ReversionDirection = Literal["reversion_up", "reversion_down", "range"]


@dataclass(frozen=True, slots=True)
class ReversionState:
    """Momentaufnahme der Reversions-Indikatoren (nur aktuelle und vergangene Kerzen)."""

    z: float  # (close[-1] - SMA50) / Std der letzten 100 Kerzen
    rsi: float  # RSI14 (Wilder-Glättung), 0..100
    pct_distance: float  # (close[-1] - SMA50) / SMA50
    sma50: float  # Gleitender Durchschnitt der letzten 50 Kerzen


class MeanReversionAgent(BaseAgent):
    """Mean-Reversion-Agent — Überdehnung (z-Score, RSI) und Reversionspotenzial."""

    MIN_BARS: int = 100
    SMA_PERIOD: int = 50
    STD_WINDOW: int = 100
    RSI_PERIOD: int = 14
    Z_THRESHOLD: float = 2.0
    RSI_OVERSOLD: float = 45.0
    RSI_OVERBOUGHT: float = 55.0

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="mean_reversion",
                agent_type=AgentType.INDICATOR,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert OHLCV-Daten auf Überdehnung und Reversionspotenzial.

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

        if len(close) < self.MIN_BARS:
            return self._short_data_report(len(close))

        state = self._compute_state(close)
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
            counter_evidence=self._build_counter_evidence(direction),
            invalidations=self._build_invalidations(state, direction),
            raw_confidence=confidence,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )

    # ── Indikatoren ─────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: NDArray[np.float64], period: int) -> float:
        """RSI mit Wilder-Glättung (nur vergangene und aktuelle Kerzen)."""
        delta = np.diff(close)
        gains = np.where(delta > 0.0, delta, 0.0)
        losses = np.where(delta < 0.0, -delta, 0.0)
        avg_gain = float(gains[:period].mean())
        avg_loss = float(losses[:period].mean())
        for i in range(period, len(delta)):
            avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
        if avg_loss < 1e-12:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _compute_state(self, close: NDArray[np.float64]) -> ReversionState:
        """Berechnet z-Score, RSI14 und Prozentabstand für die aktuelle Kerze."""
        sma50 = float(close[-self.SMA_PERIOD :].mean())
        std_window = float(close[-self.STD_WINDOW :].std())
        std_safe = std_window if std_window > 1e-12 else 1e-12
        z = float((close[-1] - sma50) / std_safe)
        rsi = self._rsi(close, self.RSI_PERIOD)
        sma_safe = sma50 if abs(sma50) > 1e-12 else 1e-12
        pct_distance = float((close[-1] - sma50) / sma_safe)
        return ReversionState(
            z=z,
            rsi=rsi,
            pct_distance=pct_distance,
            sma50=sma50,
        )

    # ── Klassifikation & Kalibrierung ───────────────────────────────────────

    def _classify(self, state: ReversionState) -> ReversionDirection:
        """Klassifiziert den Reversionszustand.

        Reversion nur bei Übereinstimmung von Ausprägung (|z| >= 2) und
        Bestätigung (RSI): z extrem negativ ABER RSI hoch (oder z extrem
        positiv ABER RSI niedrig) ist ein laufender Trend, keine
        Reversion → „range".
        """
        if state.z <= -self.Z_THRESHOLD and state.rsi < self.RSI_OVERSOLD:
            return "reversion_up"
        if state.z >= self.Z_THRESHOLD and state.rsi > self.RSI_OVERBOUGHT:
            return "reversion_down"
        return "range"

    def _calibrate(
        self, state: ReversionState, direction: ReversionDirection
    ) -> tuple[dict[str, float], float]:
        """Kalibrierte up/down/range-Verteilung plus rohe Konfidenz.

        Monoton: je extremer die Ausprägung (z) und je deutlicher die RSI-
        Bestätigung, desto höher die richtungsweisende Wahrscheinlichkeit
        (Deckel bei 0,85); „range" bleibt immer >= 0,5. Summe exakt 1,0 (± 1e-6).
        """
        match direction:
            case "reversion_up":
                return self._directional_probabilities(
                    "up",
                    self._reversion_strength(
                        -state.z,
                        self.RSI_OVERSOLD - state.rsi,
                    ),
                )
            case "reversion_down":
                return self._directional_probabilities(
                    "down",
                    self._reversion_strength(
                        state.z,
                        state.rsi - self.RSI_OVERBOUGHT,
                    ),
                )
            case "range":
                return self._range_probabilities(state)

    @staticmethod
    def _reversion_strength(extremity: float, confirmation: float) -> float:
        """Richtungs-Wahrscheinlichkeit einer Reversion (monoton, [0,65, 0,85]).

        `extremity`: |z| oberhalb des Schwellwerts; `confirmation`: RSI-
        Abstand in die Reversionsrichtung (beide trend-seitig positiv).
        """
        z_score = min(max(extremity - 2.0, 0.0), 2.0)
        rsi_score = min(max(confirmation, 0.0), 25.0) / 25.0
        raw = 0.62 + 0.16 * (z_score / 2.0) + 0.07 * rsi_score
        return float(min(0.85, max(0.65, raw)))

    def _directional_probabilities(
        self, winner: Literal["up", "down"], p_winner: float
    ) -> tuple[dict[str, float], float]:
        """Verteilung für eine Reversion: Gewinner in [0,65, 0,85]."""
        p_winner = round(p_winner, 4)
        loser = "down" if winner == "up" else "up"
        probabilities = {
            winner: p_winner,
            loser: 0.05,
            "range": round(1.0 - p_winner - 0.05, 4),
        }
        confidence = 0.35 + 0.5 * (p_winner - 0.65) / 0.20
        return probabilities, float(round(min(0.9, max(0.1, confidence)), 4))

    def _range_probabilities(
        self, state: ReversionState
    ) -> tuple[dict[str, float], float]:
        """Verteilung für Mittellage/widersprüchliche Zustände: „range" >= 0,5."""
        extremity = min(abs(state.z), 4.0) / 4.0
        p_directional = 0.20 + 0.20 * extremity
        if state.z < 0.0:
            p_up = p_directional
            p_down = 0.0
        elif state.z > 0.0:
            p_up = 0.0
            p_down = p_directional
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
        confidence = 0.10 + 0.20 * extremity
        return probabilities, float(round(min(0.9, max(0.1, confidence)), 4))

    # ── Berichtskomponenten ─────────────────────────────────────────────────

    def _build_hypothesis(
        self, state: ReversionState, direction: ReversionDirection
    ) -> str:
        """Einzeilige, menschenlesbare Zusammenfassung des Reversionszustands."""
        details = (
            f"z-score {state.z:+.2f} vs SMA50, RSI14 {state.rsi:.1f}, "
            f"distance {state.pct_distance:+.2%}"
        )
        match direction:
            case "reversion_up":
                return f"Oversold, reversion up expected: {details}"
            case "reversion_down":
                return f"Overbought, reversion down expected: {details}"
            case "range":
                return f"No reversion setup (near mean or contradictory): {details}"

    def _build_evidence(self, state: ReversionState) -> list:
        """Evidenz — Indikatorsnapshot der aktuellen Kerze, nur Scores, keine Behauptungen."""
        return [
            self._make_evidence(
                "z_score",
                f"close z-score vs SMA50 (100-bar std): {state.z:+.3f}",
                "positive" if state.z < -2.0 else "negative" if state.z > 2.0 else "neutral",
                min(abs(state.z) / 4.0, 1.0),
            ),
            self._make_evidence(
                "rsi14",
                f"RSI14 (Wilder): {state.rsi:.1f}",
                "negative" if state.rsi < 30.0 else "positive" if state.rsi > 70.0 else "neutral",
                min(abs(state.rsi - 50.0) / 50.0, 1.0),
            ),
            self._make_evidence(
                "pct_distance",
                f"distance from SMA50: {state.pct_distance:+.2%}",
                "positive" if state.pct_distance < -0.05 else "negative" if state.pct_distance > 0.05 else "neutral",
                min(abs(state.pct_distance) / 0.20, 1.0),
            ),
        ]

    _COUNTER_EXPLANATIONS: ClassVar[dict[ReversionDirection, tuple[str, str, float]]] = {
        "reversion_up": (
            "counter_falling_knife",
            "a sustained crash can continue well beyond two standard deviations",
            0.4,
        ),
        "reversion_down": (
            "counter_trend_continuation",
            "a strong pump can extend beyond two standard deviations before mean reversion",
            0.4,
        ),
        "range": (
            "counter_breakout",
            "the mean-reversion baseline can break on a regime change or news shock",
            0.3,
        ),
    }

    def _build_counter_evidence(self, direction: ReversionDirection) -> list:
        """Gegenhypothesen: alternative Erklärungen für das Ausrichtungsmaß."""
        feature, explanation, relevance = self._COUNTER_EXPLANATIONS[direction]
        return [self._make_evidence(feature, explanation, "negative", relevance)]

    _CROSS_INVALIDATIONS: ClassVar[dict[ReversionDirection, tuple[str, float, str]]] = {
        "reversion_up": ("z reverts above 0 — oversold extension closed without rebound", 0.0, "above"),
        "reversion_down": ("z reverts below 0 — overbought extension closed without pullback", 0.0, "below"),
        "range": ("|z| exceeds 2.0 with RSI confirmation — reversion setup forms", 2.0, "above"),
    }

    def _build_invalidations(
        self, state: ReversionState, direction: ReversionDirection
    ) -> list:
        """Invalidierungsbedingungen für die Reversions-Hypothese."""
        condition, threshold, inv_direction = self._CROSS_INVALIDATIONS[direction]
        return [
            self._make_invalidations(
                condition=condition,
                indicator="z_score",
                threshold=threshold,
                direction=inv_direction,
            ),
            self._make_invalidations(
                condition="RSI14 crosses the reversion-confirmation band against the signal",
                indicator="rsi14",
                threshold=self.RSI_OVERSOLD if direction == "reversion_up" else self.RSI_OVERBOUGHT,
                direction="above" if direction == "reversion_up" else "below",
            ),
            self._make_invalidations(
                condition="sample size too small for stable SMA/std estimates",
                indicator="sample_size",
                threshold=float(self.MIN_BARS),
                direction="below",
            ),
        ]

    def _short_data_report(self, n: int) -> AgentReport:
        """Bericht bei unzureichenden Daten (weniger als MIN_BARS Kerzen)."""
        hypothesis = (
            f"Insufficient data for mean-reversion analysis: {n} bars available, "
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
                "no reversion claim possible with insufficient data — by design",
                "negative",
                0.0,
            )
        ]
        invalidations = [
            self._make_invalidations(
                condition="sample size too small for stable SMA/std estimates",
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
