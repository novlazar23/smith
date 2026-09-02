# allow: SIZE_OK — Ein-Datei-Agent-Modul nach Repo-Konvention (ein Modul pro
# Agent; Referenz anomaly_agent.py ist 640 Zeilen). Agent-Klasse, Indikator-
# Berechnung und Berichtskomponenten bilden hier eine kohärente Einheit.

"""Volatility-Regime-Agent — Bollinger-Bandbreite gegen Baseline, 20-Bar-Position.

Perspektive: „In welchem Volatilitätsregime ist der Markt gerade, und was
bedeutet das für die nächste Bewegung?" Kernidee: Baseline-Bandbreite
(Bollinger-Bandbreite über die letzten ~120 Kerzen) als Maßstab.
- Squeeze (Bandbreite vor dem letzten Zehntel <= 0,5x Baseline) plus
  Ausbruch an der 20-Bar-Grenze → richtungsbestimmt („up"/„down" 0,65 - 0,85).
  Die damit einhergehende Bandverbreiterung ist Folge, kein Gegenindikator.
- Ohne Squeeze (Mittellage, Expansion, Mixed) → „range" dominant
  (>= 0,5), Konfidenz skaliert mit der Expansion. Reines NumPy, kein
  Lookahead.
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

type RegimeDirection = Literal["up", "down", "range"]


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Momentaufnahme der Regime-Indikatoren (nur aktuelle und vergangene Kerzen)."""

    squeeze_ratio: float  # Bandbreite vor dem letzten Zehntel / Baseline-Bandbreite
    expansion_ratio: float  # aktuelle Bandbreite / Baseline-Bandbreite
    position20: float  # 20-Bar-Position von close: 0 = Tief, 1 = Hoch
    atr14: float  # Mittel der letzten 14 True Ranges


def _direction_label(value: float) -> str:
    """Mapt einen signierten Wert auf die Evidenz-Richtung."""
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "neutral"


class VolatilityRegimeAgent(BaseAgent):
    """Volatility-Regime-Agent — Squeeze-Breakouts und Vol-Expansion gegen Baseline."""

    MIN_BARS: int = 150
    BB_PERIOD: int = 20
    ATR_PERIOD: int = 14
    SQUEEZE_THRESHOLD: float = 0.5
    EXPANSION_THRESHOLD: float = 1.5
    POSITION_EDGE: float = 0.8
    PRE_WINDOW: int = 20
    PRE_OFFSET: int = 10

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="volatility_regime",
                agent_type=AgentType.REGIME,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert OHLCV-Daten auf das aktuelle Volatilitätsregime.

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

    def _bandwidth_series(
        self, close: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Bollinger-Bandbreite (4*std20/sma20) pro Fensterende.

        Gibt (bw, endings) zurück: bw[i] ist die Bandbreite des 20-Kerzen-
        Fensters, das bei Index `endings[i]` endet. Jedes Fenster verwendet
        nur Kerzen <= ending (kein Lookahead).
        """
        n = len(close)
        start = max(self.BB_PERIOD - 1, n - 150)
        endings = np.arange(start, n, dtype=np.int64)
        bw = np.empty_like(endings, dtype=np.float64)
        for j, e in enumerate(endings):
            window = close[e - self.BB_PERIOD + 1 : e + 1]
            center = float(window.mean())
            spread = float(window.std())
            center_safe = center if abs(center) > 1e-12 else 1e-12
            bw[j] = 4.0 * spread / center_safe
        return bw, endings

    def _atr(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
    ) -> float:
        """ATR14: Mittel der letzten 14 True Ranges."""
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
    ) -> RegimeState:
        """Berechnet Squeeze-, Expansions- und Positions-Indikatoren."""
        n = len(close)
        bw, endings = self._bandwidth_series(close)
        hist = bw[endings <= n - 1 - self.PRE_WINDOW - self.PRE_OFFSET]
        pre = bw[
            (endings > n - 1 - self.PRE_WINDOW - self.PRE_OFFSET)
            & (endings <= n - 1 - self.PRE_OFFSET)
        ]
        now = bw[endings == n - 1]
        hist_mean = float(hist.mean()) if len(hist) else 1e-12
        hist_safe = hist_mean if hist_mean > 1e-12 else 1e-12
        pre_mean = float(pre.mean()) if len(pre) else hist_safe
        squeeze_ratio = pre_mean / hist_safe
        expansion_ratio = float(now[0]) / hist_safe if len(now) else 1.0

        low20 = float(low[-self.BB_PERIOD :].min())
        high20 = float(high[-self.BB_PERIOD :].max())
        bar_range = high20 - low20
        position20 = (
            0.5
            if bar_range < 1e-12
            else float((close[-1] - low20) / bar_range)
        )
        position20 = min(max(position20, 0.0), 1.0)

        return RegimeState(
            squeeze_ratio=float(squeeze_ratio),
            expansion_ratio=float(expansion_ratio),
            position20=position20,
            atr14=self._atr(high, low, close),
        )

    # ── Klassifikation & Kalibrierung ───────────────────────────────────────

    def _classify(self, state: RegimeState) -> RegimeDirection:
        """Klassifiziert das Regime: Squeeze-Ausbruch (richtungsbestimmt) oder range.

        Ein Squeeze-Ausbruch erfordert beide Bedingungen: die Bandbreite
        kurz vor dem aktuellen Zehntel war <= 0,5x der Baseline UND der
        Schlusskurs steht an der jeweiligen 20-Bar-Grenze (>= 0,8 / <= 0,2).
        """
        if state.squeeze_ratio <= self.SQUEEZE_THRESHOLD:
            if state.position20 >= self.POSITION_EDGE:
                return "up"
            if state.position20 <= 1.0 - self.POSITION_EDGE:
                return "down"
        return "range"

    def _calibrate(
        self, state: RegimeState, direction: RegimeDirection
    ) -> tuple[dict[str, float], float]:
        """Kalibrierte up/down/range-Verteilung plus rohe Konfidenz.

        Monoton: je enger der Squeeze und je weiter an der 20-Bar-Grenze,
        desto höher die richtungsweisende Wahrscheinlichkeit (Deckel 0,85);
        „range" bleibt immer >= 0,5. Summe exakt 1,0 (± 1e-6).
        """
        match direction:
            case "up":
                return self._directional_probabilities(
                    "up",
                    self._breakout_strength(
                        state.squeeze_ratio,
                        (state.position20 - self.POSITION_EDGE)
                        / (1.0 - self.POSITION_EDGE),
                    ),
                )
            case "down":
                return self._directional_probabilities(
                    "down",
                    self._breakout_strength(
                        state.squeeze_ratio,
                        ((1.0 - self.POSITION_EDGE) - state.position20)
                        / (1.0 - self.POSITION_EDGE),
                    ),
                )
            case "range":
                return self._range_probabilities(state)

    def _breakout_strength(self, squeeze_ratio: float, edge: float) -> float:
        """Richtungs-Wahrscheinlichkeit eines Squeeze-Ausbruchs (monoton).

        `squeeze_ratio` <= 0,5 (eng = stark), `edge` in [0, 1] (0 = an der
        Kante, 1 = Mitte des 20-Bar-Bands) — beide sind im Breakout-Fall
        bereits nur in der richtungsbestimmten Kombination erreichbar.
        """
        squeeze_score = max(0.0, min(1.0, (self.SQUEEZE_THRESHOLD - squeeze_ratio)
                                     / self.SQUEEZE_THRESHOLD))
        edge_score = max(0.0, min(1.0, edge))
        raw = 0.63 + 0.17 * squeeze_score + 0.05 * edge_score
        return float(min(0.85, max(0.65, raw)))

    def _directional_probabilities(
        self, winner: Literal["up", "down"], p_winner: float
    ) -> tuple[dict[str, float], float]:
        """Verteilung für einen Squeeze-Ausbruch: Gewinner in [0,65, 0,85]."""
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
        self, state: RegimeState
    ) -> tuple[dict[str, float], float]:
        """Verteilung für Expansion/Mittellage: „range" mit >= 0,5."""
        strength = min(state.expansion_ratio, 2.0) / 2.0
        p_directional = 0.20 + 0.20 * strength
        p_up = p_directional * state.position20
        p_down = p_directional * (1.0 - state.position20)
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

    def _build_hypothesis(self, state: RegimeState, direction: RegimeDirection) -> str:
        """Einzeilige, menschenlesbare Zusammenfassung des Regimezustands."""
        details = (
            f"squeeze ratio {state.squeeze_ratio:.2f} vs baseline, "
            f"expansion ratio {state.expansion_ratio:.2f}, "
            f"20-bar position {state.position20:.2f}, "
            f"ATR14 {state.atr14:.3f}"
        )
        match direction:
            case "up":
                return f"Squeeze breakout to the upside: {details}"
            case "down":
                return f"Squeeze breakdown to the downside: {details}"
            case "range":
                return f"No directional breakout (expansion or mid-band): {details}"

    def _build_evidence(self, state: RegimeState) -> list:
        """Evidenz — Indikatorsnapshot der aktuellen Kerze, nur Scores, keine Behauptungen."""
        return [
            self._make_evidence(
                "squeeze_ratio",
                f"pre-window bandwidth vs baseline: {state.squeeze_ratio:.3f} "
                f"(<= {self.SQUEEZE_THRESHOLD} = squeeze)",
                "negative" if state.squeeze_ratio <= self.SQUEEZE_THRESHOLD else "neutral",
                min(abs(state.squeeze_ratio - 1.0), 1.0),
            ),
            self._make_evidence(
                "expansion_ratio",
                f"current bandwidth vs baseline: {state.expansion_ratio:.3f} "
                f"(>= {self.EXPANSION_THRESHOLD} = expansion)",
                "neutral",
                min(abs(state.expansion_ratio - 1.0), 1.0),
            ),
            self._make_evidence(
                "position20",
                f"close position within 20-bar range: {state.position20:.3f}",
                "positive" if state.position20 > 0.5 else "negative",
                abs(state.position20 - 0.5) * 2.0,
            ),
            self._make_evidence(
                "atr14",
                f"ATR14 {state.atr14:.4f}",
                "neutral",
                0.3,
            ),
        ]

    _COUNTER_EXPLANATIONS: ClassVar[dict[RegimeDirection, tuple[str, str, float]]] = {
        "up": (
            "counter_false_breakout",
            "breakouts from quiet regimes frequently fail and snap back inside the band",
            0.4,
        ),
        "down": (
            "counter_false_breakdown",
            "breakdowns from quiet regimes frequently fail and snap back inside the band",
            0.4,
        ),
        "range": (
            "counter_breakout",
            "the expansion/mid-band regime can resolve into a directional move at any time",
            0.3,
        ),
    }

    def _build_counter_evidence(
        self, state: RegimeState, direction: RegimeDirection
    ) -> list:
        """Gegenhypothesen: alternative Erklärungen für das Regimebild."""
        feature, explanation, relevance = self._COUNTER_EXPLANATIONS[direction]
        return [self._make_evidence(feature, explanation, "negative", relevance)]

    _CROSS_INVALIDATIONS: ClassVar[dict[RegimeDirection, tuple[str, float, str]]] = {
        "up": ("close falls back below the 20-bar midpoint — breakout failed", 0.5, "below"),
        "down": ("close recovers above the 20-bar midpoint — breakdown failed", 0.5, "above"),
        "range": (
            "squeeze with breakout at the 20-bar edge forms",
            0.5,
            "above",
        ),
    }

    def _build_invalidations(
        self, state: RegimeState, direction: RegimeDirection
    ) -> list:
        """Invalidierungsbedingungen für die Regime-Hypothese."""
        condition, threshold, inv_direction = self._CROSS_INVALIDATIONS[direction]
        return [
            self._make_invalidations(
                condition=condition,
                indicator="position20" if direction != "range" else "squeeze_ratio",
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
                condition="sample size too small for stable bandwidth baselines",
                indicator="sample_size",
                threshold=float(self.MIN_BARS),
                direction="below",
            ),
        ]

    def _short_data_report(self, n: int) -> AgentReport:
        """Bericht bei unzureichenden Daten (weniger als MIN_BARS Kerzen)."""
        hypothesis = (
            f"Insufficient data for volatility-regime analysis: {n} bars available, "
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
                "no regime claim possible with insufficient data — by design",
                "negative",
                0.0,
            )
        ]
        invalidations = [
            self._make_invalidations(
                condition="sample size too small for stable bandwidth baselines",
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
