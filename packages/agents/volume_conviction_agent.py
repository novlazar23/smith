"""Volumen-Konviktions-Agent — Up/Down-Volumen-Verhältnis, OBV-Steigung, Partizipation; bewertet, ob die Preisbewegung von Volumen begleitet wird ("Steckt Überzeugung hinter der Preisbewegung?"). Reiner Orderflow-Proxy aus OHLCV-Daten: Kauf- gegen Verkaufsvolumen (20 Bar), normalisierte 20-Bar-OBV-Steigung, Partizipation (5/50 Bar) — nur Scores, kein Lookahead. Votings-Kalibrierung: "up" > 0,6 nur bei dominierendem Kaufvolumen (Ratio > 1,5, OBV steigend, Kurs über SMA20), "down" als Spiegelbild, sonst range-dominiert oder neutral."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import (
    AgentReport,
    EvidenceReference,
    InvalidationCondition,
)

from .base import AgentConfig, AgentType, BaseAgent, BaseParams, ParamSpace

REQUIRED_KEYS = frozenset({"open", "high", "low", "close", "volume"})


@dataclass(frozen=True, slots=True)
class VolumeConvictionParams(BaseParams):
    """Evolvierbare Volumen-Konviktions-Parameter (Defaults = bisherige Konstanten).

    Entscheidungsknobs (Fenster, Ratio-Schwellen, Partizipations-Minimum)
    plus Kalibrierung (Basis, Signal-Gewichte, Gewinner-Deckel). OBV-/
    Partizipations-Normalisierung, Evidenz-Texte und der 0,65-Boden
    bleiben fest.
    """

    window: int = 20
    ratio_up: float = 1.5
    ratio_down: float = 0.7
    participation_min: float = 0.5
    strength_base: float = 0.62
    strength_ratio_weight: float = 0.08
    strength_slope_weight: float = 0.10
    strength_part_weight: float = 0.05
    p_cap: float = 0.85

    PARAM_SPACE: ClassVar[ParamSpace] = {
        "window": ("int", 10, 50, 1),
        "ratio_up": ("float", 1.1, 3.0, 0.1),
        "ratio_down": ("float", 0.4, 1.2, 0.05),
        "participation_min": ("float", 0.3, 0.8, 0.05),
        "strength_base": ("float", 0.50, 0.75, 0.01),
        "strength_ratio_weight": ("float", 0.0, 0.20, 0.005),
        "strength_slope_weight": ("float", 0.0, 0.25, 0.005),
        "strength_part_weight": ("float", 0.0, 0.15, 0.005),
        "p_cap": ("float", 0.70, 0.95, 0.01),
    }


@dataclass(frozen=True, slots=True)
class _Features:
    """Berechnete Konviktions-Features (nur aktuelle und vergangene Kerzen)."""

    ratio: float
    obv_slope: float
    participation: float
    price_up: bool
    price_down: bool


# allow: SIZE_OK — ein einziger zusammenhängender Agent (eine Klasse, eine Perspektive)
# nach dem Repo-Muster; die Referenzimplementierung AnomalyAgent hat ~640 pure LOC.
# Eine Zerlegung würde eine künstliche Mixin-Hierarchie außerhalb des Codebase-Stils erfordern.


class VolumeConvictionAgent(BaseAgent):
    """Volumen-Konviktions-Agent — Up/Down-Volumen, OBV-Steigung, Partizipation."""

    MIN_BARS: int = 50

    def __init__(
        self, config: AgentConfig | None = None, params: VolumeConvictionParams | None = None
    ) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="volume_conviction",
                agent_type=AgentType.ORDERFLOW,
            )
        super().__init__(config)
        self._params = params if params is not None else VolumeConvictionParams()

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert aufsteigende OHLCV-Arrays (älteste → neueste Kerze, Index -1 = aktuell) auf Volumen-Überzeugung."""
        missing = REQUIRED_KEYS - data.keys()
        if missing:
            raise ValueError(f"Missing required OHLCV keys: {sorted(missing)}")

        close = np.asarray(data["close"], dtype=np.float64)
        open_ = np.asarray(data["open"], dtype=np.float64)
        volume = np.asarray(data["volume"], dtype=np.float64)

        n = len(close)
        if n < self.MIN_BARS:
            return self._short_data_report(n)

        f = self._compute_indicators(close, open_, volume)
        side = self._conviction_side(f)
        probabilities = self._probabilities(side, f)

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.now(UTC),
            hypothesis=self._hypothesis(side, f),
            probabilities=probabilities,
            evidence=self._evidence(f),
            counter_evidence=self._counter_evidence(side),
            invalidations=self._invalidations(),
            raw_confidence=self._confidence(side, probabilities),
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )

    # ── Indikatoren ─────────────────────────────────────────────────────────

    def _compute_indicators(
        self,
        close: NDArray[np.float64],
        open_: NDArray[np.float64],
        volume: NDArray[np.float64],
    ) -> _Features:
        """Up/Down-Volumen-Verhältnis, OBV-Steigung, Partizipation (kein Lookahead)."""
        w = self._params.window
        sma = float(np.mean(close[-w:]))
        up_mask = close[-w:] >= open_[-w:]
        up_volume = float(volume[-w:][up_mask].sum())
        down_volume = float(volume[-w:][~up_mask].sum())
        ratio = up_volume / max(down_volume, 1e-12)

        signed = np.sign(close - open_) * volume
        obv = np.cumsum(signed)
        scale = float(np.mean(np.abs(signed)[-50:])) * 20.0
        obv_slope = float(np.clip((obv[-1] - obv[-w - 1]) / max(scale, 1e-12), -1.5, 1.5))

        participation = float(np.mean(volume[-5:])) / max(float(np.mean(volume[-50:])), 1e-12)

        return _Features(
            ratio=ratio,
            obv_slope=obv_slope,
            participation=participation,
            price_up=bool(close[-1] > sma),
            price_down=bool(close[-1] < sma),
        )

    def _conviction_side(self, f: _Features) -> str:
        """Votingsseite: 'up', 'down', 'range' (divergent/schwach) oder 'neutral'."""
        p = self._params
        if f.price_up and f.ratio > p.ratio_up and f.obv_slope > 0.0:
            return "up"
        if f.price_down and f.ratio < p.ratio_down and f.obv_slope < 0.0:
            return "down"
        divergent = (f.price_up and f.ratio < p.ratio_down) or (
            f.price_down and f.ratio > p.ratio_up
        )
        if divergent or f.participation < p.participation_min:
            return "range"
        return "neutral"

    def _conviction_score(self, ratio_excess: float, slope: float, participation: float) -> float:
        """Monotone Konviktionsformel — steigt mit Ratio-Exzess, OBV-Steigung, Partizipation."""
        p = self._params
        score = (
            p.strength_base
            + p.strength_ratio_weight * min(max(ratio_excess, 0.0), 1.5)
            + p.strength_slope_weight * min(max(slope, 0.0), 1.0)
            + p.strength_part_weight * min(participation, 2.0) / 2.0
        )
        return float(min(max(score, 0.65), p.p_cap))

    def _probabilities(self, side: str, f: _Features) -> dict[str, float]:
        """Wahrscheinlichkeiten — summieren sich exakt auf 1.0."""
        p = self._params
        if side == "up":
            score = self._conviction_score(f.ratio - p.ratio_up, f.obv_slope, f.participation)
            up = round(score, 4)
            down = round((1.0 - up) * 0.25, 4)
            return {"up": up, "down": down, "range": round(1.0 - up - down, 4)}
        if side == "down":
            excess = (p.ratio_down - f.ratio) / p.ratio_down * 1.5
            score = self._conviction_score(excess, -f.obv_slope, f.participation)
            down = round(score, 4)
            up = round((1.0 - down) * 0.25, 4)
            return {"up": up, "down": down, "range": round(1.0 - down - up, 4)}
        if side == "range":
            return {"up": 0.25, "down": 0.25, "range": 0.50}
        return {"up": 0.33, "down": 0.33, "range": 0.34}

    # ── Report-Bausteine ────────────────────────────────────────────────────

    def _hypothesis(self, side: str, f: _Features) -> str:
        """Einzeilige Zusammenfassung der Konviktions-Signale."""
        ratio = min(f.ratio, 999.9)
        if side == "up":
            return f"Kaufvolumen dominiert: up/down={ratio:.1f}, OBV steigend ({f.obv_slope:+.2f})"
        if side == "down":
            return f"Verkaufsvolumen dominiert: up/down={ratio:.2f}, OBV fallend ({f.obv_slope:+.2f})"
        if side == "range":
            return f"Keine Überzeugung: up/down={ratio:.2f}, OBV {f.obv_slope:+.2f}, Partizipation {f.participation:.2f}"
        return f"Neutrale Lesart: up/down={ratio:.2f}, OBV {f.obv_slope:+.2f}, Partizipation {f.participation:.2f}"

    def _evidence(self, f: _Features) -> list[EvidenceReference]:
        """Evidenz — die drei Konviktions-Features als Scores, keine Behauptungen."""
        p = self._params
        if f.ratio > p.ratio_up:
            ratio_dir = "positive"
            ratio_rel = min(1.0, 0.2 + (f.ratio - p.ratio_up) / p.ratio_up * 0.8)
        elif f.ratio < p.ratio_down:
            ratio_dir = "negative"
            ratio_rel = min(1.0, 0.2 + (p.ratio_down - f.ratio) / p.ratio_down * 0.8)
        else:
            ratio_dir = "neutral"
            ratio_rel = 0.2

        if f.obv_slope > 0.1:
            slope_dir = "positive"
        elif f.obv_slope < -0.1:
            slope_dir = "negative"
        else:
            slope_dir = "neutral"

        if f.participation >= 1.2:
            part_dir = "positive"
        elif f.participation < 0.5:
            part_dir = "negative"
        else:
            part_dir = "neutral"

        ratio_disp = min(f.ratio, 999.9)
        return [
            self._make_evidence(
                "up_down_volume_ratio",
                f"up/down-Volumen (20 Bar): {ratio_disp:.2f}",
                ratio_dir,
                ratio_rel,
            ),
            self._make_evidence(
                "obv_slope",
                f"OBV-Steigung (20 Bar, normalisiert): {f.obv_slope:+.2f}",
                slope_dir,
                min(1.0, 0.2 + abs(f.obv_slope) / 1.5),
            ),
            self._make_evidence(
                "participation",
                f"Partizipation (5 Bar / 50 Bar): {f.participation:.2f}",
                part_dir,
                min(1.0, 0.2 + f.participation / 3.0),
            ),
        ]

    def _counter_evidence(self, side: str) -> list[EvidenceReference]:
        """Gegenhypothesen — alternative Erklärungen für das Volumenmuster."""
        generic = self._make_evidence(
            "counter_volume_noise",
            "Volumenmuster können durch Einzelgroßaufträge oder Auktions-Schocks verzerrt sein",
            "negative",
            0.3,
        )
        if side == "up":
            specific = self._make_evidence(
                "counter_distribution",
                "Kaufvolumen-Spitze kann vorübergehend sein — Abfluss bei Kursfortsetzung",
                "negative",
                0.4,
            )
        elif side == "down":
            specific = self._make_evidence(
                "counter_accumulation",
                "Verkaufsvolumen-Spitze kann gezielte Akkumulation institutioneller Käufer sein",
                "negative",
                0.4,
            )
        else:
            specific = self._make_evidence(
                "counter_momentum",
                "Dünnes Volumen schließt kurzfristige Momentum-Phasen nicht aus",
                "negative",
                0.2,
            )
        return [specific, generic]

    def _invalidations(self) -> list[InvalidationCondition]:
        """Invalidierungsbedingungen der Konviktions-Hypothese."""
        return [
            self._make_invalidations(
                condition="Partizipation fällt unter 0,5 — Volumen trocknet aus",
                indicator="participation",
                threshold=0.5,
                direction="below",
            ),
            self._make_invalidations(
                condition="OBV-Steigung dreht durch 0,0 gegen die Signalrichtung",
                indicator="obv_slope",
                threshold=0.0,
                direction="below",
            ),
            self._make_invalidations(
                condition="Up/Down-Volumen-Verhältnis bricht unter 1,0 (Gegenrichtung)",
                indicator="up_down_volume_ratio",
                threshold=1.0,
                direction="below",
            ),
        ]

    def _confidence(self, side: str, probabilities: dict[str, float]) -> float:
        """Roh-Konfidenz: 0,4-0,9 bei Konviktionsseite, 0,2-0,25 sonst, 0,05 bei Kurzdaten."""
        if side in ("up", "down"):
            p = probabilities[side]
            return float(round(min(max(0.4 + 0.5 * (p - 0.65) / 0.2, 0.4), 0.9), 4))
        if side == "range":
            return 0.25
        return 0.2

    def _short_data_report(self, n: int) -> AgentReport:
        """Bericht bei unzureichenden Daten (< 50 Kerzen) — neutrale Verteilung."""
        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.now(UTC),
            hypothesis=(
                f"Unzureichende Daten für Volumen-Konviktions-Analyse: {n} Kerzen verfügbar, "
                f"mindestens {self.MIN_BARS} erforderlich. Neutrale Verteilung."
            ),
            probabilities={"up": 0.33, "down": 0.33, "range": 0.34},
            evidence=[
                self._make_evidence(
                    "insufficient_data",
                    f"nur {n} Kerzen verfügbar, mindestens {self.MIN_BARS} erforderlich",
                    "neutral",
                    0.0,
                )
            ],
            counter_evidence=[
                self._make_evidence(
                    "counter_insufficient",
                    "keine Konviktionsaussage möglich bei kurzen Zeiträumen — by design",
                    "negative",
                    0.0,
                )
            ],
            invalidations=self._invalidations(),
            raw_confidence=0.05,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )
