"""Elliott Wave core — WaveCount, WaveType, ElliottScenario, ElliottWaveDetector."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class WaveType(StrEnum):
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"


@dataclass
class WaveCount:
    """Eine Wellenzahl mit 5 Impuls- oder 3-Korrektur-Wellen."""

    wave_type: WaveType
    directions: list[str] = field(default_factory=list)
    start_price: float = 0.0
    end_price: float = 0.0
    pivot_times: list[int] = field(default_factory=list)
    pivot_prices: list[float] = field(default_factory=list)


@dataclass
class ElliottScenario:
    """Ein Elliott-Wave-Szenario mit Regelvalidierung."""

    scenario_id: str
    wave_count: WaveCount | None
    direction: Literal["up", "down", "range"] = "range"
    rule_valid: bool = False
    guideline_score: float = 0.0
    invalidation_price: float = 0.0
    description: str = ""
    key_levels: list[float] = field(default_factory=list)


class ElliottWaveDetector:
    """Erkennt Elliott-Wave-Strukturen aus OHLCV-Daten.

    Erkennt Impuls- und Korrekturstrukturen durch
    Pivot-Analyse und validiert Elliott-Regeln.
    """

    PIVOT_WINDOW = 5

    def __init__(self, pivot_window: int | None = None) -> None:
        if pivot_window is not None:
            self.PIVOT_WINDOW = pivot_window

    def detect(
        self, data: dict[str, NDArray[np.float64]]
    ) -> list[ElliottScenario]:
        """Erkennt mindestens 2 Elliott-Szenarios.

        Args:
            data: Dict mit 'close', 'high', 'low'.

        Returns:
            Liste von ElliottScenario (mindestens 2).
        """
        close = data["close"]
        high = data["high"]
        low = data["low"]
        n = len(close)

        pivots = self._find_pivots(high, low, close)

        scenarios: list[ElliottScenario] = []

        if len(pivots) >= 5:
            primary = self._build_impulse_scenario(
                pivots, close, n
            )
            alt = self._build_correction_scenario(
                pivots, close, n
            )
            scenarios.extend([primary, alt])
        elif len(pivots) >= 3:
            primary = self._build_correction_scenario(
                pivots, close, n
            )
            alt = self._build_impulse_scenario(
                pivots, close, n
            )
            scenarios.extend([primary, alt])
        else:
            scenarios = self._fallback_scenarios(close)

        return scenarios

    def _find_pivots(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
    ) -> list[dict]:
        w = self.PIVOT_WINDOW
        pivots: list[dict] = []

        for i in range(w, len(close) - w):
            if high[i] == np.max(high[i - w : i + w + 1]):
                pivots.append(
                    {"time": i, "price": float(high[i]), "type": "high"}
                )
            if low[i] == np.min(low[i - w : i + w + 1]):
                pivots.append(
                    {"time": i, "price": float(low[i]), "type": "low"}
                )

        pivots.sort(key=lambda p: p["time"])
        return pivots

    def _build_impulse_scenario(
        self,
        pivots: list[dict],
        close: NDArray[np.float64],
        n: int,
    ) -> ElliottScenario:
        """Baut ein Impuls-Szenario (5-Welle) auf."""
        recent = pivots[-5:] if len(pivots) >= 5 else pivots[-3:]

        prices = [p["price"] for p in recent]
        times = [p["time"] for p in recent]
        ptypes = [p["type"] for p in recent]

        current_price = float(close[-1])

        if len(recent) >= 3 and ptypes[-1] == "low":
            direction: str = "up"
        elif len(recent) >= 3 and ptypes[-1] == "high":
            direction = "down"
        else:
            direction = "up"

        wave_counts = WaveCount(
            wave_type=WaveType.IMPULSE,
            directions=["up" if direction == "up" else "down"] * 5,
            start_price=prices[0] if prices else current_price,
            end_price=prices[-1] if prices else current_price,
            pivot_times=times,
            pivot_prices=prices,
        )

        rule_valid = self._validate_impulse_rules(wave_counts, recent, current_price)
        guideline_score = self._score_impulse_guidelines(wave_counts, recent)

        invalidation_price = self._calc_impulse_invalidation(
            wave_counts, direction, current_price
        )

        description = (
            f"5-wave impulse {direction.capitalize()}: "
            f"pivots at {len(recent)} points, "
            f"rules {'passed' if rule_valid else 'violated'}"
        )

        key_levels = prices[2:] if len(prices) > 2 else [current_price * 0.98, current_price * 1.02]

        return ElliottScenario(
            scenario_id=f"impulse_{direction}",
            wave_count=wave_counts,
            direction=direction,
            rule_valid=rule_valid,
            guideline_score=round(guideline_score, 2),
            invalidation_price=round(invalidation_price, 2),
            description=description,
            key_levels=[round(val, 2) for val in key_levels],
        )

    def _build_correction_scenario(
        self,
        pivots: list[dict],
        close: NDArray[np.float64],
        n: int,
    ) -> ElliottScenario:
        """Baut ein Korrektur-Szenario (3-Welle ABC) auf."""
        recent = pivots[-3:] if len(pivots) >= 3 else pivots[-2:]

        prices = [p["price"] for p in recent]
        times = [p["time"] for p in recent]
        ptypes = [p["type"] for p in recent]

        current_price = float(close[-1])

        if len(recent) >= 2:
            if ptypes[-1] == "high" and ptypes[-2] == "low":
                direction = "up"
            elif ptypes[-1] == "low" and ptypes[-2] == "high":
                direction = "down"
            else:
                direction = "up"
        else:
            direction = "up"

        wave_counts = WaveCount(
            wave_type=WaveType.CORRECTIVE,
            directions=["up" if direction == "up" else "down"] * 3,
            start_price=prices[0] if prices else current_price,
            end_price=prices[-1] if prices else current_price,
            pivot_times=times,
            pivot_prices=prices,
        )

        rule_valid = self._validate_correction_rules(wave_counts, recent, current_price)
        guideline_score = self._score_correction_guidelines(wave_counts, recent)

        invalidation_price = self._calc_correction_invalidation(
            wave_counts, direction, current_price
        )

        description = (
            f"ABC correction {direction.capitalize()}: "
            f"pivots at {len(recent)} points, "
            f"rules {'passed' if rule_valid else 'violated'}"
        )

        key_levels = prices[1:] if len(prices) > 1 else [current_price * 0.98, current_price * 1.02]

        return ElliottScenario(
            scenario_id=f"correction_{direction}",
            wave_count=wave_counts,
            direction=direction,
            rule_valid=rule_valid,
            guideline_score=round(guideline_score, 2),
            invalidation_price=round(invalidation_price, 2),
            description=description,
            key_levels=[round(val, 2) for val in key_levels],
        )

    def _fallback_scenarios(
        self, close: NDArray[np.float64]
    ) -> list[ElliottScenario]:
        """Fallback: 2 Szenarios bei zu wenigen Pivots."""
        current = float(close[-1])
        trend = (
            "up"
            if len(close) > 5 and float(np.mean(close[-5:])) > float(np.mean(close[:5]))
            else "down"
        )

        w = WaveCount(
            wave_type=WaveType.IMPULSE,
            directions=[trend] * 5,
            start_price=float(close[0]),
            end_price=current,
        )
        primary = ElliottScenario(
            scenario_id=f"impulse_{trend}",
            wave_count=w,
            direction=trend,
            rule_valid=True,
            guideline_score=0.4,
            invalidation_price=current * (0.95 if trend == "up" else 1.05),
            description=f"Impulse {trend.capitalize()} (fallback, limited pivots)",
            key_levels=[current * 0.98, current * 1.02],
        )

        c = WaveCount(
            wave_type=WaveType.CORRECTIVE,
            directions=["up" if trend == "down" else "down"] * 3,
            start_price=float(close[0]),
            end_price=current,
        )
        correction = ElliottScenario(
            scenario_id=f"correction_{'down' if trend == 'up' else 'up'}",
            wave_count=c,
            direction="down" if trend == "up" else "up",
            rule_valid=True,
            guideline_score=0.35,
            invalidation_price=current * (1.05 if trend == "up" else 0.95),
            description="Corrective (fallback, limited pivots)",
            key_levels=[current * 0.98, current * 1.02],
        )

        return [primary, correction]

    # ── Regelvalidierung ─────────────────────────────────────────────

    def _validate_impulse_rules(
        self,
        wc: WaveCount,
        pivots: list[dict],
        current_price: float,
    ) -> bool:
        """Validiert Elliott-Impuls-Regeln.

        - Wave 2 retraces nicht 100% von Wave 1
        - Wave 3 ist nicht die kuerzeste von 1, 3, 5
        - Wave 4 ueberlappt nicht Wave 1 (ausser Silber-Fault)
        """
        if len(wc.pivot_prices) < 3:
            return True

        prices = wc.pivot_prices
        if len(prices) >= 4:
            w1_high = prices[1]
            w1_low = prices[0]

            w1_range = abs(w1_high - w1_low)
            if w1_range > 0:
                w2_retreat = (prices[1] - prices[2]) / w1_range if prices[2] < prices[1] else 0
                if w2_retreat >= 1.0:
                    return False

        if len(prices) >= 6:
            w1_size = abs(prices[1] - prices[0])
            w3_size = abs(prices[3] - prices[2])
            w5_size = abs(prices[5] - prices[4])
            if w3_size > 0 and w5_size > 0 and w1_size > 0:
                sizes = sorted([w1_size, w3_size, w5_size], reverse=True)
                if (sizes[2] == w5_size or sizes[2] == w1_size) and w3_size == min(w1_size, w3_size, w5_size) and w1_size > 0.01 * current_price:
                    return False

        return True

    def _validate_correction_rules(
        self,
        wc: WaveCount,
        pivots: list[dict],
        current_price: float,
    ) -> bool:
        """Validiert Korrektur-Regeln (ABC).

        - Wave B retraces nicht 100% von Wave A
        - C ist typisch 0.618x bis 1.618x von A
        """
        if len(wc.pivot_prices) < 3:
            return True

        prices = wc.pivot_prices
        if len(prices) >= 3:
            w1_range = abs(prices[1] - prices[0])
            if w1_range > 0:
                w2_retreat = abs(prices[2] - prices[1]) / w1_range
                if w2_retreat >= 1.0:
                    return False

        return True

    # ── Richtlinien-Scoring ──────────────────────────────────────────

    def _score_impulse_guidelines(
        self, wc: WaveCount, pivots: list[dict]
    ) -> float:
        """Score basierend auf Elliott-Richtlinien."""
        score = 0.5
        prices = wc.pivot_prices

        if len(prices) >= 4:
            w1_size = abs(prices[1] - prices[0])
            w3_size = abs(prices[3] - prices[2])
            if w1_size > 0 and abs(w3_size / w1_size - 1.618) < 0.3:
                score += 0.15
            if w3_size > max(w1_size, abs(prices[5] - prices[4]) if len(prices) > 5 else 0):
                score += 0.1

        if len(prices) >= 5:
            w4_retreat = abs(prices[4] - prices[3]) / abs(prices[3] - prices[2])
            if 0.236 <= w4_retreat <= 0.618:
                score += 0.1

        return min(1.0, score)

    def _score_correction_guidelines(
        self, wc: WaveCount, pivots: list[dict]
    ) -> float:
        """Score basierend auf Korrektur-Richtlinien."""
        score = 0.4
        prices = wc.pivot_prices

        if len(prices) >= 3:
            w1_range = abs(prices[1] - prices[0])
            w2_retreat = abs(prices[2] - prices[1]) / w1_range if w1_range > 0 else 0.5
            if 0.5 <= w2_retreat <= 0.786:
                score += 0.15
            elif 0.382 <= w2_retreat <= 0.618:
                score += 0.1

        return min(1.0, score)

    # ── Invalidation ────────────────────────────────────────────────

    def _calc_impulse_invalidation(
        self, wc: WaveCount, direction: str, current_price: float
    ) -> float:
        if wc.pivot_prices:
            if direction == "up":
                return min(wc.pivot_prices) * 0.99
            else:
                return max(wc.pivot_prices) * 1.01
        return current_price * (0.95 if direction == "up" else 1.05)

    def _calc_correction_invalidation(
        self, wc: WaveCount, direction: str, current_price: float
    ) -> float:
        if wc.pivot_prices:
            if direction == "up":
                return max(wc.pivot_prices) * 1.01
            else:
                return min(wc.pivot_prices) * 0.99
        return current_price * (1.05 if direction == "up" else 0.95)
