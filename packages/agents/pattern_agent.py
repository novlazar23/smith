"""Pattern Analysis Agent — rule-based chart pattern detection with similarity scoring.

Spec §11.4: Pattern Analysis Agent
- Rule-based + numerical similarity pattern detection
- Report includes: candidate, quality, similarity, sample_size, distribution, invalidation
- Counter-hypothesis (Gegenhypothese) required
- Status: SHADOW (initial)
- No LLM probability estimates — purely rule-based
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from packages.chart_structure.base import ChartPattern, SwingPivot
from packages.chart_structure.patterns import PatternDetector
from packages.chart_structure.swing import SwingDetector
from packages.schemas.agent_report import AgentReport, EvidenceReference

from .base import AgentConfig, AgentType, BaseAgent

# ---------------------------------------------------------------------------
# Pattern dataclass for internal bookkeeping
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PatternCandidate:
    """Eine erkannte oder potenzielle Pattern-Instanz."""

    name: str  # e.g. "head_and_shoulders", "ascending_triangle", "range"
    direction: str  # "bullish", "bearish", "neutral"
    quality: float  # 0-1 score
    similarity: float  # 0-1 similarity to canonical pattern
    start_idx: int
    end_idx: int
    evidence_features: list[str]


# ---------------------------------------------------------------------------
# Canonical pattern shape generators (for similarity computation)
# ---------------------------------------------------------------------------

def _canonical_h_and_s(length: int, head: float, shoulder: float) -> NDArray[np.float64]:
    """Erzeugt ein kanonisches Head-and-Shoulders Preisszenario."""
    t = np.linspace(0, 1, length)
    x = np.sin(t * np.pi) * (head - shoulder) + shoulder
    return x


def _canonical_inverted_h_and_s(length: int, head: float, shoulder: float) -> NDArray[np.float64]:
    """Erzeugt ein kanonisches inverted Head-and-Shoulders Preisszenario."""
    t = np.linspace(0, 1, length)
    x = -np.sin(t * np.pi) * (shoulder - head) + shoulder
    return x


# ---------------------------------------------------------------------------
# Similarity computation helpers (module-level)
# ---------------------------------------------------------------------------

def _similarity_h_and_s(
    close: NDArray[np.float64],
    t_left: int, t_head: int, t_right: int,
    h_head: float, h_shoulder: float,
) -> float:
    """Numerische Ähnlichkeit zu einem kanonischen H&S."""
    window = close[t_left:t_right + 1]
    if len(window) < 10:
        return 0.3

    expected_profile = _canonical_h_and_s(len(window), h_head, h_shoulder)
    actual = (close[t_left:t_right + 1] - close[t_left]) / max(close[t_left], 1e-10)
    expected_norm = (expected_profile - expected_profile[0]) / max(
        abs(expected_profile.max() - expected_profile.min()), 1e-10
    )

    if len(actual) != len(expected_norm):
        min_len = min(len(actual), len(expected_norm))
        actual = actual[:min_len]
        expected_norm = expected_norm[:min_len]

    correlation = float(np.corrcoef(actual, expected_norm)[0, 1])
    if np.isnan(correlation):
        return 0.2

    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def _similarity_inverted_h_and_s(
    close: NDArray[np.float64],
    t_left: int, t_head: int, t_right: int,
    h_head: float, h_shoulder: float,
) -> float:
    """Numerische Ähnlichkeit zu einem inverted H&S."""
    window = close[t_left:t_right + 1]
    if len(window) < 10:
        return 0.3

    expected_profile = _canonical_inverted_h_and_s(len(window), h_head, h_shoulder)
    actual = (close[t_left:t_right + 1] - close[t_left]) / max(close[t_left], 1e-10)
    expected_norm = (expected_profile - expected_profile[0]) / max(
        abs(expected_profile.max() - expected_profile.min()), 1e-10
    )

    if len(actual) != len(expected_norm):
        min_len = min(len(actual), len(expected_norm))
        actual = actual[:min_len]
        expected_norm = expected_norm[:min_len]

    correlation = float(np.corrcoef(actual, expected_norm)[0, 1])
    if np.isnan(correlation):
        return 0.2

    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def _triangle_similarity(
    close: NDArray[np.float64],
    high_pivots: list[SwingPivot],
    low_pivots: list[SwingPivot],
    type_str: str,
) -> float:
    """Numerische Ähnlichkeit zu einem kanonischen Dreieck."""
    if len(high_pivots) < 2 or len(low_pivots) < 2:
        return 0.3

    start_t = min(h.time for h in high_pivots + low_pivots)
    end_t = max(h.time for h in high_pivots + low_pivots)
    n = end_t - start_t + 1
    if n < 10:
        return 0.3

    actual = close[start_t:end_t + 1]

    high_prices = np.array([p.price for p in high_pivots])
    low_prices = np.array([p.price for p in low_pivots])
    high_times = np.array([p.time for p in high_pivots])
    low_times = np.array([p.time for p in low_pivots])

    # Linear interpolation of upper/lower bounds
    t = np.arange(start_t, end_t + 1)
    upper = np.interp(t, high_times, high_prices)
    lower = np.interp(t, low_times, low_prices)
    mid = (upper + lower) / 2
    width = upper - lower

    actual_relative = (actual - mid) / width

    if type_str == "ascending":
        expected = np.linspace(-0.8, 0.8, n)
    elif type_str == "descending":
        expected = np.linspace(0.8, -0.8, n)
    else:  # symmetric
        expected = np.linspace(0.0, 0.0, n)

    actual_norm = (actual_relative - actual_relative.mean()) / max(actual_relative.std(), 1e-10)
    expected_norm = (expected - expected.mean()) / max(expected.std(), 1e-10)

    correlation = float(np.corrcoef(actual_norm, expected_norm)[0, 1])
    if np.isnan(correlation):
        return 0.2

    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


# ---------------------------------------------------------------------------
# Pattern detection algorithms (rule-based)
# ---------------------------------------------------------------------------

class _PatternScanner:
    """Regelbasierte Erkennung klassischer Chart-Patterns."""

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback
        self.swing_detector = SwingDetector(lookback=lookback)

    # ---- Head & Shoulders (regular + inverted) ----

    @staticmethod
    def detect_head_and_shoulders(
        pivots: list[SwingPivot], close: NDArray[np.float64],
    ) -> list[PatternCandidate]:
        """Erkennt Head-and-Shoulders und Inverted-H&S.

        H&S: 3 Peaks — mittlerer (Head) höher als die beiden Schultern.
        Inverted H&S: 3 Troughs — mittlerer (Head) tiefer als die Schultern.
        """
        if len(pivots) < 5:
            return []

        candidates: list[PatternCandidate] = []

        # --- Regular H&S (bearish reversal) ---
        for i in range(2, len(pivots) - 1):
            p_prev = pivots[i - 2]
            p_head = pivots[i - 1]
            p_curr = pivots[i]

            if (
                p_prev.direction == "high"
                and p_head.direction == "high"
                and p_curr.direction == "high"
            ):
                head_height = p_head.price
                shoulder1 = p_prev.price
                shoulder2 = p_curr.price

                # Head must be higher than both shoulders
                if head_height > shoulder1 * 1.01 and head_height > shoulder2 * 1.01:
                    # Shoulders should be roughly equal height
                    shoulder_diff = abs(shoulder1 - shoulder2) / max(shoulder1, shoulder2)
                    quality = max(
                        0.0,
                        1.0 - shoulder_diff,
                    ) * min(
                        1.0,
                        (head_height - shoulder1) / head_height
                        + (head_height - shoulder2) / head_height,
                    )
                    quality = min(quality, 1.0)

                    similarity = _similarity_h_and_s(
                        close, p_prev.time, p_head.time, p_curr.time,
                        head_height, shoulder1,
                    )

                    candidates.append(
                        PatternCandidate(
                            name="head_and_shoulders",
                            direction="bearish",
                            quality=round(quality, 4),
                            similarity=round(similarity, 4),
                            start_idx=p_prev.time,
                            end_idx=p_curr.time,
                            evidence_features=[
                                f"left_shoulder={shoulder1:.2f}",
                                f"head={head_height:.2f}",
                                f"right_shoulder={shoulder2:.2f}",
                                f"shoulder_diff={shoulder_diff:.3f}",
                            ],
                        )
                    )

        # --- Inverted H&S (bullish reversal) ---
        for i in range(2, len(pivots) - 1):
            p_prev = pivots[i - 2]
            p_head = pivots[i - 1]
            p_curr = pivots[i]

            if (
                p_prev.direction == "low"
                and p_head.direction == "low"
                and p_curr.direction == "low"
            ):
                head_low = p_head.price
                shoulder1 = p_prev.price
                shoulder2 = p_curr.price

                if head_low < shoulder1 * 0.99 and head_low < shoulder2 * 0.99:
                    shoulder_diff = abs(shoulder1 - shoulder2) / max(shoulder1, shoulder2)
                    quality = max(0.0, 1.0 - shoulder_diff) * min(
                        1.0,
                        (shoulder1 - head_low) / shoulder1
                        + (shoulder2 - head_low) / shoulder2,
                    )
                    quality = min(quality, 1.0)

                    similarity = _similarity_inverted_h_and_s(
                        close, p_prev.time, p_head.time, p_curr.time,
                        head_low, shoulder1,
                    )

                    candidates.append(
                        PatternCandidate(
                            name="inverted_head_and_shoulders",
                            direction="bullish",
                            quality=round(quality, 4),
                            similarity=round(similarity, 4),
                            start_idx=p_prev.time,
                            end_idx=p_curr.time,
                            evidence_features=[
                                f"left_shoulder={shoulder1:.2f}",
                                f"head={head_low:.2f}",
                                f"right_shoulder={shoulder2:.2f}",
                                f"shoulder_diff={shoulder_diff:.3f}",
                            ],
                        )
                    )

        return candidates

    # ---- Double Top / Bottom ----

    @staticmethod
    def detect_double_top_bottom(pivots: list[SwingPivot]) -> list[PatternCandidate]:
        """Erkennt Double Top (bearish) und Double Bottom (bullish)."""
        if len(pivots) < 3:
            return []

        candidates: list[PatternCandidate] = []

        # Double Top
        for i in range(1, len(pivots) - 1):
            p1 = pivots[i - 1]
            p2 = pivots[i]
            p3 = pivots[i + 1]

            if (
                p1.direction == "high"
                and p2.direction == "low"
                and p3.direction == "high"
            ):
                top1 = p1.price
                top2 = p3.price
                valley = p2.price

                if abs(top1 - top2) / max(top1, top2) < 0.02:
                    quality = min(
                        1.0,
                        0.5 + 0.5 * (max(top1, top2) - valley) / max(top1, top2),
                    )
                    candidates.append(
                        PatternCandidate(
                            name="double_top",
                            direction="bearish",
                            quality=round(quality, 4),
                            similarity=0.75,
                            start_idx=p1.time,
                            end_idx=p3.time,
                            evidence_features=[
                                f"top1={top1:.2f}",
                                f"top2={top2:.2f}",
                                f"valley={valley:.2f}",
                                f"deviation={abs(top1 - top2) / max(top1, top2):.4f}",
                            ],
                        )
                    )

        # Double Bottom
        for i in range(1, len(pivots) - 1):
            p1 = pivots[i - 1]
            p2 = pivots[i]
            p3 = pivots[i + 1]

            if (
                p1.direction == "low"
                and p2.direction == "high"
                and p3.direction == "low"
            ):
                bot1 = p1.price
                bot2 = p3.price
                peak = p2.price

                if abs(bot1 - bot2) / max(bot1, bot2) < 0.02:
                    quality = min(
                        1.0,
                        0.5 + 0.5 * (peak - min(bot1, bot2)) / max(bot1, bot2),
                    )
                    candidates.append(
                        PatternCandidate(
                            name="double_bottom",
                            direction="bullish",
                            quality=round(quality, 4),
                            similarity=0.75,
                            start_idx=p1.time,
                            end_idx=p3.time,
                            evidence_features=[
                                f"bottom1={bot1:.2f}",
                                f"bottom2={bot2:.2f}",
                                f"peak={peak:.2f}",
                                f"deviation={abs(bot1 - bot2) / max(bot1, bot2):.4f}",
                            ],
                        )
                    )

        return candidates

    # ---- Triangles (ascending, descending, symmetric) ----

    @staticmethod
    def detect_triangles(
        pivots: list[SwingPivot], close: NDArray[np.float64],
    ) -> list[PatternCandidate]:
        """Erkennt aufsteigende, absteigende und symmetrische Triangles."""
        if len(pivots) < 5:
            return []

        candidates: list[PatternCandidate] = []

        high_pivots = [p for p in pivots if p.direction == "high"]
        low_pivots = [p for p in pivots if p.direction == "low"]

        if len(high_pivots) < 2 or len(low_pivots) < 2:
            return candidates

        # Take the last 3-5 relevant pivots
        last_n = min(5, len(high_pivots))
        h_pivots = high_pivots[-last_n:]
        l_pivots = low_pivots[-last_n:]

        high_prices = np.array([p.price for p in h_pivots])
        low_prices = np.array([p.price for p in l_pivots])
        np.array([p.time for p in h_pivots])
        np.array([p.time for p in l_pivots])

        # Ascending triangle: flat resistance, rising support
        if len(h_pivots) >= 2 and len(l_pivots) >= 2:
            resistance_trend = (
                high_prices[-1] - high_prices[0]
            ) / max(high_prices[0], 1e-10)
            support_trend = (
                low_prices[-1] - low_prices[0]
            ) / max(low_prices[0], 1e-10)

            if abs(resistance_trend) < 0.03 and support_trend > 0.02:
                quality = min(
                    1.0,
                    max(0.0, support_trend * 10) * (1 - abs(resistance_trend) * 50),
                )
                similarity = _triangle_similarity(
                    close, h_pivots, l_pivots, "ascending",
                )
                candidates.append(
                    PatternCandidate(
                        name="ascending_triangle",
                        direction="bullish",
                        quality=round(quality, 4),
                        similarity=round(similarity, 4),
                        start_idx=l_pivots[0].time,
                        end_idx=h_pivots[-1].time,
                        evidence_features=[
                            f"resistance_trend={resistance_trend:.4f}",
                            f"support_trend={support_trend:.4f}",
                            f"flatness={1 - abs(resistance_trend) * 50:.3f}",
                        ],
                    )
                )

        # Descending triangle: flat support, falling resistance
        if len(h_pivots) >= 2 and len(l_pivots) >= 2:
            high_prices2 = np.array([p.price for p in h_pivots])
            low_prices2 = np.array([p.price for p in l_pivots])
            resistance_trend2 = (
                high_prices2[-1] - high_prices2[0]
            ) / max(high_prices2[0], 1e-10)
            support_trend2 = (
                low_prices2[-1] - low_prices2[0]
            ) / max(low_prices2[0], 1e-10)

            if abs(support_trend2) < 0.03 and resistance_trend2 < -0.02:
                quality = min(
                    1.0,
                    max(0.0, abs(resistance_trend2) * 10)
                    * (1 - abs(support_trend2) * 50),
                )
                similarity = _triangle_similarity(
                    close, h_pivots, l_pivots, "descending",
                )
                candidates.append(
                    PatternCandidate(
                        name="descending_triangle",
                        direction="bearish",
                        quality=round(quality, 4),
                        similarity=round(similarity, 4),
                        start_idx=l_pivots[0].time,
                        end_idx=h_pivots[-1].time,
                        evidence_features=[
                            f"support_trend={support_trend2:.4f}",
                            f"resistance_trend={resistance_trend2:.4f}",
                            f"flatness={1 - abs(support_trend2) * 50:.3f}",
                        ],
                    )
                )

        # Symmetric triangle: converging highs and lows
        if len(h_pivots) >= 2 and len(l_pivots) >= 2:
            high_prices3 = np.array([p.price for p in h_pivots])
            low_prices3 = np.array([p.price for p in l_pivots])
            high_converge = (
                high_prices3[-1] - high_prices3[0]
            ) / max(high_prices3[0], 1e-10)
            low_converge = (
                low_prices3[-1] - low_prices3[0]
            ) / max(low_prices3[0], 1e-10)

            if high_converge < -0.02 and low_converge > 0.02:
                squeeze = abs(high_converge) + abs(low_converge)
                quality = min(1.0, squeeze * 5)
                similarity = _triangle_similarity(
                    close, h_pivots, l_pivots, "symmetric",
                )
                candidates.append(
                    PatternCandidate(
                        name="symmetric_triangle",
                        direction="neutral",
                        quality=round(quality, 4),
                        similarity=round(similarity, 4),
                        start_idx=min(h_pivots[0].time, l_pivots[0].time),
                        end_idx=max(h_pivots[-1].time, l_pivots[-1].time),
                        evidence_features=[
                            f"high_converge={high_converge:.4f}",
                            f"low_converge={low_converge:.4f}",
                            f"squeeze={squeeze:.4f}",
                        ],
                    )
                )

        return candidates

    # ---- Range / Consolidation ----

    @staticmethod
    def detect_range(
        pivots: list[SwingPivot], close: NDArray[np.float64],
    ) -> list[PatternCandidate]:
        """Erkennt Range / Konsolidierung (flache Preisbewegung)."""
        if len(pivots) < 3:
            return []

        last = pivots[-3:]
        all_directions = [p.direction for p in last]

        # If pivots alternate frequently → range
        changes = sum(
            1
            for i in range(1, len(all_directions))
            if all_directions[i] != all_directions[i - 1]
        )
        if changes >= len(all_directions) - 1 and len(pivots) >= 4:
            prices = [p.price for p in pivots[-5:]]
            price_range = (max(prices) - min(prices)) / max(min(prices), 1e-10)

            if price_range < 0.05:  # < 5% range over last 5 pivots
                n_pivots = len(pivots)
                quality = max(0.0, 1.0 - price_range * 20)
                return [
                    PatternCandidate(
                        name="consolidation_range",
                        direction="neutral",
                        quality=round(quality, 4),
                        similarity=0.6,
                        start_idx=pivots[0].time,
                        end_idx=pivots[-1].time,
                        evidence_features=[
                            f"price_range={price_range:.4f}",
                            f"pivot_changes={changes}",
                            f"range_duration={n_pivots} pivots",
                        ],
                    )
                ]

        return []

    # ---- Full scan ----

    def scan(
        self, data: dict[str, NDArray[np.float64]],
    ) -> tuple[list[PatternCandidate], dict]:
        """Führt alle Pattern-Erkennungen durch und gibt Kandidaten + Metadaten zurück."""
        pivots = self.swing_detector.detect_swings(data)
        close = data["close"]

        all_candidates: list[PatternCandidate] = []

        # 1. Head & Shoulders
        all_candidates.extend(self.detect_head_and_shoulders(pivots, close))

        # 2. Double Top / Bottom
        all_candidates.extend(self.detect_double_top_bottom(pivots))

        # 3. Triangles
        all_candidates.extend(self.detect_triangles(pivots, close))

        # 4. Range
        all_candidates.extend(self.detect_range(pivots, close))

        # 5. Use existing PatternDetector for BOS/CHoCH/FailedBreakout
        pd = PatternDetector()
        structure = pd.detect_all_patterns(data)
        for pat in structure.patterns:
            if pat == ChartPattern.BOS:
                if pivots:
                    last = pivots[-1]
                    direction = (
                        "bullish" if last.direction == "low" else "bearish"
                    )
                else:
                    direction = "neutral"
                all_candidates.append(
                    PatternCandidate(
                        name="break_of_structure",
                        direction=direction,
                        quality=0.5,
                        similarity=0.5,
                        start_idx=0,
                        end_idx=len(close) - 1,
                        evidence_features=["BOS detected by PatternDetector"],
                    )
                )
            elif pat == ChartPattern.CHoCH:
                all_candidates.append(
                    PatternCandidate(
                        name="change_of_character",
                        direction="neutral",
                        quality=0.6,
                        similarity=0.6,
                        start_idx=0,
                        end_idx=len(close) - 1,
                        evidence_features=["CHoCH detected by PatternDetector"],
                    )
                )
            elif pat == ChartPattern.FAILED_BREAKOUT:
                all_candidates.append(
                    PatternCandidate(
                        name="failed_breakout",
                        direction="neutral",
                        quality=0.4,
                        similarity=0.5,
                        start_idx=0,
                        end_idx=len(close) - 1,
                        evidence_features=["Failed breakout by PatternDetector"],
                    )
                )

        # Sort by quality descending
        all_candidates.sort(key=lambda c: c.quality, reverse=True)

        # Compute distribution
        dir_counts: dict[str, int] = {}
        for c in all_candidates:
            dir_counts[c.direction] = dir_counts.get(c.direction, 0) + 1

        distribution = dict(sorted(dir_counts.items()))

        # Sample size based on number of bars used
        if all_candidates:
            sample_size = max(c.end_idx - c.start_idx for c in all_candidates) + 1
        else:
            sample_size = len(close)

        metadata = {
            "sample_size": sample_size,
            "distribution": distribution,
            "total_pivots": len(pivots),
            "high_pivots": sum(1 for p in pivots if p.direction == "high"),
            "low_pivots": sum(1 for p in pivots if p.direction == "low"),
            "pattern_count": len(all_candidates),
        }

        return all_candidates, metadata


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class PatternAgent(BaseAgent):
    """Pattern Analysis Agent — rule-based chart pattern detection with similarity scoring.

    Produces a standardized AgentReport with:
    - candidate patterns (name, direction, quality)
    - numerical similarity to canonical patterns
    - sample_size and distribution info
    - counter_evidence (Gegenhypothese)
    - invalidation conditions
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="pattern",
                agent_type=AgentType.PATTERN,
            )
        super().__init__(config)

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert Marktdaten auf Chart-Patterns.

        Required keys:
            close (NDArray) - required
            high (NDArray) - required
            low (NDArray) - required
            volume (NDArray) - optional

        Returns:
            AgentReport mit Pattern-basierten Wahrscheinlichkeiten,
            Qualitätsscores, Ähnlichkeitswerten und Invalidierungen.
        """
        required = ("close", "high", "low")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data.get("volume")
        n_bars = len(close)

        # --- Pattern Scanning ---
        scanner = _PatternScanner(lookback=3)
        candidates, metadata = scanner.scan(data)

        # --- Directional probability from pattern signals ---
        bull_count = sum(1 for c in candidates if c.direction == "bullish")
        bear_count = sum(1 for c in candidates if c.direction == "bearish")
        neutral_count = sum(1 for c in candidates if c.direction == "neutral")

        # Weight by quality
        bull_weight = sum(c.quality for c in candidates if c.direction == "bullish")
        bear_weight = sum(c.quality for c in candidates if c.direction == "bearish")
        neutral_weight = sum(c.quality for c in candidates if c.direction == "neutral")

        total_weight = bull_weight + bear_weight + neutral_weight + 0.01
        up_prob = round(bull_weight / total_weight, 4)
        down_prob = round(bear_weight / total_weight, 4)
        range_prob = round(neutral_weight / total_weight, 4)

        # Ensure exact sum = 1.0
        total = up_prob + down_prob + range_prob
        if total > 0:
            up_prob = round(up_prob / total, 4)
            down_prob = round(down_prob / total, 4)
            range_prob = round(range_prob / total, 4)
            # Final adjustment for floating point
            diff = 1.0 - (up_prob + down_prob + range_prob)
            if abs(diff) > 0:
                range_prob = round(range_prob + diff, 4)

        # --- Evidence ---
        evidence: list[EvidenceReference] = []
        counter_evidence: list[EvidenceReference] = []

        for i, c in enumerate(candidates[:5]):
            # Primary evidence from top candidates
            evidence.append(
                EvidenceReference(
                    reference=f"pattern:{c.name}:{i}",
                    feature=c.name,
                    value=(
                        f"quality={c.quality:.3f} sim={c.similarity:.3f} "
                        f"start={c.start_idx} end={c.end_idx}"
                    ),
                    direction=(
                        "positive"
                        if c.direction == "bullish"
                        else ("negative" if c.direction == "bearish" else "neutral")
                    ),
                    relevance=round(c.quality * c.similarity, 4),
                )
            )
            # Counter-evidence: note the opposing possibility
            opposite_dir = "bearish" if c.direction == "bullish" else "bullish"
            counter_evidence.append(
                EvidenceReference(
                    reference=f"pattern:counter:{c.name}:{i}",
                    feature=f"counter_{c.name}",
                    value=(
                        f"opposite_direction={opposite_dir} "
                        f"quality={c.quality:.3f}"
                    ),
                    direction="negative",
                    relevance=round(c.quality * 0.3, 4),
                )
            )

        # Always include distribution info as evidence
        evidence.append(
            EvidenceReference(
                reference="pattern:distribution",
                feature="pattern_distribution",
                value=(
                    f"bullish={metadata.get('distribution', {}).get('bullish', 0)} "
                    f"bearish={metadata.get('distribution', {}).get('bearish', 0)} "
                    f"neutral={metadata.get('distribution', {}).get('neutral', 0)}"
                ),
                direction="neutral",
                relevance=0.3,
            )
        )

        # Fallback if no candidates found
        if not evidence:
            price_range = (
                float(close.max()) - float(close.min())
            ) / max(float(close.min()), 1e-10)
            evidence.append(
                EvidenceReference(
                    reference="pattern:price_action",
                    feature="price_range",
                    value=f"range={price_range:.4f} bars={n_bars}",
                    direction="neutral",
                    relevance=0.2,
                )
            )
            counter_evidence.append(
                EvidenceReference(
                    reference="pattern:counter_no_signal",
                    feature="no_pattern_detected",
                    value="insufficient_data_or_no_clear_pattern",
                    direction="negative",
                    relevance=0.4,
                )
            )

        # --- Invalidation Conditions ---
        invalidations: list = []

        if candidates:
            best = candidates[0]
            if best.direction == "bullish":
                invalidations.append(
                    self._make_invalidations(
                        condition=f"Preis fällt unter {best.name}-Support",
                        indicator=best.name,
                        threshold=float(low.min()),
                        direction="below",
                    )
                )
            elif best.direction == "bearish":
                invalidations.append(
                    self._make_invalidations(
                        condition=f"Preis steigt über {best.name}-Resistance",
                        indicator=best.name,
                        threshold=float(high.max()),
                        direction="above",
                    )
                )

        # Always include general invalidations
        invalidations.append(
            self._make_invalidations(
                condition="Volume bricht zusammen (kein Trend-Bestätigungs-Volume)",
                indicator="volume",
                threshold=(
                    float(volume.mean()) * 0.5
                    if volume is not None
                    else 0.0
                ),
                direction="below",
            )
        )
        invalidations.append(
            self._make_invalidations(
                condition="Neues Swing-High/Low verändert Struktur",
                indicator="swing_pivot",
                threshold=0.0,
                direction=(
                    "above"
                    if candidates
                    and any(c.direction == "bearish" for c in candidates)
                    else "below"
                ),
            )
        )

        # --- Hypothesis Narrative ---
        top_names = [c.name for c in candidates[:3]]
        if top_names:
            hypothesis = (
                f"Erkannte Patterns: {', '.join(top_names)}. "
                f"Bullish={bull_count} Bearish={bear_count} "
                f"Neutral={neutral_count}. "
                f"Pivots: {metadata['total_pivots']} "
                f"(H:{metadata['high_pivots']} L:{metadata['low_pivots']}). "
                f"Sample: {metadata['sample_size']} Bars."
            )
        else:
            hypothesis = (
                f"Kein klares Chart-Pattern erkannt über {n_bars} Bars. "
                f"Pivots: {metadata['total_pivots']}. "
                f"Range: {close.min():.2f}-{close.max():.2f}."
            )

        # --- Raw confidence from pattern quality ---
        if candidates:
            best_quality = candidates[0].quality
            best_similarity = candidates[0].similarity
            raw_confidence = round(
                min(0.9, 0.3 + best_quality * 0.4 + best_similarity * 0.2), 4
            )
        else:
            raw_confidence = 0.15

        # --- Sample size ---
        sample_size = metadata["sample_size"]

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(),
            hypothesis=hypothesis,
            probabilities={
                "up": up_prob,
                "down": down_prob,
                "range": range_prob,
            },
            expected_return=None,
            evidence=evidence,
            counter_evidence=counter_evidence,
            invalidations=invalidations,
            sample_size=sample_size,
            raw_confidence=raw_confidence,
            calibrated_confidence=None,
            status=self.config.status,
        )
