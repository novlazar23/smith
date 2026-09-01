"""Historical Analogy Agent — k-NN similarity search on OHLCV data.

Uses Dynamic Time Warping (DTW) for pattern matching against
historical periods, regime-filtered, with frequency-based probability.
"""

from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent

SUPPORTED_KEYS = frozenset({"close", "high", "low", "volume"})
MIN_WINDOW = 20
MIN_MATCHES = 3


# ── DTW (Dynamic Time Warping) ────────────────────────────────────────────

def _dtw_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute DTW distance between two 1-D sequences.

    Uses the standard Sakoe-Chiba band with bandwidth = max(2, len(min_seq)//4).
    Returns infinity if paths are blocked.
    """
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return float("inf")

    band = max(2, min(n, m) // 4)
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            dist = abs(a[i - 1] - b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n, m])


# ── Regime Detection (simple, no external deps) ───────────────────────────

def _detect_regime(returns: NDArray[np.float64]) -> str:
    """Classify a return series into 'bull', 'bear', or 'choppy'."""
    if len(returns) < 5:
        return "choppy"

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    if std_ret == 0:
        return "choppy"

    # Sharpe-like ratio for classification
    ratio = mean_ret / std_ret
    if ratio > 0.5:
        return "bull"
    elif ratio < -0.5:
        return "bear"
    return "choppy"


# ── Feature extraction ────────────────────────────────────────────────────

def _extract_features(
    close: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract multi-dimensional feature vector for a price series window.

    Features: [close, high, low, volume, volume_weighted_price, daily_range]
    """
    vwap = (high + low + close) / 3.0
    daily_range = (high - low) / close
    features = np.stack([close, high, low, volume, vwap, daily_range], axis=-1)
    return features


def _normalize_features(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z-score normalize each feature dimension."""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0  # avoid div by zero
    return (features - mean) / std


# ── Agent ─────────────────────────────────────────────────────────────────

class HistoricalAnalogyAgent(BaseAgent):
    """Historical-Analogy-Agent — sucht aehnliche Muster in historischen Daten.

    Verwendet DTW (Dynamic Time Warping) als aehnlichkeitsmass, filtert
    nach regime und gibt top-k historische Periode zurueck.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        k: int = 5,
        window: int = 20,
        min_window: int = 20,
        minimum_matches: int = 3,
        historical_data: dict[str, NDArray[np.float64] | None] | None = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="historical_analogy",
                agent_type=AgentType.HISTORICAL_ANALOGY,
            )
        super().__init__(config)
        self._k = k
        self._window = window
        self._min_window = min_window
        self._minimum_matches = minimum_matches
        self._historical_data = historical_data

    def analyze(
        self, data: dict[str, NDArray[np.float64]]
    ) -> AgentReport:
        """Analysiert Marktdaten mit historischem Vergleich.

        Required keys:
            close (NDArray) — required
            high   (NDArray) — optional
            low    (NDArray) — optional
            volume (NDArray) — optional

        Returns:
            AgentReport mit top-k historischen Matches.

        Raises:
            ValueError: Wenn erforderliche Fehlt.
        """
        if "close" not in data:
            raise ValueError("Missing required data keys: ['close']")

        close = data["close"]
        high = data.get("high")
        low = data.get("low")
        volume = data.get("volume")

        if len(close) < self._min_window + 1:
            raise ValueError(
                f"Need at least {self._min_window + 1} data points, "
                f"got {len(close)}"
            )

        # Combine current data with optional historical dataset
        hist = self._resolve_historical(close, high, low, volume)

        # Current period features
        current = self._build_period(close, high, low, volume, start=0, end=None)

        # Sliding window over historical data
        candidates = self._scan_history(hist, current)

        # Regime filtering
        current_regime = _detect_regime(current["returns"])
        candidates = self._filter_by_regime(candidates, current_regime)

        # k-NN selection
        matches = self._select_top_k(candidates, current["features"], self._k)

        # Ensure minimum matches
        matches = self._ensure_min_matches(matches, hist, current, close, high, low, volume)

        # Build report
        probability = self._compute_prob(matches)
        evidence = self._build_evidence(matches)
        counter_evidence = self._build_counter_evidence(matches, probability)
        invalidations = self._build_invalidations(matches, current_regime)
        hypothesis = self._build_hypothesis(matches, probability)
        confidence = self._compute_confidence(matches)
        sample_size = len(matches)

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(),
            hypothesis=hypothesis,
            probabilities=probability,
            evidence=evidence,
            counter_evidence=counter_evidence,
            invalidations=invalidations,
            sample_size=sample_size,
            raw_confidence=confidence,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _resolve_historical(
        self,
        close: NDArray[np.float64],
        high: NDArray[np.float64] | None,
        low: NDArray[np.float64] | None,
        volume: NDArray[np.float64] | None,
    ) -> dict[str, NDArray[np.float64] | None]:
        """Resolve data source — external historical or pseudo-historical from own data."""
        if self._historical_data is not None:
            return self._historical_data

        # Pseudo-historical: use the tail of current data as history
        min_len = self._min_window * 5
        if len(close) > min_len:
            tail_close = close[:-min_len]
            tail_high = high[:-min_len] if high is not None else None
            tail_low = low[:-min_len] if low is not None else None
            tail_volume = volume[:-min_len] if volume is not None else None
            return {"close": tail_close, "high": tail_high, "low": tail_low, "volume": tail_volume}

        return {"close": close, "high": high, "low": low, "volume": volume}

    def _build_period(
        self,
        close: NDArray[np.float64],
        high: NDArray[np.float64] | None,
        low: NDArray[np.float64] | None,
        volume: NDArray[np.float64] | None,
        start: int,
        end: int | None,
    ) -> dict:
        """Build a period dict with features, returns, and regime."""
        c = close[start:end]
        h = high[start:end] if high is not None else c
        lo = low[start:end] if low is not None else c
        v = volume[start:end] if volume is not None else np.ones(len(c))

        # Handle length mismatches
        min_len = min(len(c), len(h), len(lo), len(v))
        c, h, lo, v = c[:min_len], h[:min_len], lo[:min_len], v[:min_len]

        returns = np.diff(c) / c[:-1] if len(c) > 1 else np.array([0.0])
        features = _extract_features(c, h, lo, v)

        regime = _detect_regime(returns)
        outcome = "up" if c[-1] > c[0] else ("down" if c[-1] < c[0] else "range")
        outcome_pct = float((c[-1] - c[0]) / c[0]) if c[0] != 0 else 0.0

        return {
            "close": c,
            "high": h,
            "low": lo,
            "volume": v,
            "features": features,
            "returns": returns,
            "regime": regime,
            "outcome": outcome,
            "outcome_pct": outcome_pct,
        }

    def _scan_history(
        self,
        hist: dict[str, NDArray[np.float64] | None],
        current: dict,
    ) -> list[dict]:
        """Sliding-window scan over historical data for candidate periods."""
        h_close = hist["close"]
        if h_close is None:
            return []
        h_high = hist.get("high")
        h_low = hist.get("low")
        h_volume = hist.get("volume")
        min_len = self._min_window
        step = max(1, min_len // 4)

        candidates: list[dict] = []
        end = min_len
        while end <= len(h_close):
            period = self._build_period(
                h_close, h_high, h_low, h_volume,
                start=end - min_len, end=end,
            )
            candidates.append(period)
            end += step

        return candidates

    def _filter_by_regime(
        self,
        candidates: list[dict],
        current_regime: str,
    ) -> list[dict]:
        """Keep only candidates in the same regime as current."""
        # First pass: exact regime match
        same_regime = [c for c in candidates if c["regime"] == current_regime]

        # If too few, relax to any regime but penalize mismatched
        if len(same_regime) >= 5:
            return same_regime

        # Fallback: include all but mark regime mismatch
        for c in candidates:
            c["regime_matched"] = c["regime"] == current_regime
        return candidates

    def _select_top_k(
        self,
        candidates: list[dict],
        current_features: NDArray[np.float64],
        k: int,
    ) -> list[dict]:
        """Compute DTW distances and return top-k most similar candidates."""
        current_norm = _normalize_features(current_features)
        # Flatten multi-dimensional features for DTW comparison
        current_flat = current_norm.reshape(-1)

        scored: list[tuple[float, dict]] = []
        for cand in candidates:
            cand_feats = cand["features"]
            cand_norm = _normalize_features(cand_feats)
            cand_flat = cand_norm.reshape(-1)

            # DTW distance
            d = _dtw_distance(current_flat, cand_flat)
            scored.append((d, cand))

        # Sort by distance (lower = more similar)
        scored.sort(key=lambda x: x[0])
        matches = scored[:k]

        # Convert to match dicts with similarity score
        result = []
        for dist, cand in matches:
            # Convert distance to similarity (0-1, higher = more similar)
            similarity = float(np.clip(1.0 / (1.0 + dist), 0.0, 1.0))
            result.append({
                **cand,
                "distance": dist,
                "similarity": similarity,
            })

        return result

    def _ensure_min_matches(
        self,
        matches: list[dict],
        hist: dict,
        current: dict,
        close: NDArray[np.float64],
        high: NDArray[np.float64] | None,
        low: NDArray[np.float64] | None,
        volume: NDArray[np.float64] | None,
    ) -> list[dict]:
        """Ensure at least minimum_matches candidates."""
        if len(matches) >= self._minimum_matches:
            return matches

        # Fallback: create synthetic matches with uniform low similarity
        for _i in range(self._minimum_matches - len(matches)):
            matches.append({
                "close": close[:self._min_window],
                "high": high[:self._min_window] if high is not None else close[:self._min_window],
                "low": low[:self._min_window] if low is not None else close[:self._min_window],
                "volume": volume[:self._min_window] if volume is not None else np.ones(self._min_window),
                "features": np.zeros((self._min_window, 6)),
                "returns": np.zeros(self._min_window - 1),
                "regime": "choppy",
                "outcome": "up",
                "outcome_pct": 0.0,
                "distance": 999.0,
                "similarity": 0.01,
                "synthetic": True,
            })

        return matches

    def _compute_prob(
        self,
        matches: list[dict],
    ) -> dict[str, float]:
        """Compute up/down/range probabilities from historical outcome frequencies."""
        if not matches:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        up_count = 0
        down_count = 0
        range_count = 0
        total_weight = 0.0

        for m in matches:
            weight = m["similarity"]
            total_weight += weight
            outcome = m["outcome"]

            if outcome == "up":
                up_count += weight
            elif outcome == "down":
                down_count += weight
            else:
                range_count += weight

        total = up_count + down_count + range_count
        if total == 0:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        up_prob = round(up_count / total, 4)
        down_prob = round(down_count / total, 4)
        range_prob = round(1.0 - up_prob - down_prob, 4)

        return {"up": up_prob, "down": down_prob, "range": range_prob}

    def _build_evidence(self, matches: list[dict]) -> list:
        """Evidence from top matches."""
        evidence: list = []

        for i, m in enumerate(matches[:3]):  # Top 3
            evidence.append(
                self._make_evidence(
                    f"match_{i}",
                    f"{m['outcome']} period, similarity={m['similarity']:.3f}, "
                    f"outcome={m['outcome_pct']:+.1%}",
                    "positive",
                    m["similarity"],
                )
            )

        if not evidence:
            evidence.append(
                self._make_evidence(
                    "no_matches",
                    "no historical matches found",
                    "neutral",
                    0.0,
                )
            )

        return evidence

    def _build_counter_evidence(
        self, matches: list[dict], probability: dict[str, float]
    ) -> list:
        """Counter-evidence: least similar match in opposite direction."""
        counter: list = []

        dominant = max(probability, key=lambda k: probability.get(k, 0.0))
        opposing = "down" if dominant == "up" else ("up" if dominant == "down" else "range")

        # Find least similar match with opposing outcome
        opposing_matches = [
            m for m in matches if m["outcome"] == opposing
        ]
        if opposing_matches:
            worst = min(opposing_matches, key=lambda m: m["similarity"])
            counter.append(
                self._make_evidence(
                    f"counter_{worst['outcome']}",
                    f"opposing outcome ({worst['outcome']}), "
                    f"similarity={worst['similarity']:.3f}, "
                    f"outcome={worst['outcome_pct']:+.1%}",
                    "negative",
                    1.0 - worst["similarity"],
                )
            )

        # If no opposing outcome, include lowest-similarity match as weak evidence
        if not counter and matches:
            worst = min(matches, key=lambda m: m["similarity"])
            counter.append(
                self._make_evidence(
                    "counter_low_sim",
                    f"lowest similarity match ({worst['similarity']:.3f})",
                    "negative",
                    0.2,
                )
            )

        # Fallback: always include at least one counter
        if not counter:
            counter.append(
                self._make_evidence(
                    "counter_no_data",
                    "insufficient historical data for counter-evidence",
                    "negative",
                    0.1,
                )
            )

        return counter

    def _build_invalidations(
        self,
        matches: list[dict],
        current_regime: str,
    ) -> list:
        """Invalidation conditions."""
        invalidations: list = []

        # Regime change invalidation
        invalidations.append(
            self._make_invalidations(
                condition=f"Regime change from '{current_regime}' invalidates analogy",
                indicator="regime",
                threshold=0.0,
                direction="above",
            )
        )

        # Data quality check
        if matches:
            invalidations.append(
                self._make_invalidations(
                    condition="Average similarity below 0.01 makes analogy invalid",
                    indicator="avg_similarity",
                    threshold=0.01,
                    direction="below",
                )
            )

        # Sample size
        invalidations.append(
            self._make_invalidations(
                condition="Fewer than 3 matches reduces statistical validity",
                indicator="match_count",
                threshold=self._minimum_matches,
                direction="below",
            )
        )

        return invalidations

    def _build_hypothesis(
        self,
        matches: list[dict],
        probability: dict[str, float],
    ) -> str:
        """Build hypothesis string."""
        dominant = max(probability, key=lambda k: probability.get(k, 0.0))
        avg_sim = (
            sum(m["similarity"] for m in matches) / len(matches)
            if matches else 0.0
        )
        outcomes = [m["outcome"] for m in matches]

        return (
            f"Historical analogy: {len(matches)} matches, "
            f"avg similarity={avg_sim:.3f}. "
            f"Dominant outcome: {dominant} ({probability[dominant]:.0%}). "
            f"Outcomes: {outcomes[:3]}"
        )

    def _compute_confidence(self, matches: list[dict]) -> float:
        """Compute raw confidence from similarity and match count."""
        if not matches:
            return 0.1

        avg_sim = sum(m["similarity"] for m in matches) / len(matches)
        count_factor = min(1.0, len(matches) / self._minimum_matches)
        confidence = 0.2 + 0.4 * avg_sim + 0.2 * count_factor
        return round(min(0.9, confidence), 4)
