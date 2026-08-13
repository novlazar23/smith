"""Anomaly Detection Agent — statistical anomaly scoring, no factual claims.

Detects anomalous patterns in OHLCV data using purely statistical methods:
  1. ANOMALY - volume spike (>3 std), price gap (>2 std), sudden volatility change
  2. SPOOFING_LIKE - depth imbalance / large orders at edge (simulated via volume)
  3. LIQUIDITY_WITHDRAWAL - volume thinning before price move, spread widening
  4. CROSS_VENUE_DIVERGENCE - contradictory volume patterns (simulated)

Only scores, no factual claims (Nur Scores, keine Tatsachenbehauptungen).
Conservative scoring: only flag when score > 0.6.
"""

from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent

REQUIRED_KEYS = frozenset({"close", "high", "low", "volume", "open"})


class AnomalyAgent(BaseAgent):
    """Statistical Anomalie-Erkennung — Nur Scores, keine Tatsachenbehauptungen."""

    THRESHOLD: float = 0.6
    LOOKBACK: int = 30  # minimum bars for rolling statistics

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        threshold: float | None = None,
        lookback: int | None = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="anomaly",
                agent_type=AgentType.ANOMALY,
            )
        super().__init__(config)
        self._threshold = threshold if threshold is not None else self.THRESHOLD
        self._lookback = lookback if lookback is not None else self.LOOKBACK

    def analyze(
        self, data: dict[str, NDArray[np.float64]]
    ) -> AgentReport:
        """Analysiert OHLCV-Daten auf statistische Anomalien.

        Accepts a dict with OHLCV arrays:
            close: NDArray[float64] — closing prices
            high: NDArray[float64] — high prices
            low: NDArray[float64] — low prices
            volume: NDArray[float64] — trading volume
            open: NDArray[float64] — opening prices

        Returns:
            AgentReport with anomaly scores and evidence.
        """
        # Validate required keys
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required OHLCV keys: {sorted(missing)}"
            )

        close = np.asarray(data["close"], dtype=np.float64)
        high = np.asarray(data["high"], dtype=np.float64)
        low = np.asarray(data["low"], dtype=np.float64)
        volume = np.asarray(data["volume"], dtype=np.float64)
        open_ = np.asarray(data["open"], dtype=np.float64)

        n = len(close)
        lb = self._lookback

        # Ensure we have enough data
        if n < lb:
            return self._short_data_report(
                close, high, low, volume, open_, n
            )

        # Compute derived series
        returns = np.diff(close) / close[:-1]
        realized_vol = np.diff(high) - np.diff(low)  # intrabar range
        gaps = close[1:] - open_[:-1]  # overnight / inter-bar gaps

        # ── 1. ANOMALY scores ───────────────────────────────────────────
        anomaly_score = self._score_anomaly(
            returns, realized_vol, gaps, volume, lb
        )

        # ── 2. SPOOFING_LIKE scores ─────────────────────────────────────
        spoofing_score = self._score_spoofing(
            close, open_, high, low, volume, lb
        )

        # ── 3. LIQUIDITY_WITHDRAWAL scores ──────────────────────────────
        liquidity_score = self._score_liquidity(
            close, open_, volume, realized_vol, lb
        )

        # ── 4. CROSS_VENUE_DIVERGENCE scores ────────────────────────────
        cross_venue_score = self._score_cross_venue(
            close, open_, high, low, volume, lb
        )

        # ── Build report components ─────────────────────────────────────
        scores = {
            "anomaly": anomaly_score,
            "spoofing_like": spoofing_score,
            "liquidity_withdrawal": liquidity_score,
            "cross_venue_divergence": cross_venue_score,
        }

        hypothesis = self._build_hypothesis(scores)
        probabilities = self._compute_probabilities(scores)
        evidence = self._build_evidence(scores, close, volume, lb)
        counter_evidence = self._build_counter_evidence(scores)
        invalidations = self._build_invalidations()
        confidence = self._compute_confidence(scores)

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(),
            hypothesis=hypothesis,
            probabilities=probabilities,
            evidence=evidence,
            counter_evidence=counter_evidence,
            invalidations=invalidations,
            raw_confidence=confidence,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )

    # ── Score computation ──────────────────────────────────────────────────

    def _score_anomaly(
        self,
        returns: NDArray[np.float64],
        realized_vol: NDArray[np.float64],
        gaps: NDArray[np.float64],
        volume: NDArray[np.float64],
        lb: int,
    ) -> float:
        """Score for ANOMALY type: volume spike, price gap, vol change."""
        if len(returns) < lb:
            return 0.0

        # Rolling statistics on volume (look back window up to last bar)
        vol_window = volume[-lb:]
        vol_mean = np.mean(vol_window)
        vol_std = np.std(vol_window)
        if vol_std < 1e-12:
            vol_std = 1e-12

        # Volume spike: current volume > 3 std above mean
        vol_z = (volume[-1] - vol_mean) / vol_std
        vol_score = max(0.0, min(1.0, (vol_z - 3.0) / 3.0))

        # Price gap: last gap > 2 std of recent gaps
        gap_window = gaps[-lb:]
        gap_std = np.std(gap_window)
        if gap_std < 1e-12:
            gap_std = 1e-12
        gap_z = abs(gaps[-1]) / gap_std
        gap_score = max(0.0, min(1.0, (gap_z - 2.0) / 3.0))

        # Sudden volatility change: compare last-bar range to rolling avg
        rv_window = realized_vol[-lb:]
        rv_mean = np.mean(rv_window)
        rv_std = np.std(rv_window)
        if rv_std < 1e-12:
            rv_std = 1e-12
        vol_change_z = (abs(realized_vol[-1]) - rv_mean) / rv_std
        vol_change_score = max(0.0, min(1.0, (vol_change_z - 2.0) / 3.0))

        # Return-based: current return vs rolling
        ret_window = returns[-lb:]
        ret_std = np.std(ret_window)
        if ret_std < 1e-12:
            ret_std = 1e-12
        ret_z = abs(returns[-1]) / ret_std
        ret_score = max(0.0, min(1.0, (ret_z - 2.0) / 3.0))

        # Weighted combination — volume spike is primary signal
        return float(round(
            0.35 * vol_score
            + 0.25 * gap_score
            + 0.25 * vol_change_score
            + 0.15 * ret_score,
            4,
        ))

    def _score_spoofing(
        self,
        close: NDArray[np.float64],
        open_: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        volume: NDArray[np.float64],
        lb: int,
    ) -> float:
        """Score for SPOOFING_LIKE: depth imbalance / large orders at edge.

        Since we only have OHLCV, we simulate spoofing detection via:
        - Wicks to volume ratio (large wicks with thin volume = potential spoof)
        - Repeated rejections at round numbers
        - Volume concentration at session extremes
        """
        if len(close) < lb:
            return 0.0

        wick_up = high - np.maximum(close, open_)
        wick_down = np.maximum(close, open_) - low
        total_range = high - low

        # Avoid division by zero

        vol_safe = np.where(
            volume[-lb:] > 1e-12, volume[-lb:], 1e-12
        )
        wick_to_volume_ratio = (wick_up + wick_down)[-lb:] / vol_safe * 1000
        avg_wick_v = np.mean(wick_to_volume_ratio)

        # High wick-to-volume ratio suggests price rejection without follow-through
        wick_score = min(1.0, avg_wick_v / 10.0)

        # Repeated long wicks (rejections at levels)
        long_wick_ratio = np.mean(
            (wick_up[-lb:] + wick_down[-lb:]) > 0.3 * total_range[-lb:]
        )
        wick_freq_score = long_wick_ratio

        # Volume distribution: if most volume is at edges (not center),
        # that suggests order-book manipulation pattern
        top_quarter_mask = np.arange(lb) >= (3 * lb // 4)
        top_vol_concentration = (
            volume[-lb:][top_quarter_mask].sum() / volume[-lb:].sum()
            if volume[-lb:].sum() > 0
            else 0.5
        )
        edge_bias = min(1.0, abs(top_vol_concentration - 0.25) / 0.25)

        return float(round(
            0.4 * wick_score
            + 0.3 * wick_freq_score
            + 0.3 * edge_bias,
            4,
        ))

    def _score_liquidity(
        self,
        close: NDArray[np.float64],
        open_: NDArray[np.float64],
        volume: NDArray[np.float64],
        realized_vol: NDArray[np.float64],
        lb: int,
    ) -> float:
        """Score for LIQUIDITY_WITHDRAWAL: thinning volume before price move.

        Detects:
        - Gradually declining volume in recent bars before a price swing
        - Spread widening (via realized range) without volume support
        - Price move on declining volume (divergence)
        """
        if len(close) < lb:
            return 0.0

        # Volume thinning: compare last-3-bar avg to preceding window
        vol_thinning = 0.0
        if lb >= 7:
            recent_vol = np.mean(volume[-3:])
            prior_vol = np.mean(volume[-min(7, lb) : -3])
            if prior_vol > 1e-12:
                vol_thin_ratio = recent_vol / prior_vol
                vol_thinning = max(0.0, 1.0 - vol_thin_ratio)

        # Price move on declining volume
        price_move = abs(close[-1] - close[-lb]) / close[-lb]
        ret_window = np.abs(np.diff(close)[-lb:]) / close[-lb - 1 : -1]
        avg_ret = np.mean(ret_window) if len(ret_window) > 0 else 1e-12

        # Price move is large but volume is thin → suspicious
        move_on_thin = min(
            1.0, (price_move / max(avg_ret, 1e-12)) * (1.0 - vol_thinning)
        )

        # Spread widening (realized range increase without volume)
        range_window = realized_vol[-lb:]
        range_mean = np.mean(range_window[:-1])
        range_std = np.std(range_window[:-1])
        if range_std < 1e-12:
            range_std = 1e-12
        spread_z = (abs(range_window[-1]) - range_mean) / range_std
        spread_score = max(0.0, min(1.0, (spread_z - 1.0) / 2.0))

        # Volume-price divergence: price up but volume down
        price_dir = np.sign(close[-1] - close[-lb])
        vol_dir = np.sign(volume[-1] - np.mean(volume[-lb:-1]))
        divergence = 1.0 if price_dir * vol_dir < 0 else 0.0

        return float(round(
            0.35 * vol_thinning
            + 0.25 * move_on_thin
            + 0.20 * spread_score
            + 0.20 * divergence,
            4,
        ))

    def _score_cross_venue(
        self,
        close: NDArray[np.float64],
        open_: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        volume: NDArray[np.float64],
        lb: int,
    ) -> float:
        """Score for CROSS_VENUE_DIVERGENCE: contradictory patterns.

        Simulated via volume distribution analysis:
        - If volume clusters at specific price levels → single venue dominance
        - If price and volume disagree on trend → cross-venue divergence
        - Large range bars with minimal volume → thin venue liquidity
        """
        if len(close) < lb:
            return 0.0

        # Price direction per bar
        # Volume distribution across bars
        vol_per_bar = volume[-lb:]
        total_vol = vol_per_bar.sum()
        if total_vol < 1e-12:
            return 0.0

        # Herfindahl index of volume concentration
        vol_share = vol_per_bar / total_vol
        herfindahl = np.sum(vol_share ** 2)
        # High concentration (>0.15) suggests one venue driving volume
        venue_concentration = min(1.0, max(0.0, (herfindahl - 0.1) / 0.15))

        # Price-volume disagreement: bars with price move but no volume
        range_bars = np.abs(np.diff(close)[-lb:]) / close[-lb - 1 : -1]
        low_vol_mask = vol_per_bar < np.mean(vol_per_bar)
        big_move_on_low_vol = np.mean(
            range_bars[low_vol_mask]
        ) if np.any(low_vol_mask) else 0.0
        avg_range = np.mean(range_bars)
        if avg_range > 1e-12:
            divergence_score = min(
                1.0, (big_move_on_low_vol / avg_range - 0.5) / 0.5
            )
        else:
            divergence_score = 0.0

        # Spread-to-volume ratio — wide spread with low volume = fragmented
        wicks = np.abs(high[-lb:] - low[-lb:])
        safe_vol = np.where(vol_per_bar > 1e-12, vol_per_bar, 1e-12)
        sv_ratio = wicks / safe_vol
        avg_sv = np.mean(sv_ratio)
        sv_score = min(1.0, avg_sv / 5.0)

        return float(round(
            0.35 * venue_concentration
            + 0.35 * max(0.0, divergence_score)
            + 0.30 * sv_score,
            4,
        ))

    # ── Report construction ─────────────────────────────────────────────────

    def _build_hypothesis(self, scores: dict[str, float]) -> str:
        """Summarize anomaly scores."""
        flagged = {k: v for k, v in scores.items() if v >= self._threshold}
        if not flagged:
            return (
                f"No anomalies detected above threshold {self._threshold:.1f}. "
                f"All anomaly scores below conservative threshold."
            )

        parts = [f"Anomaly scores — {len(flagged)} type(s) above threshold {self._threshold:.1f}:"]
        for name, score in sorted(flagged.items(), key=lambda x: -x[1]):
            label = name.replace("_", " ").title()
            parts.append(f"  {label}: {score:.3f}")
        return " ".join(parts)

    def _compute_probabilities(
        self, scores: dict[str, float]
    ) -> dict[str, float]:
        """Derive up/down/range probabilities from anomaly scores.

        Conservative: high anomaly score → more range (uncertainty).
        Direction comes from which anomaly type dominates.
        """
        anomaly = scores.get("anomaly", 0.0)
        spoofing = scores.get("spoofing_like", 0.0)
        liquidity = scores.get("liquidity_withdrawal", 0.0)
        cross_venue = scores.get("cross_venue_divergence", 0.0)

        # Anomaly signals can be directional; use volume-weighted aggregation
        # Anomaly and spoofing suggest directional moves
        directional_weight = 0.5 * anomaly + 0.3 * spoofing + 0.2 * liquidity
        # Cross-venue divergence suggests uncertainty (range)
        uncertainty_weight = 0.6 * cross_venue + 0.4 * (1.0 - directional_weight)

        # Normalize to probabilities
        total = directional_weight + uncertainty_weight
        if total < 1e-12:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        # Split directional into up/down roughly equally with slight bias
        up_prob = round(directional_weight / total * 0.55, 4)
        down_prob = round(directional_weight / total * 0.45, 4)
        range_prob = round(uncertainty_weight / total, 4)

        # Ensure sum = 1.0
        total_p = up_prob + down_prob + range_prob
        if abs(total_p - 1.0) > 1e-6:
            range_prob = round(1.0 - up_prob - down_prob, 4)

        return {"up": float(up_prob), "down": float(down_prob), "range": float(range_prob)}

    def _build_evidence(
        self,
        scores: dict[str, float],
        close: NDArray[np.float64],
        volume: NDArray[np.float64],
        lb: int,
    ) -> list:
        """Evidence — all scored, no factual claims."""
        evidence: list = []
        now = datetime.datetime.now().isoformat()

        for score_name, score_val in scores.items():
            if score_val >= self._threshold:
                label = score_name.replace("_", " ").title()
                evidence.append(
                    self._make_evidence(
                        score_name,
                        f"{label} score: {score_val:.3f} (above {self._threshold:.1f} threshold, t={now})",
                        "positive",
                        score_val,
                    )
                )
            elif score_val > 0.0:
                # Always report non-zero scores — they are data, not claims
                label = score_name.replace("_", " ").title()
                evidence.append(
                    self._make_evidence(
                        score_name,
                        f"{label} score: {score_val:.3f} (below {self._threshold:.1f} threshold, t={now})",
                        "neutral",
                        score_val * 0.5,
                    )
                )

        # Always include at least one evidence entry
        if not evidence:
            latest_vol = volume[-1]
            latest_close = close[-1]
            evidence.append(
                self._make_evidence(
                    "no_anomaly",
                    f"all anomaly scores below {self._threshold:.1f}; "
                    f"last close: {latest_close:.2f}, last volume: {latest_vol:.1f}",
                    "neutral",
                    0.0,
                )
            )

        return evidence

    def _build_counter_evidence(self, scores: dict[str, float]) -> list:
        """Counter-evidence: alternative explanations for anomaly-like patterns.

        Required by spec — anomalies could be legitimate market events,
        not necessarily manipulation or structural breaks.
        """
        counter: list = []
        flagged = {k: v for k, v in scores.items() if v >= self._threshold}

        if not flagged:
            # No anomalies — counter is simply "normal market conditions"
            counter.append(
                self._make_evidence(
                    "counter_normal",
                    "no anomalies detected — consistent with normal market conditions",
                    "negative",
                    0.0,
                )
            )
            return counter

        # Provide alternative explanations for each flagged anomaly
        counter_explanations = {
            "anomaly": [
                "volume spike may reflect scheduled market events (earnings, FOMC, options expiry)",
                "price gap could be due to after-hours information release",
                "volatility increase may follow normal mean-reversion patterns",
            ],
            "spoofing_like": [
                "long wicks may reflect genuine liquidity sweeps by institutional players",
                "wick-to-volume ratio spikes can occur during news-driven volatility",
                "volume concentration at extremes may be algorithmic rebalancing",
            ],
            "liquidity_withdrawal": [
                "volume thinning may reflect regular market hours transition",
                "price moves on declining volume are common in low-liquidity sessions",
                "spread widening may be normal bid-ask bounce, not liquidity withdrawal",
            ],
            "cross_venue_divergence": [
                "volume concentration may reflect a single active trading session",
                "Herfindahl index changes are normal across timezone boundaries",
                "price-volume disagreement can follow scheduled economic data releases",
            ],
        }

        for name in flagged:
            alternatives = counter_explanations.get(name, ["no alternative explanation available"])
            counter.append(
                self._make_evidence(
                    f"counter_{name}_alt",
                    f"alternative: {'; '.join(alternatives[:2])}",
                    "negative",
                    0.4,
                )
            )

        return counter

    def _build_invalidations(self) -> list:
        """Invalidation conditions for anomaly detection."""
        return [
            self._make_invalidations(
                condition="Market regime change (e.g. crisis, halting) invalidates rolling statistics",
                indicator="regime_shift",
                threshold=1.0,
                direction="above",
            ),
            self._make_invalidations(
                condition="Data quality below threshold (gaps, out-of-range values)",
                indicator="data_quality",
                threshold=0.9,
                direction="below",
            ),
            self._make_invalidations(
                condition="Sample size too small for rolling statistics",
                indicator="sample_size",
                threshold=10,
                direction="below",
            ),
        ]

    def _compute_confidence(self, scores: dict[str, float]) -> float:
        """Compute raw confidence from score magnitudes.

        Conservative: confidence increases with score magnitude
        but is capped to penalize ambiguity.
        """
        score_values = list(scores.values())
        if not score_values:
            return 0.1

        # Mean of flagged scores (0 if none flagged)
        flagged_scores = [s for s in score_values if s >= self._threshold]
        if flagged_scores:
            mean_flagged = np.mean(flagged_scores)
            base_confidence = 0.3 + 0.4 * mean_flagged
        else:
            # No flagged anomalies — low confidence
            max_score = max(score_values)
            base_confidence = 0.1 + 0.15 * max_score

        # Penalty for having many low scores (ambiguous)
        near_threshold = sum(1 for s in score_values if s > 0.3 and s < self._threshold)
        if near_threshold > 1:
            base_confidence -= 0.05 * near_threshold

        return float(round(max(0.1, min(0.9, base_confidence)), 4))

    def _short_data_report(
        self,
        close: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        volume: NDArray[np.float64],
        open_: NDArray[np.float64],
        n: int,
    ) -> AgentReport:
        """Report for insufficient data."""
        hypothesis = (
            f"Insufficient data for anomaly detection: {n} bars available, "
            f"minimum {self._lookback} required. Scores default to 0.0."
        )
        probabilities = {"up": 0.33, "down": 0.33, "range": 0.34}
        evidence = [
            self._make_evidence(
                "insufficient_data",
                f"only {n} bars available, need at least {self._lookback}",
                "neutral",
                0.0,
            )
        ]
        counter_evidence = [
            self._make_evidence(
                "counter_insufficient",
                "no anomalies possible with insufficient data — by design",
                "negative",
                0.0,
            )
        ]
        confidence = 0.05
        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(),
            hypothesis=hypothesis,
            probabilities=probabilities,
            evidence=evidence,
            counter_evidence=counter_evidence,
            invalidations=self._build_invalidations(),
            raw_confidence=confidence,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )
