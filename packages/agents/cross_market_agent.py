"""Cross-Market Analysis Agent — multi-market correlation and signal aggregation."""

from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent

SUPPORTED_KEYS = frozenset({
    "btc_dominance", "btc_dominance_arr",
    "eth_btc", "eth_btc_arr",
    "sp500_ret", "sp500_arr",
    "nasdaq_ret", "nasdaq_arr",
    "dxy", "dxy_arr",
    "gold_ret", "gold_arr",
    "stablecoin_flow",
})

# Market definitions: (scalar_key, array_key, signal_map, weight)
_MARKET_DEFS = [
    {
        "name": "btc_dominance",
        "display": "BTC.D",
        "scalar_key": "btc_dominance",
        "array_key": "btc_dominance_arr",
        "description": "BTC Market Dominance",
        "signal_fn": lambda val: 1.0 - 2.0 * val,  # high dom = alt weakness
        "weight": 0.18,
        "up_range": (0.45, 0.55),  # rising dominance -> down
    },
    {
        "name": "eth_btc",
        "display": "ETH/BTC",
        "scalar_key": "eth_btc",
        "array_key": "eth_btc_arr",
        "description": "ETH/BTC Ratio",
        "signal_fn": lambda val: 2.0 * (val - 0.04),  # rising = ETH strength
        "weight": 0.16,
        "up_range": (0.035, 0.045),
    },
    {
        "name": "sp500_ret",
        "display": "S&P 500",
        "scalar_key": "sp500_ret",
        "array_key": "sp500_arr",
        "description": "S&P 500 Return",
        "signal_fn": lambda val: np.sign(val) if val != 0 else 0.0,
        "weight": 0.16,
        "up_range": (-0.5, 0.5),
    },
    {
        "name": "nasdaq_ret",
        "display": "Nasdaq",
        "scalar_key": "nasdaq_ret",
        "array_key": "nasdaq_arr",
        "description": "Nasdaq Return",
        "signal_fn": lambda val: np.sign(val) if val != 0 else 0.0,
        "weight": 0.14,
        "up_range": (-0.5, 0.5),
    },
    {
        "name": "dxy",
        "display": "DXY (USD Index)",
        "scalar_key": "dxy",
        "array_key": "dxy_arr",
        "description": "US Dollar Index",
        "signal_fn": lambda val: 1.0 - 2.0 * (val - 100) / 100,  # inverse
        "weight": 0.16,
        "up_range": (98, 102),
    },
    {
        "name": "gold_ret",
        "display": "Gold",
        "scalar_key": "gold_ret",
        "array_key": "gold_arr",
        "description": "Gold Return",
        "signal_fn": lambda val: np.sign(val) if val != 0 else 0.0,
        "weight": 0.10,
        "up_range": (-0.5, 0.5),
    },
    {
        "name": "stablecoin_flow",
        "display": "Stablecoin Flow",
        "scalar_key": "stablecoin_flow",
        "array_key": "stablecoin_flow_arr",
        "description": "Stablecoin Net Flow",
        "signal_fn": lambda val: np.sign(val) if val != 0 else 0.0,
        "weight": 0.10,
        "up_range": (-1e9, 1e9),
    },
]


class CrossMarketAgent(BaseAgent):
    """Cross-Market-Analyse-Agent — Korrelationen und Signale aus multiplen M\u00e4rkten."""

    def __init__(
        self,
        config: AgentConfig | None = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="cross_market",
                agent_type=AgentType.CROSS_MARKET,
            )
        super().__init__(config)

    def analyze(
        self, data: dict[str, NDArray[np.float64]]
    ) -> AgentReport:
        """Analysiert Quermarktdaten auf Signale und Korrelationen.

        Accepts a dict with cross-market data:
            scalar keys (one or more required):
                btc_dominance: float, 0-1 (e.g. 0.52)
                eth_btc: float, ratio (e.g. 0.041)
                sp500_ret: float, daily return % (e.g. 0.5)
                nasdaq_ret: float, daily return % (e.g. 1.2)
                dxy: float, index level (e.g. 104.5)
                gold_ret: float, daily return % (e.g. -0.3)
                stablecoin_flow: float, net flow in USD (e.g. 1e9)
            array keys (optional, used for correlation):
                btc_dominance_arr: NDArray — time series of BTC dominance
                eth_btc_arr: NDArray — time series of ETH/BTC ratio
                sp500_arr: NDArray — time series of S&P 500 returns
                nasdaq_arr: NDArray — time series of Nasdaq returns
                dxy_arr: NDArray — time series of DXY index
                gold_arr: NDArray — time series of gold returns
                stablecoin_flow_arr: NDArray — time series of stablecoin flows

        Returns:
            AgentReport with aggregated cross-market probabilities.
        """
        scalar_keys = {mdef["scalar_key"] for mdef in _MARKET_DEFS}
        has_news = "news" in data
        has_any_scalar = any(k in data for k in scalar_keys)
        if not has_news and not has_any_scalar:
            raise ValueError(
                "Missing required data keys: "
                "need at least one scalar (e.g. 'btc_dominance', 'eth_btc') or 'news'"
            )

        # Extract scalar values
        scalars: dict[str, float] = {}
        for mdef in _MARKET_DEFS:
            key = mdef["scalar_key"]
            val = data.get(key)
            if val is not None:
                scalars[key] = float(val)

        # Extract arrays for correlation
        arrays: dict[str, NDArray[np.float64]] = {}
        for mdef in _MARKET_DEFS:
            key = mdef["array_key"]
            arr = data.get(key)
            if arr is not None:
                arrays[key] = np.asarray(arr, dtype=np.float64)

        # Build market signals
        market_signals: list[dict] = []
        for mdef in _MARKET_DEFS:
            signal = self._evaluate_market(
                mdef, scalars, arrays
            )
            if signal is not None:
                market_signals.append(signal)

        # Sort by weight descending
        market_signals.sort(key=lambda s: -s["weight"])

        # Build report components
        hypothesis = self._build_hypothesis(market_signals)
        probabilities = self._compute_probabilities(market_signals)
        evidence = self._build_evidence(market_signals)
        counter_evidence = self._build_counter_evidence(market_signals)
        invalidations = self._build_invalidations(market_signals)
        confidence = self._compute_confidence(market_signals)

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

    # ── private helpers ──────────────────────────────────────────────────

    def _evaluate_market(
        self,
        mdef: dict,
        scalars: dict[str, float],
        arrays: dict[str, NDArray[np.float64]],
    ) -> dict | None:
        """Evaluate a single market's signal.

        Returns dict with signal info or None if insufficient data.
        """
        scalar_key = mdef["scalar_key"]
        array_key = mdef["array_key"]

        # Skip markets with no data at all
        if scalar_key not in scalars and array_key not in arrays:
            return None

        # Compute direction and magnitude from scalar
        direction = "neutral"
        magnitude = 0.0
        value = None
        range_context = mdef.get("up_range", (0, 1))

        if scalar_key in scalars:
            val = scalars[scalar_key]
            value = val
            if mdef["name"] == "btc_dominance":
                lower_bound, upper_bound = mdef["up_range"]
                if val > upper_bound:
                    direction = "down"  # high dom = alt weakness
                    magnitude = min(1.0, (val - upper_bound) / 0.1)
                elif val < lower_bound:
                    direction = "up"  # low dom = alt strength
                    magnitude = min(1.0, (lower_bound - val) / 0.1)
            elif mdef["name"] == "eth_btc":
                lower_bound, upper_bound = mdef["up_range"]
                if val > upper_bound:
                    direction = "up"  # rising ratio = ETH strength
                    magnitude = min(1.0, (val - upper_bound) / 0.01)
                elif val < lower_bound:
                    direction = "down"  # falling ratio = ETH weakness
                    magnitude = min(1.0, (lower_bound - val) / 0.01)
            elif mdef["name"] == "dxy":
                lower_bound, upper_bound = mdef["up_range"]
                if val > upper_bound:
                    direction = "down"  # strong dollar = crypto weakness
                    magnitude = min(1.0, (val - upper_bound) / 5.0)
                elif val < lower_bound:
                    direction = "up"  # weak dollar = crypto strength
                    magnitude = min(1.0, (lower_bound - val) / 5.0)
            elif mdef["name"] == "stablecoin_flow":
                if val > 1e8:
                    direction = "up"  # inflow = bullish
                    magnitude = min(1.0, abs(val) / 5e9)
                elif val < -1e8:
                    direction = "down"  # outflow = bearish
                    magnitude = min(1.0, abs(val) / 5e9)
            else:
                # sp500_ret, nasdaq_ret, gold_ret — return-based
                if val > 0.1:
                    direction = "up"
                    magnitude = min(1.0, abs(val) / 2.0)
                elif val < -0.1:
                    direction = "down"
                    magnitude = min(1.0, abs(val) / 2.0)

        # Compute correlation from arrays
        correlation = 0.0
        correlation_available = False

        if array_key in arrays:
            arr = arrays[array_key]
            if len(arr) >= 10:
                correlation = float(np.corrcoef(
                    arr, np.arange(len(arr), dtype=np.float64)
                )[0, 1]) if len(arr) > 1 else 0.0
                if np.isnan(correlation):
                    correlation = 0.0
                correlation_available = True

        return {
            "name": mdef["name"],
            "display": mdef["display"],
            "description": mdef["description"],
            "direction": direction,
            "magnitude": magnitude,
            "weight": mdef["weight"],
            "value": value,
            "range_context": range_context,
            "correlation": correlation if correlation_available else None,
            "correlation_available": correlation_available,
            "time_reference": datetime.datetime.now().isoformat(),
        }

    def _build_hypothesis(self, signals: list[dict]) -> str:
        """Build hypothesis summary."""
        if not signals:
            return "No cross-market data available for analysis."

        bull_count = sum(1 for s in signals if s["direction"] == "up")
        bear_count = sum(1 for s in signals if s["direction"] == "down")
        correlated = sum(
            1 for s in signals
            if s["correlation_available"] and abs(s["correlation"]) > 0.3
        )

        top_up = next(
            (s for s in signals if s["direction"] == "up"), None
        )
        top_down = next(
            (s for s in signals if s["direction"] == "down"), None
        )

        parts = [
            f"Cross-market analysis: {len(signals)} market(s) analyzed, "
            f"{bull_count} bullish, {bear_count} bearish, "
            f"{correlated} with strong correlation.",
        ]

        if top_up:
            parts.append(
                f"Top bullish: {top_up['display']} "
                f"({top_up['direction']}, mag={top_up['magnitude']:.2f})"
            )
        if top_down:
            parts.append(
                f"Top bearish: {top_down['display']} "
                f"({top_down['direction']}, mag={top_down['magnitude']:.2f})"
            )

        return " ".join(parts)

    def _compute_probabilities(
        self, signals: list[dict]
    ) -> dict[str, float]:
        """Compute up/down/range from aggregated market signals."""
        if not signals:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        up_weight = 0.0
        down_weight = 0.0

        for sig in signals:
            w = sig["weight"]
            m = sig["magnitude"]

            if sig["direction"] == "up":
                up_weight += w * m
            elif sig["direction"] == "down":
                down_weight += w * m

            # Correlation boost: if signal shows strong trend,
            # amplify it slightly
            if sig["correlation_available"] and sig["correlation"]:
                corr_factor = 1.0 + 0.2 * abs(sig["correlation"])
                if sig["direction"] == "up":
                    up_weight *= corr_factor
                elif sig["direction"] == "down":
                    down_weight *= corr_factor

        if up_weight == 0 and down_weight == 0:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        total = up_weight + down_weight + 0.20  # range buffer
        up_prob = round(up_weight / total, 4)
        down_prob = round(down_weight / total, 4)
        range_prob = round(1.0 - up_prob - down_prob, 4)

        return {"up": up_prob, "down": down_prob, "range": range_prob}

    def _build_evidence(self, signals: list[dict]) -> list:
        """Evidence from market signals with correlations."""
        evidence: list = []

        for sig in signals:
            corr_str = (
                f", corr={sig['correlation']:.3f}"
                if sig["correlation_available"] and sig["correlation"] is not None
                else ""
            )
            val_str = (
                f", val={sig['value']:.4f}"
                if sig["value"] is not None
                else ""
            )
            evidence.append(
                self._make_evidence(
                    sig["name"],
                    f"{sig['display']}: {sig['direction']} "
                    f"(mag={sig['magnitude']:.2f}, w={sig['weight']:.2f}"
                    f"{corr_str}{val_str}, t={sig['time_reference']})",
                    "positive" if sig["direction"] == "up" else "negative",
                    sig["weight"] * max(sig["magnitude"], 0.1),
                )
            )

        if not evidence:
            evidence.append(
                self._make_evidence(
                    "no_market_data",
                    "no cross-market data available",
                    "neutral",
                    0.0,
                )
            )

        return evidence

    def _build_counter_evidence(
        self, signals: list[dict]
    ) -> list:
        """Counter-evidence: markets with conflicting signals."""
        counter: list = []

        if len(signals) < 2:
            counter.append(
                self._make_evidence(
                    "counter_single_market",
                    "only one market available, insufficient for counter-evidence",
                    "negative",
                    0.1,
                )
            )
            return counter

        up_signals = [s for s in signals if s["direction"] == "up"]
        down_signals = [s for s in signals if s["direction"] == "down"]

        if up_signals and down_signals:
            best_up = max(up_signals, key=lambda s: s["magnitude"])
            best_down = max(down_signals, key=lambda s: s["magnitude"])
            counter.append(
                self._make_evidence(
                    "counter_conflict",
                    f"conflicting: {best_up['display']} {best_up['direction']} "
                    f"(mag={best_up['magnitude']:.2f}) vs "
                    f"{best_down['display']} {best_down['direction']} "
                    f"(mag={best_down['magnitude']:.2f})",
                    "negative",
                    0.6,
                )
            )
        else:
            # All same direction — use weakest signal as counter
            weakest = min(signals, key=lambda s: s["magnitude"])
            counter.append(
                self._make_evidence(
                    "counter_weak_signal",
                    f"weakest signal: {weakest['display']} "
                    f"(mag={weakest['magnitude']:.2f}, "
                    f"correlation={weakest.get('correlation', 'N/A')})",
                    "negative",
                    0.3,
                )
            )

        return counter

    def _build_invalidations(
        self, signals: list[dict]
    ) -> list:
        """Invalidation conditions for cross-market analysis."""
        invalidations: list = []

        if signals:
            # Check for correlation breakdown
            corrs = [
                s["correlation"]
                for s in signals
                if s["correlation_available"] and s["correlation"] is not None
            ]
            if corrs and max(abs(c) for c in corrs) > 0.8:
                invalidations.append(
                    self._make_invalidations(
                        condition="Correlation breakdown > 0.8 undermines trend signal",
                        indicator="max_correlation",
                        threshold=0.8,
                        direction="above",
                    )
                )

            # Data staleness
            invalidations.append(
                self._make_invalidations(
                    condition="Market data stale beyond 24h",
                    indicator="data_recency",
                    threshold=0.0,
                    direction="below",
                )
            )

        # No data at all
        invalidations.append(
            self._make_invalidations(
                condition="No cross-market data received",
                indicator="market_count",
                threshold=1.0,
                direction="below",
            )
        )

        return invalidations

    def _compute_confidence(self, signals: list[dict]) -> float:
        """Compute raw confidence from signal agreement."""
        if not signals:
            return 0.1

        # Agreement score: ratio of strongest direction
        up_count = sum(1 for s in signals if s["direction"] == "up")
        down_count = sum(1 for s in signals if s["direction"] == "down")

        total = len(signals)
        max_dir = max(up_count, down_count)
        agreement = max_dir / total if total > 0 else 0.0

        # Correlation coverage bonus
        corr_count = sum(
            1 for s in signals
            if s["correlation_available"] and s["correlation"] is not None
        )
        corr_coverage = corr_count / total if total > 0 else 0.0

        confidence = 0.2 + 0.4 * agreement + 0.2 * corr_coverage
        return round(min(0.9, confidence), 4)
