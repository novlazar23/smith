"""Unit tests for the Cross-Market Analysis Agent."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from packages.agents import AgentType, CrossMarketAgent
from packages.agents.base import AgentConfig
from packages.schemas.agent_report import AgentReport

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def full_market_data() -> dict:
    """Complete cross-market dataset with scalars and arrays."""
    return {
        "btc_dominance": 0.58,
        "eth_btc": 0.052,
        "sp500_ret": 0.8,
        "nasdaq_ret": 1.2,
        "dxy": 97.5,
        "gold_ret": -0.4,
        "stablecoin_flow": 2e9,
        "btc_dominance_arr": np.linspace(0.50, 0.58, 30),
        "eth_btc_arr": np.linspace(0.040, 0.052, 30),
        "sp500_arr": np.random.randn(30) * 0.5,
        "dxy_arr": np.linspace(100.0, 97.5, 30),
    }


@pytest.fixture
def scalar_only_data() -> dict:
    """Scalar values only, no time-series arrays."""
    return {
        "btc_dominance": 0.55,
        "eth_btc": 0.041,
        "sp500_ret": 0.3,
        "dxy": 101.0,
    }


@pytest.fixture
def bullish_data() -> dict:
    """All markets pointing up."""
    return {
        "btc_dominance": 0.40,
        "eth_btc": 0.048,
        "sp500_ret": 1.5,
        "nasdaq_ret": 2.0,
        "dxy": 96.0,
        "gold_ret": -0.5,
        "stablecoin_flow": 5e9,
    }


@pytest.fixture
def bearish_data() -> dict:
    """All markets pointing down."""
    return {
        "btc_dominance": 0.62,
        "eth_btc": 0.032,
        "sp500_ret": -1.5,
        "nasdaq_ret": -2.0,
        "dxy": 108.0,
        "gold_ret": 1.0,
        "stablecoin_flow": -5e9,
    }


@pytest.fixture
def conflicting_data() -> dict:
    """Mixed directional signals."""
    return {
        "btc_dominance": 0.60,
        "eth_btc": 0.048,
        "sp500_ret": 1.0,
        "nasdaq_ret": -0.8,
        "dxy": 100.0,
        "gold_ret": 0.5,
        "stablecoin_flow": 0.0,
    }


@pytest.fixture
def news_data() -> dict:
    """Data with news key only (valid per validation logic)."""
    return {
        "news": [{"title": "test", "body": "test", "source_name": "test"}],
    }


# ── AgentType & Config ───────────────────────────────────────────────────

class TestAgentTypeConfig:
    def test_cross_market_type_exists(self) -> None:
        assert hasattr(AgentType, "CROSS_MARKET")
        assert AgentType.CROSS_MARKET == "cross_market"

    def test_agent_config_defaults(self) -> None:
        a = CrossMarketAgent()
        assert a.agent_id == "cross_market"
        assert a.config.agent_type == AgentType.CROSS_MARKET
        assert a.config.agent_version == "0.1.0"
        assert a.config.status.value == "shadow"

    def test_agent_custom_config(self) -> None:
        custom = AgentConfig(
            agent_id="custom_cm",
            agent_type=AgentType.CROSS_MARKET,
            instrument="BTCUSD",
            horizon="4h",
        )
        a = CrossMarketAgent(config=custom)
        assert a.agent_id == "custom_cm"
        assert a.config.instrument == "BTCUSD"
        assert a.config.horizon == "4h"


# ── Initialization ──────────────────────────────────────────────────────

class TestInit:
    def test_no_config_creates_default(self) -> None:
        CrossMarketAgent()

    def test_none_config_creates_default(self) -> None:
        CrossMarketAgent(config=None)


# ── Validation ──────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_required_keys_raises(self) -> None:
        a = CrossMarketAgent()
        with pytest.raises(ValueError, match="Missing required data keys"):
            a.analyze({})

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing required"):
            CrossMarketAgent().analyze({})

    def test_news_only_valid(self, news_data: dict) -> None:
        report = CrossMarketAgent().analyze(news_data)
        assert report is not None
        assert report.agent_id == "cross_market"


# ── Scalar Analysis ─────────────────────────────────────────────────────

class TestScalarAnalysis:
    def test_btc_dominance_down_signal(self) -> None:
        data = {"btc_dominance": 0.58}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "btc_dominance" in e.reference]
        assert len(evidence) == 1
        assert "down" in evidence[0].value

    def test_btc_dominance_up_signal(self) -> None:
        data = {"btc_dominance": 0.40}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "btc_dominance" in e.reference]
        assert len(evidence) == 1
        assert "up" in evidence[0].value

    def test_eth_btc_up_signal(self) -> None:
        data = {"eth_btc": 0.052}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert len(evidence) == 1
        assert "up" in evidence[0].value

    def test_eth_btc_down_signal(self) -> None:
        data = {"eth_btc": 0.030}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert len(evidence) == 1
        assert "down" in evidence[0].value

    def test_dxy_up_signal(self) -> None:
        data = {"dxy": 96.0}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "dxy" in e.reference]
        assert len(evidence) == 1
        assert "up" in evidence[0].value

    def test_dxy_down_signal(self) -> None:
        data = {"dxy": 105.0}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "dxy" in e.reference]
        assert len(evidence) == 1
        assert "down" in evidence[0].value

    def test_stablecoin_up_signal(self) -> None:
        data = {"stablecoin_flow": 5e9}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "stablecoin_flow" in e.reference]
        assert len(evidence) == 1
        assert "up" in evidence[0].value

    def test_stablecoin_down_signal(self) -> None:
        data = {"stablecoin_flow": -5e9}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "stablecoin_flow" in e.reference]
        assert len(evidence) == 1
        assert "down" in evidence[0].value

    def test_sp500_up_signal(self) -> None:
        data = {"sp500_ret": 1.5}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "sp500" in e.reference]
        assert len(evidence) == 1
        assert "up" in evidence[0].value

    def test_sp500_down_signal(self) -> None:
        data = {"sp500_ret": -1.5}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "sp500" in e.reference]
        assert len(evidence) == 1
        assert "down" in evidence[0].value

    def test_gold_up_signal(self) -> None:
        data = {"gold_ret": 1.0}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "gold" in e.reference]
        assert len(evidence) == 1
        assert "up" in evidence[0].value

    def test_gold_down_signal(self) -> None:
        data = {"gold_ret": -1.0}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "gold" in e.reference]
        assert len(evidence) == 1
        assert "down" in evidence[0].value

    def test_neutral_btc_dominance(self) -> None:
        data = {"btc_dominance": 0.50}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "btc_dominance" in e.reference]
        assert len(evidence) == 1
        assert "neutral" in evidence[0].value

    def test_neutral_sp500_return(self) -> None:
        data = {"sp500_ret": 0.05}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "sp500" in e.reference]
        assert len(evidence) == 1
        assert "neutral" in evidence[0].value

    def test_neutral_dxy(self) -> None:
        data = {"dxy": 100.0}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "dxy" in e.reference]
        assert len(evidence) == 1
        assert "neutral" in evidence[0].value

    def test_neutral_stablecoin_flow(self) -> None:
        data = {"stablecoin_flow": 1e7}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "stablecoin_flow" in e.reference]
        assert len(evidence) == 1
        assert "neutral" in evidence[0].value

    def test_all_scalar_evidence_count(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert len(report.evidence) == 7


# ── Array Correlation ───────────────────────────────────────────────────

class TestCorrelation:
    def test_correlation_from_array(self) -> None:
        data = {"btc_dominance": 0.55, "btc_dominance_arr": np.linspace(0.48, 0.56, 30)}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "btc_dominance" in e.reference]
        assert len(evidence) == 1
        assert "corr=" in evidence[0].value

    def test_correlation_negative(self) -> None:
        x = np.linspace(0, 10, 50)
        arr = -x + np.random.randn(50) * 0.1
        data = {"eth_btc": 0.040, "eth_btc_arr": arr}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert "corr=" in evidence[0].value
        corr_val = float(evidence[0].value.split("corr=")[1].split(",")[0])
        assert corr_val < 0

    def test_correlation_positive(self) -> None:
        x = np.linspace(0, 10, 50)
        arr = x + np.random.randn(50) * 0.1
        data = {"eth_btc": 0.045, "eth_btc_arr": arr}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert "corr=" in evidence[0].value
        corr_val = float(evidence[0].value.split("corr=")[1].split(",")[0])
        assert corr_val > 0

    def test_correlation_zero(self) -> None:
        data = {"eth_btc": 0.045, "eth_btc_arr": np.random.randn(50)}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        corr_val = float(evidence[0].value.split("corr=")[1].split(",")[0])
        assert abs(corr_val) < 0.3

    def test_short_array_no_correlation(self) -> None:
        data = {"eth_btc": 0.045, "eth_btc_arr": np.linspace(0.040, 0.050, 5)}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert "corr=" not in evidence[0].value

    def test_no_array_no_correlation(self) -> None:
        data = {"eth_btc": 0.045}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert "corr=" not in evidence[0].value

    def test_correlation_boosts_probability(self) -> None:
        x = np.linspace(0, 100, 100)
        arr = 0.040 + 0.015 * x / 100 + np.random.randn(100) * 0.0001
        data = {
            "btc_dominance": 0.40,
            "eth_btc": 0.040,
            "eth_btc_arr": arr,
            "sp500_ret": 0.3,
            "dxy": 99.0,
            "stablecoin_flow": 1e9,
        }
        report = CrossMarketAgent().analyze(data)
        assert report.probabilities["up"] > 0.35


# ── Signal Aggregation ──────────────────────────────────────────────────

class TestSignalAggregation:
    def test_bullish_all_up(self, bullish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bullish_data)
        assert report.probabilities["up"] > 0.5

    def test_bearish_all_down(self, bearish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bearish_data)
        assert report.probabilities["down"] > 0.5

    def test_conflicting_probabilities(self, conflicting_data: dict) -> None:
        report = CrossMarketAgent().analyze(conflicting_data)
        assert report.probabilities["range"] >= 0.15

    def test_probability_sum_close_to_1(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        total = (
            report.probabilities["up"]
            + report.probabilities["down"]
            + report.probabilities["range"]
        )
        assert abs(total - 1.0) < 0.0001

    def test_probability_sum_scalar_only(self, scalar_only_data: dict) -> None:
        report = CrossMarketAgent().analyze(scalar_only_data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) < 0.0001

    def test_probability_sum_bullish(self, bullish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bullish_data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) < 0.0001

    def test_probability_sum_bearish(self, bearish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bearish_data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) < 0.0001

    def test_probability_sum_conflicting(self, conflicting_data: dict) -> None:
        report = CrossMarketAgent().analyze(conflicting_data)
        total = sum(report.probabilities.values())
        assert abs(total - 1.0) < 0.0001

    def test_all_neutral_gives_range_bias(self) -> None:
        data = {
            "btc_dominance": 0.50,
            "eth_btc": 0.040,
            "sp500_ret": 0.0,
            "nasdaq_ret": 0.0,
            "dxy": 100.0,
            "gold_ret": 0.0,
            "stablecoin_flow": 0.0,
        }
        report = CrossMarketAgent().analyze(data)
        assert report.probabilities["range"] > 0.30

    def test_probability_up_positive(self, bullish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bullish_data)
        assert report.probabilities["up"] > 0

    def test_probability_down_positive(self, bearish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bearish_data)
        assert report.probabilities["down"] > 0

    def test_probability_range_positive(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert report.probabilities["range"] > 0


# ── Evidence ─────────────────────────────────────────────────────────────

class TestEvidence:
    def test_evidence_min_1(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert len(report.evidence) >= 1

    def test_evidence_min_1_scalar_only(self, scalar_only_data: dict) -> None:
        report = CrossMarketAgent().analyze(scalar_only_data)
        assert len(report.evidence) >= 1

    def test_evidence_min_1_bullish(self, bullish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bullish_data)
        assert len(report.evidence) >= 1

    def test_evidence_min_1_bearish(self, bearish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bearish_data)
        assert len(report.evidence) >= 1

    def test_evidence_min_1_conflicting(self, conflicting_data: dict) -> None:
        report = CrossMarketAgent().analyze(conflicting_data)
        assert len(report.evidence) >= 1

    def test_evidence_contains_correlation(self) -> None:
        data = {"btc_dominance": 0.55, "btc_dominance_arr": np.linspace(0.48, 0.56, 30)}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "btc_dominance" in e.reference]
        assert "corr=" in evidence[0].value

    def test_evidence_time_referenced(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        for e in report.evidence:
            assert "t=" in e.value

    def test_evidence_market_display_names(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        values = [e.value for e in report.evidence]
        full_str = " ".join(values)
        assert "BTC.D" in full_str
        assert "ETH/BTC" in full_str
        assert "S&P 500" in full_str
        assert "Nasdaq" in full_str
        assert "DXY" in full_str or "USD Index" in full_str

    def test_evidence_has_direction(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        for e in report.evidence:
            assert any(d in e.value for d in ["up", "down", "neutral"])

    def test_evidence_has_magnitude(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        for e in report.evidence:
            assert "mag=" in e.value

    def test_evidence_has_weight(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        for e in report.evidence:
            assert "w=" in e.value

    def test_evidence_reference_format(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        for e in report.evidence:
            assert e.reference.startswith("cross_market:")

    def test_evidence_no_array_no_correlation(self, scalar_only_data: dict) -> None:
        report = CrossMarketAgent().analyze(scalar_only_data)
        for e in report.evidence:
            if "no_market_data" not in e.reference:
                assert "corr=" not in e.value

    def test_evidence_from_single_scalar(self) -> None:
        data = {"btc_dominance": 0.58}
        report = CrossMarketAgent().analyze(data)
        assert len(report.evidence) >= 1


# ── Counter Evidence ─────────────────────────────────────────────────────

class TestCounterEvidence:
    def test_counter_evidence_required(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert len(report.counter_evidence) >= 1

    def test_counter_evidence_scalar_only(self, scalar_only_data: dict) -> None:
        report = CrossMarketAgent().analyze(scalar_only_data)
        assert len(report.counter_evidence) >= 1

    def test_counter_evidence_bullish(self, bullish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bullish_data)
        assert len(report.counter_evidence) >= 1

    def test_counter_evidence_bearish(self, bearish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bearish_data)
        assert len(report.counter_evidence) >= 1

    def test_counter_evidence_conflicting(self, conflicting_data: dict) -> None:
        report = CrossMarketAgent().analyze(conflicting_data)
        counter = report.counter_evidence[0]
        assert "conflict" in counter.reference.lower() or "conflicting" in counter.value.lower()

    def test_counter_evidence_all_same_direction(self) -> None:
        """All markets pointing same direction → weakest signal as counter."""
        data = {
            "btc_dominance": 0.40,
            "eth_btc": 0.048,
            "sp500_ret": 1.5,
            "nasdaq_ret": 2.0,
            "dxy": 96.0,
            "gold_ret": 0.5,  # positive = up
            "stablecoin_flow": 5e9,
        }
        report = CrossMarketAgent().analyze(data)
        counter = report.counter_evidence[0]
        assert "weakest" in counter.reference.lower() or "weakest" in counter.value.lower()

    def test_counter_evidence_single_market(self) -> None:
        data = {"btc_dominance": 0.58}
        report = CrossMarketAgent().analyze(data)
        assert len(report.counter_evidence) >= 1
        assert "single" in report.counter_evidence[0].reference.lower() or \
               "weak_signal" in report.counter_evidence[0].reference.lower()


# ── Invalidation ─────────────────────────────────────────────────────────

class TestInvalidations:
    def test_invalidations_present(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert len(report.invalidations) >= 1

    def test_invalidations_no_data_condition(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        conditions = [inv.condition for inv in report.invalidations]
        assert any("No cross-market data" in c for c in conditions)

    def test_invalidations_staleness_condition(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        conditions = [inv.condition for inv in report.invalidations]
        assert any("stale" in c.lower() or "24h" in c for c in conditions)

    def test_invalidations_correlation_breakdown(self) -> None:
        data = {
            "btc_dominance": 0.55,
            "btc_dominance_arr": np.linspace(0.48, 0.56, 50),
        }
        report = CrossMarketAgent().analyze(data)
        conditions = [inv.condition for inv in report.invalidations]
        has_corr_condition = any("Correlation" in c for c in conditions)
        assert has_corr_condition


# ── Hypothesis ───────────────────────────────────────────────────────────

class TestHypothesis:
    def test_hypothesis_non_empty(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert len(report.hypothesis) > 0

    def test_hypothesis_mentions_markets(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert "market" in report.hypothesis.lower() or "7" in report.hypothesis

    def test_hypothesis_mentions_bearish_bullish(self, bullish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bullish_data)
        assert "bullish" in report.hypothesis.lower() or "bearish" in report.hypothesis.lower()

    def test_hypothesis_no_data(self, news_data: dict) -> None:
        report = CrossMarketAgent().analyze(news_data)
        assert "no cross-market data" in report.hypothesis.lower() or "news" in report.hypothesis.lower()


# ── Confidence ───────────────────────────────────────────────────────────

class TestConfidence:
    def test_confidence_positive(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert report.raw_confidence > 0

    def test_confidence_max_capped_at_0_9(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert report.raw_confidence <= 0.9

    def test_confidence_high_agreement(self, bullish_data: dict) -> None:
        report = CrossMarketAgent().analyze(bullish_data)
        assert report.raw_confidence > 0.5

    def test_confidence_low_agreement(self, conflicting_data: dict) -> None:
        report = CrossMarketAgent().analyze(conflicting_data)
        assert report.raw_confidence < 0.7

    def test_confidence_with_correlation_coverage(
        self, scalar_only_data: dict, full_market_data: dict
    ) -> None:
        report_no_corr = CrossMarketAgent().analyze(scalar_only_data)
        report_with_corr = CrossMarketAgent().analyze(full_market_data)
        assert report_with_corr.raw_confidence >= report_no_corr.raw_confidence

    def test_confidence_single_market(self) -> None:
        data = {"btc_dominance": 0.58}
        report = CrossMarketAgent().analyze(data)
        assert 0.1 <= report.raw_confidence <= 0.9


# ── Report Structure ─────────────────────────────────────────────────────

class TestReportStructure:
    def test_report_has_all_fields(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert report.report_id is not None
        assert report.run_id is not None
        assert report.agent_id == "cross_market"
        assert report.agent_version == "0.1.0"
        assert report.horizon == "1h"
        assert report.hypothesis is not None
        assert isinstance(report.probabilities, dict)
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1
        assert report.raw_confidence is not None
        assert report.as_of is not None

    def test_report_as_of_datetime(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert isinstance(report.as_of, datetime.datetime)

    def test_report_agent_id_match(self, full_market_data: dict) -> None:
        a = CrossMarketAgent()
        report = a.analyze(full_market_data)
        assert report.agent_id == a.agent_id

    def test_report_agent_version(self, full_market_data: dict) -> None:
        a = CrossMarketAgent()
        report = a.analyze(full_market_data)
        assert report.agent_version == a.config.agent_version

    def test_report_status(self, full_market_data: dict) -> None:
        a = CrossMarketAgent()
        report = a.analyze(full_market_data)
        assert report.status == a.config.status

    def test_report_run_id_unique(self, full_market_data: dict, bullish_data: dict) -> None:
        a = CrossMarketAgent()
        report1 = a.analyze(full_market_data)
        report2 = a.analyze(bullish_data)
        assert report1.run_id != report2.run_id

    def test_report_agent_version_format(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)
        assert isinstance(report.agent_version, str)
        assert "." in report.agent_version


# ── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_partial_market_data(self) -> None:
        data = {"btc_dominance": 0.55, "eth_btc": 0.050}
        report = CrossMarketAgent().analyze(data)
        assert len(report.evidence) == 2
        assert abs(sum(report.probabilities.values()) - 1.0) < 0.0001

    def test_only_btc_dominance(self) -> None:
        data = {"btc_dominance": 0.58}
        report = CrossMarketAgent().analyze(data)
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1

    def test_zero_values(self) -> None:
        data = {
            "btc_dominance": 0.0,
            "eth_btc": 0.0,
            "sp500_ret": 0.0,
            "nasdaq_ret": 0.0,
            "dxy": 100.0,
            "gold_ret": 0.0,
            "stablecoin_flow": 0.0,
        }
        report = CrossMarketAgent().analyze(data)
        assert abs(sum(report.probabilities.values()) - 1.0) < 0.0001

    def test_very_large_values(self) -> None:
        data = {"btc_dominance": 0.50, "stablecoin_flow": 1e12}
        report = CrossMarketAgent().analyze(data)
        assert report.probabilities["up"] > 0.3

    def test_very_small_values(self) -> None:
        data = {
            "btc_dominance": 0.50,
            "stablecoin_flow": 1e5,
            "sp500_ret": 0.05,
        }
        report = CrossMarketAgent().analyze(data)
        assert abs(sum(report.probabilities.values()) - 1.0) < 0.0001

    def test_extreme_dominance(self) -> None:
        data = {"btc_dominance": 0.90}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "btc_dominance" in e.reference]
        assert "down" in evidence[0].value
        assert report.probabilities["down"] > 0.4

    def test_extreme_eth_btc(self) -> None:
        data = {"eth_btc": 0.10}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert "up" in evidence[0].value

    def test_very_long_array(self) -> None:
        data = {
            "btc_dominance": 0.55,
            "btc_dominance_arr": np.linspace(0.48, 0.60, 1000),
        }
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "btc_dominance" in e.reference]
        assert "corr=" in evidence[0].value

    def test_nan_array_handling(self) -> None:
        arr = np.linspace(0.040, 0.050, 30)
        arr[15] = np.nan
        data = {"eth_btc": 0.045, "eth_btc_arr": arr}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert "corr=" in evidence[0].value
        corr_val = float(evidence[0].value.split("corr=")[1].split(",")[0])
        assert not np.isnan(corr_val)

    def test_single_element_array(self) -> None:
        data = {"eth_btc": 0.045, "eth_btc_arr": np.array([0.045])}
        report = CrossMarketAgent().analyze(data)
        evidence = [e for e in report.evidence if "eth_btc" in e.reference]
        assert "corr=" not in evidence[0].value


# ── Signal Weights ───────────────────────────────────────────────────────

class TestSignalWeights:
    def test_btc_dominance_heaviest(self) -> None:
        data = {
            "btc_dominance": 0.60,
            "eth_btc": 0.030,
        }
        report = CrossMarketAgent().analyze(data)
        assert report.probabilities["down"] > 0.4

    def test_weighted_aggregation(self) -> None:
        data = {
            "btc_dominance": 0.40,
            "eth_btc": 0.050,
            "sp500_ret": 1.0,
            "nasdaq_ret": 1.5,
            "dxy": 96.0,
            "gold_ret": -0.5,
            "stablecoin_flow": 3e9,
        }
        report = CrossMarketAgent().analyze(data)
        assert report.probabilities["up"] > report.probabilities["down"]
        assert abs(sum(report.probabilities.values()) - 1.0) < 0.0001

    def test_weighted_bearish_dominance(self) -> None:
        data = {
            "btc_dominance": 0.65,
            "eth_btc": 0.030,
            "dxy": 108.0,
        }
        report = CrossMarketAgent().analyze(data)
        assert report.probabilities["down"] > 0.4
        assert abs(sum(report.probabilities.values()) - 1.0) < 0.0001


# ── Integration ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self, full_market_data: dict) -> None:
        report = CrossMarketAgent().analyze(full_market_data)

        assert isinstance(report, AgentReport)
        assert report.agent_id == "cross_market"
        assert len(report.evidence) == 7
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1
        assert abs(sum(report.probabilities.values()) - 1.0) < 0.0001
        assert report.raw_confidence > 0.3

        for e in report.evidence:
            assert e.reference.startswith("cross_market:")
            assert e.relevance > 0
            assert any(d in e.value for d in ["up", "down", "neutral"])

        for ce in report.counter_evidence:
            assert ce.relevance > 0

    def test_hypothesis_includes_all_metrics(self) -> None:
        data = {
            "btc_dominance": 0.60,
            "eth_btc": 0.030,
            "sp500_ret": -1.0,
            "nasdaq_ret": -1.5,
            "dxy": 105.0,
            "gold_ret": 0.5,
            "stablecoin_flow": -2e9,
            "btc_dominance_arr": np.linspace(0.55, 0.62, 30),
        }
        report = CrossMarketAgent().analyze(data)
        h = report.hypothesis
        assert "market" in h.lower()
        assert "bullish" in h.lower() or "bearish" in h.lower()

    def test_counter_evidence_identifies_conflict(self) -> None:
        data = {
            "btc_dominance": 0.60,
            "eth_btc": 0.050,
            "sp500_ret": 1.0,
            "dxy": 97.0,
        }
        report = CrossMarketAgent().analyze(data)
        counter = report.counter_evidence[0]
        assert len(counter.value) > 10
        assert "conflict" in counter.reference.lower() or "conflicting" in counter.value.lower()
