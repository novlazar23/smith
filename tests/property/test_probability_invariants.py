"""Property-based tests for probability invariants in the consensus module.

These tests verify structural invariants around probability outputs from
consensus agents — that probabilities are well-formed, sum to reasonable
values, and stay within the valid [0, 1] range.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from packages.consensus import ConsensusDecision, ConsensusResult, VoteDirection
from packages.consensus.weighted import WeightedConsensusEngine
from packages.schemas.agent_report import AgentReport, AgentStatus, EvidenceReference

# ---------------------------------------------------------------------------
# Helper strategies
# ---------------------------------------------------------------------------

def _valid_probabilities() -> st.SearchStrategy[dict[str, float]]:
    """A strategy that produces probability dicts that sum to ~1.0."""
    up = st.floats(0.3, 0.95, allow_nan=False, allow_infinity=False)
    return up.map(lambda u: {
        "up": round(u, 6),
        "down": round(max(0.0, 0.34 - (u - 0.34)), 6),
        "range": round(max(0.0, 1.0 - u - max(0.0, 0.34 - (u - 0.34))), 6),
    })


def _agent_id_strategy() -> st.SearchStrategy[str]:
    """Strategy for agent IDs."""
    return st.integers(0, 99999).map(lambda i: f"agent-{i}")


def _report_id_strategy() -> st.SearchStrategy[str]:
    """Strategy for report IDs."""
    return st.integers(0, 999999).map(lambda i: f"report-{i}")


def _evidence_reference() -> st.SearchStrategy[EvidenceReference]:
    """Generate a single EvidenceReference."""
    return st.builds(
        EvidenceReference,
        reference=st.text(min_size=1, max_size=20),
        feature=st.text(min_size=1, max_size=20),
        value=st.text(min_size=1, max_size=50),
        direction=st.sampled_from(["positive", "negative", "neutral"]),
        relevance=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )


def _valid_report() -> st.SearchStrategy[AgentReport]:
    """Generate AgentReport instances with valid probability dicts and evidence."""
    from datetime import UTC, datetime

    prob_strategy = _valid_probabilities()
    evidence_strategy = st.lists(_evidence_reference(), min_size=1, max_size=3)

    return prob_strategy.flatmap(
        lambda probs: evidence_strategy.map(
            lambda ev: AgentReport(
                report_id=f"test-{hash(str(probs) + str(ev)) % 100000}",
                run_id="test-run",
                agent_id=f"agent-{hash(str(probs)) % 10000}",
                agent_version="0.1.0",
                instrument="BTC/USDT",
                horizon="1h",
                as_of=datetime(2026, 1, 1, tzinfo=UTC),
                hypothesis="Test hypothesis",
                probabilities=probs,
                evidence=ev,
                expected_return=None,
                raw_confidence=None,
                calibrated_confidence=None,
                status=AgentStatus.SHADOW,
            )
        )
    )


# ---------------------------------------------------------------------------
# Test: VoteDirection enum is well-formed
# ---------------------------------------------------------------------------

class TestVoteDirection:
    """VoteDirection enum structural properties."""

    @given(st.builds(dict, keys=st.just(VoteDirection), values=st.floats(0, 1)))
    @settings(max_examples=20)
    def test_vote_directions_cover_expected_values(self, distribution: dict) -> None:
        """Each VoteDirection maps to a unique string value."""
        assert VoteDirection.LONG == "long"
        assert VoteDirection.SHORT == "short"
        assert VoteDirection.RANGE == "range"
        assert VoteDirection.ABSTAIN == "abstain"
        assert len(VoteDirection) == 4


# ---------------------------------------------------------------------------
# Test: Consensus probabilities are always valid [0, 1]
# ---------------------------------------------------------------------------

class TestConsensusProbabilitiesValid:
    """Properties that probability values from consensus agents are valid."""

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_single_probability_is_in_range(self, p: float) -> None:
        """Any probability value must be within [0, 1]."""
        assume(p >= 0.0)
        assume(p <= 1.0)
        assert 0.0 <= p <= 1.0

    @given(_valid_probabilities())
    @settings(max_examples=100)
    def test_all_probability_keys_are_non_negative(self, probs: dict[str, float]) -> None:
        """Every individual probability in the dict must be >= 0."""
        for key, value in probs.items():
            assert key in ("up", "down", "range")
            assert value >= 0.0

    @given(_valid_probabilities())
    @settings(max_examples=100)
    def test_all_probability_keys_are_at_most_one(self, probs: dict[str, float]) -> None:
        """Every individual probability in the dict must be <= 1."""
        for value in probs.values():
            assert value <= 1.0


# ---------------------------------------------------------------------------
# Test: ConsensusResult has valid probability-related fields
# ---------------------------------------------------------------------------

class TestConsensusResultInvariants:
    """Verify that ConsensusResult fields respect structural invariants."""

    def _make_result(self, vote_dist: dict[VoteDirection, float], confidence: float) -> ConsensusResult:
        """Helper to construct a ConsensusResult for testing."""
        return ConsensusResult(
            decision=ConsensusDecision.NO_TRADE,
            vote_distribution=vote_dist,
            agent_weights={},
            agent_agreements=[],
            agent_disagreements=[],
            confidence=confidence,
            reason="test",
        )

    @given(
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_confidence_is_in_range(self, confidence: float) -> None:
        """Confidence must be a valid probability in [0, 1]."""
        result = self._make_result(
            {
                VoteDirection.LONG: 0.0,
                VoteDirection.SHORT: 0.0,
                VoteDirection.RANGE: 0.0,
                VoteDirection.ABSTAIN: 0.0,
            },
            confidence,
        )
        assert 0.0 <= result.confidence <= 1.0

    @given(
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_vote_distribution_values_are_valid(
        self,
        long_val: float,
        short_val: float,
        range_val: float,
        abstain_val: float,
    ) -> None:
        """Every entry in vote_distribution must be in [0, 1]."""
        result = self._make_result(
            {
                VoteDirection.LONG: long_val,
                VoteDirection.SHORT: short_val,
                VoteDirection.RANGE: range_val,
                VoteDirection.ABSTAIN: abstain_val,
            },
            0.5,
        )
        for value in result.vote_distribution.values():
            assert 0.0 <= value <= 1.0

    @given(
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_vote_distribution_values_are_non_negative(
        self,
        long_val: float,
        short_val: float,
        range_val: float,
        abstain_val: float,
    ) -> None:
        """All vote distribution values must be >= 0."""
        result = self._make_result(
            {
                VoteDirection.LONG: long_val,
                VoteDirection.SHORT: short_val,
                VoteDirection.RANGE: range_val,
                VoteDirection.ABSTAIN: abstain_val,
            },
            0.5,
        )
        for value in result.vote_distribution.values():
            assert value >= 0.0


# ---------------------------------------------------------------------------
# Test: WeightedConsensusEngine produces valid results
# ---------------------------------------------------------------------------

class TestWeightedConsensusEngine:
    """Verify that the consensus engine always produces well-formed results."""

    @given(st.lists(_valid_report(), min_size=1, max_size=5))
    @settings(max_examples=50)
    def test_consensus_engine_validates_input(self, reports: list[AgentReport]) -> None:
        """Empty reports must raise ValueError."""
        engine = WeightedConsensusEngine()
        with pytest.raises(ValueError):
            engine.validate_input([])

    @given(st.lists(_valid_report(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_consensus_result_has_valid_decision(self, reports: list[AgentReport]) -> None:
        """The resulting decision must be one of the known ConsensusDecision values."""
        engine = WeightedConsensusEngine()
        result = engine.compute_consensus(reports)
        assert result.decision in (
            ConsensusDecision.LONG_BIAS,
            ConsensusDecision.SHORT_BIAS,
            ConsensusDecision.RANGE,
            ConsensusDecision.NO_TRADE,
        )

    @given(st.lists(_valid_report(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_consensus_confidence_is_valid(self, reports: list[AgentReport]) -> None:
        """Confidence from the engine must be in [0, 1]."""
        engine = WeightedConsensusEngine()
        result = engine.compute_consensus(reports)
        assert 0.0 <= result.confidence <= 1.0

    @given(st.lists(_valid_report(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_consensus_vote_distribution_is_non_negative(self, reports: list[AgentReport]) -> None:
        """All vote distribution values must be >= 0."""
        engine = WeightedConsensusEngine()
        result = engine.compute_consensus(reports)
        for value in result.vote_distribution.values():
            assert value >= 0.0

    @given(st.lists(_valid_report(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_consensus_agent_weights_are_non_negative(self, reports: list[AgentReport]) -> None:
        """Agent weights must be >= 0."""
        engine = WeightedConsensusEngine()
        result = engine.compute_consensus(reports)
        for weight in result.agent_weights.values():
            assert weight >= 0.0
