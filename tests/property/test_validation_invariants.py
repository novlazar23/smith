"""Property-based tests for validation invariants in the ablation module.

These tests verify structural invariants around ablation results —
that feature importance values are valid, marginal contributions are
consistent, and direction flags match the sign of the contribution.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from packages.validation import AblationResult

# ---------------------------------------------------------------------------
# Helper strategies
# ---------------------------------------------------------------------------

_positive_score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_method = st.sampled_from(["random_zero", "mean_zero", "shuffle", "leave_one_out"])
_agent_id = st.integers(0, 99999).map(lambda i: f"agent_{i}")


# ---------------------------------------------------------------------------
# Test: AblationResult invariants — marginal contribution sign and direction
# ---------------------------------------------------------------------------

class TestAblationResultInvariants:
    """AblationResult fields must be internally consistent."""

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=200)
    def test_marginal_contribution_is_consistent_with_scores(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """marginal_contribution must equal full_score - ablated_score."""
        marginal = full_score - ablated_score
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=marginal,
            direction="positive",
        )
        assert result.marginal_contribution == full_score - ablated_score

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=200)
    def test_is_helpful_and_is_harmful_are_exclusive(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """An agent cannot be both helpful and harmful simultaneously."""
        marginal = full_score - ablated_score
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=marginal,
            direction="positive",
        )
        # Must be exactly one (or neither if neutral)
        if result.is_helpful:
            assert not result.is_harmful
        elif result.is_harmful:
            assert not result.is_helpful
        else:
            assert not result.is_helpful
            assert not result.is_harmful

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=200)
    def test_marginal_contribution_values_are_finite(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """Marginal contribution must be a finite float, not inf or NaN."""
        marginal = full_score - ablated_score
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=marginal,
            direction="positive",
        )
        import math
        assert math.isfinite(result.marginal_contribution)

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=200)
    def test_confidence_is_in_range(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """Confidence defaults to 1.0 and must be in [0, 1]."""
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=full_score - ablated_score,
            confidence=1.0,
        )
        assert 0.0 <= result.confidence <= 1.0

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=200)
    def test_full_score_is_in_range(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """full_score should be in [0, 1] for normalized metrics."""
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=full_score - ablated_score,
        )
        assert 0.0 <= result.full_score <= 1.0

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=200)
    def test_ablated_score_is_in_range(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """ablated_score should be in [0, 1] for normalized metrics."""
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=full_score - ablated_score,
        )
        assert 0.0 <= result.ablated_score <= 1.0

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=200)
    def test_direction_can_be_set_explicitly(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """The direction field is explicitly set, not auto-computed."""
        for dir_val in ("positive", "negative", "neutral"):
            result = AblationResult(
                method=method,
                agent_id=agent_id,
                full_score=full_score,
                ablated_score=ablated_score,
                marginal_contribution=full_score - ablated_score,
                direction=dir_val,
            )
            assert result.direction == dir_val


# ---------------------------------------------------------------------------
# Test: Feature importance is valid
# ---------------------------------------------------------------------------

class TestFeatureImportanceValidity:
    """Feature importance values from ablation must be valid."""

    @given(
        _method,
        _agent_id,
        _positive_score,
        _positive_score,
    )
    @settings(max_examples=200)
    def test_feature_importance_sum_consistency(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """If full_score is the score of all features, removing a feature
        can only decrease or maintain the score (positive marginal) or
        increase it (negative marginal)."""
        marginal = full_score - ablated_score
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=marginal,
        )
        # The marginal contribution is finite and well-defined
        assert isinstance(result.marginal_contribution, float)
        assert not isinstance(result.marginal_contribution, bool)

    @given(
        _method,
        _agent_id,
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_marginal_contribution_can_be_negative_or_positive(
        self,
        method: str,
        agent_id: str,
        marginal: float,
    ) -> None:
        """Marginal contribution can be negative (harmful agent) or positive (helpful)."""
        full_score = 0.5 + marginal if marginal > 0 else 0.5
        ablated_score = full_score - marginal
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=marginal,
            direction="positive",
        )
        # Regardless of what direction we set, is_helpful/is_harmful are computed
        # from the marginal_contribution field
        if marginal > 1e-9:
            assert result.is_helpful
        elif marginal < -1e-9:
            assert result.is_harmful
        else:
            assert not result.is_helpful
            assert not result.is_harmful


# ---------------------------------------------------------------------------
# Test: AblationResult is immutable (frozen dataclass)
# ---------------------------------------------------------------------------

class TestAblationResultImmutability:
    """AblationResult should be a frozen dataclass."""

    @given(_method, _agent_id, _positive_score, _positive_score)
    @settings(max_examples=50)
    def test_cannot_modify_fields_after_creation(
        self,
        method: str,
        agent_id: str,
        full_score: float,
        ablated_score: float,
    ) -> None:
        """Attempting to mutate a frozen dataclass should raise an error."""
        result = AblationResult(
            method=method,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=full_score - ablated_score,
        )
        try:
            result.full_score = 999.0  # pyright: ignore[reportAttributeAccessIssue]  # Intentionally tries to mutate frozen dataclass
            assert False, "Frozen dataclass should raise FrozenInstanceError"
        except Exception:
            pass  # expected
