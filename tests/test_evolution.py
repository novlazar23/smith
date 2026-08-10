from trading_harness.models import ChallengerEvaluation
from trading_harness.services.evolution import PromotionPolicy

POLICY = {
    "promotion": {
        "minimum_observations": 100,
        "relative_improvement_min": 0.03,
        "require_out_of_sample": True,
        "require_walk_forward": True,
        "require_shadow_mode": True,
        "require_positive_ensemble_contribution": True,
        "require_security_pass": True,
    }
}


def item(**updates):
    values = {
        "challenger_id": "tech-2",
        "incumbent_id": "tech-1",
        "category": "technical",
        "incumbent_category": "technical",
        "observations": 200,
        "incumbent_score": 0.70,
        "challenger_score": 0.75,
        "out_of_sample_pass": True,
        "walk_forward_pass": True,
        "shadow_pass": True,
        "ensemble_contribution": 0.01,
        "security_pass": True,
    }
    values.update(updates)
    return ChallengerEvaluation(**values)


def test_good_challenger_promotes():
    result = PromotionPolicy(POLICY).evaluate(item())
    assert result.promote is True


def test_category_mismatch_rejects():
    result = PromotionPolicy(POLICY).evaluate(item(incumbent_category="elliott"))
    assert result.promote is False
    assert result.reason == "CATEGORY_MISMATCH"


def test_insufficient_observations_rejects():
    result = PromotionPolicy(POLICY).evaluate(item(observations=50))
    assert result.promote is False
    assert result.reason == "INSUFFICIENT_OBSERVATIONS"


def test_negative_ensemble_contribution_rejects():
    result = PromotionPolicy(POLICY).evaluate(item(ensemble_contribution=-0.01))
    assert result.promote is False
    assert result.reason == "NEGATIVE_ENSEMBLE_CONTRIBUTION"


def test_small_improvement_rejects():
    result = PromotionPolicy(POLICY).evaluate(item(challenger_score=0.71))
    assert result.promote is False
    assert result.reason == "PROMOTION_MARGIN_NOT_MET"
