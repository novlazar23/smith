from __future__ import annotations

from trading_harness.models import ChallengerEvaluation, PromotionDecision


class PromotionPolicy:
    def __init__(self, policy: dict):
        self.policy = policy
        self.promotion = policy["promotion"]

    def evaluate(self, item: ChallengerEvaluation) -> PromotionDecision:
        if item.category != item.incumbent_category:
            return PromotionDecision(promote=False, reason="CATEGORY_MISMATCH")

        minimum_observations = int(self.promotion["minimum_observations"])
        if item.observations < minimum_observations:
            return PromotionDecision(promote=False, reason="INSUFFICIENT_OBSERVATIONS")

        if self.promotion.get("require_out_of_sample", True) and not item.out_of_sample_pass:
            return PromotionDecision(promote=False, reason="OUT_OF_SAMPLE_FAILED")

        if self.promotion.get("require_walk_forward", True) and not item.walk_forward_pass:
            return PromotionDecision(promote=False, reason="WALK_FORWARD_FAILED")

        if self.promotion.get("require_shadow_mode", True) and not item.shadow_pass:
            return PromotionDecision(promote=False, reason="SHADOW_MODE_FAILED")

        if self.promotion.get("require_security_pass", True) and not item.security_pass:
            return PromotionDecision(promote=False, reason="SECURITY_FAILED")

        if (
            self.promotion.get("require_positive_ensemble_contribution", True)
            and item.ensemble_contribution < 0
        ):
            return PromotionDecision(promote=False, reason="NEGATIVE_ENSEMBLE_CONTRIBUTION")

        denominator = abs(item.incumbent_score)
        if denominator < 1e-12:
            relative_improvement = item.challenger_score - item.incumbent_score
        else:
            relative_improvement = (
                item.challenger_score - item.incumbent_score
            ) / denominator

        minimum_improvement = float(self.promotion["relative_improvement_min"])
        if relative_improvement < minimum_improvement:
            return PromotionDecision(
                promote=False,
                reason="PROMOTION_MARGIN_NOT_MET",
                relative_improvement=relative_improvement,
            )

        return PromotionDecision(
            promote=True,
            reason="PROMOTION_APPROVED",
            relative_improvement=relative_improvement,
        )
