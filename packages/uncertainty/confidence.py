"""Bayesian confidence intervals and bootstrap methods."""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats as scipy_stats


@dataclass(frozen=True, slots=True)
class BayesianConfidence:
    """Bayesian confidence interval from Beta distribution.

    Attributes:
        mean: Posterior mean.
        lower: Lower bound of confidence interval.
        upper: Upper bound of confidence interval.
        width: Width of the confidence interval.
    """

    mean: float
    lower: float
    upper: float
    width: float

    @classmethod
    def from_alpha_beta(
        cls,
        alpha: float,
        beta: float,
        confidence_level: float = 0.95,
    ) -> BayesianConfidence:
        """Create from Beta distribution parameters.

        Args:
            alpha: Alpha parameter (successes + 1).
            beta: Beta parameter (failures + 1).
            confidence_level: Desired confidence level.

        Returns:
            BayesianConfidence with credible interval.
        """
        if alpha <= 0 or beta <= 0:
            return cls(mean=0.5, lower=0.0, upper=1.0, width=1.0)

        alpha_corrected = max(alpha, 1.0)
        beta_corrected = max(beta, 1.0)

        mean = alpha_corrected / (alpha_corrected + beta_corrected)
        alpha_ci = alpha_corrected - 1.0
        beta_ci = beta_corrected - 1.0

        if alpha_ci <= 0 or beta_ci <= 0:
            return cls(mean=mean, lower=0.0, upper=1.0, width=1.0)

        lower = scipy_stats.beta.ppf(
            (1 - confidence_level) / 2, alpha_ci, beta_ci
        )
        upper = scipy_stats.beta.ppf(
            1 - (1 - confidence_level) / 2, alpha_ci, beta_ci
        )

        return cls(
            mean=mean,
            lower=max(lower, 0.0),
            upper=min(upper, 1.0),
            width=upper - lower,
        )


def bootstrap_confidence_interval(
    samples: list[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> BayesianConfidence:
    """Compute bootstrap confidence interval for a sample mean.

    Args:
        samples: List of observed values.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Desired confidence level.

    Returns:
        BayesianConfidence with bootstrap interval.

    Raises:
        ValueError: If samples is empty.
    """
    if not samples:
        raise ValueError("samples must not be empty")

    n = len(samples)
    means: list[float] = []
    rng_state = 42  # deterministic
    import random
    rng = random.Random(rng_state)

    for _ in range(n_bootstrap):
        bootstrap_sample = [rng.choice(samples) for _ in range(n)]
        means.append(sum(bootstrap_sample) / n)

    means.sort()
    lower_idx = int(n_bootstrap * (1 - confidence_level) / 2)
    upper_idx = int(n_bootstrap * (1 - confidence_level / 2))

    lower = means[lower_idx]
    upper = means[upper_idx]
    mean_val = sum(means) / len(means)

    return BayesianConfidence(
        mean=mean_val,
        lower=lower,
        upper=upper,
        width=upper - lower,
    )
