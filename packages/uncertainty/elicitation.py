"""Expert elicitation for prior distributions.

Provides helpers to create Beta distribution priors from
human-interpretable parameters like confidence scores
and historical accuracy rates.
"""

from __future__ import annotations


def prior_from_confidence(
    confidence: float,
    strength: float = 1.0,
) -> tuple[float, float]:
    """Create Beta(alpha, beta) prior from confidence score.

    Args:
        confidence: Confidence score in [0.0, 1.0].
        strength: Effective sample size (higher = more confident prior).

    Returns:
        Tuple of (alpha, beta) for Beta distribution.

    Raises:
        ValueError: If confidence is out of range.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    if strength <= 0:
        raise ValueError(f"strength must be positive, got {strength}")

    alpha = max(confidence * strength + 1.0, 1.001)
    beta = max((1 - confidence) * strength + 1.0, 1.001)

    return (alpha, beta)


def prior_from_historical(
    accuracy: float,
    n_observed: int,
) -> tuple[float, float]:
    """Create Beta prior from historical accuracy rate.

    Args:
        accuracy: Historical accuracy rate in [0.0, 1.0].
        n_observed: Number of past observations.

    Returns:
        Tuple of (alpha, beta) for Beta distribution.

    Raises:
        ValueError: If accuracy is out of range or n_observed <= 0.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
    if n_observed <= 0:
        raise ValueError(f"n_observed must be positive, got {n_observed}")

    alpha = accuracy * n_observed + 1.0
    beta = (1 - accuracy) * n_observed + 1.0

    return (max(alpha, 1.001), max(beta, 1.001))
