"""Shannon entropy for probability distributions.

Used to measure uncertainty in agent probability outputs.
"""

from __future__ import annotations

import math


def entropy_score(probabilities: dict[str, float]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Higher entropy means higher uncertainty (more uniform distribution).

    Args:
        probabilities: Dict of category -> probability.

    Returns:
        Shannon entropy in bits.
    """
    total = sum(probabilities.values())
    if total == 0:
        return 0.0

    probs = [p / total for p in probabilities.values() if p > 0]
    if not probs:
        return 0.0

    return -sum(p * math.log2(p) for p in probs)


def normalized_entropy(
    probabilities: dict[str, float],
) -> float:
    """Compute normalized entropy in [0.0, 1.0].

    0.0 = certainty (one category has all probability)
    1.0 = maximum uncertainty (uniform distribution)

    Args:
        probabilities: Dict of category -> probability.

    Returns:
        Normalized entropy in [0.0, 1.0].
    """
    total = sum(probabilities.values())
    if total == 0:
        return 0.0

    probs = [p / total for p in probabilities.values() if p > 0]
    n = len(probs)
    if n <= 1:
        return 0.0

    max_entropy = math.log2(n)
    if max_entropy == 0:
        return 0.0

    return entropy_score(probabilities) / max_entropy
