"""Uncertainty Package — Entropy, Confidence, Elicitation.

Quantifiziert Unsicherheit in Agenten-Berichten mittels Shannon-Entropie,
Bayesianischen Konfidenzintervallen und Experten-Elicitation.
"""

from __future__ import annotations

from .confidence import (
    BayesianConfidence,
    bootstrap_confidence_interval,
)
from .elicitation import (
    prior_from_confidence,
    prior_from_historical,
)
from .entropy import entropy_score, normalized_entropy

__all__ = [
    "BayesianConfidence",
    "bootstrap_confidence_interval",
    "entropy_score",
    "normalized_entropy",
    "prior_from_confidence",
    "prior_from_historical",
]
