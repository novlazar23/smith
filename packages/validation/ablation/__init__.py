"""Ablation Analysis for Agent Evaluation.

Measures each agent's marginal contribution through:
- Leave-One-Out (LOO): remove one agent, measure performance drop
- Feature Importance: measure feature contribution via permutation
- Marginal Brier Score: measure per-agent Brier score improvement
"""

from __future__ import annotations

from .base import AblationAnalyzer, AblationResult
from .feature_importance import FeatureImportanceAnalyzer
from .loo import LeaveOneOutAblation
from .marginal_brier import MarginalBrierAnalyzer

__all__ = [
    "AblationAnalyzer",
    "AblationResult",
    "FeatureImportanceAnalyzer",
    "LeaveOneOutAblation",
    "MarginalBrierAnalyzer",
]
