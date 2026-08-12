"""Bayesian optimization for backtesting parameter tuning.

Provides a BayesianOptimizer stub with a random-search fallback.
Designed as a drop-in placeholder for future hyperopt / scikit-optimize
integration (EPIC-14 WP05, optional).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ─── Result Data Structure ──────────────────────────────────────────────────


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable result of an optimization run."""

    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]] = field(default_factory=list)


# ─── Optimizer ──────────────────────────────────────────────────────────────


class BayesianOptimizer:
    """Parameter optimizer with Bayesian optimization intent.

    Currently implements random search as a practical fallback.
    The interface is designed for a future scikit-optimize or hyperopt
    backend without requiring callers to change.

    Parameters
    ----------
    param_grid : dict
        Mapping of parameter names to iterables of candidate values,
        e.g. ``{"window_short": [5, 10, 20], "threshold": [0.1, 0.5, 0.9]}``.
    n_iterations : int
        Maximum number of sampling iterations (default 20).
    random_state : int
        Seed for reproducibility (default 42).
    """

    def __init__(
        self,
        param_grid: dict[str, Any],
        n_iterations: int = 20,
        random_state: int = 42,
    ) -> None:
        self.param_grid = param_grid
        self.n_iterations = n_iterations
        self._rng = random.Random(random_state)

    def optimize(
        self,
        objective_function: Callable[[dict[str, Any]], float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimization and return ``{"best_params": ..., "best_score": ...}``.

        Parameters
        ----------
        objective_function : callable
            A function that accepts a ``dict`` of parameter values and
            returns a scalar score to maximise (higher is better).
        **kwargs
            Extra keyword arguments forwarded to ``objective_function``.

        Returns
        -------
        dict
            ``{"best_params": <dict>, "best_score": <float>}``
        """
        results: list[dict[str, Any]] = []
        best_params: dict[str, Any] = {}
        best_score: float = float("-inf")

        for _ in range(self.n_iterations):
            params = self._sample_params()
            score = objective_function(params, **kwargs)
            entry = {"params": params, "score": score}
            results.append(entry)
            if score > best_score:
                best_score = score
                best_params = params

        return {"best_params": best_params, "best_score": best_score}

    def _sample_params(self) -> dict[str, Any]:
        """Draw one random parameter combination from the grid."""
        return {
            key: self._rng.choice(values)
            for key, values in self.param_grid.items()
        }
