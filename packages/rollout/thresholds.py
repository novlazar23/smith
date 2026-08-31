"""Configurable thresholds for phased rollout evaluation.

Defines the metric gates that drive promotion, demotion, and kill-switch
decisions across all rollout phases (SHADOW → PAPER → LIVE_SMALL →
LIVE_MEDIUM → LIVE_FULL).

All values live in code and can be overridden at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RolloutThresholds:
    """All configurable thresholds for the PhasedRolloutController.

    Parameters are chosen as sensible defaults aligned to the rollout
    progression: capital_ramp_pct maps to 1 %, 5 %, 25 %, 50 %, 100 %.
    """

    # ── Capital ramp per phase (fraction of total authorised capital) ──
    capital_ramp_pct: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 1.00)

    # ── Shadow → PAPER ──
    shadow_duration_pct: float = 0.25  # at least 25 % of planned shadow window

    # ── Model-quality gates ──
    min_brier_score: float = 0.30  # mean Brier score must be _at or below_ this

    # ── Risk gates ──
    max_drawdown_pct: float = 0.05  # hard kill-switch threshold (5 %)
    max_spread_anomaly_ratio: float = 2.5  # bid-ask spread / baseline

    # ── Circuit-breaker triggers ──
    max_exchange_error_rate: float = 0.10  # 10 % errors opens the circuit

    # ── Exchange-error tracking ──
    exchange_error_window_seconds: float = 300.0  # 5-minute sliding window

    # ── Manual kill-switch flag ──
    manual_kill_enabled: bool = True

    # ── Optional: overrideable per-phase (resolved by RolloutPhase index) ──
    _capital_ramp: tuple[float, ...] = field(default=capital_ramp_pct, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_capital_ramp", self.capital_ramp_pct)

    @property
    def capital_ramp(self) -> tuple[float, ...]:
        """Read-only access to the capital ramp values."""
        return self._capital_ramp
