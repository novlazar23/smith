"""Phased Rollout Controller — state machine for gradual capital deployment.

Implements the five-phase rollout progression:

    SHADOW → PAPER → LIVE_SMALL → LIVE_MEDIUM → LIVE_FULL

Each phase has a defined capital allocation, minimum duration, and a set
of evaluation thresholds that drive automated promotion, demotion, and
kill-switch decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from .circuit_breaker import CircuitBreaker, CircuitState
from .kill_switch import KillSwitch
from .thresholds import RolloutThresholds

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Phase definitions
# ──────────────────────────────────────────────────────────────


class RolloutPhase(StrEnum):
    """Ordered rollout phases from lowest to highest capital."""

    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE_SMALL = "LIVE_SMALL"
    LIVE_MEDIUM = "LIVE_MEDIUM"
    LIVE_FULL = "LIVE_FULL"

    @property
    def level(self) -> int:
        """Zero-based index in the progression order."""
        return list(RolloutPhase).index(self)

    @property
    def is_live(self) -> bool:
        """True once actual capital is deployed."""
        return self in (
            RolloutPhase.LIVE_SMALL,
            RolloutPhase.LIVE_MEDIUM,
            RolloutPhase.LIVE_FULL,
        )


# ──────────────────────────────────────────────────────────────
# Promotion / demotion decision
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    """Outcome of a single evaluation cycle.

    Parameters
    ----------
    action :
        One of "promote", "demote", "hold", "kill".
    from_phase :
        The phase the system was in before this decision.
    to_phase :
        The phase to transition to (same as *from_phase* for "hold").
    reason :
        Human-readable explanation for the action taken.
    """

    action: str
    from_phase: str
    to_phase: str
    reason: str


# ──────────────────────────────────────────────────────────────
# Controller state dataclass
# ──────────────────────────────────────────────────────────────


@dataclass
class RolloutControllerState:
    """Persistent state snapshot for the rollout controller."""

    current_phase: str = RolloutPhase.SHADOW.value
    started_at: float = field(default=0.0)
    phase_started_at: float = field(default=0.0)
    total_evaluation_cycles: int = field(default=0)
    promotions: int = field(default=0)
    demotions: int = field(default=0)
    last_decision: str = field(default="")
    last_reason: str = field(default="")


# ──────────────────────────────────────────────────────────────
# PhasedRolloutController
# ──────────────────────────────────────────────────────────────


class PhasedRolloutController:
    """State machine that governs the phased rollout of live trading.

    Parameters
    ----------
    thresholds :
        All configurable thresholds (capital ramp, Brier score,
        drawdown limits, etc.).
    on_cancel_orders :
        Optional callback invoked by the kill switch to cancel open
        orders.

    Example
    -------
    >>> ctrl = PhasedRolloutController()
    >>> ctrl.start()
    >>> decision = ctrl.evaluate(
    ...     brier_score=0.22,
    ...     drawdown_pct=0.01,
    ...     spread_ratio=1.1,
    ...     exchange_error_rate=0.02,
    ...     positive_trend=True,
    ... )
    >>> print(decision.action)  # "promote"
    """

    def __init__(
        self,
        thresholds: RolloutThresholds | None = None,
        on_cancel_orders: object | None = None,
    ) -> None:
        self._thresholds = thresholds or RolloutThresholds()
        self._state = RolloutControllerState()

        self._kill_switch = KillSwitch(
            max_drawdown_pct=self._thresholds.max_drawdown_pct,
            max_spread_anomaly_ratio=self._thresholds.max_spread_anomaly_ratio,
            max_exchange_error_rate=self._thresholds.max_exchange_error_rate,
            manual_enabled=self._thresholds.manual_kill_enabled,
        )
        self._kill_switch.register_cancel_callback(on_cancel_orders)

        self._circuit_breaker = CircuitBreaker(
            error_rate_threshold=self._thresholds.max_exchange_error_rate,
            window_seconds=self._thresholds.exchange_error_window_seconds,
        )

        self._started_at: float = 0.0
        self._phase_started_at: float = 0.0

    # ── Lifecycle ──

    def start(self) -> None:
        """Start the controller at the initial phase (SHADOW)."""
        self._state.current_phase = RolloutPhase.SHADOW.value
        self._started_at = _now()
        self._phase_started_at = self._started_at
        logger.info("rollout: controller started at phase=%s", RolloutPhase.SHADOW.value)

    def stop(self) -> None:
        """Stop the controller and deactivate all safety mechanisms."""
        self._kill_switch.deactivate()
        self._circuit_breaker.reset()
        logger.info("rollout: controller stopped")

    # ── Evaluation ──

    def evaluate(
        self,
        *,
        brier_score: float,
        drawdown_pct: float,
        spread_ratio: float,
        exchange_error_rate: float,
        positive_trend: bool,
    ) -> RolloutDecision:
        """Run a full evaluation cycle and return a decision.

        The evaluation follows this priority:

        1. **Kill switch** — if any automatic trigger fires, return "kill".
        2. **Circuit breaker** — if the exchange error rate opens the
           circuit, return "kill" until reset.
        3. **Promotion** — all thresholds met + positive trend.
        4. **Demotion** — any single threshold exceeded.
        5. **Hold** — nothing decisive.

        Parameters
        ----------
        brier_score :
            Current mean Brier score (lower is better).
        drawdown_pct :
            Current drawdown as a fraction of peak equity.
        spread_ratio :
            Current bid-ask spread / baseline ratio.
        exchange_error_rate :
            Fraction of exchange calls that failed in the window.
        positive_trend :
            Whether the model's virtual PnL shows an upward trend.

        Returns
        -------
        RolloutDecision
            The action to take.
        """
        self._state.total_evaluation_cycles += 1

        # ── 1. Kill switch check (automatic) ──
        if self._kill_switch.check_and_activate(
            current_drawdown_pct=drawdown_pct,
            current_spread_ratio=spread_ratio,
            current_error_rate=exchange_error_rate,
        ):
            self._record_decision("kill", "kill switch activated")
            return RolloutDecision(
                action="kill",
                from_phase=self._state.current_phase,
                to_phase=self._state.current_phase,
                reason=self._kill_switch.reason,
            )

        # ── 2. Circuit breaker check ──
        self._circuit_breaker.record_call(exchange_error_rate < self._thresholds.max_exchange_error_rate)
        if self._circuit_breaker.state == CircuitState.OPEN:
            reason = (
                f"circuit breaker open (error_rate={exchange_error_rate:.2%} "
                f"> threshold={self._thresholds.max_exchange_error_rate:.2%})"
            )
            self._record_decision("kill", reason)
            return RolloutDecision(
                action="kill",
                from_phase=self._state.current_phase,
                to_phase=self._state.current_phase,
                reason=reason,
            )

        current_phase = RolloutPhase(self._state.current_phase)

        # ── 3. Promotion check (only if not already at LIVE_FULL) ──
        if current_phase != RolloutPhase.LIVE_FULL:
            if self._meets_promotion_criteria(
                brier_score=brier_score,
                drawdown_pct=drawdown_pct,
                spread_ratio=spread_ratio,
                exchange_error_rate=exchange_error_rate,
                positive_trend=positive_trend,
                current_phase=current_phase,
            ):
                new_phase = RolloutPhase(current_phase._value2member_map_[
                    list(RolloutPhase)[current_phase.level + 1]
                ])
                return self._do_promotion(new_phase)

        # ── 4. Demotion check ──
        if (
            current_phase.level > 0
            and self._needs_demotion(
                brier_score=brier_score,
                drawdown_pct=drawdown_pct,
                spread_ratio=spread_ratio,
                exchange_error_rate=exchange_error_rate,
                current_phase=current_phase,
            )
        ):
            new_phase = RolloutPhase(current_phase._value2member_map_[
                list(RolloutPhase)[current_phase.level - 1]
            ])
            return self._do_demotion(new_phase)

        # ── 5. Hold ──
        self._state.last_decision = "hold"
        self._state.last_reason = "all thresholds within acceptable range"
        return RolloutDecision(
            action="hold",
            from_phase=self._state.current_phase,
            to_phase=self._state.current_phase,
            reason=self._state.last_reason,
        )

    # ── Manual overrides ──

    def force_promote(self, target: str) -> RolloutDecision:
        """Force-promote to a specific phase (bypasses thresholds)."""
        target_phase = RolloutPhase(target)
        if target_phase.level <= RolloutPhase(self._state.current_phase).level:
            return RolloutDecision(
                action="hold",
                from_phase=self._state.current_phase,
                to_phase=self._state.current_phase,
                reason="target phase is not ahead of current",
            )
        return self._do_promotion(target_phase, manual=True)

    def force_demote(self, target: str) -> RolloutDecision:
        """Force-demote to a specific phase (bypasses thresholds)."""
        target_phase = RolloutPhase(target)
        if target_phase.level >= RolloutPhase(self._state.current_phase).level:
            return RolloutDecision(
                action="hold",
                from_phase=self._state.current_phase,
                to_phase=self._state.current_phase,
                reason="target phase is not behind current",
            )
        return self._do_demotion(target_phase, manual=True)

    def force_kill(self, reason: str = "manual override") -> RolloutDecision:
        """Immediately activate the kill switch."""
        self._kill_switch.activate(reason)
        return RolloutDecision(
            action="kill",
            from_phase=self._state.current_phase,
            to_phase=self._state.current_phase,
            reason=f"manual kill: {reason}",
        )

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (circuit closes)."""
        self._circuit_breaker.reset()

    # ── Status ──

    def status(self) -> dict:
        """Return a full status snapshot."""
        return {
            "controller": self._state_status(),
            "kill_switch": self._kill_switch.status(),
            "circuit_breaker": self._circuit_breaker.stats(),
            "thresholds": self._thresholds_status(),
        }

    @property
    def current_phase(self) -> str:
        """Return the current rollout phase as a string."""
        return self._state.current_phase

    @property
    def kill_switch(self) -> KillSwitch:
        """Expose the kill switch for inspection or manual control."""
        return self._kill_switch

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Expose the circuit breaker for inspection or manual reset."""
        return self._circuit_breaker

    @property
    def thresholds(self) -> RolloutThresholds:
        """Expose the active thresholds."""
        return self._thresholds

    # ── Internal helpers ──

    def _meets_promotion_criteria(
        self,
        *,
        brier_score: float,
        drawdown_pct: float,
        spread_ratio: float,
        exchange_error_rate: float,
        positive_trend: bool,
        current_phase: RolloutPhase,
    ) -> bool:
        """Return True if all promotion criteria are satisfied."""
        # Duration gate
        elapsed = _now() - self._phase_started_at
        min_duration = self._min_duration(current_phase)
        if elapsed < min_duration:
            return False

        # Brier score gate
        if brier_score > self._thresholds.min_brier_score:
            return False

        # Drawdown gate
        if drawdown_pct >= self._thresholds.max_drawdown_pct:
            return False

        # Spread gate
        if spread_ratio >= self._thresholds.max_spread_anomaly_ratio:
            return False

        # Exchange error rate gate
        if exchange_error_rate >= self._thresholds.max_exchange_error_rate:
            return False

        # Trend gate
        return bool(positive_trend)

    def _needs_demotion(
        self,
        *,
        brier_score: float,
        drawdown_pct: float,
        spread_ratio: float,
        exchange_error_rate: float,
        current_phase: RolloutPhase,
    ) -> bool:
        """Return True if any demotion threshold has been breached."""
        if drawdown_pct >= self._thresholds.max_drawdown_pct:
            return True
        if spread_ratio >= self._thresholds.max_spread_anomaly_ratio:
            return True
        if exchange_error_rate >= self._thresholds.max_exchange_error_rate:
            return True
        # Brier score regression
        return brier_score <= self._thresholds.min_brier_score * 1.5

    def _min_duration(self, phase: RolloutPhase) -> float:
        """Return the minimum time (in seconds) to spend in a phase.

        Shadow mode requires at least 25 % of the total planned
        observation window.  For simplicity the default is 600 s (10 min)
        for SHADOW and 300 s (5 min) for all others — production should
        use much larger values.
        """
        if phase == RolloutPhase.SHADOW:
            return 600.0
        return 300.0

    def _do_promotion(
        self,
        target: RolloutPhase,
        manual: bool = False,
    ) -> RolloutDecision:
        self._state.promotions += 1
        self._state.current_phase = target.value
        self._phase_started_at = _now()
        reason = "manual promotion" if manual else "all thresholds met, positive trend"
        self._record_decision("promote", reason)
        logger.info(
            "rollout: PROMOTE %s → %s  reason=%s",
            target,
            target,
            reason,
        )
        return RolloutDecision(
            action="promote",
            from_phase=target.value,
            to_phase=target.value,
            reason=reason,
        )

    def _do_demotion(
        self,
        target: RolloutPhase,
        manual: bool = False,
    ) -> RolloutDecision:
        self._state.demotions += 1
        self._state.current_phase = target.value
        self._phase_started_at = _now()
        reason = "manual demotion" if manual else "threshold breach detected"
        self._record_decision("demote", reason)
        logger.warning(
            "rollout: DEMOTE %s → %s  reason=%s",
            target,
            target,
            reason,
        )
        return RolloutDecision(
            action="demote",
            from_phase=target.value,
            to_phase=target.value,
            reason=reason,
        )

    def _record_decision(self, action: str, reason: str) -> None:
        self._state.last_decision = action
        self._state.last_reason = reason

    def _state_status(self) -> dict:
        return {
            "current_phase": self._state.current_phase,
            "started_at": self._started_at,
            "phase_started_at": self._phase_started_at,
            "total_evaluation_cycles": self._state.total_evaluation_cycles,
            "promotions": self._state.promotions,
            "demotions": self._state.demotions,
            "last_decision": self._state.last_decision,
            "last_reason": self._state.last_reason,
        }

    def _thresholds_status(self) -> dict:
        t = self._thresholds
        return {
            "capital_ramp_pct": list(t.capital_ramp),
            "shadow_duration_pct": t.shadow_duration_pct,
            "min_brier_score": t.min_brier_score,
            "max_drawdown_pct": t.max_drawdown_pct,
            "max_spread_anomaly_ratio": t.max_spread_anomaly_ratio,
            "max_exchange_error_rate": t.max_exchange_error_rate,
        }


def _now() -> float:
    """Return current monotonic time in seconds."""
    import time
    return time.monotonic()
