"""Kill switch — immediate halt mechanism for the rollout controller.

Supports both manual activation (via config / API) and automatic
activation when risk thresholds are breached (drawdown, spread anomaly,
exchange error rate).  On activation the controller must cancel all
open orders and transition the system to a halted state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class KillSwitchState:
    """States of the kill switch."""

    DISABLED = "disabled"
    ACTIVATED = "activated"


@dataclass
class KillSwitch:
    """Immediate-stop mechanism for the Phased Rollout Controller.

    Parameters
    ----------
    max_drawdown_pct :
        Hard drawdown limit that triggers automatic activation.
    max_spread_anomaly_ratio :
        Bid-ask spread / baseline threshold that triggers automatic
        activation.
    max_exchange_error_rate :
        Exchange error-rate threshold (same as CircuitBreaker config)
        that triggers automatic activation.
    manual_enabled :
        Whether the kill switch can be activated manually.

    Notes
    -----
    The ``on_cancel_orders`` callback is invoked automatically on
    activation.  Callers must implement order cancellation logic there.
    """

    max_drawdown_pct: float = 0.05
    max_spread_anomaly_ratio: float = 2.5
    max_exchange_error_rate: float = 0.10
    manual_enabled: bool = True

    _state: str = field(default=KillSwitchState.DISABLED, init=False)
    _reason: str = field(default="", init=False)
    _activated_at: float = field(default=0.0, init=False)
    _on_cancel_orders: object | None = field(default=None, init=False)

    # ── Public state ──

    @property
    def state(self) -> str:
        """Current kill switch state ('disabled' or 'activated')."""
        return self._state

    @property
    def reason(self) -> str:
        """Reason the kill switch was activated (empty if disabled)."""
        return self._reason

    @property
    def activated_at(self) -> float:
        """Monotonic timestamp of activation (0.0 if not activated)."""
        return self._activated_at

    # ── Core API ──

    def activate(self, reason: str) -> None:
        """Activate the kill switch immediately.

        Parameters
        ----------
        reason :
            Human-readable explanation for why the switch was activated.
        """
        if self._state == KillSwitchState.ACTIVATED:
            logger.warning("kill_switch: already activated (reason=%s)", self._reason)
            return

        self._state = KillSwitchState.ACTIVATED
        self._reason = reason
        self._activated_at = _monotonic_now()

        logger.critical("kill_switch ACTIVATED — reason=%s", reason)

        # Invoke the order cancellation callback if registered.
        if self._on_cancel_orders is not None:
            try:
                cancel_fn = self._on_cancel_orders
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    getattr(cancel_fn, "cancel_open_orders", lambda: None)()
            except Exception:
                logger.exception("kill_switch: order cancellation callback failed")

    def check_and_activate(
        self,
        *,
        current_drawdown_pct: float = 0.0,
        current_spread_ratio: float = 0.0,
        current_error_rate: float = 0.0,
    ) -> bool:
        """Evaluate all automatic triggers and activate if any fire.

        Returns *True* if the kill switch was (or already was) activated
        as a result of this check.
        """
        # ── Drawdown gate ──
        if current_drawdown_pct >= self.max_drawdown_pct:
            self.activate(
                f"drawdown {current_drawdown_pct:.2%} >= "
                f"limit {self.max_drawdown_pct:.2%}"
            )
            return True

        # ── Spread anomaly gate ──
        if current_spread_ratio >= self.max_spread_anomaly_ratio:
            self.activate(
                f"spread ratio {current_spread_ratio:.2f} >= "
                f"limit {self.max_spread_anomaly_ratio:.2f}"
            )
            return True

        # ── Exchange error rate gate ──
        if current_error_rate >= self.max_exchange_error_rate:
            self.activate(
                f"exchange error rate {current_error_rate:.2%} >= "
                f"limit {self.max_exchange_error_rate:.2%}"
            )
            return True

        return False

    def deactivate(self) -> None:
        """Reset the kill switch to disabled (manual action only)."""
        if self._state == KillSwitchState.DISABLED:
            return
        logger.warning("kill_switch: manually deactivated (was reason=%s)", self._reason)
        self._state = KillSwitchState.DISABLED
        self._reason = ""
        self._activated_at = 0.0

    def register_cancel_callback(self, fn: object) -> None:
        """Register a callback for cancelling open orders on activation.

        Parameters
        ----------
        fn :
            Either a callable with no arguments or an object with a
            ``cancel_open_orders()`` method.
        """
        self._on_cancel_orders = fn

    # ── Diagnostics ──

    def status(self) -> dict:
        """Return a snapshot of the kill switch state."""
        return {
            "state": self._state,
            "reason": self._reason,
            "activated_at": self._activated_at,
        }


def _monotonic_now() -> float:
    """Return the current monotonic clock value in seconds."""
    import time
    return time.monotonic()
