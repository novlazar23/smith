"""Pre-submission order validator.

Before any order reaches the exchange, it passes through a pipeline of
validation checks.  Each check is independent and returns a list of
validation errors.

Checks
------
1. **Size validation** — order quantity must be positive and within
   configurable bounds.
2. **Price validation** — limit orders must have a valid price; stop
   orders must have a stop price.
3. **Account balance / equity check** — estimated notional must not
   exceed a configured fraction of account equity.
4. **Risk gates** — composite risk decision from the existing
   ``packages.risk`` module.

Usage
-----

.. code-block:: python

    validator = OrderValidator(
        min_order_size=0.001,
        max_order_size=10.0,
        max_notional_ratio=0.1,
        account_equity=50000.0,
    )

    errors = await validator.validate(
        symbol="BTC/USDT",
        venue="binance",
        side="buy",
        quantity=0.5,
        price=45000.0,
        order_type="limit",
    )
    if errors:
        for err in errors:
            print(f"Rejected: {err}")
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─── Validation Error ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ValidationError:
    """A single validation failure.

    Attributes:
        code: Machine-readable error code.
        message: Human-readable description.
        severity: ``"hard"`` blocks submission; ``"soft"`` logs a warning.
        field: The field that failed validation (optional).
    """

    code: str
    message: str
    severity: str = "hard"
    field: str | None = None

    @property
    def is_blocking(self) -> bool:
        """Return ``True`` if this error blocks submission."""
        return self.severity == "hard"


# ─── Risk Gate Result ───────────────────────────────────────────────────────


@dataclass
class RiskGateResult:
    """Simplified risk gate decision.

    Attributes:
        approved: ``True`` if the order passes all risk gates.
        blocking_reasons: List of blocking reasons if not approved.
    """

    approved: bool = True
    blocking_reasons: list[str] = field(default_factory=list)


# ─── Validator Config ───────────────────────────────────────────────────────


@dataclass
class ValidationConfig:
    """Configuration for order validation.

    Args:
        min_order_size: Minimum allowed order quantity.
        max_order_size: Maximum allowed order quantity.
        min_price: Minimum allowed limit price (``0`` for market orders).
        max_price: Maximum allowed limit price.
        max_notional_ratio: Maximum order notional as fraction of equity.
        account_equity: Account equity for notional checks.
        require_stop_price: Whether stop orders must have a stop price.
    """

    min_order_size: float = 0.001
    max_order_size: float = 100.0
    min_price: float = 0.0
    max_price: float = 10_000_000.0
    max_notional_ratio: float = 0.1
    account_equity: float = 50_000.0
    require_stop_price: bool = True


# ─── Validator ──────────────────────────────────────────────────────────────


class OrderValidator:
    """Pre-submission order validator.

    Validates size, price, account balance, and risk gates before an order
    is submitted to a venue.

    Args:
        config: Validation configuration.  A default is used if ``None``.
        risk_gate_fn: Optional callable that returns a
            :class:`RiskGateResult`.  If ``None``, no risk-gate check is performed.
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        risk_gate_fn: Any = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self._risk_gate_fn = risk_gate_fn

    # ── main entry point ─────────────────────────────────────────────────

    async def validate(
        self,
        symbol: str,
        venue: str,
        side: str,
        quantity: float,
        price: float | None,
        order_type: str = "limit",
        stop_price: float | None = None,
    ) -> list[ValidationError]:
        """Run the full validation pipeline.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            venue: Venue identifier.
            side: ``"buy"`` or ``"sell"``.
            quantity: Order quantity.
            price: Limit price (``None`` for market orders).
            order_type: Order type string.
            stop_price: Stop price for stop-limit / stop-market orders.

        Returns:
            List of ``ValidationError``s.  Empty list means the order passes.
        """
        errors: list[ValidationError] = []

        errors.extend(self._validate_size(quantity))
        errors.extend(
            self._validate_price(price, order_type, stop_price),
        )
        errors.extend(
            self._validate_notional(price, quantity, side),
        )
        errors.extend(
            await self._validate_risk(symbol, venue, side, quantity, price),
        )

        if errors:
            blocking = [e for e in errors if e.is_blocking]
            soft = [e for e in errors if not e.is_blocking]
            if blocking:
                logger.warning(
                    "Order validation FAILED for %s/%s %s %.4f @ %s: %d blocking errors",
                    symbol,
                    venue,
                    side,
                    quantity,
                    price,
                    len(blocking),
                )
            if soft:
                logger.info(
                    "Order validation soft-warnings for %s/%s: %d",
                    symbol,
                    venue,
                    len(soft),
                )

        return errors

    # ── individual check methods ─────────────────────────────────────────

    def _validate_size(self, quantity: float) -> list[ValidationError]:
        """Check order quantity against configured bounds."""
        errors: list[ValidationError] = []

        if quantity <= 0:
            errors.append(
                ValidationError(
                    code="INVALID_SIZE",
                    message=f"Order quantity must be positive, got {quantity}",
                    field="quantity",
                ),
            )
        elif quantity < self.config.min_order_size:
            errors.append(
                ValidationError(
                    code="BELOW_MIN_SIZE",
                    message=(
                        f"Order size {quantity} below minimum "
                        f"{self.config.min_order_size}"
                    ),
                    field="quantity",
                ),
            )
        elif quantity > self.config.max_order_size:
            errors.append(
                ValidationError(
                    code="ABOVE_MAX_SIZE",
                    message=(
                        f"Order size {quantity} exceeds maximum "
                        f"{self.config.max_order_size}"
                    ),
                    field="quantity",
                ),
            )

        return errors

    def _validate_price(
        self,
        price: float | None,
        order_type: str,
        stop_price: float | None,
    ) -> list[ValidationError]:
        """Check price for limit / stop orders."""
        errors: list[ValidationError] = []
        ot = order_type.lower()

        # Market orders don't need a price
        if ot == "market":
            return errors

        # Limit orders need a price
        if price is None:
            if "limit" in ot:
                errors.append(
                    ValidationError(
                        code="MISSING_PRICE",
                        message=f"{order_type} order requires a price",
                        field="price",
                    ),
                )
            return errors

        if price <= self.config.min_price:
            errors.append(
                ValidationError(
                    code="INVALID_PRICE",
                    message=f"Price {price} is at or below minimum {self.config.min_price}",
                    field="price",
                ),
            )
        elif price > self.config.max_price:
            errors.append(
                ValidationError(
                    code="PRICE_TOO_HIGH",
                    message=f"Price {price} exceeds maximum {self.config.max_price}",
                    field="price",
                ),
            )

        # Stop orders need a stop price
        if (
            ("stop" in ot or "stp" in ot)
            and self.config.require_stop_price
            and (stop_price is None or stop_price <= 0)
        ):
                errors.append(
                    ValidationError(
                        code="MISSING_STOP_PRICE",
                        message=f"{order_type} order requires a valid stop price",
                        field="stop_price",
                    ),
                )

        return errors

    def _validate_notional(
        self,
        price: float | None,
        quantity: float,
        side: str,
    ) -> list[ValidationError]:
        """Check estimated notional against account equity."""
        errors: list[ValidationError] = []

        if price is None:
            # Market order — skip notional check (unknown price)
            return errors

        notional = price * quantity
        max_notional = self.config.account_equity * self.config.max_notional_ratio

        if notional > max_notional:
            errors.append(
                ValidationError(
                    code="NOTIONAL_EXCEEDED",
                    message=(
                        f"Estimated notional {notional:.2f} exceeds "
                        f"max {max_notional:.2f} "
                        f"({self.config.max_notional_ratio * 100:.0f}% of "
                        f"equity {self.config.account_equity:.2f})"
                    ),
                    field="notional",
                ),
            )

        return errors

    async def _validate_risk(
        self,
        symbol: str,
        venue: str,
        side: str,
        quantity: float,
        price: float | None,
    ) -> list[ValidationError]:
        """Run the risk-gate pipeline if configured."""
        if self._risk_gate_fn is None:
            return []

        try:
            if callable(self._risk_gate_fn):
                result = self._risk_gate_fn(
                    symbol=symbol,
                    venue=venue,
                    side=side,
                    quantity=quantity,
                    price=price,
                )
                # Handle both sync and async risk gate functions
                if inspect.iscoroutine(result):
                    result = await result

                if isinstance(result, RiskGateResult):
                    if not result.approved:
                        return [
                            ValidationError(
                                code="RISK_GATE_BLOCKED",
                                message="; ".join(result.blocking_reasons),
                                field="risk_gate",
                            ),
                        ]
                elif isinstance(result, dict) and not result.get(
                    "approved", True
                ):
                    reasons = result.get("blocking_reasons", ["risk gate rejected"])
                    return [
                        ValidationError(
                            code="RISK_GATE_BLOCKED",
                            message="; ".join(reasons),
                            field="risk_gate",
                        ),
                    ]
        except Exception as exc:
            logger.error("Risk-gate check failed: %s", exc)
            # On risk-gate failure, do NOT block — log and proceed

        return []
