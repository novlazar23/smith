"""LiveExecutionService — orchestrates KillSwitch, RateLimiter, Deduplicator, ExchangeAdapter.

Standardmäßig deaktiviert. Alle Sicherheitsgrenzen werden vor Execution再次 geprüft.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from trading_harness.services.exchange_adapter import ExchangeAdapter, StubExchangeAdapter
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.order_deduplicator import OrderDeduplicator
from trading_harness.services.rate_limiter import RateLimiter


class ExecutionConfig(BaseModel):
    """Konfiguration des Execution Service."""

    live_execution_enabled: bool = False
    global_rate_limit: int = 10
    symbol_rate_limit: int = 2
    min_capital: float = 0.01
    allowed_endpoints: list[str] = Field(default_factory=list)


class ExecutionLog(BaseModel):
    """Protokoll-Eintrag für jeden Trade-Versuch."""

    id: str
    decision_id: str
    run_id: str
    symbol: str
    side: str
    status: str
    order_id: str | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LiveExecutionService:
    """Orchestrates KillSwitch -> RateLimiter -> Deduplicator -> ExchangeAdapter.

    Alle Komponenten werden im Konstruktor injiziert. Standardmäßig deaktiviert.
    """

    def __init__(
        self,
        kill_switch: KillSwitch | None = None,
        rate_limiter: RateLimiter | None = None,
        deduplicator: OrderDeduplicator | None = None,
        exchange_adapter: ExchangeAdapter | None = None,
        config: ExecutionConfig | None = None,
    ) -> None:
        self._kill_switch = kill_switch or KillSwitch(enabled=False)
        self._rate_limiter = rate_limiter or RateLimiter(
            global_limit=10, symbol_limit=2
        )
        self._deduplicator = deduplicator or OrderDeduplicator()
        self._exchange_adapter = exchange_adapter or StubExchangeAdapter()
        self._config = config or ExecutionConfig()
        self._enabled = self._config.live_execution_enabled
        self._lock = threading.Lock()
        self._logs: list[ExecutionLog] = []

    def submit_order(
        self,
        decision_id: str,
        run_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> dict[str, Any]:
        """Order durch den Execution Pipeline schicken.

        Pipeline: KillSwitch -> RateLimit -> Dedup -> Exchange -> Log
        """
        # 1. Check if live execution is enabled
        if not self._enabled:
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                error="LIVE_EXECUTION_DISABLED",
            )

        # 2. Check KillSwitch (muss <100ms sein)
        if self._kill_switch.is_active():
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                error="KILL_SWITCH_ACTIVE",
            )

        # 3. Check Rate Limit
        if not self._rate_limiter.allow(symbol):
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                error="RATE_LIMIT_EXCEEDED",
            )

        # 4. Check for duplicates
        if self._deduplicator.is_duplicate(decision_id, symbol, side):
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                error="DUPLICATE_DECISION_ID",
            )

        # 5. Check min capital
        if quantity < self._config.min_capital:
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                error="MIN_CAPITAL_NOT_MET",
            )

        # 6. Submit to Exchange
        try:
            result = self._exchange_adapter.submit_order(
                symbol=symbol,
                side="BUY" if side == "LONG" else "SELL",
                quantity=quantity,
                price=price,
                order_type="MARKET",
            )
            order_id = result.get("order_id")
            status = result.get("status", "ERROR")
            error = result.get("error")

            if status == "NOT_IMPLEMENTED":
                status = "ERROR"
                error = "NO_EXCHANGE_ADAPTER_IMPLEMENTED"

            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                status,
                order_id=order_id,
                error=error,
            )
        except Exception as e:
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "ERROR",
                error=str(e),
            )

    def activate_live(self) -> None:
        """Live Execution aktivieren (muss explizit aufgerufen werden)."""
        with self._lock:
            self._enabled = True

    def deactivate_live(self) -> None:
        """Live Execution deaktivieren."""
        with self._lock:
            self._enabled = False

    @property
    def is_live_enabled(self) -> bool:
        """Prüfen ob Live Execution aktiviert ist."""
        with self._lock:
            return self._enabled

    def get_logs(self, decision_id: str | None = None) -> list[dict[str, Any]]:
        """Execution Logs abrufen."""
        with self._lock:
            if decision_id is None:
                return [log.model_dump() for log in self._logs]
            return [
                log.model_dump()
                for log in self._logs
                if log.decision_id == decision_id
            ]

    def _log_and_return(
        self,
        decision_id: str,
        run_id: str,
        symbol: str,
        side: str,
        status: str,
        order_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Execution loggen und Antwort zurückgeben."""
        log = ExecutionLog(
            id=f"exec-{int(time.time() * 1000)}",
            decision_id=decision_id,
            run_id=run_id,
            symbol=symbol,
            side=side,
            status=status,
            order_id=order_id,
            error=error,
        )
        with self._lock:
            self._logs.append(log)

        return {
            "decision_id": decision_id,
            "run_id": run_id,
            "symbol": symbol,
            "side": side,
            "status": status,
            "order_id": order_id,
            "error": error,
            "timestamp": log.timestamp.isoformat(),
        }