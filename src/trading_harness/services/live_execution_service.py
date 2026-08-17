"""LiveExecutionService — orchestrates KillSwitch, RateLimiter, Deduplicator, ExchangeAdapter.

Standardmäßig deaktiviert. Alle Sicherheitsgrenzen werden vor Execution再次 geprüft.

Pipeline:
  KillSwitch → RateLimit → Dedup → SymbolWhitelist → RiskEngine → NetworkPolicy →
  CredentialCheck → Exchange → Log
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from trading_harness.models import TradeProposal
from trading_harness.services.credential_manager import CredentialManager
from trading_harness.services.exchange_adapter import ExchangeAdapter, StubExchangeAdapter
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.network_policy import NetworkPolicy
from trading_harness.services.order_deduplicator import OrderDeduplicator
from trading_harness.services.rate_limiter import RateLimiter
from trading_harness.services.risk_engine import RiskEngine


class ExecutionConfig(BaseModel):
    """Konfiguration des Execution Service."""

    live_execution_enabled: bool = False
    global_rate_limit: int = 10
    symbol_rate_limit: int = 2
    min_capital: float = 0.01
    allowed_endpoints: list[str] = Field(default_factory=list)
    allowed_exchanges: list[str] = Field(default_factory=list)
    symbol_whitelist: list[str] = Field(default_factory=list)


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
    risk_approved: bool = False
    risk_max_position_size: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LiveExecutionService:
    """Orchestrates KillSwitch → RateLimiter → Deduplicator → ExchangeAdapter.

    Alle Komponenten werden im Konstruktor injiziert. Standardmäßig deaktiviert.
    """

    def __init__(
        self,
        kill_switch: KillSwitch | None = None,
        rate_limiter: RateLimiter | None = None,
        deduplicator: OrderDeduplicator | None = None,
        exchange_adapter: ExchangeAdapter | None = None,
        risk_engine: RiskEngine | None = None,
        network_policy: NetworkPolicy | None = None,
        credential_manager: CredentialManager | None = None,
        config: ExecutionConfig | None = None,
    ) -> None:
        self._kill_switch = kill_switch or KillSwitch(enabled=False)
        self._rate_limiter = rate_limiter or RateLimiter(
            global_limit=10, symbol_limit=2
        )
        self._deduplicator = deduplicator or OrderDeduplicator()
        self._exchange_adapter = exchange_adapter or StubExchangeAdapter()
        self._risk_engine = risk_engine
        self._network_policy = network_policy
        self._credential_manager = credential_manager
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

        Pipeline: KillSwitch → RateLimit → Dedup → SymbolWhitelist → RiskEngine →
        NetworkPolicy → CredentialCheck → Exchange → Log
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

        # 5. Check symbol whitelist (R5.6, Sicherheitsgrenze)
        if self._config.symbol_whitelist and symbol not in self._config.symbol_whitelist:
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                error="SYMBOL_NOT_WHITELISTED",
            )

        # 6. Check min capital
        if quantity < self._config.min_capital:
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                error="MIN_CAPITAL_NOT_MET",
            )

        # 7. Risk Policy Re-Check (R5.6, Sicherheitsgrenze)
        risk_approved = False
        max_position_size = quantity
        if self._risk_engine:
            equity = quantity * price if price > 0 else 100000.0
            stop_distance = price * 0.02 if price > 0 else 100.0
            trade_proposal = TradeProposal(
                decision_id=decision_id,
                symbol=symbol,
                side=side,
                equity=equity,
                entry_price=price,
                stop_price=max(0, price - stop_distance),
                target_price=min(price * 1.5, price + stop_distance * 2),
                requested_leverage=1.0,
                open_positions=0,
                current_daily_loss_fraction=0.0,
                current_portfolio_risk_fraction=0.0,
                expected_slippage_bps=10.0,
                requested_quantity=quantity,
            )
            risk_result = self._risk_engine.evaluate(trade_proposal, kill_switch=False)
            risk_approved = risk_result.approved
            max_position_size = risk_result.max_position_size if risk_result.max_position_size > 0 else quantity
            if not risk_approved:
                return self._log_and_return(
                    decision_id,
                    run_id,
                    symbol,
                    side,
                    "REJECTED",
                    error=risk_result.reason,
                )

        # 8. Enforce RiskEngine max_position_size
        if max_position_size > 0 and quantity > max_position_size:
            quantity = max_position_size

        # 9. Network Policy Check (R5.15–R5.17)
        if self._network_policy:
            exchange_url = self._get_exchange_url()
            if not self._network_policy.is_allowed("POST", exchange_url):
                return self._log_and_return(
                    decision_id,
                    run_id,
                    symbol,
                    side,
                    "REJECTED",
                    error="NETWORK_POLICY_VIOLATION",
                )

        # 10. Credential Check (R5.18–R5.20)
        if self._credential_manager:
            api_key = self._credential_manager.get("API_KEY")
            api_secret = self._credential_manager.get("API_SECRET")
            if not api_key or not api_secret:
                return self._log_and_return(
                    decision_id,
                    run_id,
                    symbol,
                    side,
                    "REJECTED",
                    error="CREDENTIALS_NOT_CONFIGURED",
                )

        # 11. Submit to Exchange
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
                risk_approved=risk_approved,
                risk_max_position_size=max_position_size,
            )
        except Exception as e:  # noqa: BLE001 — execution pipeline: any unexpected adapter/IO error
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "ERROR",
                error=str(e),
            )

    def _get_exchange_url(self) -> str:
        """Get the base URL of the exchange adapter for network policy check."""
        adapter_type = type(self._exchange_adapter).__name__.lower()
        if "bybit" in adapter_type:
            return "https://api.bybit.com/*"
        elif "bitget" in adapter_type:
            return "https://api.bitget.com/*"
        return "*"

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
        risk_approved: bool = False,
        risk_max_position_size: float = 0.0,
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
            risk_approved=risk_approved,
            risk_max_position_size=risk_max_position_size,
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
            "risk_approved": risk_approved,
            "risk_max_position_size": risk_max_position_size,
            "timestamp": log.timestamp.isoformat(),
        }