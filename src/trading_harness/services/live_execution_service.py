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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from trading_harness.models import TradeProposal
from trading_harness.services.credential_manager import CredentialManager
from trading_harness.services.exchange_adapter import (
    ExchangeAdapter,
    ExchangeAdapterError,
    StubExchangeAdapter,
)
from trading_harness.services.execution_store import ExecutionLogStore
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.network_policy import NetworkPolicy
from trading_harness.services.order_deduplicator import OrderDeduplicator
from trading_harness.services.rate_limiter import RateLimiter
from trading_harness.services.risk_engine import RiskEngine

if TYPE_CHECKING:
    from trading_harness.services.shadow_mode_logger import ShadowModeLogger


class ExecutionConfig(BaseModel):
    """Konfiguration des Execution Service.

    R5.21–R5.22: Separate Auth für Trade-Aktionen vs. Lese-Zugriffe.
    """

    live_execution_enabled: bool = False
    global_rate_limit: int = 10
    symbol_rate_limit: int = 2
    min_capital: float = 0.01
    # R5.23/R5.24: Maximaler Kapitaleinsatz pro Order (Einheiten); None = min_capital.
    max_capital: float | None = None
    allowed_endpoints: list[str] = Field(default_factory=list)
    allowed_exchanges: list[str] = Field(default_factory=list)
    symbol_whitelist: list[str] = Field(default_factory=list)
    # R5.21–R5.22: Read/Trade API Separation — Key-Namen für CredentialManager
    trade_api_key_ref: str = "TRADE_API_KEY"
    trade_api_secret_ref: str = "TRADE_API_SECRET"
    read_api_key_ref: str = "READ_API_KEY"
    read_api_secret_ref: str = "READ_API_SECRET"


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


class SafetyGateCheck(BaseModel):
    """Ergebnis eines einzelnen Safety-Gate-Checks."""

    name: str
    passed: bool
    detail: str = ""


class SafetyGateResult(BaseModel):
    """Ergebnis der Safety-Gate-Prüfung vor der Live-Aktivierung."""

    ready: bool
    checks: list[SafetyGateCheck] = Field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        """Namen der fehlgeschlagenen Checks."""
        return [check.name for check in self.checks if not check.passed]


class LiveExecutionService:
    """Orchestrates KillSwitch → RateLimiter → Deduplicator → ExchangeAdapter.

    Alle Komponenten werden im Konstruktor injiziert. Standardmäßig deaktiviert.
    Optional: ShadowModeLogger für Backtesting.
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
        shadow_logger: ShadowModeLogger | None = None,
        log_store: ExecutionLogStore | None = None,
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
        self._shadow_logger = shadow_logger
        self._log_store = log_store

    def submit_order(
        self,
        decision_id: str,
        run_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        exchange_name: str | None = None,
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
                quantity=quantity,
                price=price,
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
                quantity=quantity,
                price=price,
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
                quantity=quantity,
                price=price,
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
                quantity=quantity,
                price=price,
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
                quantity=quantity,
                price=price,
                error="SYMBOL_NOT_WHITELISTED",
            )

        # 5b. Check allowed exchanges (R5.21, erlaubte Exchanges enforced)
        if self._config.allowed_exchanges and exchange_name and exchange_name not in self._config.allowed_exchanges:
                return self._log_and_return(
                    decision_id,
                    run_id,
                    symbol,
                    side,
                    "REJECTED",
                    quantity=quantity,
                    price=price,
                    error="EXCHANGE_NOT_ALLOWED",
                )

        # 6. Check min capital
        if quantity < self._config.min_capital:
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                quantity=quantity,
                price=price,
                error="MIN_CAPITAL_NOT_MET",
            )

        # 6b. Check max capital (R5.23/R5.24: standardmäßig = min_capital)
        effective_max_capital = (
            self._config.max_capital
            if self._config.max_capital is not None
            else self._config.min_capital
        )
        if quantity > effective_max_capital:
            return self._log_and_return(
                decision_id,
                run_id,
                symbol,
                side,
                "REJECTED",
                quantity=quantity,
                price=price,
                error="MAX_CAPITAL_EXCEEDED",
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
                    quantity=quantity,
                    price=price,
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
                    quantity=quantity,
                    price=price,
                    error="NETWORK_POLICY_VIOLATION",
                )

        # 10. Credential Check — Trade Operations (R5.18–R5.20, R5.21–R5.22)
        if self._credential_manager:
            trade_key = self._credential_manager.get(self._config.trade_api_key_ref)
            trade_secret = self._credential_manager.get(self._config.trade_api_secret_ref)
            if not trade_key or not trade_secret:
                return self._log_and_return(
                    decision_id,
                    run_id,
                    symbol,
                    side,
                    "REJECTED",
                    quantity=quantity,
                    price=price,
                    error="TRADE_CREDENTIALS_NOT_CONFIGURED",
                )

        # 11. Submit to Exchange
        try:
            result = self._exchange_adapter.submit_order(
                symbol=symbol,
                side="BUY" if side == "LONG" else "SELL",
                quantity=quantity,
                price=price,
                order_type="MARKET",
                exchange_name=exchange_name,
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
                quantity=quantity,
                price=price,
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
                quantity=quantity,
                price=price,
                error=str(e),
            )

    def _get_exchange_url(self) -> str:
        """Get the base URL of the exchange adapter for network policy check."""
        adapter_type = type(self._exchange_adapter).__name__.lower()
        if "bybit" in adapter_type:
            return "https://api.bybit.com/*"
        elif "bitget" in adapter_type:
            return "https://api.bitget.com/*"
        elif "binance" in adapter_type:
            return "https://api.binance.com/*"
        elif "coinbase" in adapter_type:
            return "https://api.coinbase.com/*"
        elif "crypto" in adapter_type:
            # CryptoExecutionRouter: allow all 4 exchange domains
            return "https://api.*/*"
        return "*"

    def verify_safety_gate(self) -> SafetyGateResult:
        """Prüft die Sicherheitsvoraussetzungen für die Live-Aktivierung."""
        checks: list[SafetyGateCheck] = []

        checks.append(
            SafetyGateCheck(
                name="kill_switch_present",
                passed=self._kill_switch is not None,
                detail="Kill Switch verdrahtet"
                if self._kill_switch is not None
                else "Kein Kill Switch verdrahtet",
            )
        )

        min_capital = self._config.min_capital
        checks.append(
            SafetyGateCheck(
                name="min_capital_positive",
                passed=min_capital > 0,
                detail=f"min_capital={min_capital}",
            )
        )

        max_capital = self._config.max_capital
        checks.append(
            SafetyGateCheck(
                name="max_capital_valid",
                passed=max_capital is None or max_capital > 0,
                detail="max_capital=unset (Default: min_capital)"
                if max_capital is None
                else f"max_capital={max_capital}",
            )
        )

        return SafetyGateResult(
            ready=all(check.passed for check in checks), checks=checks
        )

    def kill_switch_status(self) -> dict[str, Any]:
        """Monitoring-Zustand des Kill Switches (R5.7)."""
        return self._kill_switch.config.model_dump()

    def activate_live(self) -> SafetyGateResult:
        """Live Execution aktivieren — nur bei bestandenem Safety Gate.

        Fail-closed: Bei nicht bestandenem Gate bleibt Live Execution
        deaktiviert; das Ergebnis dokumentiert die fehlgeschlagenen Checks.
        """
        result = self.verify_safety_gate()
        if result.ready:
            with self._lock:
                self._enabled = True
        return result

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
        quantity: float = 0.0,
        price: float = 0.0,
    ) -> dict[str, Any]:
        """Execution loggen und Antwort zurückgeben.
        
        Wenn ShadowMode aktiv und Order nicht ausgeführt wurde, wird sie
        shadow-geloggt für Backtesting.
        """
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

        # R5.3: Audit-Log persistieren (ohne Credentials)
        if self._log_store is not None:
            self._log_store.add(
                decision_id=decision_id,
                run_id=run_id,
                symbol=symbol,
                side=side,
                status=status,
                order_id=order_id,
                error=error,
            )

        # Shadow Mode: logge nicht-ausgeführte Orders für Backtesting
        if self._shadow_logger and status in ("REJECTED", "ERROR"):
            self._shadow_logger.log_rejection(
                decision_id=decision_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                run_id=run_id,
                error=error,
            )

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
            "shadow_mode": bool(self._shadow_logger),
            "timestamp": log.timestamp.isoformat(),
        }

    def get_order_status(
        self,
        order_id: str,
        exchange_name: str | None = None,
    ) -> dict[str, Any]:
        """Order-Status durch die Pipeline (KillSwitch, NetworkPolicy)."""
        if not self._enabled:
            return {"status": "REJECTED", "order_id": order_id, "error": "LIVE_EXECUTION_DISABLED"}
        if self._kill_switch.is_active():
            return {"status": "REJECTED", "order_id": order_id, "error": "KILL_SWITCH_ACTIVE"}
        # R5.21–R5.22: Read operations require READ API credentials
        if self._credential_manager:
            read_key = self._credential_manager.get(self._config.read_api_key_ref)
            read_secret = self._credential_manager.get(self._config.read_api_secret_ref)
            if not read_key or not read_secret:
                return {"status": "REJECTED", "order_id": order_id, "error": "READ_CREDENTIALS_NOT_CONFIGURED"}
        if self._network_policy:
            exchange_url = self._get_exchange_url()
            if not self._network_policy.is_allowed("GET", exchange_url):
                return {"status": "REJECTED", "order_id": order_id, "error": "NETWORK_POLICY_VIOLATION"}
        try:
            return self._exchange_adapter.get_order_status(order_id, exchange_name=exchange_name)
        except ExchangeAdapterError as exc:
            return {"status": "ERROR", "order_id": order_id, "error": str(exc)}

    def cancel_order(
        self,
        order_id: str,
        exchange_name: str | None = None,
    ) -> dict[str, Any]:
        """Order stornieren durch die Pipeline (KillSwitch, NetworkPolicy)."""
        if not self._enabled:
            return {"success": False, "order_id": order_id, "error": "LIVE_EXECUTION_DISABLED"}
        if self._kill_switch.is_active():
            return {"success": False, "order_id": order_id, "error": "KILL_SWITCH_ACTIVE"}
        # R5.21–R5.22: Cancel is a write operation — require TRADE API credentials
        if self._credential_manager:
            trade_key = self._credential_manager.get(self._config.trade_api_key_ref)
            trade_secret = self._credential_manager.get(self._config.trade_api_secret_ref)
            if not trade_key or not trade_secret:
                return {"success": False, "order_id": order_id, "error": "TRADE_CREDENTIALS_NOT_CONFIGURED"}
        if self._network_policy:
            exchange_url = self._get_exchange_url()
            if not self._network_policy.is_allowed("DELETE", exchange_url):
                return {"success": False, "order_id": order_id, "error": "NETWORK_POLICY_VIOLATION"}
        try:
            return self._exchange_adapter.cancel_order(order_id, exchange_name=exchange_name)
        except ExchangeAdapterError as exc:
            return {"success": False, "order_id": order_id, "error": str(exc)}