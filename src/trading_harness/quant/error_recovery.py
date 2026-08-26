"""Error Recovery (Phase 11).

Graceful Degradation, Retry-Logik und Fallback-Werte für die Quant-Plattform.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Konfiguration für Retry-Logik."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


@dataclass
class RecoveryResult:
    """Ergebnis einer Recovery-Operation."""
    success: bool
    value: Any = None
    error: str | None = None
    retries: int = 0


class ErrorRecovery:
    """Error-Handler mit Retry-Logik und Graceful Degradation."""

    def __init__(self, retry_config: RetryConfig | None = None) -> None:
        self.retry_config = retry_config or RetryConfig()
        self._fallback_values: dict[str, Any] = {}

    def with_retry(
        self,
        func: Callable[[], Any],
        fallback: Any = None,
        operation_name: str = "operation",
    ) -> RecoveryResult:
        """Führt eine Operation mit Retry-Logik aus."""
        last_error = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = func()
                return RecoveryResult(success=True, value=result, retries=attempt)
            except Exception as e:  # noqa: BLE001 - Retry-Modul fängt gezielt alle Fehler
                last_error = str(e)
                logger.warning(f"{operation_name} failed (attempt {attempt + 1}): {e}")
                if attempt < self.retry_config.max_retries:
                    delay = min(
                        self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                        self.retry_config.max_delay,
                    )
                    time.sleep(delay)

        return RecoveryResult(
            success=False, error=last_error,
            retries=self.retry_config.max_retries,
        )

    def with_fallback(self, func: Callable[[], Any], fallback: Any, name: str = "") -> Any:
        """Führt eine Operation mit Fallback aus."""
        try:
            return func()
        except Exception as e:  # noqa: BLE001 - Fallback-Modul fängt gezielt alle Fehler
            logger.warning(f"{name or 'operation'} failed, using fallback: {e}")
            return fallback

    def register_fallback(self, key: str, value: Any) -> None:
        """Registriert einen Fallback-Wert."""
        self._fallback_values[key] = value

    def get_fallback(self, key: str, default: Any = None) -> Any:
        """Holt einen Fallback-Wert."""
        return self._fallback_values.get(key, default)

    def safe_execute(
        self,
        func: Callable[[], Any],
        default: Any = None,
        log_error: bool = True,
    ) -> Any:
        """Führt eine Operation sicher aus, fängt alle Exceptions ab."""
        try:
            return func()
        except Exception as e:  # noqa: BLE001 - Safe-Execute fängt gezielt alle Fehler
            if log_error:
                logger.error(f"Safe execute failed: {e}")
            return default

    def validate_and_fix(self, data: dict, schema: dict[str, Any]) -> dict:
        """Validiert Daten gegen Schema und füllt fehlende Werte mit Defaults."""
        result = {}
        for field_name, field_type in schema.items():
            if field_name in data:
                result[field_name] = data[field_name]
            elif "default" in field_type:
                result[field_name] = field_type["default"]
            else:
                logger.warning(f"Missing required field: {field_name}")
        return result
