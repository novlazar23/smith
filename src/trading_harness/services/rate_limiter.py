"""RateLimiter — global + per-symbol Token Bucket.

Semantik "N Orders/Minute" (Spec R5.10, Epic WI-P5-2): Kapazität = Limit
(Burst), Refill-Rate = Limit/60 Tokens pro Sekunde, d.h. die
Sustained-Rate entspricht exakt dem konfigurierten Limit.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Token-Bucket Rate Limiter mit globalen und pro-Symbol Limits in N/min.

    Verhindert Überlastung durch zu viele Orders in einem Zeitraum.
    """

    def __init__(self, global_limit: int = 10, symbol_limit: int = 2) -> None:
        self._global_limit = global_limit
        self._symbol_limit = symbol_limit
        self._global_tokens: float = float(global_limit)
        self._symbol_tokens: dict[str, float] = {}
        self._lock = threading.RLock()
        self._last_refill_time = time.monotonic()

    def _refill(self) -> None:
        """Tokens auffüllen (basierend auf verstrichener Zeit).

        Refill-Rate skaliert mit dem Limit: global_limit/60 bzw.
        symbol_limit/60 Tokens pro Sekunde (N/min-Semantik). Die
        Kapazität bleibt der Limit-Wert (Burst unverändert).
        """
        now = time.monotonic()
        elapsed = now - self._last_refill_time

        with self._lock:
            global_refill = elapsed * (self._global_limit / 60.0)
            self._global_tokens = min(
                self._global_limit, self._global_tokens + global_refill
            )
            symbol_refill = elapsed * (self._symbol_limit / 60.0)
            for symbol in self._symbol_tokens:
                self._symbol_tokens[symbol] = min(
                    self._symbol_limit, self._symbol_tokens[symbol] + symbol_refill
                )
            self._last_refill_time = now

    def allow(self, symbol: str) -> bool:
        """Prüfen ob eine Order für das Symbol erlaubt ist.

        Args:
            symbol: Das Symbol der Order

        Returns:
            True wenn Order erlaubt, False wenn Rate Limit erreicht
        """
        with self._lock:
            # Refill Tokens
            self._refill()

            # Prüfe globales Limit
            if self._global_tokens < 1.0:
                return False

            # pro-Symbol Limit (initialisiere bei Bedarf)
            if symbol not in self._symbol_tokens:
                self._symbol_tokens[symbol] = float(self._symbol_limit)

            current_symbol_tokens = self._symbol_tokens[symbol]
            if current_symbol_tokens < 1.0:
                return False

            # Tokens verbrauchen
            self._global_tokens -= 1.0
            self._symbol_tokens[symbol] = current_symbol_tokens - 1.0
            return True

    def reset(self, symbol: str | None = None) -> None:
        """Rate Limits zurücksetzen.

        Args:
            symbol: Wenn gesetzt, nur dieses Symbol zurücksetzen.
                    None = alle zurücksetzen.
        """
        with self._lock:
            if symbol is None:
                self._global_tokens = float(self._global_limit)
                self._symbol_tokens.clear()
            else:
                self._symbol_tokens[symbol] = float(self._symbol_limit)

    @property
    def global_limit(self) -> int:
        """Globales Rate Limit."""
        return self._global_limit

    @property
    def symbol_limit(self) -> int:
        """Pro-Symbol Rate Limit."""
        return self._symbol_limit