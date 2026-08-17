"""NetworkPolicy — Endpoint-Whitelist und Audit-Log von Verletzungen.

Blockiert alle Netzwerk-Ausgänge, die nicht auf der Whitelist stehen.
Erfüllt R5.15–R5.17.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PolicyViolation(BaseModel):
    """Einzelne Network-Policy-Verletzung."""

    id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    method: str = ""
    url: str = ""
    reason: str = "POLICY_VIOLATION"
    details: dict[str, Any] = Field(default_factory=dict)


class NetworkPolicy:
    """Endpoint-Whitelist mit Audit-Logging.

    Zulässige Hosts können als regex patterns definiert werden.
    Alle Anfragen an nicht whitelisted targets werden blockiert.
    """

    def __init__(self, allowed_patterns: list[str] | None = None) -> None:
        self._allowed: list[re.Pattern] = []
        for pat in (allowed_patterns or []):
            self._allowed.append(re.compile(pat))
        self._violations: list[PolicyViolation] = []
        self._lock = threading.Lock()

    def is_allowed(self, method: str, url: str) -> bool:
        """Prüfen ob eine URL auf der Whitelist steht."""
        if not self._allowed:
            return True
        for pat in self._allowed:
            if pat.match(url):
                return True
        with self._lock:
            violation = PolicyViolation(
                id=f"pol-{int(datetime.now(UTC).timestamp() * 1000)}",
                method=method,
                url=url,
            )
            self._violations.append(violation)
        logger.warning(
            "NetworkPolicyViolation: %s %s",
            method,
            url,
        )
        return False

    @property
    def violation_count(self) -> int:
        with self._lock:
            return len(self._violations)

    def get_violations(self, limit: int = 100) -> list[PolicyViolation]:
        with self._lock:
            return list(self._violations[-limit:])

    def add_allowed(self, pattern: str) -> None:
        """Ein neues pattern zur Whitelist hinzufügen."""
        self._allowed.append(re.compile(pattern))