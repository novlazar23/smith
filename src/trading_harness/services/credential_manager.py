"""CredentialManager — Sichere Credential-Verwaltung.

Liest Credentials aus env vars oder externem Secret Store.
Nie in Logs/Audit/Debug-Output schreiben (R5.18–R5.20).
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CredentialRef(BaseModel):
    """Referenz auf ein Credential ohne den Wert."""

    key: str
    source: str = "env"  # env | vault | aws_secrets_manager | etc.
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CredentialManager:
    """Zentrale Credential-Verwaltung.

    - Liest Credentials nur aus env vars oder Secret Stores
    - Speichert NIE raw values in logs/audit/debug output
    - Unterstützt rotation über env var updates
    """

    def __init__(self, secret_store: Any | None = None) -> None:
        self._refs: dict[str, CredentialRef] = {}
        self._cache: dict[str, str] = {}
        self._secret_store = secret_store
        self._lock = threading.Lock()

    def register(self, key: str, source: str = "env") -> CredentialRef:
        """Credential key registrieren."""
        with self._lock:
            self._refs[key] = CredentialRef(key=key, source=source)
        return self._refs[key]

    def get(self, key: str) -> str | None:
        """Credential value lesen (nie logging)."""
        if key in self._cache:
            cached = self._cache.get(key)
            return cached if cached else None

        value: str | None = None
        if self._secret_store:
            value = self._secret_store.get(key)

        if value is None:
            value = os.environ.get(key)

        with self._lock:
            if value is not None:
                self._cache[key] = value
        return value

    def is_configured(self, key: str) -> bool:
        """Prüfen ob ein Credential gesetzt ist."""
        return self.get(key) is not None and len(self.get(key) or "") > 0

    @property
    def configured_keys(self) -> list[str]:
        """Keys deren Credentials aktuell konfiguriert sind."""
        result: list[str] = []
        with self._lock:
            for key in self._refs:
                val = os.environ.get(key)
                if val:
                    result.append(key)
        return result

    def summary(self) -> dict[str, Any]:
        """Zusammenfassung — NIE raw values."""
        with self._lock:
            refs = self._refs
        return {
            "registered": list(refs.keys()),
            "configured": self.configured_keys,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Cache leeren (z.B. bei credential rotation)."""
        with self._lock:
            self._cache.clear()