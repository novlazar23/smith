"""Secret-/Key-Rotation mit Zero Downtime (EPIC-16 WP06).

- :class:`SecretRotationPolicy` — konfigurierbares Intervall, injectable
  Clock, deterministisch testbar.
- :func:`rotate_key` — Zero-downtime-Rotation: neue Version hinzufügen,
  als current setzen; die alte Version bleibt bis zur expliziten
  Deprecation verwendbar. Die Rotation wird im Live-Audit-Trail
  protokolliert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from packages.security.hardening.audit_live import LiveAuditTrail
from packages.security.hardening.encryption import KeyRing

__all__ = ["RotationEvent", "SecretRotationPolicy", "rotate_key"]


@dataclass(frozen=True)
class RotationEvent:
    """Ergebnis einer Key-Rotation."""

    old_version: int | None
    new_version: int
    timestamp: datetime


class SecretRotationPolicy:
    """Rotation-Policy: fällig, wenn das Intervall seit letzter Rotation abgelaufen ist."""

    def __init__(
        self,
        interval_days: float,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if interval_days <= 0:
            raise ValueError("interval_days muss größer als 0 sein")
        self.interval_days = interval_days
        self._clock = clock or (lambda: datetime.now(UTC))

    def is_due(
        self,
        last_rotated: datetime | None,
        now: datetime | None = None,
    ) -> bool:
        """True wenn noch nie rotiert wurde oder das Intervall abgelaufen ist."""
        if last_rotated is None:
            return True
        current = now if now is not None else self._clock()
        return (current - last_rotated) >= timedelta(days=self.interval_days)

    def next_due(self, last_rotated: datetime | None) -> datetime | None:
        """Nächstes Fälligkeits-Datum (None wenn noch nie rotiert)."""
        if last_rotated is None:
            return None
        return last_rotated + timedelta(days=self.interval_days)


def rotate_key(
    keyring: KeyRing,
    new_version: int,
    new_key_b64: str,
    *,
    audit: LiveAuditTrail,
    actor: str = "system",
    now: datetime | None = None,
) -> RotationEvent:
    """Zero-downtime Rotation.

    1. Neue Key-Version wird hinzugefügt.
    2. Sie wird aktuelle Version (neue Verschlüsselungen nutzen sie).
    3. Die alte Version bleibt verwendbar, bis sie explizit über
       ``keyring.deprecate(old_version)`` entfernt wird.
    4. Die Rotation wird im Live-Audit-Trail protokolliert.
    """
    old_version = keyring.current_version
    keyring.add_key(new_version, new_key_b64)
    keyring.set_current(new_version)
    ts = now if now is not None else datetime.now(UTC)
    audit.record(
        "secret_rotation",
        actor=actor,
        resource="keyring",
        status="ok",
        details={"old_version": old_version, "new_version": new_version},
        timestamp=ts,
    )
    return RotationEvent(
        old_version=old_version, new_version=new_version, timestamp=ts
    )
