"""Append-only Live-Audit-Trail mit Hash-Kette (EPIC-16 WP06).

Jeder Eintrag ist unveränderlich (frozen dataclass) und in eine SHA-256-
Hash-Kette eingebettet (``previous_hash`` + ``entry_hash``), sodass
Nachträgliche Manipulationen über :meth:`LiveAuditTrail.verify_integrity`
erkennbar sind.

Secrets (API-Keys, Passwörter, Tokens, …) werden aus den Details
herausgeredacted, bevor sie gespeichert werden.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "GENESIS_HASH",
    "LiveAuditEntry",
    "LiveAuditTrail",
    "get_live_audit",
    "reset_live_audit",
]

GENESIS_HASH = "0" * 64

#: Details-Keys, deren Werte niemals roh im Audit-Trail landen dürfen.
_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|credential)", re.IGNORECASE
)


def _sanitize_details(details: dict[str, Any]) -> dict[str, Any]:
    """Redactet sensible Werte aus dem Details-Dict."""
    clean: dict[str, Any] = {}
    for key, value in details.items():
        if (
            isinstance(key, str)
            and _SENSITIVE_KEY.search(key)
            and isinstance(value, str)
            and value
        ):
            clean[key] = "[REDACTED]"
        else:
            clean[key] = value
    return clean


def _canonical(payload: dict[str, Any]) -> str:
    """Kanonicalisierte JSON-Repräsentation für die Hash-Berechnung."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_entry(
    previous_hash: str,
    sequence: int,
    timestamp: datetime,
    actor: str,
    action: str,
    resource: str,
    status: str,
    details: dict[str, Any],
) -> str:
    payload = {
        "previous_hash": previous_hash,
        "sequence": sequence,
        "timestamp": timestamp.isoformat(),
        "actor": actor,
        "action": action,
        "resource": resource,
        "status": status,
        "details": details,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LiveAuditEntry:
    """Ein unveränderlicher Eintrag im Live-Audit-Trail."""

    sequence: int
    timestamp: datetime
    actor: str
    action: str
    resource: str
    status: str
    details: dict[str, Any]
    previous_hash: str
    entry_hash: str


class LiveAuditTrail:
    """Append-only, in-memory Audit-Trail mit SHA-256-Hash-Kette.

    Der Clock ist injectierbar, damit Tests deterministisch laufen.
    """

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._entries: list[LiveAuditEntry] = []
        self._clock = clock or (lambda: datetime.now(UTC))

    def record(
        self,
        action: str,
        *,
        actor: str = "system",
        resource: str = "",
        status: str = "ok",
        details: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> LiveAuditEntry:
        """Hängt einen neuen Eintrag an und liefert ihn zurück."""
        ts = timestamp if timestamp is not None else self._clock()
        sequence = len(self._entries)
        previous_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        clean = _sanitize_details(dict(details) if details is not None else {})
        entry_hash = _hash_entry(
            previous_hash, sequence, ts, actor, action, resource, status, clean
        )
        entry = LiveAuditEntry(
            sequence=sequence,
            timestamp=ts,
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            details=clean,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def record_order_submit(
        self,
        *,
        actor: str,
        order_id: str,
        instrument: str,
        venue: str,
        direction: str,
        quantity: float,
        status: str = "submitted",
        order_type: str = "",
        idempotency_key: str = "",
    ) -> LiveAuditEntry:
        details: dict[str, Any] = {
            "instrument": instrument,
            "venue": venue,
            "direction": direction,
            "quantity": quantity,
            "order_type": order_type,
        }
        if idempotency_key:
            details["idempotency_key"] = idempotency_key
        return self.record(
            "order_submit",
            actor=actor,
            resource=order_id,
            status=status,
            details=details,
        )

    def record_order_cancel(
        self,
        *,
        actor: str,
        order_id: str,
        status: str,
        reason: str | None = None,
        error: str | None = None,
    ) -> LiveAuditEntry:
        details: dict[str, Any] = {"reason": reason}
        if error:
            details["error"] = error
        return self.record(
            "order_cancel",
            actor=actor,
            resource=order_id,
            status=status,
            details=details,
        )

    def record_order_fill(
        self,
        *,
        actor: str,
        order_id: str,
        filled_quantity: float,
        price: float | None = None,
    ) -> LiveAuditEntry:
        details: dict[str, Any] = {"filled_quantity": filled_quantity}
        if price is not None:
            details["price"] = price
        return self.record(
            "order_fill",
            actor=actor,
            resource=order_id,
            status="filled",
            details=details,
        )

    def record_kill_switch(
        self,
        *,
        actor: str,
        action: str,
        reason: str,
        affected_order_count: int | None = None,
    ) -> LiveAuditEntry:
        details: dict[str, Any] = {"action": action, "reason": reason}
        if affected_order_count is not None:
            details["affected_order_count"] = affected_order_count
        return self.record(
            "kill_switch",
            actor=actor,
            resource="kill_switch",
            status="ok",
            details=details,
        )

    def record_rollout_state_change(
        self,
        *,
        actor: str,
        previous_state: str,
        new_state: str,
    ) -> LiveAuditEntry:
        return self.record(
            "rollout_state_change",
            actor=actor,
            resource="rollout",
            status="ok",
            details={
                "previous_state": previous_state,
                "new_state": new_state,
            },
        )

    def record_secret_rotation(
        self,
        *,
        actor: str,
        old_version: int | None,
        new_version: int,
    ) -> LiveAuditEntry:
        return self.record(
            "secret_rotation",
            actor=actor,
            resource="keyring",
            status="ok",
            details={
                "old_version": old_version,
                "new_version": new_version,
            },
        )

    @property
    def entries(self) -> list[LiveAuditEntry]:
        """Kopie der Einträge (append-only bleibt intern gewahrt)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def verify_integrity(self) -> bool:
        """Prüft Sequenz- und Hash-Kette; False bei jeder Manipulation."""
        previous = GENESIS_HASH
        for expected_sequence, entry in enumerate(self._entries):
            if entry.sequence != expected_sequence or entry.previous_hash != previous:
                return False
            expected = _hash_entry(
                entry.previous_hash,
                entry.sequence,
                entry.timestamp,
                entry.actor,
                entry.action,
                entry.resource,
                entry.status,
                entry.details,
            )
            if expected != entry.entry_hash:
                return False
            previous = entry.entry_hash
        return True


_trail: LiveAuditTrail = LiveAuditTrail()


def get_live_audit() -> LiveAuditTrail:
    """App-weiter accessor — geeignet für FastAPI-Dependency-Injection."""
    return _trail


def reset_live_audit() -> LiveAuditTrail:
    """Ersetzt den geteilten Trail durch einen frischen (für Tests)."""
    global _trail
    _trail = LiveAuditTrail()
    return _trail
