"""Deterministic tests for the secret rotation policy (EPIC-16 WP06)."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

from packages.security.hardening.audit_live import LiveAuditTrail
from packages.security.hardening.encryption import KeyRing
from packages.security.hardening.secret_rotation import (
    SecretRotationPolicy,
    rotate_key,
)

KEY_V1 = base64.urlsafe_b64encode(b"\x01" * 32).decode()
KEY_V2 = base64.urlsafe_b64encode(b"\x02" * 32).decode()
KEY_V3 = base64.urlsafe_b64encode(b"\x03" * 32).decode()


class TestPolicyIsDue:
    def test_never_rotated_is_due(self) -> None:
        policy = SecretRotationPolicy(interval_days=90)
        assert policy.is_due(None, now=datetime(2025, 6, 1, tzinfo=UTC)) is True

    def test_recently_rotated_not_due(self) -> None:
        policy = SecretRotationPolicy(interval_days=90)
        last = datetime(2025, 1, 1, tzinfo=UTC)
        assert policy.is_due(last, now=last + timedelta(days=89)) is False

    def test_interval_elapsed_is_due(self) -> None:
        policy = SecretRotationPolicy(interval_days=90)
        last = datetime(2025, 1, 1, tzinfo=UTC)
        assert policy.is_due(last, now=last + timedelta(days=90)) is True

    def test_uses_injected_clock_when_now_absent(self) -> None:
        last = datetime(2025, 1, 1, tzinfo=UTC)
        policy = SecretRotationPolicy(
            interval_days=90, clock=lambda: last + timedelta(days=90)
        )
        assert policy.is_due(last) is True


class TestRotationEvent:
    def test_zero_downtime_rotation(self) -> None:
        """New key active while old one stays usable until deprecated."""
        ring = KeyRing()
        ring.add_key(1, KEY_V1)
        old_token = ring.encrypt("legacy")
        ring.add_key(2, KEY_V2)
        ring.set_current(2)

        # Old token still decodes before deprecation.
        assert ring.decrypt(old_token) == "legacy"
        # New tokens are written with the new version.
        new_token = ring.encrypt("fresh")
        assert ring.decrypt(new_token) == "fresh"
        # After deprecating v1, old tokens fail.
        ring.deprecate(1)
        assert ring.current_version == 2

    def test_policy_records_rotation_in_audit(self) -> None:
        trail = LiveAuditTrail()
        ring = KeyRing()
        ring.add_key(1, KEY_V1)
        event = rotate_key(
            ring, 2, KEY_V2, audit=trail, actor="security-ops",
            now=datetime(2025, 3, 1, tzinfo=UTC),
        )
        assert event.old_version == 1
        assert event.new_version == 2
        assert ring.current_version == 2
        assert len(trail) == 1
        entry = trail.entries[0]
        assert entry.action == "secret_rotation"
        assert entry.actor == "security-ops"
        assert entry.details == {"old_version": 1, "new_version": 2}

    def test_rotation_is_deterministic(self) -> None:
        """Same inputs, same event (no wall clock, no randomness)."""
        ts = datetime(2025, 3, 1, tzinfo=UTC)

        def make() -> object:
            trail = LiveAuditTrail()
            ring = KeyRing(nonce_factory=lambda: b"\x00" * 12)
            ring.add_key(1, KEY_V1)
            t = ring.encrypt("same")
            rotate_key(ring, 2, KEY_V2, audit=trail, actor="x", now=ts)
            return t, trail.entries[0].entry_hash

        t1, h1 = make()
        t2, h2 = make()
        assert t1 == t2
        assert h1 == h2
