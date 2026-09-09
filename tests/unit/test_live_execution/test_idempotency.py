"""Unit tests for packages.live_execution.idempotency.

Covers record/get/exists, venue isolation, TTL expiry, clear, and key
generation.  TTL tests manipulate the private ``expires_at`` field for
determinism (no real sleeps).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from packages.live_execution.idempotency import IdempotencyStore


class TestRecordGetExists:
    async def test_record_then_get_returns_same_result(self) -> None:
        store = IdempotencyStore()
        result = await store.record("key-1", "binance", {"order_id": "123"})
        assert result == {"order_id": "123"}
        assert await store.get("key-1", "binance") == {"order_id": "123"}

    async def test_get_unknown_key_returns_none(self) -> None:
        store = IdempotencyStore()
        assert await store.get("missing", "binance") is None

    async def test_exists_true_after_record(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"a": 1})
        assert await store.exists("key-1", "binance") is True

    async def test_exists_false_for_unknown(self) -> None:
        store = IdempotencyStore()
        assert await store.exists("missing", "binance") is False

    async def test_duplicate_record_returns_original_cached_result(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"order_id": "first"})
        cached = await store.record("key-1", "binance", {"order_id": "second"})
        assert cached == {"order_id": "first"}
        assert await store.get("key-1", "binance") == {"order_id": "first"}

    async def test_record_stores_copy_not_reference(self) -> None:
        store = IdempotencyStore()
        original = {"order_id": "123"}
        await store.record("key-1", "binance", original)
        original["order_id"] = "mutated"
        assert await store.get("key-1", "binance") == {"order_id": "123"}


class TestVenueIsolation:
    async def test_same_key_different_venues_are_independent(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"venue": "binance"})
        assert await store.get("key-1", "bybit") is None
        assert await store.exists("key-1", "binance") is True
        assert await store.exists("key-1", "bybit") is False

    async def test_record_under_other_venue_does_not_clobber(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"venue": "binance"})
        await store.record("key-1", "bybit", {"venue": "bybit"})
        assert await store.get("key-1", "binance") == {"venue": "binance"}
        assert await store.get("key-1", "bybit") == {"venue": "bybit"}


class TestTtlExpiry:
    async def test_expired_entry_is_not_returned(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"order_id": "123"})
        entry = store._store[("binance", "key-1")]
        entry.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert await store.get("key-1", "binance") is None
        assert await store.exists("key-1", "binance") is False

    async def test_record_after_expiry_stores_fresh_result(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"order_id": "old"})
        store._store[("binance", "key-1")].expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await store.record("key-1", "binance", {"order_id": "new"})
        assert await store.get("key-1", "binance") == {"order_id": "new"}

    async def test_future_expiry_is_not_expired(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"order_id": "123"})
        entry = store._store[("binance", "key-1")]
        entry.expires_at = datetime.now(UTC) + timedelta(hours=1)
        assert entry.is_expired() is False
        assert await store.get("key-1", "binance") is not None

    async def test_entry_without_expiry_never_expires(self) -> None:
        from packages.live_execution.idempotency import IdempotencyEntry

        fresh = IdempotencyEntry(key="k", venue="binance", result={}, expires_at=None)
        assert fresh.is_expired() is False


class TestClear:
    async def test_clear_removes_all_entries(self) -> None:
        store = IdempotencyStore()
        await store.record("key-1", "binance", {"a": 1})
        await store.record("key-2", "bybit", {"b": 2})
        await store.clear()
        assert await store.get("key-1", "binance") is None
        assert await store.get("key-2", "bybit") is None


class TestGenerateKey:
    async def test_generate_key_returns_uuid4_string(self) -> None:
        store = IdempotencyStore()
        key = await store.generate_key("binance")
        UUID(key)  # raises if not a valid UUID

    async def test_generate_key_is_unique(self) -> None:
        store = IdempotencyStore()
        keys = {await store.generate_key("binance") for _ in range(50)}
        assert len(keys) == 50
