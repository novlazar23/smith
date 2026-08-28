from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from trading_harness.models import MarketSnapshot
from trading_harness.services.db import Database
from trading_harness.services.snapshot_store import SnapshotStore


def _parse_ts(value: str | Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


class PersistedSnapshotStore:
    """PostgreSQL-backed snapshot store.

    Falls back to in-memory store when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback = SnapshotStore()

    @staticmethod
    def _hash(snapshot: MarketSnapshot) -> str:
        payload = {
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp.isoformat(),
            "data": snapshot.data,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def add(self, snapshot: MarketSnapshot) -> MarketSnapshot:
        if not snapshot.content_hash:
            snapshot.content_hash = self._hash(snapshot)
        if self._db and self._db.is_available:
            self._db.execute_write(
                "INSERT INTO market_snapshots (id, symbol, timestamp, data, content_hash) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "symbol = EXCLUDED.symbol, "
                "timestamp = EXCLUDED.timestamp, "
                "data = EXCLUDED.data, "
                "content_hash = EXCLUDED.content_hash",
                snapshot.id,
                snapshot.symbol,
                snapshot.timestamp.isoformat(),
                snapshot.data,
                snapshot.content_hash,
            )
        else:
            self._fallback.add(snapshot)
        return snapshot

    def get(self, snapshot_id: str) -> MarketSnapshot | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM market_snapshots WHERE id = %s", (snapshot_id,)
            )
            if rows:
                row = rows[0]
                return MarketSnapshot(
                    id=row["id"],
                    symbol=row["symbol"],
                    timestamp=_parse_ts(row["timestamp"]),
                    data=row["data"],
                    content_hash=row.get("content_hash"),
                )
            return None
        return self._fallback.get(snapshot_id)