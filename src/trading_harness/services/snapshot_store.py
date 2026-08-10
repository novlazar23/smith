from __future__ import annotations

import hashlib
import json
from threading import RLock

from trading_harness.models import MarketSnapshot


class SnapshotStore:
    def __init__(self) -> None:
        self._items: dict[str, MarketSnapshot] = {}
        self._lock = RLock()

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
        snapshot.content_hash = self._hash(snapshot)
        with self._lock:
            self._items[snapshot.id] = snapshot
        return snapshot

    def get(self, snapshot_id: str) -> MarketSnapshot | None:
        with self._lock:
            return self._items.get(snapshot_id)
