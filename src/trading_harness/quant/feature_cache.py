"""Feature Cache (Phase 10).

LRU-Cache für berechnete Features mit TTL-basiertem Ablauf.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """Einzelner Cache-Eintrag."""
    key: str
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache-Statistiken."""
    hits: int
    misses: int
    size: int
    max_size: int
    hit_rate: float
    total_accesses: int


class FeatureCache:
    """LRU-Cache für Features mit TTL."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Holt Wert aus Cache."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.access_count += 1
        self._hits += 1
        return entry.value

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Speichert Wert im Cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].value = value
            self._cache[key].created_at = time.time()
            self._cache[key].ttl = ttl or self.default_ttl
            return
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = CacheEntry(
            key=key, value=value, created_at=time.time(),
            ttl=ttl or self.default_ttl,
        )

    def invalidate(self, key: str) -> bool:
        """Entfernt einen Eintrag."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Leert den Cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> CacheStats:
        """Gibt Cache-Statistiken zurück."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return CacheStats(
            hits=self._hits, misses=self._misses,
            size=len(self._cache), max_size=self.max_size,
            hit_rate=hit_rate, total_accesses=total,
        )

    def cleanup_expired(self) -> int:
        """Entfernt abgelaufene Einträge. Gibt Anzahl entfernter Einträge zurück."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)

    @property
    def size(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            del self._cache[key]
            return False
        return True

    def __len__(self) -> int:
        return len(self._cache)
