"""Tests für Feature Cache."""
from __future__ import annotations

import time
import pytest
from trading_harness.quant.feature_cache import FeatureCache, CacheStats


class TestFeatureCache:
    def test_put_and_get(self):
        cache = FeatureCache(max_size=10)
        cache.put("key1", {"rsi": 65.0})
        result = cache.get("key1")
        assert result == {"rsi": 65.0}

    def test_get_miss(self):
        cache = FeatureCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_lru_eviction(self):
        cache = FeatureCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict 'a'
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_ttl_expiration(self):
        cache = FeatureCache(default_ttl=0.1)
        cache.put("key1", "value1")
        time.sleep(0.15)
        result = cache.get("key1")
        assert result is None

    def test_stats(self):
        cache = FeatureCache(max_size=10)
        cache.put("a", 1)
        cache.get("a")
        cache.get("b")
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1
        assert stats.hit_rate == pytest.approx(0.5)

    def test_invalidate(self):
        cache = FeatureCache()
        cache.put("key1", "value1")
        removed = cache.invalidate("key1")
        assert removed is True
        assert cache.get("key1") is None

    def test_clear(self):
        cache = FeatureCache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size == 0

    def test_cleanup_expired(self):
        cache = FeatureCache(default_ttl=0.1)
        cache.put("a", 1)
        cache.put("b", 2, ttl=10)
        time.sleep(0.15)
        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.size == 1

    def test_contains(self):
        cache = FeatureCache()
        cache.put("key1", "value1")
        assert "key1" in cache
        assert "key2" not in cache

    def test_update_existing(self):
        cache = FeatureCache()
        cache.put("key1", "old")
        cache.put("key1", "new")
        assert cache.get("key1") == "new"

    def test_max_size_respected(self):
        cache = FeatureCache(max_size=3)
        for i in range(5):
            cache.put(f"key{i}", i)
        assert cache.size == 3
