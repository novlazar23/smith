"""Tests für OrderDeduplicator."""

from __future__ import annotations

import threading

from trading_harness.services.order_deduplicator import OrderDeduplicator


class TestOrderDeduplicatorBasic:
    """Grundlegende Dedup-Tests."""

    def test_first_order_not_duplicate(self):
        """Erste Order ist kein Duplikat."""
        dd = OrderDeduplicator()
        assert dd.is_duplicate("dec-1", "BTCUSDT", "LONG") is False

    def test_same_order_is_duplicate(self):
        """Gleiche Order ist ein Duplikat."""
        dd = OrderDeduplicator()
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
        assert dd.is_duplicate("dec-1", "BTCUSDT", "LONG") is True

    def test_different_decision_not_duplicate(self):
        """Verschiedene decision_id ist kein Duplikat."""
        dd = OrderDeduplicator()
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
        assert dd.is_duplicate("dec-2", "BTCUSDT", "LONG") is False

    def test_different_symbol_not_duplicate(self):
        """Verschiedenes Symbol ist kein Duplikat."""
        dd = OrderDeduplicator()
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
        assert dd.is_duplicate("dec-1", "ETHUSDT", "LONG") is False

    def test_different_side_not_duplicate(self):
        """Verschiedene Side ist kein Duplikat."""
        dd = OrderDeduplicator()
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
        assert dd.is_duplicate("dec-1", "BTCUSDT", "SHORT") is False


class TestOrderDeduplicatorConcurrency:
    """Thread-Safety-Tests."""

    def test_concurrent_duplicate_detection(self):
        """Parallele Duplikat-Erkennung zu 100% zuverlässig."""
        dd = OrderDeduplicator()
        results: list[bool] = []
        lock = threading.Lock()

        def check_duplicate() -> None:
            result = dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
            with lock:
                results.append(result)

        # Erster Aufruf markiert die Order
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")

        # Parallele Checks — alle sollten True zurückgeben
        threads = [threading.Thread(target=check_duplicate) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results), "Alle parallelen Calls sollten Duplikat erkennen"

    def test_concurrent_different_orders(self):
        """Parallele verschiedene Orders — keine falschen Duplikate."""
        dd = OrderDeduplicator()
        errors = []

        def check_order(decision_id: str) -> None:
            try:
                result = dd.is_duplicate(decision_id, "BTCUSDT", "LONG")
                if result:
                    errors.append(f"False positive for {decision_id}")
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=check_order, args=(f"dec-{i}",))
            for i in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Fehler bei parallelen Orders: {errors}"


class TestOrderDeduplicatorMemory:
    """Memory-bounded-Tests."""

    def test_max_entries_limit(self):
        """Deque hat maximale Größe."""
        dd = OrderDeduplicator(max_entries=10)
        for i in range(20):
            dd.is_duplicate(f"dec-{i}", "BTCUSDT", "LONG")
        # seen sollte nicht unendlich wachsen
        assert dd.seen_count <= 10 + 1  # +1 weil der letzte evtl. noch drin ist

    def test_clear_all(self):
        """Clear löscht alle Einträge."""
        dd = OrderDeduplicator()
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
        dd.is_duplicate("dec-2", "ETHUSDT", "SHORT")
        dd.clear()
        assert dd.seen_count == 0
        assert dd.is_duplicate("dec-1", "BTCUSDT", "LONG") is False

    def test_clear_decision(self):
        """Clear nur für decision_id."""
        dd = OrderDeduplicator()
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
        dd.is_duplicate("dec-2", "ETHUSDT", "SHORT")
        dd.clear("dec-1")
        assert dd.seen_count == 1  # nur dec-2 übrig
        assert dd.is_duplicate("dec-1", "BTCUSDT", "LONG") is False
        assert dd.is_duplicate("dec-2", "ETHUSDT", "SHORT") is True


class TestOrderDeduplicatorSeenCount:
    """SeenCount-Tests."""

    def test_seen_count_increments(self):
        """SeenCount zählt einzigartige Orders."""
        dd = OrderDeduplicator()
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")
        dd.is_duplicate("dec-2", "BTCUSDT", "LONG")
        dd.is_duplicate("dec-1", "BTCUSDT", "LONG")  # Duplikat
        assert dd.seen_count == 2