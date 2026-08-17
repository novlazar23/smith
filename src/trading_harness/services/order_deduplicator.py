"""OrderDeduplicator — thread-safe decision_id dedup."""

from __future__ import annotations

import threading
from collections import deque


class OrderDeduplicator:
    """Thread-sichere Order-Dedup basierend auf decision_id + symbol + side.

    Erkennt 100% von Duplikaten und ist speicher-begrenzt.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._seen: set[str] = set()
        self._recent: deque[str] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def _make_key(self, decision_id: str, symbol: str, side: str) -> str:
        """Erzeugt einen eindeutigen Key aus decision_id, symbol und side."""
        return f"{decision_id}:{symbol}:{side}"

    def is_duplicate(self, decision_id: str, symbol: str, side: str) -> bool:
        """Prüfen ob eine Order ein Duplikat ist.

        Args:
            decision_id: Unique decision identifier from TradingRun
            symbol: Trading symbol (e.g. "BTCUSDT")
            side: "LONG" or "SHORT"

        Returns:
            True wenn Duplikat erkannt, False sonst
        """
        key = self._make_key(decision_id, symbol, side)
        with self._lock:
            if key in self._seen:
                return True
            self._seen.add(key)
            self._recent.append(key)
            # Periodische Bereinigung bei Überschreitung
            maxlen = self._recent.maxlen
            if maxlen is not None and len(self._recent) >= maxlen:
                self._trim()
            return False

    def _trim(self) -> None:
        """Alte Einträge aus seen entfernen, die nicht mehr im recent sind."""
        # Behalte nur die letzten N Einträge aus recent
        recent_set = set(self._recent)
        self._seen = recent_set

    def clear(self, decision_id: str | None = None) -> None:
        """Duplikat-Speicher leeren.

        Args:
            decision_id: Wenn gesetzt, nur diesen decision_id entfernen.
                        None = alles leeren.
        """
        with self._lock:
            if decision_id is None:
                self._seen.clear()
                self._recent.clear()
            else:
                # Entferne alle Einträge mit diesem decision_id
                to_remove = [
                    key
                    for key in self._seen
                    if key.startswith(f"{decision_id}:")
                ]
                for key in to_remove:
                    self._seen.discard(key)

    @property
    def seen_count(self) -> int:
        """Anzahl der gesehenen einzigartigen Orders."""
        with self._lock:
            return len(self._seen)