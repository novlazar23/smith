"""Deterministische JSONL-Response-Cache für LLM-Antworten.

Annahme: **Einzelprozess-Zugriff** — es wird kein Lock gesetzt, ein
Multi-Writer-Betrieb ist nicht vorgesehen. Innerhalb eines Prozesses ist
das Verhalten deterministisch: `put` aktualisiert zuerst das In-Memory-
Dictionary und hängt danach eine JSON-Zeile an die Datei (append + flush),
wobei der letzte Schreibzug pro Key gewinnt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LLMResponseCache:
    """JSONL-Cache: Key → Antworttext, letzter Eintrag pro Key gewinnt."""

    def __init__(self, path: str) -> None:
        """Legt das Elternverzeichnis an und lädt vorhandene Einträge aus der Datei."""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, str] = {}
        if self._path.is_file():
            self._load()

    def _load(self) -> None:
        """Lädt JSON-Zeilen; kaputte/leere Zeilen werden still übersprungen."""
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    key = record.get("key")
                    value = record.get("value")
                    if isinstance(key, str) and isinstance(value, str):
                        self._entries[key] = value

    @staticmethod
    def key(*parts: str) -> str:
        """Stabiler SHA-256-Hash über alle Teile (Unit-Separator-getrennt)."""
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        """Liefert den gecachten Wert oder None (Cache-Miss)."""
        return self._entries.get(key)

    def put(self, key: str, value: str) -> None:
        """Speichert den Wert im Dictionary und als JSON-Zeile (append + flush)."""
        self._entries[key] = value
        line = json.dumps({"key": key, "value": value}, ensure_ascii=False, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
