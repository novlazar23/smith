"""Instrument Pool — Pool of instruments to be processed together.

Provides validation, correlation heuristics, and shared-computation
pair detection for batch analysis runs.
"""

from __future__ import annotations

import re
from typing import Any

# Known cross-correlated pairs that always count as correlated.
_KNOWN_CROSS_PAIRS: set[tuple[str, str]] = {
    ("BTC", "ETH"),
    ("ETH", "BTC"),
}

# Family groupings — instruments sharing a family prefix are correlated.
_FAMILY_GROUPS: dict[str, set[str]] = {
    "BTC": {"BTC", "BTC/USD", "BTC-EUR", "BTC_USD"},
    "ETH": {"ETH", "ETH/USD", "ETH-EUR", "ETH_USD"},
    "SOL": {"SOL", "SOL/USD", "SOL-EUR", "SOL_USD"},
    "XRP": {"XRP", "XRP/USD", "XRP-EUR", "XRP_USD"},
    "ADA": {"ADA", "ADA/USD", "ADA-EUR", "ADA_USD"},
}


def _normalize(name: str) -> str:
    """Return a normalised key for family lookup."""
    return re.sub(r"[^A-Za-z0-9_]", "", name).upper()


class InstrumentPool:
    """Reprsentiert einen Pool von Instrumenten, die gemeinsam verarbeitet werden.

    Bietet Validierung, Korrelationsheuristik und Erkennung gemeinsamer
    Feature-Computing-Paare fuer Batch-Analyselaufe.
    """

    def __init__(
        self,
        max_instruments: int = 20,
        memory_limit_mb: int = 4096,
        correlation_threshold: float = 0.8,
    ) -> None:
        """Initialisiert den Instrumentenpool.

        Args:
            max_instruments: Maximal erlaubte Anzahl Instrumente (default 20).
            memory_limit_mb: Speicherlimit in MB pro Batch (default 4096).
            correlation_threshold: Schwellwert fuer Korrelationspaare (default 0.8).
        """
        if max_instruments < 1:
            raise ValueError("max_instruments must be >= 1")
        if not (0.0 <= correlation_threshold <= 1.0):
            raise ValueError("correlation_threshold must be between 0 and 1")

        self._max_instruments: int = max_instruments
        self._memory_limit_mb: int = memory_limit_mb
        self._correlation_threshold: float = correlation_threshold
        self._instruments: list[str] = []

    @property
    def max_instruments(self) -> int:
        """Maximal erlaubte Anzahl Instrumente."""
        return self._max_instruments

    @property
    def memory_limit_mb(self) -> int:
        """Speicherlimit in MB."""
        return self._memory_limit_mb

    @property
    def correlation_threshold(self) -> float:
        """Korrelationsschwellwert."""
        return self._correlation_threshold

    @property
    def count(self) -> int:
        """Aktuelle Anzahl Instrumente im Pool."""
        return len(self._instruments)

    @staticmethod
    def _validate_instrument_name(name: str) -> None:
        """Validiert einen Instrumentennamen.

        Muss nicht-leer sein und max 20 Zeichen enthalten.
        Nur alphanumerische Zeichen und Unterstriche erlaubt.

        Args:
            name: Zu pruefender Name.

        Raises:
            ValueError: Wenn der Name ungultig ist.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Instrument name must be a non-empty string")
        if len(name) > 20:
            raise ValueError(
                f"Instrument name too long: {name!r} ({len(name)} chars, max 20)"
            )
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(
                f"Instrument name contains invalid characters: {name!r}"
            )

    def add_instruments(self, instruments: list[str]) -> None:
        """Fuegt Instrumente zum Pool hinzu.

        Validiert jeden Namen und stellt sicher, dass die Gesamtanzahl
        max_instruments nicht ueberschreitet.

        Args:
            instruments: Liste von Instrumentennamen.

        Raises:
            ValueError: Wenn ein Name ungultig ist oder die Grenze ueberschritten wird.
        """
        for name in instruments:
            self._validate_instrument_name(name)

        if self.count + len(instruments) > self._max_instruments:
            raise ValueError(
                f"Adding {len(instruments)} instruments would exceed "
                f"the maximum of {self._max_instruments}"
            )

        # Verhindere Duplikate
        existing = set(self._instruments)
        for name in instruments:
            if name in existing:
                raise ValueError(f"Duplicate instrument: {name!r}")
            self._instruments.append(name)
            existing.add(name)

    def get_instruments(self) -> list[str]:
        """Gibt die aktuelle Liste der Instrumente zurueck."""
        return list(self._instruments)

    def clear(self) -> None:
        """Loescht alle Instrumente aus dem Pool."""
        self._instruments.clear()

    @staticmethod
    def is_correlated(inst_a: str, inst_b: str) -> bool:
        """Prueft ob zwei Instrumente korreliert sind.

        Heuristik:
        1. Bekannte Kreuzpaare (BTC/ETH)
        2. Gleiche Familie (z.B. BTC/* und BTC/USD)

        Args:
            inst_a: Erster Instrumentenname.
            inst_b: Zweiter Instrumentenname.

        Returns:
            True wenn die Instrumente als korreliert gelten.
        """
        key_a, key_b = inst_a.upper(), inst_b.upper()

        # Bekannte Kreuzpaare
        if (key_a, key_b) in _KNOWN_CROSS_PAIRS:
            return True

        # Gleiche Familie
        norm_a = _normalize(inst_a)
        norm_b = _normalize(inst_b)

        # Pruefe ob beide zur gleichen Familie gehoeren
        for family_symbols in _FAMILY_GROUPS.values():
            if (
                any(s in norm_a for s in family_symbols)
                and any(s in norm_b for s in family_symbols)
            ):
                return True

        return False

    def get_correlated_pairs(self) -> list[tuple[str, str]]:
        """Gibt Paare korrelierter Instrumente im Pool zurueck.

        Nutzt die is_correlated-Heuristik, um alle Paare zu finden,
        die gemeinsame Feature-Berechnung rechtfertigen.

        Returns:
            Liste von Tupeln korrelierter Instrumentennamen.
        """
        correlated: list[tuple[str, str]] = []
        instruments = self._instruments
        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                if self.is_correlated(instruments[i], instruments[j]):
                    correlated.append((instruments[i], instruments[j]))
        return correlated

    def model_dump(self) -> dict[str, Any]:  # Alias for Pydantic compatibility
        """Gibt Pool-Konfiguration als Dict zurueck."""
        return {
            "max_instruments": self._max_instruments,
            "memory_limit_mb": self._memory_limit_mb,
            "correlation_threshold": self._correlation_threshold,
            "instruments": list(self._instruments),
            "count": self.count,
        }
