"""SourceMetadata — Zeitpunkt- und Herkunftsinformationen für jedes Ereignis.

Enthält die drei kritischen Zeitstempel für Point-in-Time-Korrektheit:
  - event_time:      Das Ereignis selbst ist zum Zeitpunkt eingetreten.
  - ingestion_time:  Der Zeitpunkt der Aufnahme ins System (UTC).
  - availability_time: Der Zeitpunkt, ab dem die Daten allgemein verfügbar waren.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SourceMetadata(BaseModel):
    """Metadaten zur Herkunft und Zeitlinie eines Ereignisses."""

    model_config = ConfigDict(frozen=True)

    source: str
    """Name der Datenquelle (z. B. ``binance``, ``coingecko``)."""

    venue: str | None = None
    """Handelsplatz / Börse, falls zutreffend."""

    event_time: datetime
    """Das Ereignis selbst ist zum Zeitpunkt eingetreten. ISO 8601."""

    ingestion_time: datetime
    """Zeitpunkt der Aufnahme ins System (UTC)."""

    availability_time: datetime
    """Zeitpunkt, ab dem die Daten allgemein öffentlich verfügbar waren.

    Dient der Point-in-Time-Validität von Feature-Berechnungen.
    ``availability_time <= analysis_time`` muss geprüft werden.
    """

    sequence: int | None = None
    """Sequenznummer des Ereignisses bei quellenpezifischer Reihenfolge
    (z. B. Orderbook-Sequenz, Kandelaber-Sequenz)."""

    revision: int = 1
    """Versionsnummer für revidierte Daten. Startet bei 1."""

    quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Qualitätsbewertung der Quelle (0.0 = unbrauchbar, 1.0 = voll vertrauenswürdig).",
    )
