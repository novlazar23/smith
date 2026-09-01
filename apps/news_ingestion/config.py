"""Konfiguration für den News Ingestion Service.

Definiert Feed-URLs, Quellen und Verarbeitungseinstellungen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FeedType(StrEnum):
    """Typ eines News-Feeds."""

    RSS = "RSS"
    JSON = "JSON"
    API = "API"


@dataclass(frozen=True)
class SourceConfig:
    """Konfiguration einer einzelnen Nachrichtenquelle.

    Felder:
        name: Benutzerdefinierten Name der Quelle.
        url: URL des Feeds / der API.
        feed_type: RSS, JSON oder API.
        update_interval: Sekunden zwischen Updates.
        priority: 1 = hoch, 5 = niedrig.
        enabled: Ob die Quelle aktiv ist.
        headers: Optionale Header für API-Anfragen.
    """

    name: str
    url: str
    feed_type: FeedType = FeedType.RSS
    update_interval: int = 300
    priority: int = 3
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingSettings:
    """Einstellungen für die Nachrichtenverarbeitung.

    Felder:
        max_items_per_feed: Maximale Anzahl Items pro Feed-Abruf.
        dedup_window_hours: Zeitspanne in Stunden für Deduplikation.
        entity_match_threshold: Ähnlichkeitsschwelle für Entity Matching (0-1).
        impact_levels: Schwellschwellen für Impact-Berechnung.
        retry_max_attempts: Maximale Wiederholungsversuche bei Fehlern.
        retry_base_delay: Basis-Verzögerung in Sekunden für exponentielles Backoff.
    """

    max_items_per_feed: int = 50
    dedup_window_hours: int = 24
    entity_match_threshold: float = 0.8
    impact_levels: dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }
    )
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0


@dataclass(frozen=True)
class NewsConfig:
    """Gesamtkonfiguration des News Ingestion Service.

    Felder:
        sources: Liste der zu überwachenden Quellen.
        processing: Einstellungen für die Nachrichtenverarbeitung.
        feed_urls: Zusätzliche Feed-URLs (werden automatisch erstellt).
    """

    sources: list[SourceConfig] = field(default_factory=list)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    feed_urls: list[str] = field(default_factory=list)


def default_source_configs() -> list[SourceConfig]:
    """Standard-Quellen für Krypto-Trading bereitstellen.

    Alle Feeds wurden auf Erreichbarkeit (HTTP 200, gültiges RSS-XML)
    aus der Laufzeitumgebung verifiziert.

    Gibt eine Liste vordefinierter Quellen zurück:
        - CoinDesk RSS
        - Cointelegraph RSS
        - Decrypt RSS
        - The Block RSS
        - Bitcoin Magazine RSS
        - Crypto Potato RSS
    """
    return [
        SourceConfig(
            name="CoinDesk",
            url="https://www.coindesk.com/arc/outboundfeeds/rss/",
            feed_type=FeedType.RSS,
            update_interval=300,
            priority=1,
        ),
        SourceConfig(
            name="Cointelegraph",
            url="https://cointelegraph.com/rss",
            feed_type=FeedType.RSS,
            update_interval=300,
            priority=1,
        ),
        SourceConfig(
            name="Decrypt",
            url="https://decrypt.co/feed",
            feed_type=FeedType.RSS,
            update_interval=600,
            priority=2,
        ),
        SourceConfig(
            name="The Block",
            url="https://www.theblock.co/rss.xml",
            feed_type=FeedType.RSS,
            update_interval=600,
            priority=2,
        ),
        SourceConfig(
            name="Bitcoin Magazine",
            url="https://news.bitcoin.com/feed/",
            feed_type=FeedType.RSS,
            update_interval=900,
            priority=3,
        ),
        SourceConfig(
            name="Crypto Potato",
            url="https://cryptopotato.com/feed/",
            feed_type=FeedType.RSS,
            update_interval=900,
            priority=3,
        ),
    ]


def build_news_config(
    extra_sources: list[SourceConfig] | None = None,
    processing: ProcessingSettings | None = None,
) -> NewsConfig:
    """Konfiguration mit Standard-Quellen und optionalen Erweiterungen erstellen.

    Args:
        extra_sources: Zusätzliche Quellen neben den Standard-Quellen.
        processing: Eigene Verarbeitungseinstellungen.

    Returns:
        Vollständig konfigurierte NewsConfig-Instanz.
    """
    sources = default_source_configs()
    if extra_sources:
        sources = sources + extra_sources
    return NewsConfig(sources=sources, processing=processing or ProcessingSettings())
