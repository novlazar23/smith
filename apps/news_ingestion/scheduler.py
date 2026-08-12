"""Zeitgesteuerte Ingestion für News-Quellen.

Bereitgestellt:
    - run_ingestion_cycle(config) -> list[NewsEvent] — Vollständiger Zyklus
    - schedule_sources(config) -> dict[str, datetime] — Zeitplan aktualisieren
    - get_due_sources(last_run_times, config) -> list[SourceConfig] — Fällige Quellen
    - retry_delay(attempt, base_delay) — Exponentielles Backoff
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from apps.news_ingestion.classifier import classify_news
from apps.news_ingestion.config import NewsConfig, ProcessingSettings, SourceConfig
from apps.news_ingestion.ingest_rss import (
    NewsRawItem,
    deduplicate,
    ingest_feed,
)
from apps.news_ingestion.normalize import normalize_item

logger = logging.getLogger(__name__)


def retry_delay(attempt: int, base_delay: float = 1.0) -> float:
    """Berechnet die Verzögerung für exponentielles Backoff.

    Formel: base_delay * (2 ** attempt), maximal 300 Sekunden.

    Args:
        attempt: Nummer des Versuchs (0-basiert).
        base_delay: Basis-Verzögerung in Sekunden.

    Returns:
        Verzögerung in Sekunden.
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, 300.0)


def get_due_sources(
    last_run_times: dict[str, datetime],
    config: NewsConfig,
) -> list[SourceConfig]:
    """Ermittelt Quellen, die wegen einer Aktualisierung fällig sind.

    Eine Quelle ist fällig, wenn:
        - Sie nicht in last_run_times existiert (noch nie gelaufen)
        - Der letzte Lauf mehr als update_interval Sekunden zurückliegt

    Args:
        last_run_times: Mapping von Quellenname → letzter Lauf-Zeitpunkt.
        config: NewsConfig mit den Quelkonfigurationen.

    Returns:
        Liste der fälligen (nicht deaktivierten) Quellen.
    """
    now = datetime.now(UTC)
    due: list[SourceConfig] = []

    for source in config.sources:
        if not source.enabled:
            continue

        last_run = last_run_times.get(source.name)
        if last_run is None:
            # Noch nie gelaufen → sofort
            due.append(source)
            continue

        elapsed = (now - last_run).total_seconds()
        if elapsed >= source.update_interval:
            due.append(source)

    return due


def run_ingestion_cycle(
    config: NewsConfig,
    last_run_times: dict[str, datetime] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Führt einen vollständigen Ingestion-Zyklus durch.

    Pipeline: Fetch → Dedup → Normalize → Classify

    Args:
        config: NewsConfig mit den Quellen.
        last_run_times: Mapping von Quellenname → letzter Lauf.
        history: Liste historischer Events für Klassifikation.

    Returns:
        Liste von normalisierten und klassifizierten News-Events.
    """
    if last_run_times is None:
        last_run_times = {}

    all_events: list[dict[str, Any]] = []
    due_sources = get_due_sources(last_run_times, config)

    logger.info(
        "Starting ingestion cycle: %d/%d sources due",
        len(due_sources),
        len(config.sources),
    )

    for source in due_sources:
        items = _fetch_with_retry(source, config.processing)
        if not items:
            logger.warning("No items from '%s', skipping", source.name)
            continue

        # Dedup
        raw_items = deduplicate(items)
        logger.info("After dedup: %d items from '%s'", len(raw_items), source.name)

        # Normalize + Classify
        for raw in raw_items:
            event = normalize_item(raw, config)
            event["status"] = classify_news(event, history).value
            all_events.append(event)

        last_run_times[source.name] = datetime.now(UTC)

    logger.info("Ingestion cycle complete: %d events", len(all_events))
    return all_events


def _fetch_with_retry(
    source: SourceConfig,
    processing: ProcessingSettings,
) -> list[NewsRawItem]:
    """Holt Feed mit exponentiellem Backoff bei Fehlern.

    Args:
        source: Die abzurufende Quelle.
        processing: ProcessingSettings für retry-Konfiguration.

    Returns:
        Liste von NewsRawItem oder leere Liste bei Fehlschlag.
    """
    max_attempts = processing.retry_max_attempts
    base_delay = processing.retry_base_delay
    items: list[NewsRawItem] = []

    for attempt in range(max_attempts):
        try:
            items = ingest_feed(source)
            if items:
                return items

            if attempt < max_attempts - 1:
                delay = retry_delay(attempt, base_delay)
                logger.info(
                    "No items from '%s', retrying in %.1fs (attempt %d/%d)",
                    source.name,
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(delay)

        except Exception as exc:
            if attempt < max_attempts - 1:
                delay = retry_delay(attempt, base_delay)
                logger.warning(
                    "Error fetching '%s': %s, retrying in %.1fs",
                    source.name,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Failed to fetch '%s' after %d attempts: %s",
                    source.name,
                    max_attempts,
                    exc,
                )

    return items


def schedule_sources(
    config: NewsConfig,
    last_run_times: dict[str, datetime] | None = None,
) -> dict[str, datetime]:
    """Aktualisiert den Zeitplan für alle Quellen basierend auf
    ihren update_intervals und dem aktuellen Zeitpunkt.

    Args:
        config: NewsConfig mit den Quelkonfigurationen.
        last_run_times: Bisherige Laufzeiten (wird aktualisiert).

    Returns:
        Aktualisiertes Mapping von Quellenname → nächster Lauf-Zeitpunkt.
    """
    if last_run_times is None:
        last_run_times = {}

    now = datetime.now(UTC)
    schedule: dict[str, datetime] = {}

    for source in config.sources:
        last_run = last_run_times.get(source.name, now)
        next_run = last_run + timedelta(seconds=source.update_interval)
        schedule[source.name] = next_run

    return schedule
