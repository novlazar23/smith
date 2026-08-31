"""News Ingestion Service — periodische RSS-Zyklen mit Persistenz.

Bereitgestellt:
    - session_factory() -> sessionmaker — Session-Factory aus Env-Vars
    - persist_events(events, factory) -> int — Bulk-Insert in news_events
    - run_forever(config, tick_seconds, once) -> dict[str, datetime] — Endlosschleife
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from apps.news_ingestion.config import NewsConfig, build_news_config
from apps.news_ingestion.scheduler import run_ingestion_cycle
from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine
from packages.persistence.sqlalchemy.news import NewsEventModel
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = Path("/tmp/heartbeat")

# Maximale Anzahl historischer Events für die Klassifikation
HISTORY_LIMIT = 200


def _db_config() -> DatabaseConfig:
    """PostgreSQL-Config aus Env-Vars mit Compose-Defaults."""
    return DatabaseConfig(
        host=os.environ.get("DB_HOST", "postgres"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ.get("DB_NAME", "trading"),
        user=os.environ.get("DB_USER", "orchestra"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


def session_factory() -> sessionmaker[Session]:
    """Erstellt die Session-Factory für die news_events-Tabelle.

    Verbindung aus Env-Vars (DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD)
    mit Compose-Defaults.

    Returns:
        sessionmaker, der Sessions auf die news_events-Tabelle bindet.
    """
    engine = SQLAlchemyEngine(_db_config())
    return engine.session_factory


def _event_row(event: dict[str, Any]) -> dict[str, Any]:
    """Mappt ein News-Event-Dict auf die news_events-Spalten."""
    return {
        "news_id": event["news_id"],
        "event_identity": event["event_identity"],
        "title": event["title"],
        "body": event["body"],
        "source_name": event["source_name"],
        "source_type": event["source_type"],
        "url_hash": event["url_hash"],
        "published_at": event["published_at"],
        "received_at": event["received_at"],
        "entities": event["entities"],
        "instruments": event["instruments"],
        "language": event.get("language", "en"),
        "revision": event.get("revision", 1),
        "status": event["status"],
    }


def persist_events(events: list[dict[str, Any]], factory: sessionmaker[Session]) -> int:
    """Persistiert News-Events bulk in die news_events-Tabelle.

    Verwendet ON CONFLICT (news_id) DO NOTHING, damit wiederholt empfangene
    Nachrichten keine Fehler verursachen und nicht dupliziert werden.

    Args:
        events: Liste normalisierter News-Event-Dicts (aus run_ingestion_cycle).
        factory: Session-Factory (siehe session_factory).

    Returns:
        Anzahl der zur Persistenz übergebenen Events.
    """
    if not events:
        return 0

    rows = [_event_row(event) for event in events]
    with factory() as session:
        stmt = (
            insert(NewsEventModel)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["news_id"])
        )
        session.execute(stmt)
        session.commit()

    logger.info("persisted %d news events", len(events))
    return len(events)


def _touch_heartbeat() -> None:
    """Aktualisiert die Heartbeat-Datei für den Compose-Healthcheck."""
    try:
        HEARTBEAT_PATH.touch()
    except OSError:
        logger.warning("Heartbeat-Datei nicht updatbar", exc_info=True)


def run_forever(
    config: NewsConfig | None = None,
    tick_seconds: float = 30.0,
    once: bool = False,  # noqa: FBT001,FBT002
) -> dict[str, datetime]:
    """Führt periodische Ingestion-Zyklen aus und persistiert die Events.

    Jeder Zyklus: run_ingestion_cycle (Fetch → Dedup → Normalize → Classify)
    → persist_events. Zyklus-Fehler werden geloggt, die Schleife läuft weiter.

    Args:
        config: NewsConfig; bei None werden die Standard-Quellen verwendet.
        tick_seconds: Pause zwischen den Zyklen in Sekunden.
        once: Falls True, wird nach dem ersten Zyklus beendet.

    Returns:
        Mapping von Quellenname → letzter Lauf-Zeitpunkt.
    """
    if config is None:
        config = build_news_config()

    factory = session_factory()
    last_run_times: dict[str, datetime] = {}
    history: list[dict[str, Any]] = []

    logger.info(
        "news ingestion starting: %d sources, tick=%.1fs",
        len(config.sources),
        tick_seconds,
    )

    try:
        while True:
            try:
                events = run_ingestion_cycle(config, last_run_times, history)
                if events:
                    persisted = persist_events(events, factory)
                    logger.info("cycle persisted %d events", persisted)
                    history.extend(events)
                    if len(history) > HISTORY_LIMIT:
                        del history[: len(history) - HISTORY_LIMIT]
            except Exception:
                logger.exception("Ingestion-Zyklus fehlgeschlagen, nächster Versuch")

            if once:
                break

            _touch_heartbeat()
            time.sleep(tick_seconds)
    except KeyboardInterrupt:
        logger.info("News-Ingestion wird beendet (KeyboardInterrupt)")

    return last_run_times
