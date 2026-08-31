"""News-Model — SQLAlchemy-Model für News-Events.

Mapping der News-Events aus dem News-Ingestion-Service auf die
PostgreSQL-Tabelle ``news_events``:
  - NewsEvent (dict aus run_ingestion_cycle) -> news_events
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base, _generate_uuid


class NewsEventModel(Base):
    """SQLAlchemy-Model für NewsEvents.

    Persistiert ein normalisiertes und klassifiziertes News-Event.
    ``news_id`` ist eindeutig (unique) — Wiederholungen derselben
    Nachricht werden beim Insert ignoriert (ON CONFLICT DO NOTHING).
    """

    __tablename__ = "news_events"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: _generate_uuid()
    )
    news_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    event_identity: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    received_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    entities: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    instruments: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
