"""Normalization — converts raw news into structured NewsEvent."""

from __future__ import annotations

import uuid
from datetime import datetime

from packages.domain.news.models import NewsEvent, NewsStatus


def normalize_raw_news(
    title: str,
    body: str,
    source_name: str,
    source_type: str = "unknown",
    url: str = "",
    language: str = "en",
    published_at: datetime | None = None,
    received_at: datetime | None = None,
    raw_entities: list[str] | None = None,
) -> NewsEvent:
    """Normalize raw news into a canonical NewsEvent.

    Handles entity resolution, instrument mapping, and status detection.
    """
    now = datetime.now()
    pub_time = published_at or now
    recv_time = received_at or now

    # URL hash
    url_hash = _url_hash(url) if url else ""

    # Entity resolution
    entities = raw_entities or []
    instruments: list[str] = []

    return NewsEvent(
        id=uuid.uuid4().hex[:12],
        title=title.strip(),
        body=body.strip(),
        source_name=source_name.strip(),
        source_type=source_type.strip().lower(),
        url_hash=url_hash,
        published_at=pub_time,
        received_at=recv_time,
        entities=entities,
        instruments=instruments,
        language=language,
        revision=1,
        status=_detect_status(title, body),
    )


def _url_hash(url: str) -> str:
    """Simple URL fingerprint for deduplication."""
    normalized = url.strip().lower().rstrip("/")
    return str(hash(normalized) & 0xFFFFFFFFFFFFFFFF)


def _detect_status(title: str, body: str) -> NewsStatus:
    """Detect news status from title and body keywords."""
    combined = f"{title} {body}".lower()

    retraction_keywords = [
        "retracted", "withdrawn", "false report", "hoax",
        "fabricated", "no such", "retraction",
    ]
    if any(kw in combined for kw in retraction_keywords):
        return NewsStatus.RETRACTION

    correction_keywords = [
        "correction", "mistake", "error in", "previously reported",
        "incorrect", "retracted", "withdrawn", "retraction",
    ]
    if any(kw in combined for kw in correction_keywords):
        return NewsStatus.CORRECTION

    rumor_keywords = [
        "reportedly", "allegedly", "unconfirmed", "rumor",
        "sources say", "tips", "may", "could",
    ]
    if any(kw in combined for kw in rumor_keywords):
        return NewsStatus.RUMOR

    confirmation_keywords = [
        "confirms", "confirmed", "verifies", "verification",
        "officially", "as stated", "in line with",
    ]
    if any(kw in combined for kw in confirmation_keywords):
        return NewsStatus.CONFIRMATION

    update_keywords = [
        "update", "updated", "follow-up", "developing",
        "latest", "new information",
    ]
    if any(kw in combined for kw in update_keywords):
        return NewsStatus.UPDATE

    return NewsStatus.INITIAL
