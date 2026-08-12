"""Source scoring — rates news sources by reliability and relevance."""

from __future__ import annotations

from datetime import datetime

from packages.domain.news.models import NewsEvent

# Source reliability scores (0-1):
# Higher = more reliable, faster, more accurate
SOURCE_RELIABILITY: dict[str, float] = {
    "reuters": 0.95,
    "ap news": 0.93,
    "bloomberg": 0.90,
    "coindesk": 0.85,
    "cointelegraph": 0.80,
    "the block": 0.82,
    "decrypt": 0.78,
    "cryptonews": 0.70,
    "twitter": 0.40,
    "reddit": 0.35,
    "telegram": 0.30,
    "unknown": 0.20,
}

# Source type multipliers
SOURCE_TYPE_WEIGHT: dict[str, float] = {
    "wire_service": 1.0,
    "exchange": 0.95,
    "regulatory": 0.90,
    "exchange_filing": 0.85,
    "financial_media": 0.80,
    "crypto_media": 0.75,
    "blog": 0.60,
    "social_media": 0.40,
    "forum": 0.30,
    "unknown": 0.20,
}


def score_source(
    source_name: str,
    source_type: str,
    news_event: NewsEvent | None = None,
) -> float:
    """Compute a composite source reliability score (0-1).

    Combines base reliability with type weight.
    """
    base = _base_reliability(source_name)
    type_weight = SOURCE_TYPE_WEIGHT.get(source_type.lower(), 0.20)
    score = base * 0.6 + type_weight * 0.4
    return round(min(1.0, max(0.0, score)), 4)


def score_news_event(event: NewsEvent) -> dict[str, float]:
    """Score a complete news event: source, novelty, recency, impact.

    Returns composite dict with breakdown.
    """
    source_score = score_source(event.source_name, event.source_type, event)

    # Recency score: decays over 24h
    hours_ago = _hours_since(event.published_at, event.received_at)
    recency_score = max(0.0, 1.0 - hours_ago / 24.0)

    # Novelty: based on revision
    novelty_score = 1.0 if event.revision == 1 else max(0.1, 1.0 / event.revision)

    # Impact proxy: entity count * source reliability
    entity_count = len(event.entities) if event.entities else 0
    impact_proxy = min(1.0, 0.3 + 0.2 * entity_count) * source_score

    # Composite
    composite = (
        source_score * 0.35
        + recency_score * 0.20
        + novelty_score * 0.15
        + impact_proxy * 0.30
    )

    return {
        "source": source_score,
        "recency": round(recency_score, 4),
        "novelty": round(novelty_score, 4),
        "impact": round(impact_proxy, 4),
        "composite": round(composite, 4),
    }


def _base_reliability(source_name: str) -> float:
    """Look up base reliability for a source name."""
    name = source_name.lower().strip()
    for key, val in SOURCE_RELIABILITY.items():
        if key in name:
            return val
    return SOURCE_RELIABILITY["unknown"]


def _hours_since(published_at: datetime, received_at: datetime) -> float:
    """Compute hours between published and received."""
    delta = received_at - published_at
    return abs(delta.total_seconds()) / 3600.0
