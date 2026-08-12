"""News domain package — models, normalization, dedup, clustering, entity resolution, source scoring."""

from .dedup import Deduplicator
from .entity_resolution import resolve_entities
from .models import EntityMatch, NewsCluster, NewsEvent, NewsStatus
from .normalization import normalize_raw_news
from .source_scoring import score_news_event, score_source

__all__ = [
    "Deduplicator",
    "EntityMatch",
    "NewsCluster",
    "NewsEvent",
    "NewsStatus",
    "normalize_raw_news",
    "resolve_entities",
    "score_news_event",
    "score_source",
]
