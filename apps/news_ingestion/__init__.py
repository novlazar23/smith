"""News Ingestion — RSS, Börsenankündigungen, regulatorische Quellen, Wirtschaftskalender.

Pipeline: Ingestion → Normalisierung → Dedup → Clustering → Entity Resolution
          → Source Scoring → Novelty → Impact → Historical Comparison.

Siehe Spec §7 und §13 (News-Agent Pipeline).
"""

from __future__ import annotations

from apps.news_ingestion.classifier import NewsStatus, classify_news
from apps.news_ingestion.config import NewsConfig, SourceConfig, default_source_configs
from apps.news_ingestion.ingest_rss import (
    deduplicate,
    ingest_feed,
    rss_fetch,
    url_hash,
)
from apps.news_ingestion.normalize import (
    calculate_event_identity,
    extract_entities,
    extract_instruments,
    normalize_item,
    resolve_entities,
)
from apps.news_ingestion.scheduler import (
    get_due_sources,
    run_ingestion_cycle,
    schedule_sources,
)

__all__ = [
    "NewsConfig",
    "NewsStatus",
    "SourceConfig",
    "calculate_event_identity",
    "classify_news",
    "deduplicate",
    "default_source_configs",
    "extract_entities",
    "extract_instruments",
    "get_due_sources",
    "ingest_feed",
    "normalize_item",
    "resolve_entities",
    "rss_fetch",
    "run_ingestion_cycle",
    "schedule_sources",
    "url_hash",
]
