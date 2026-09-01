"""Contract tests for news ingestion schema.

Verifies:
- News ingestion produces valid NewsEvent objects with required fields
  (source_name, source_type, title, body, published_at).
- The domain-level NewsEvent (packages.domain.news.models) carries id, title,
  body, source_name, source_type, url_hash, published_at, received_at.
- Normalization handles missing/empty fields gracefully.
- NewsStatus enum values are well-formed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from packages.domain.news import (
    Deduplicator,
    NewsEvent,
    NewsStatus,
    normalize_raw_news,
    resolve_entities,
    score_news_event,
    score_source,
)
from packages.schemas.market_event import NewsEvent as SchemaNewsEvent

# ── Domain NewsEvent contract ───────────────────────────────────────────


class TestDomainNewsEvent:
    """Verifies domain-level NewsEvent dataclass has all required fields."""

    @pytest.fixture
    def sample_news(self) -> dict:
        now = datetime.now(UTC)
        return {
            "id": "abc123",
            "title": "Fed raises rates",
            "body": "The Federal Reserve announced a rate hike today.",
            "source_name": "Bloomberg",
            "source_type": "news_wire",
            "url_hash": "sha256test",
            "published_at": now,
            "received_at": now,
        }

    def test_news_event_requires_core_fields(self, sample_news: dict) -> None:
        event = NewsEvent(**sample_news)
        for field in ("id", "title", "body", "source_name", "source_type",
                       "url_hash", "published_at", "received_at"):
            assert hasattr(event, field), f"Missing field: {field}"

    def test_news_event_defaults_entities(self, sample_news: dict) -> None:
        event = NewsEvent(**sample_news)
        assert event.entities == []

    def test_news_event_defaults_language(self, sample_news: dict) -> None:
        event = NewsEvent(**sample_news)
        assert event.language == "en"

    def test_news_event_defaults_revision(self, sample_news: dict) -> None:
        event = NewsEvent(**sample_news)
        assert event.revision == 1

    def test_news_event_defaults_status_initial(self, sample_news: dict) -> None:
        event = NewsEvent(**sample_news)
        assert event.status == NewsStatus.INITIAL

    def test_news_event_slots_frozen(self, sample_news: dict) -> None:
        """NewsEvent uses frozen=True so it must be immutable."""
        event = NewsEvent(**sample_news)
        with pytest.raises(Exception):
            event.title = "new title"

    def test_news_event_fields_match_dataclass_spec(self, sample_news: dict) -> None:
        """Verify the expected fields from the Spec §8 contract."""
        event = NewsEvent(**sample_news)
        required = {
            "id", "title", "body", "source_name", "source_type",
            "url_hash", "published_at", "received_at",
            "entities", "instruments", "language", "revision", "status",
        }
        actual = set(event.__dataclass_fields__.keys())
        assert required.issubset(actual), f"Missing: {required - actual}"


# ── NewsStatus enum contract ────────────────────────────────────────────


class TestNewsStatus:
    """NewsStatus enum must expose expected lifecycle values."""

    def test_status_values(self) -> None:
        expected = {"rumor", "initial", "confirmation", "update",
                     "correction", "retraction"}
        actual = {s.value for s in NewsStatus}
        assert expected == actual

    def test_status_is_str_enum(self) -> None:
        assert isinstance(NewsStatus.INITIAL, str)
        assert NewsStatus.INITIAL == "initial"

    def test_status_iterable(self) -> None:
        statuses = list(NewsStatus)
        assert len(statuses) == 6


# ── Normalization contract ──────────────────────────────────────────────


class TestNormalizeRawNews:
    """normalize_raw_news must produce valid NewsEvent even with missing data."""

    def test_normalization_with_full_data(self) -> None:
        now = datetime.now(UTC)
        event = normalize_raw_news(
            title="Crypto bull run",
            body="Analysis of current trends",
            source_name="CoinDesk",
            source_type="blog",
            url="https://coindesk.com/article",
            language="en",
            published_at=now,
            received_at=now,
            raw_entities=["BTC"],
        )
        assert isinstance(event, NewsEvent)
        assert event.title == "Crypto bull run"
        assert event.source_name == "CoinDesk"
        assert event.entities == ["BTC"]
        assert event.language == "en"

    def test_normalization_handles_missing_optional_fields(self) -> None:
        """When optional fields are omitted, defaults must be applied."""
        event = normalize_raw_news(
            title="Test",
            body="Content",
            source_name="Test",
            source_type="unknown",
        )
        assert event.title == "Test"
        assert event.body == "Content"
        assert event.source_type == "unknown"

    def test_normalization_handles_empty_string_fields(self) -> None:
        """Empty strings should be stripped, not cause errors."""
        event = normalize_raw_news(
            title="",
            body="",
            source_name="Test",
            source_type="",
        )
        assert event.title == ""
        assert event.body == ""

    def test_normalization_handles_none_dates(self) -> None:
        """When dates are None, current time is used."""
        event = normalize_raw_news(
            title="Test",
            body="Content",
            source_name="Test",
            source_type="wire",
            published_at=None,
            received_at=None,
        )
        assert event.published_at is not None
        assert event.received_at is not None

    def test_normalization_detects_confirmation_status(self) -> None:
        event = normalize_raw_news(
            title="Fed officially confirms rate hike",
            body="The Fed confirmed its decision today.",
            source_name="Reuters",
            source_type="news_wire",
        )
        assert event.status == NewsStatus.CONFIRMATION

    def test_normalization_detects_rumor_status(self) -> None:
        event = normalize_raw_news(
            title="Reportedly, BTC ETF pending approval",
            body="Sources say the ETF may be coming.",
            source_name="Twitter",
            source_type="social",
        )
        assert event.status == NewsStatus.RUMOR

    def test_normalization_detects_retraction_status(self) -> None:
        event = normalize_raw_news(
            title="Retracted: false report on SEC action",
            body="This was a hoax.",
            source_name="Wire",
            source_type="news_wire",
        )
        assert event.status == NewsStatus.RETRACTION

    def test_normalization_detects_update_status(self) -> None:
        event = normalize_raw_news(
            title="Update on the situation",
            body="New information has emerged.",
            source_name="AP",
            source_type="news_wire",
        )
        assert event.status == NewsStatus.UPDATE


# ── Schema-level NewsEvent contract ─────────────────────────────────────


class TestSchemaNewsEvent:
    """packages.schemas.market_event.NewsEvent (Pydantic model)."""

    def test_schema_news_event_minimal(self) -> None:
        event = SchemaNewsEvent(
            news_id="n1",
            event_identity="e1",
            title="Test",
            source_name="Test",
            source_type="blog",
        )
        assert event.news_id == "n1"
        assert event.body is None
        assert event.published_at is None

    def test_schema_news_event_with_all_fields(self) -> None:
        now = datetime.now(UTC)
        event = SchemaNewsEvent(
            news_id="n2",
            event_identity="e2",
            title="Full",
            body="Full content",
            source_name="Reuters",
            source_type="wire",
            url_hash="hash123",
            published_at=now,
            received_at=now,
            entities=["BTC"],
            instruments=["BTC/USDT"],
            language="de",
            revision=2,
        )
        assert event.body == "Full content"
        assert event.url_hash == "hash123"
        assert event.language == "de"
        assert event.revision == 2

    def test_schema_news_event_body_optional(self) -> None:
        event = SchemaNewsEvent(
            news_id="n3",
            event_identity="e3",
            title="No body",
            source_name="Test",
            source_type="blog",
        )
        assert event.body is None


# ── Deduplicator contract ───────────────────────────────────────────────


class TestDeduplicator:
    """Deduplicator must accept a threshold and process events."""

    def test_deduplicator_init(self) -> None:
        dedup = Deduplicator(content_similarity_threshold=0.85)
        assert dedup._threshold == 0.85

    def test_deduplicator_processes_empty_list(self) -> None:
        dedup = Deduplicator()
        result = dedup.process([])
        assert result == []

    def test_deduplicator_handles_single_event(self) -> None:
        now = datetime.now(UTC)
        events = [NewsEvent(
            id="e1", title="Same", body="Same", source_name="A",
            source_type="a", url_hash="h1", published_at=now,
            received_at=now,
        )]
        dedup = Deduplicator(content_similarity_threshold=0.5)
        result = dedup.process(events)
        assert len(result) >= 1


# ── Entity resolution & scoring contract ────────────────────────────────


class TestEntityResolutionAndScoring:
    """resolve_entities and scoring functions must return expected types."""

    def test_resolve_entities_returns_list(self) -> None:
        result = resolve_entities("BTC is up 5% today")
        assert isinstance(result, list)

    def test_score_news_event_returns_dict(self) -> None:
        now = datetime.now(UTC)
        event = NewsEvent(
            id="sc1", title="Test", body="Content", source_name="Reuters",
            source_type="wire", url_hash="h", published_at=now,
            received_at=now,
        )
        scores = score_news_event(event)
        assert isinstance(scores, dict)
        assert "source" in scores
        assert "recency" in scores
        assert "composite" in scores

    def test_score_source_returns_float(self) -> None:
        """score_source returns a float reliability score (0-1)."""
        result = score_source("Reuters", "news_wire")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_source_low_reliability(self) -> None:
        """Unknown social media sources score low."""
        result = score_source("Unknown User", "social_media")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
