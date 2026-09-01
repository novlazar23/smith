"""Tests für den News Ingestion Service.

Abdeckung:
  - Config: NewsConfig, SourceConfig, Default-Feeds
  - RSS Ingestion: fetch, parse, deduplicate
  - Normalization: normalize_item, extract_entities, extract_instruments
  - Classifier: keyword matching, historical comparison
  - Scheduler: due sources, retry logic
  - Integration: full pipeline
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from apps.news_ingestion.classifier import (
    NewsStatus,
    classify_news,
)
from apps.news_ingestion.config import (
    FeedType,
    NewsConfig,
    ProcessingSettings,
    SourceConfig,
    build_news_config,
    default_source_configs,
)
from apps.news_ingestion.ingest_rss import (
    NewsRawItem,
    _title_similarity,
    deduplicate,
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
    retry_delay,
    run_ingestion_cycle,
    schedule_sources,
)

# ═══════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════


class TestSourceConfig:
    """Tests für SourceConfig."""

    def test_default_values(self) -> None:
        cfg = SourceConfig(name="Test", url="https://example.com")
        assert cfg.feed_type == FeedType.RSS
        assert cfg.update_interval == 300
        assert cfg.priority == 3
        assert cfg.enabled is True
        assert cfg.headers == {}

    def test_custom_values(self) -> None:
        cfg = SourceConfig(
            name="SEC",
            url="https://sec.gov/rss.xml",
            feed_type=FeedType.RSS,
            update_interval=900,
            priority=2,
            enabled=True,
            headers={"Authorization": "Bearer xyz"},
        )
        assert cfg.name == "SEC"
        assert cfg.feed_type == FeedType.RSS
        assert cfg.update_interval == 900
        assert cfg.priority == 2
        assert cfg.headers == {"Authorization": "Bearer xyz"}

    def test_frozen_dataclass(self) -> None:
        from dataclasses import FrozenInstanceError

        cfg = SourceConfig(name="Test", url="https://example.com")
        with pytest.raises(FrozenInstanceError):
            cfg.name = "Changed"

    def test_json_feed_type(self) -> None:
        cfg = SourceConfig(name="API", url="https://api.example.com", feed_type=FeedType.JSON)
        assert cfg.feed_type == FeedType.JSON

    def test_api_feed_type(self) -> None:
        cfg = SourceConfig(name="API", url="https://api.example.com", feed_type=FeedType.API)
        assert cfg.feed_type == FeedType.API


class TestProcessingSettings:
    """Tests für ProcessingSettings."""

    def test_defaults(self) -> None:
        ps = ProcessingSettings()
        assert ps.max_items_per_feed == 50
        assert ps.dedup_window_hours == 24
        assert ps.entity_match_threshold == 0.8
        assert ps.retry_max_attempts == 3
        assert ps.retry_base_delay == 1.0

    def test_impact_levels(self) -> None:
        ps = ProcessingSettings()
        assert ps.impact_levels["low"] == 0.3
        assert ps.impact_levels["medium"] == 0.5
        assert ps.impact_levels["high"] == 0.7
        assert ps.impact_levels["critical"] == 0.9


class TestNewsConfig:
    """Tests für NewsConfig."""

    def test_empty_config(self) -> None:
        cfg = NewsConfig()
        assert cfg.sources == []
        assert cfg.feed_urls == []

    def test_with_sources(self) -> None:
        sources = [SourceConfig(name="A", url="https://a.com")]
        cfg = NewsConfig(sources=sources)
        assert len(cfg.sources) == 1
        assert cfg.sources[0].name == "A"


class TestDefaultSourceConfigs:
    """Tests für Standard-Quellen."""

    def test_returns_list(self) -> None:
        configs = default_source_configs()
        assert isinstance(configs, list)

    def test_has_expected_sources(self) -> None:
        configs = default_source_configs()
        names = [c.name for c in configs]
        assert "CoinDesk" in names
        assert "Cointelegraph" in names
        assert "Decrypt" in names
        assert "The Block" in names
        assert "Bitcoin Magazine" in names
        assert "Crypto Potato" in names

    def test_cookdesk_is_rss(self) -> None:
        configs = default_source_configs()
        coindesk = next(c for c in configs if c.name == "CoinDesk")
        assert coindesk.feed_type == FeedType.RSS

    def test_all_have_urls(self) -> None:
        configs = default_source_configs()
        for cfg in configs:
            assert cfg.url.startswith("http")


class TestBuildNewsConfig:
    """Tests für build_news_config."""

    def test_defaults(self) -> None:
        cfg = build_news_config()
        assert len(cfg.sources) > 0

    def test_with_extra_sources(self) -> None:
        extra = [SourceConfig(name="Extra", url="https://extra.com")]
        cfg = build_news_config(extra_sources=extra)
        names = [c.name for c in cfg.sources]
        assert "Extra" in names
        assert len(cfg.sources) > len(default_source_configs())

    def test_custom_processing(self) -> None:
        ps = ProcessingSettings(retry_max_attempts=5)
        cfg = build_news_config(processing=ps)
        assert cfg.processing.retry_max_attempts == 5


# ═══════════════════════════════════════════════
# RSS Ingestion Tests
# ═══════════════════════════════════════════════


class TestUrlHash:
    """Tests für url_hash."""

    def test_deterministic(self) -> None:
        h1 = url_hash("https://example.com/rss")
        h2 = url_hash("https://example.com/rss")
        assert h1 == h2

    def test_different_urls_different_hash(self) -> None:
        h1 = url_hash("https://a.com/rss")
        h2 = url_hash("https://b.com/rss")
        assert h1 != h2

    def test_returns_64_chars(self) -> None:
        h = url_hash("https://example.com/rss")
        assert len(h) == 64

    def test_is_hex(self) -> None:
        h = url_hash("https://example.com/rss")
        int(h, 16)  # sollte keine ValueError werfen


class TestDeduplicate:
    """Tests für deduplicate."""

    def test_exact_duplicate(self) -> None:
        items = [
            {"link": "https://example.com/1", "title": "Test"},
            {"link": "https://example.com/1", "title": "Test"},
        ]
        result = deduplicate(items)
        assert len(result) == 1

    def test_different_links_kept(self) -> None:
        items = [
            {"link": "https://example.com/1", "title": "Test A"},
            {"link": "https://example.com/2", "title": "Test B"},
        ]
        result = deduplicate(items)
        assert len(result) == 2

    def test_similar_titles_deduped(self) -> None:
        items = [
            {"link": "https://example.com/1", "title": "Bitcoin hits new high"},
            {"link": "https://example.com/2", "title": "Bitcoin Hits New High Today"},
        ]
        result = deduplicate(items)
        assert len(result) == 1

    def test_empty_list(self) -> None:
        assert deduplicate([]) == []

    def test_first_kept_over_duplicates(self) -> None:
        items = [
            {"link": "https://a.com/1", "title": "Original"},
            {"link": "https://b.com/2", "title": "Original"},
        ]
        result = deduplicate(items)
        assert result[0]["link"] == "https://a.com/1"

    def test_no_similarity_threshold(self) -> None:
        items = [
            {"link": "https://a.com/1", "title": "Bitcoin price"},
            {"link": "https://b.com/2", "title": "Banana bread recipe"},
        ]
        result = deduplicate(items)
        assert len(result) == 2


class TestTitleSimilarity:
    """Tests für _title_similarity."""

    def test_identical_titles(self) -> None:
        assert _title_similarity("Bitcoin", "Bitcoin") == 1.0

    def test_completely_different(self) -> None:
        s = _title_similarity("Bitcoin price", "Banana bread")
        assert s == 0.0

    def test_partial_overlap(self) -> None:
        s = _title_similarity("Bitcoin price update", "Bitcoin price goes up")
        assert s > 0.0

    def test_empty_titles(self) -> None:
        assert _title_similarity("", "Bitcoin") == 0.0
        assert _title_similarity("Bitcoin", "") == 0.0


class TestRssFetch:
    """Tests für rss_fetch."""

    @patch("apps.news_ingestion.ingest_rss.HTTPX_AVAILABLE", new=False)
    def test_requires_httpx(self) -> None:
        with patch("apps.news_ingestion.ingest_rss.httpx", None), pytest.raises(
            RuntimeError, match="httpx required"
        ):
            rss_fetch("https://example.com/rss")

    @patch("apps.news_ingestion.ingest_rss.httpx")
    def test_returns_empty_on_error(self, mock_httpx: MagicMock) -> None:
        mock_httpx.get.side_effect = Exception("Network error")
        result = rss_fetch("https://example.com/rss")
        assert result == []

    @patch("apps.news_ingestion.ingest_rss.httpx")
    def test_returns_empty_on_http_error(self, mock_httpx: MagicMock) -> None:
        import httpx
        mock_httpx.get.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )
        result = rss_fetch("https://example.com/rss")
        assert result == []

    @patch("apps.news_ingestion.ingest_rss.httpx")
    def test_parses_rss_feed(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = (
            '<?xml version="1.0"?><rss version="2.0">'
            "<channel><title>Test</title>"
            "<item><title>Bitcoin News</title><link>https://example.com/1</link>"
            "<description>BTC update</description>"
            "<pubDate>Mon, 01 Jan 2024 00:00:00 +0000</pubDate></item>"
            "</channel></rss>"
        )
        mock_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = mock_response

        result = rss_fetch("https://example.com/rss")
        assert len(result) == 1
        assert result[0]["title"] == "Bitcoin News"
        assert result[0]["link"] == "https://example.com/1"


# ═══════════════════════════════════════════════
# Normalization Tests
# ═══════════════════════════════════════════════


class TestNormalizeItem:
    """Tests für normalize_item."""

    def test_basic_normalization(self) -> None:
        raw = NewsRawItem(
            title="Bitcoin Hits $100K",
            body="Bitcoin reaches milestone",
            source_url="https://example.com/1",
            published_at=datetime.now(UTC),
            source_name="Test Source",
            source_type="RSS",
        )
        result = normalize_item(raw)
        assert result["title"] == "Bitcoin Hits $100K"
        assert result["source_name"] == "Test Source"
        assert result["source_type"] == "RSS"
        assert result["language"] == "en"
        assert result["revision"] == 1
        assert len(result["news_id"]) > 0
        assert len(result["event_identity"]) > 0

    def test_url_hash_field(self) -> None:
        raw = NewsRawItem(
            title="Test", body="Body", source_url="https://example.com/1",
            published_at=None, source_name="Src", source_type="RSS",
        )
        result = normalize_item(raw)
        assert result["url_hash"] == url_hash("https://example.com/1")

    def test_deterministic_identity(self) -> None:
        raw1 = NewsRawItem(
            title="Same", body="Same body", source_url="https://example.com/1",
            published_at=None, source_name="Src", source_type="RSS",
        )
        raw2 = NewsRawItem(
            title="Same", body="Same body", source_url="https://example.com/1",
            published_at=None, source_name="Src", source_type="RSS",
        )
        r1 = normalize_item(raw1)
        r2 = normalize_item(raw2)
        assert r1["event_identity"] == r2["event_identity"]
        assert r1["news_id"] == r2["news_id"]


class TestExtractEntities:
    """Tests für extract_entities."""

    def test_sec_entity(self) -> None:
        entities = extract_entities("SEC announces new regulations for crypto")
        assert "U.S. Securities and Exchange Commission" in entities

    def test_ecb_entity(self) -> None:
        entities = extract_entities("The ECB raises interest rates")
        assert "European Central Bank" in entities

    def test_multiple_entities(self) -> None:
        entities = extract_entities("SEC and CFTC work together on crypto regulation")
        assert "U.S. Securities and Exchange Commission" in entities
        assert "Commodity Futures Trading Commission" in entities

    def test_no_entities(self) -> None:
        entities = extract_entities("Just a normal news article about weather")
        assert entities == []

    def test_empty_text(self) -> None:
        assert extract_entities("") == []

    def test_case_insensitive(self) -> None:
        entities = extract_entities("the sec announces new rules")
        assert "U.S. Securities and Exchange Commission" in entities


class TestExtractInstruments:
    """Tests für extract_instruments."""

    def test_btc_mention(self) -> None:
        instruments = extract_instruments("Bitcoin price surge continues")
        assert "BTC" in instruments

    def test_eth_mention(self) -> None:
        instruments = extract_instruments("Ethereum upgrade announced")
        assert "ETH" in instruments

    def test_multiple_instruments(self) -> None:
        instruments = extract_instruments("BTC and ETH both rising today")
        assert "BTC" in instruments
        assert "ETH" in instruments

    def test_no_instruments(self) -> None:
        instruments = extract_instruments("The weather is nice today")
        assert instruments == []

    def test_empty_text(self) -> None:
        assert extract_instruments("") == []

    def test_btcusdt_longer_match(self) -> None:
        instruments = extract_instruments("Trading BTCUSDT pairs")
        # BTCUSDT sollte zu BTC auflösen
        assert "BTC" in instruments

    def test_sol_mention(self) -> None:
        instruments = extract_instruments("Solana SOL price update")
        assert "SOL" in instruments

    def test_xrp_mention(self) -> None:
        instruments = extract_instruments("XRP court case decision")
        assert "XRP" in instruments

    def test_doge_mention(self) -> None:
        instruments = extract_instruments("Dogecoin DOGE community news")
        assert "DOGE" in instruments

    def test_polygon_mention(self) -> None:
        instruments = extract_instruments("Polygon MATIC network upgrade")
        assert "MATIC" in instruments


class TestResolveEntities:
    """Tests für resolve_entities."""

    def test_resolve_sec(self) -> None:
        entities = resolve_entities(["SEC", "some other entity"])
        assert "U.S. Securities and Exchange Commission" in entities

    def test_resolve_bitcoin(self) -> None:
        entities = resolve_entities(["BITCOIN"])
        assert "BTC" in entities

    def test_resolve_no_synonym(self) -> None:
        entities = resolve_entities(["UnknownEntity"])
        assert "UnknownEntity" in entities

    def test_empty_list(self) -> None:
        assert resolve_entities([]) == []

    def test_duplicates_removed(self) -> None:
        entities = resolve_entities(["SEC", "SEC", "SEC"])
        result = [e for e in entities if "Securities" in e]
        assert len(result) == 1


class TestCalculateEventIdentity:
    """Tests für calculate_event_identity."""

    def test_deterministic(self) -> None:
        i1 = calculate_event_identity("Title", "Body", "Source")
        i2 = calculate_event_identity("Title", "Body", "Source")
        assert i1 == i2

    def test_different_title_different_identity(self) -> None:
        i1 = calculate_event_identity("Title A", "Body", "Source")
        i2 = calculate_event_identity("Title B", "Body", "Source")
        assert i1 != i2

    def test_different_source_different_identity(self) -> None:
        i1 = calculate_event_identity("Title", "Body", "Source A")
        i2 = calculate_event_identity("Title", "Body", "Source B")
        assert i1 != i2

    def test_returns_32_chars(self) -> None:
        identity = calculate_event_identity("T", "B", "S")
        assert len(identity) == 32

    def test_empty_title_still_works(self) -> None:
        identity = calculate_event_identity("", "Body", "Source")
        assert len(identity) > 0


# ═══════════════════════════════════════════════
# Classifier Tests
# ═══════════════════════════════════════════════


class TestNewsStatusEnum:
    """Tests für NewsStatus Enum."""

    def test_all_statuses_present(self) -> None:
        assert NewsStatus.RUMOR == "RUMOR"
        assert NewsStatus.INITIAL == "INITIAL"
        assert NewsStatus.CONFIRMATION == "CONFIRMATION"
        assert NewsStatus.UPDATE == "UPDATE"
        assert NewsStatus.CORRECTION == "CORRECTION"
        assert NewsStatus.RETRACTION == "RETRACTION"

    def test_is_str_enum(self) -> None:
        assert isinstance(NewsStatus.INITIAL, str)


class TestClassifyNewsKeywords:
    """Tests für Keyword-basierte Klassifikation."""

    def test_rumor_keywords(self) -> None:
        item = {"title": "Rumor: SEC to ban Bitcoin", "body": "Unconfirmed report"}
        status = classify_news(item)
        assert status == NewsStatus.RUMOR

    def test_initial_keywords(self) -> None:
        item = {"title": "Binance announces new listing", "body": "First launch"}
        status = classify_news(item)
        assert status == NewsStatus.INITIAL

    def test_confirmation_keywords(self) -> None:
        item = {"title": "SEC confirms new rule", "body": "Verified official statement"}
        status = classify_news(item)
        assert status == NewsStatus.CONFIRMATION

    def test_update_keywords(self) -> None:
        item = {"title": "Project update: Phase Two", "body": "Further development"}
        status = classify_news(item)
        assert status == NewsStatus.UPDATE

    def test_correction_keywords(self) -> None:
        item = {"title": "Correction: Error in report", "body": "Clarification needed"}
        status = classify_news(item)
        assert status == NewsStatus.CORRECTION

    def test_retraction_keywords(self) -> None:
        item = {"title": "Service to be retracted", "body": "Withdrawal announced"}
        status = classify_news(item)
        assert status == NewsStatus.RETRACTION

    def test_empty_item_defaults_initial(self) -> None:
        item = {"title": "", "body": ""}
        status = classify_news(item)
        assert status == NewsStatus.INITIAL


class TestClassifyNewsHistory:
    """Tests für history-basierte Klassifikation."""

    def test_same_source_similar_title_update(self) -> None:
        history = [{
            "title": "SEC Announces New Rule",
            "source_name": "CoinDesk",
            "event_identity": "abc123",
            "status": "CONFIRMATION",
        }]
        item = {
            "title": "SEC Announces New Rule Update",
            "source_name": "CoinDesk",
            "event_identity": "def456",
        }
        status = classify_news(item, history)
        assert status == NewsStatus.UPDATE

    def test_same_source_similar_title_confirmation(self) -> None:
        history = [{
            "title": "Binance Lists New Token",
            "source_name": "Binance Announcements",
            "event_identity": "abc123",
            "status": "INITIAL",
        }]
        item = {
            "title": "Binance Lists New Token Again",
            "source_name": "Binance Announcements",
            "event_identity": "def456",
        }
        status = classify_news(item, history)
        assert status == NewsStatus.CONFIRMATION

    def test_different_source_no_history_match(self) -> None:
        history = [{
            "title": "SEC News",
            "source_name": "CoinDesk",
            "event_identity": "abc123",
            "status": "INITIAL",
        }]
        item = {
            "title": "SEC News",
            "source_name": "Binance Announcements",
            "event_identity": "different",
        }
        status = classify_news(item, history)
        assert status == NewsStatus.INITIAL

    def test_same_identity_update(self) -> None:
        history = [{
            "title": "Original",
            "source_name": "Source",
            "event_identity": "same123",
            "status": "INITIAL",
        }]
        item = {
            "title": "Same Original",
            "source_name": "Source",
            "event_identity": "same123",
        }
        status = classify_news(item, history)
        assert status == NewsStatus.UPDATE

    def test_no_history(self) -> None:
        item = {"title": "Test", "body": "Just testing"}
        status = classify_news(item, None)
        assert status == NewsStatus.INITIAL

    def test_empty_history(self) -> None:
        item = {"title": "Test", "body": "Just testing"}
        status = classify_news(item, [])
        assert status == NewsStatus.INITIAL


# ═══════════════════════════════════════════════
# Scheduler Tests
# ═══════════════════════════════════════════════


class TestRetryDelay:
    """Tests für exponentielles Backoff."""

    def test_base_delay(self) -> None:
        assert retry_delay(0, 1.0) == 1.0

    def test_exponential_growth(self) -> None:
        assert retry_delay(1, 1.0) == 2.0
        assert retry_delay(2, 1.0) == 4.0
        assert retry_delay(3, 1.0) == 8.0

    def test_capped_at_300(self) -> None:
        assert retry_delay(10, 1.0) == 300.0
        assert retry_delay(20, 1.0) == 300.0

    def test_custom_base_delay(self) -> None:
        assert retry_delay(0, 5.0) == 5.0
        assert retry_delay(1, 5.0) == 10.0


class TestGetDueSources:
    """Tests für get_due_sources."""

    def test_first_run_all_due(self) -> None:
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", update_interval=300),
            SourceConfig(name="B", url="https://b.com", update_interval=600),
        ])
        due = get_due_sources({}, config)
        assert len(due) == 2
        names = [s.name for s in due]
        assert "A" in names
        assert "B" in names

    def test_not_due_yet(self) -> None:
        now = datetime.now(UTC)
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", update_interval=300),
        ])
        last_times = {"A": now}  # gerade eben gelaufen
        due = get_due_sources(last_times, config)
        assert len(due) == 0

    def test_disabled_source_not_due(self) -> None:
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", enabled=False),
        ])
        due = get_due_sources({}, config)
        assert len(due) == 0

    def test_past_run_due(self) -> None:
        now = datetime.now(UTC)
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", update_interval=60),
        ])
        old_time = now - timedelta(seconds=120)
        due = get_due_sources({"A": old_time}, config)
        assert len(due) == 1

    def test_exact_boundary(self) -> None:
        now = datetime.now(UTC)
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", update_interval=300),
        ])
        past = now - timedelta(seconds=300)
        due = get_due_sources({"A": past}, config)
        assert len(due) == 1


class TestScheduleSources:
    """Tests für schedule_sources."""

    def test_initial_schedule(self) -> None:
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", update_interval=300),
            SourceConfig(name="B", url="https://b.com", update_interval=600),
        ])
        schedule = schedule_sources(config)
        assert "A" in schedule
        assert "B" in schedule

    def test_schedule_increases_with_last_run(self) -> None:
        now = datetime.now(UTC)
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", update_interval=300),
        ])
        old_run = now - timedelta(seconds=100)
        schedule = schedule_sources(config, {"A": old_run})
        assert schedule["A"] == old_run + timedelta(seconds=300)


class TestRunIngestionCycle:
    """Tests für den vollständigen Ingestion-Zyklus."""

    @patch("apps.news_ingestion.scheduler.ingest_feed")
    def test_empty_cycle(self, mock_ingest: MagicMock) -> None:
        mock_ingest.return_value = []
        config = build_news_config()
        result = run_ingestion_cycle(config, {})
        assert result == []

    @patch("apps.news_ingestion.scheduler.ingest_feed")
    def test_returns_events(self, mock_ingest: MagicMock) -> None:
        raw_item = NewsRawItem(
            title="Test News",
            body="Test body",
            source_url="https://example.com/1",
            published_at=datetime.now(UTC),
            source_name="TestSource",
            source_type="RSS",
        )
        mock_ingest.return_value = [raw_item]
        config = NewsConfig(sources=[
            SourceConfig(name="TestSource", url="https://test.com", update_interval=60),
        ])
        result = run_ingestion_cycle(config, {})
        assert len(result) == 1
        assert result[0]["title"] == "Test News"
        assert isinstance(result[0]["status"], str)


# ═══════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════


class TestFullPipeline:
    """Integrationstests für die komplette Pipeline."""

    def test_full_pipeline_flow(self) -> None:
        """Fetch → Dedup → Normalize → Classify."""
        raw = NewsRawItem(
            title="SEC Announces Bitcoin ETF Approval",
            body="The SEC has officially approved the new Bitcoin ETF.",
            source_url="https://example.com/sec-btc-etf",
            published_at=datetime.now(UTC),
            source_name="CoinDesk",
            source_type="RSS",
        )

        # Normalize
        event = normalize_item(raw)
        assert event["title"] == "SEC Announces Bitcoin ETF Approval"
        assert event["source_name"] == "CoinDesk"
        assert "U.S. Securities and Exchange Commission" in event["entities"]
        assert "BTC" in event["instruments"]
        assert event["language"] == "en"
        assert event["revision"] == 1

        # Classify
        status = classify_news(event, None)
        assert status in (NewsStatus.INITIAL, NewsStatus.CONFIRMATION)

    def test_dedup_before_normalize(self) -> None:
        """Deduplizierung entfernt Duplikate vor der Normalisierung."""
        items = [
            {"link": "https://example.com/1", "title": "Same News"},
            {"link": "https://example.com/1", "title": "Same News"},
            {"link": "https://example.com/2", "title": "Different News"},
        ]
        deduped = deduplicate(items)
        assert len(deduped) == 2

    def test_normalization_preserves_original_data(self) -> None:
        """Normalisierung verändert Originaldaten nicht."""
        raw = NewsRawItem(
            title="Original Title",
            body="Original Body",
            source_url="https://example.com/1",
            published_at=datetime.now(UTC),
            source_name="Test",
            source_type="RSS",
        )
        title = raw.title
        _ = normalize_item(raw)
        assert raw.title == title

    def test_scheduler_integration(self) -> None:
        """Scheduler ermittelt korrekt fällige Quellen."""
        config = NewsConfig(sources=[
            SourceConfig(name="A", url="https://a.com", update_interval=60),
            SourceConfig(name="B", url="https://b.com", update_interval=300),
        ])
        now = datetime.now(UTC)
        last_times = {
            "A": now - timedelta(seconds=120),
            "B": now - timedelta(seconds=10),
        }
        due = get_due_sources(last_times, config)
        names = [s.name for s in due]
        assert "A" in names
        assert "B" not in names

    def test_news_raw_item_url_hash(self) -> None:
        """NewsRawItem.url_hash korrekter SHA256."""
        item = NewsRawItem(
            title="Test", body="Body", source_url="https://example.com/1",
            published_at=None, source_name="Src", source_type="RSS",
        )
        assert item.url_hash == url_hash("https://example.com/1")

    def test_normalize_with_none_body(self) -> None:
        """Normalisierung mit leerem Body funktioniert."""
        raw = NewsRawItem(
            title="Title Only",
            body="",
            source_url="https://example.com/1",
            published_at=None,
            source_name="Src",
            source_type="RSS",
        )
        result = normalize_item(raw)
        assert result["body"] == ""
        assert result["news_id"] != ""

    def test_classify_with_history_and_keywords(self) -> None:
        """Klassifikation kombiniert History und Keywords."""
        history = [{
            "title": "BTC Price Update",
            "source_name": "Source",
            "event_identity": "old_id",
            "status": "INITIAL",
        }]
        item = {
            "title": "BTC Price Update Confirmed",
            "body": "SEC confirms the previous report",
            "source_name": "Source",
            "event_identity": "new_id",
        }
        status = classify_news(item, history)
        # Sollte CONFIRMATION sein (history + confirmation keywords)
        assert status in (NewsStatus.CONFIRMATION, NewsStatus.UPDATE)

    def test_extract_instruments_case_insensitive(self) -> None:
        """Instrument-Erkennung ist case-insensitive."""
        assert "BTC" in extract_instruments("bitcoin price")
        assert "BTC" in extract_instruments("BITCOIN price")
        assert "ETH" in extract_instruments("ethereum upgrade")

    def test_extract_entities_handles_special_chars(self) -> None:
        """Entity-Erkennung mit Sonderzeichen."""
        entities = extract_entities("The SEC. announced new rules!")
        assert "U.S. Securities and Exchange Commission" in entities
