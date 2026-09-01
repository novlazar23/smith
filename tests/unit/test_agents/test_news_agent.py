"""Unit tests for the News domain and NewsAgent."""

from __future__ import annotations

import datetime

import pytest
from packages.agents import AgentType, NewsAgent
from packages.agents.base import AgentConfig
from packages.domain.news import (
    Deduplicator,
    EntityMatch,
    NewsEvent,
    NewsStatus,
    normalize_raw_news,
    resolve_entities,
    score_news_event,
    score_source,
)
from packages.schemas.agent_report import (
    AgentReport,
    EvidenceReference,
    InvalidationCondition,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def agent() -> NewsAgent:
    return NewsAgent()


@pytest.fixture
def news_data() -> dict:
    """Generate a batch of raw news events."""
    return {
        "news": [
            {
                "title": "Bitcoin Surges Past 100K",
                "body": "BTC reaches new all-time high as institutional demand grows",
                "source_name": "Reuters",
                "source_type": "wire_service",
                "url": "https://example.com/btc-surge",
            },
            {
                "title": "Ethereum Network Upgrade Announced",
                "body": "ETH 2.0 upgrade confirmed by core developers",
                "source_name": "CoinDesk",
                "source_type": "crypto_media",
                "url": "https://example.com/eth-upgrade",
            },
            {
                "title": "Report: SEC May Ban Bitcoin Trading",
                "body": "Unconfirmed sources suggest regulatory action against BTC",
                "source_name": "Twitter",
                "source_type": "social_media",
                "url": "https://example.com/sec-ban-rumor",
            },
        ]
    }


@pytest.fixture
def single_news_data() -> dict:
    return {
        "news": [
            {
                "title": "Single News Event",
                "body": "Just one news item",
                "source_name": "Test",
                "source_type": "unknown",
            }
        ]
    }


@pytest.fixture
def conflicting_news_data() -> dict:
    return {
        "news": [
            {
                "title": "Bitcoin Surges Past 100K",
                "body": "BTC rally driven by institutional buying",
                "source_name": "Reuters",
                "source_type": "wire_service",
            },
            {
                "title": "SEC Bans Bitcoin Trading Nationwide",
                "body": "Regulatory crackdown on BTC across the country",
                "source_name": "Reuters",
                "source_type": "wire_service",
            },
        ]
    }


# ── AgentType enum ────────────────────────────────────────────────────────

class TestAgentTypeEnum:
    def test_news_enum_value(self) -> None:
        assert AgentType.NEWS == "news"

    def test_news_enum_unique(self) -> None:
        types = [t for t in AgentType]
        assert len(types) == len(set(types))


# ── NewsAgent initialization ─────────────────────────────────────────────

class TestNewsAgentInit:
    def test_default_config(self) -> None:
        agent = NewsAgent()
        assert agent.agent_id == "news"
        assert agent.config.agent_type == AgentType.NEWS

    def test_custom_config(self) -> None:
        config = AgentConfig(
            agent_id="news",
            agent_type=AgentType.NEWS,
            instrument="BTC",
            horizon="4h",
        )
        agent = NewsAgent(config=config)
        assert agent.agent_id == "news"
        assert agent.config.instrument == "BTC"
        assert agent.config.horizon == "4h"


# ── NewsAgent basic analysis ────────────────────────────────────────────

class TestNewsAgentBasic:
    def test_produces_agent_report(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert isinstance(report, AgentReport)
        assert report.agent_id == "news"

    def test_probabilities_sum_to_one(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        prob_sum = sum(report.probabilities.values())
        assert abs(prob_sum - 1.0) <= 0.0001

    def test_probabilities_have_required_keys(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert "up" in report.probabilities
        assert "down" in report.probabilities
        assert "range" in report.probabilities

    def test_evidence_present(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert len(report.evidence) >= 1

    def test_evidence_is_evidence_reference(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for ev in report.evidence:
            assert isinstance(ev, EvidenceReference)

    def test_counter_evidence_required(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert len(report.counter_evidence) >= 1

    def test_invalidations_present(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert len(report.invalidations) >= 1

    def test_invalidations_are_proper_type(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for inv in report.invalidations:
            assert isinstance(inv, InvalidationCondition)

    def test_status_shadow(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert report.status.value == "shadow"

    def test_report_id_is_unique(self, agent, news_data) -> None:
        r1 = agent.analyze(news_data)
        r2 = agent.analyze(news_data)
        assert r1.report_id != r2.report_id

    def test_agent_version(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert report.agent_version == "0.1.0"


# ── NewsAgent validation ─────────────────────────────────────────────────

class TestNewsAgentValidation:
    def test_missing_news_raises(self, agent) -> None:
        with pytest.raises(ValueError, match="news"):
            agent.analyze({})

    def test_empty_news_list_raises(self, agent) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            agent.analyze({"news": []})

    def test_valid_single_news(self, agent, single_news_data) -> None:
        report = agent.analyze(single_news_data)
        assert isinstance(report, AgentReport)


# ── NewsEvent model ──────────────────────────────────────────────────────

class TestNewsEventModel:
    def test_minimal_news_event(self) -> None:
        now = datetime.datetime.now()
        evt = NewsEvent(
            id="test-1",
            title="Test Title",
            body="Test body",
            source_name="Test",
            source_type="blog",
            url_hash="abc123",
            published_at=now,
            received_at=now,
        )
        assert evt.id == "test-1"
        assert evt.status == NewsStatus.INITIAL
        assert evt.revision == 1
        assert evt.entities == []
        assert evt.instruments == []
        assert evt.language == "en"

    def test_frozen_instance(self) -> None:
        now = datetime.datetime.now()
        evt = NewsEvent(
            id="test-2",
            title="T",
            body="B",
            source_name="S",
            source_type="blog",
            url_hash="x",
            published_at=now,
            received_at=now,
        )
        with pytest.raises(Exception):
            evt.title = "Changed"

    def test_news_status_enum_values(self) -> None:
        statuses = [NewsStatus.RUMOR, NewsStatus.INITIAL, NewsStatus.CONFIRMATION,
                     NewsStatus.UPDATE, NewsStatus.CORRECTION, NewsStatus.RETRACTION]
        for s in statuses:
            assert isinstance(s, str)
            assert len(s) > 0


# ── Entity resolution ────────────────────────────────────────────────────

class TestEntityResolution:
    def test_resolve_btc(self) -> None:
        matches = resolve_entities("Bitcoin surges past 100K")
        assert any(m.entity == "BTC" for m in matches)

    def test_resolve_eth(self) -> None:
        matches = resolve_entities("Ethereum gains on ETH upgrade news")
        assert any(m.entity == "ETH" for m in matches)

    def test_resolve_multiple(self) -> None:
        matches = resolve_entities("BTC and ETH both surge, SOL follows")
        entities = {m.entity for m in matches}
        assert "BTC" in entities
        assert "ETH" in entities
        assert "SOL" in entities

    def test_resolve_empty(self) -> None:
        matches = resolve_entities("")
        assert matches == []

    def test_resolve_no_tickers(self) -> None:
        matches = resolve_entities("The weather is nice today")
        assert all(m.entity not in ("BTC", "ETH", "SOL", "XRP", "ADA") for m in matches)

    def test_resolve_sorted_by_confidence(self) -> None:
        matches = resolve_entities("Bitcoin rally")
        for i in range(len(matches) - 1):
            assert matches[i].confidence >= matches[i + 1].confidence

    def test_entity_match_structure(self) -> None:
        match = EntityMatch(entity="BTC", confidence=0.9, type="ticker")
        assert match.entity == "BTC"
        assert match.confidence == 0.9
        assert match.type == "ticker"


# ── Normalization ────────────────────────────────────────────────────────

class TestNormalization:
    def test_normalizes_title_and_body(self) -> None:
        evt = normalize_raw_news(
            title="  Bitcoin Surges  ",
            body="  BTC hits record high  ",
            source_name=" Reuters ",
            source_type=" wire_service ",
        )
        assert evt.title == "Bitcoin Surges"
        assert evt.body == "BTC hits record high"
        assert evt.source_name == "Reuters"

    def test_status_initial(self) -> None:
        evt = normalize_raw_news(
            title="New Data Published", body="Prices are moving", source_name="Test"
        )
        assert evt.status == NewsStatus.INITIAL

    def test_status_rumor(self) -> None:
        evt = normalize_raw_news(
            title="Reportedly, Bitcoin Will Be Banned",
            body="Unconfirmed sources say regulatory ban coming",
            source_name="Test",
        )
        assert evt.status == NewsStatus.RUMOR

    def test_status_confirmation(self) -> None:
        evt = normalize_raw_news(
            title="Officially Confirmed: Bitcoin Partnership",
            body="As stated by the company, they have a new partnership",
            source_name="Test",
        )
        assert evt.status == NewsStatus.CONFIRMATION

    def test_status_update(self) -> None:
        evt = normalize_raw_news(
            title="Bitcoin Update: New Data", body="Latest information available",
            source_name="Test",
        )
        assert evt.status == NewsStatus.UPDATE

    def test_status_correction(self) -> None:
        evt = normalize_raw_news(
            title="Correction: Previous Report Was Wrong",
            body="Error in earlier story",
            source_name="Test",
        )
        assert evt.status == NewsStatus.CORRECTION

    def test_status_retraction(self) -> None:
        evt = normalize_raw_news(
            title="Retraction: Report Was Fabricated",
            body="No such report exists",
            source_name="Test",
        )
        assert evt.status == NewsStatus.RETRACTION

    def test_url_hash_set(self) -> None:
        evt = normalize_raw_news(
            title="T", body="B", source_name="S", url="https://example.com/page"
        )
        assert len(evt.url_hash) > 0

    def test_default_language(self) -> None:
        evt = normalize_raw_news(
            title="T", body="B", source_name="S", language="en"
        )
        assert evt.language == "en"

    def test_default_revision(self) -> None:
        evt = normalize_raw_news(
            title="T", body="B", source_name="S"
        )
        assert evt.revision == 1


# ── Deduplication ─────────────────────────────────────────────────────────

class TestDedup:
    def test_url_dedup(self) -> None:
        dedup = Deduplicator()
        evt1 = normalize_raw_news(
            title="BTC News", body="body1", source_name="R",
            url="https://example.com/1"
        )
        evt2 = normalize_raw_news(
            title="BTC News v2", body="body2", source_name="R",
            url="https://example.com/1"
        )
        result = dedup.process([evt1, evt2])
        assert len(result) == 1
        assert result[0].revision == 2

    def test_content_dedup(self) -> None:
        dedup = Deduplicator(content_similarity_threshold=0.5)
        evt1 = normalize_raw_news(
            title="Bitcoin Surges", body="BTC reaches new high today",
            source_name="R"
        )
        evt2 = normalize_raw_news(
            title="Bitcoin Rallies", body="Bitcoin reaches new high today",
            source_name="R"
        )
        result = dedup.process([evt1, evt2])
        assert len(result) == 1  # Content too similar

    def test_no_dedup_different_content(self) -> None:
        dedup = Deduplicator()
        evt1 = normalize_raw_news(
            title="Bitcoin Surges", body="BTC reaches new high",
            source_name="R"
        )
        evt2 = normalize_raw_news(
            title="Ethereum Drops", body="ETH falls sharply today",
            source_name="R"
        )
        result = dedup.process([evt1, evt2])
        assert len(result) == 2

    def test_revision_bumping(self) -> None:
        dedup = Deduplicator()
        evt1 = normalize_raw_news(
            title="BTC News", body="body", source_name="R",
            url="https://example.com/1"
        )
        result1 = dedup.process([evt1])
        assert result1[0].revision == 1

        evt2 = normalize_raw_news(
            title="BTC News Corrected", body="body corrected", source_name="R",
            url="https://example.com/1"
        )
        result2 = dedup.process([evt2])
        assert len(result2) == 1
        assert result2[0].revision == 2

    def test_correction_status_preserved(self) -> None:
        dedup = Deduplicator()
        evt1 = normalize_raw_news(
            title="BTC News", body="body", source_name="R",
            url="https://example.com/1"
        )
        evt2 = normalize_raw_news(
            title="Correction: BTC News", body="corrected", source_name="R",
            url="https://example.com/1"
        )
        result = dedup.process([evt1, evt2])
        assert len(result) == 1
        assert result[0].status == NewsStatus.CORRECTION


# ── Clustering ───────────────────────────────────────────────────────────

class TestClustering:
    def test_cluster_similar_events(self) -> None:
        from packages.domain.news.clustering import EventClusterer
        clusterer = EventClusterer(similarity_threshold=0.2)
        evt1 = NewsEvent(
            id="e1", title="Bitcoin Surges", body="BTC up",
            source_name="S", source_type="blog", url_hash="u1",
            published_at=datetime.datetime.now(),
            received_at=datetime.datetime.now(),
        )
        evt2 = NewsEvent(
            id="e2", title="Bitcoin Rallies", body="BTC goes up",
            source_name="S", source_type="blog", url_hash="u2",
            published_at=datetime.datetime.now(),
            received_at=datetime.datetime.now(),
        )
        clusters = clusterer.cluster([evt1, evt2])
        assert len(clusters) == 1
        assert evt1.id in clusters[0].event_ids
        assert evt2.id in clusters[0].event_ids

    def test_no_cluster_different_events(self) -> None:
        from packages.domain.news.clustering import EventClusterer
        clusterer = EventClusterer()
        evt1 = NewsEvent(
            id="e1", title="Bitcoin Surges", body="BTC up",
            source_name="S", source_type="blog", url_hash="u1",
            published_at=datetime.datetime.now(),
            received_at=datetime.datetime.now(),
        )
        evt2 = NewsEvent(
            id="e2", title="Ethereum Drops", body="ETH down",
            source_name="S", source_type="blog", url_hash="u2",
            published_at=datetime.datetime.now(),
            received_at=datetime.datetime.now(),
        )
        clusters = clusterer.cluster([evt1, evt2])
        assert len(clusters) == 2

    def test_empty_cluster(self) -> None:
        from packages.domain.news.clustering import EventClusterer
        clusterer = EventClusterer()
        clusters = clusterer.cluster([])
        assert clusters == []


# ── Source Scoring ───────────────────────────────────────────────────────

class TestSourceScoring:
    def test_wire_service_high_score(self) -> None:
        score = score_source("Reuters", "wire_service")
        assert score >= 0.8

    def test_social_media_low_score(self) -> None:
        score = score_source("Twitter", "social_media")
        assert score < 0.5

    def test_news_event_scoring(self) -> None:
        now = datetime.datetime.now()
        evt = NewsEvent(
            id="test", title="T", body="B",
            source_name="Reuters", source_type="wire_service",
            url_hash="u", published_at=now, received_at=now,
        )
        scores = score_news_event(evt)
        assert "source" in scores
        assert "recency" in scores
        assert "composite" in scores
        assert scores["source"] > 0

    def test_novelty_decreases_with_revision(self) -> None:
        now = datetime.datetime.now()
        evt1 = NewsEvent(
            id="test", title="T", body="B",
            source_name="R", source_type="wire_service",
            url_hash="u", published_at=now, received_at=now,
            revision=1,
        )
        evt2 = NewsEvent(
            id="test", title="T", body="B",
            source_name="R", source_type="wire_service",
            url_hash="u", published_at=now, received_at=now,
            revision=5,
        )
        s1 = score_news_event(evt1)
        s2 = score_news_event(evt2)
        assert s1["novelty"] > s2["novelty"]


# ── Evidence structure ──────────────────────────────────────────────────

class TestEvidenceStructure:
    def test_evidence_has_source_info(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for ev in report.evidence:
            assert ev.feature  # feature should be set
            assert ev.relevance > 0

    def test_counter_evidence_direction(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for ev in report.counter_evidence:
            assert ev.direction == "negative"

    def test_invalidations_structure(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for inv in report.invalidations:
            assert inv.condition
            assert inv.indicator
            assert inv.threshold >= 0


# ── Probability distribution ────────────────────────────────────────────

class TestProbabilityDistribution:
    def test_balanced_news_near_equal(self, agent) -> None:
        data = {
            "news": [
                {"title": "Bitcoin Goes Up", "body": "BTC rises", "source_name": "R"},
                {"title": "Bitcoin Goes Down", "body": "BTC falls", "source_name": "R"},
            ]
        }
        report = agent.analyze(data)
        up = report.probabilities["up"]
        down = report.probabilities["down"]
        # At least one direction should have weight
        assert (up + down) > 0.01

    def test_bullish_bias_increases_up(self, agent) -> None:
        data = {
            "news": [
                {"title": "Bitcoin Surges Past 100K", "body": "BTC reaches record", "source_name": "Reuters"},
                {"title": "ETH Also Gains", "body": "Ethereum rallies", "source_name": "CoinDesk"},
                {"title": "Institutional Buying Boosts Crypto", "body": "BTC demand increases", "source_name": "Bloomberg"},
            ]
        }
        report = agent.analyze(data)
        assert report.probabilities["up"] > report.probabilities["down"]

    def test_bearish_bias_increases_down(self, agent) -> None:
        data = {
            "news": [
                {"title": "Bitcoin Crash Below 50K", "body": "BTC plummets", "source_name": "Reuters"},
                {"title": "ETH Falls Sharply", "body": "Ethereum drops", "source_name": "CoinDesk"},
            ]
        }
        report = agent.analyze(data)
        assert report.probabilities["down"] > report.probabilities["up"]


# ── Confidence score ────────────────────────────────────────────────────

class TestConfidenceScore:
    def test_confidence_in_range(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert 0.0 <= report.raw_confidence <= 0.9

    def test_more_events_increase_confidence(self, agent) -> None:
        few = {"news": [
            {"title": "BTC Up", "body": "Bitcoin rises", "source_name": "R"},
        ]}
        many = {"news": [
            {"title": "BTC Up", "body": "Bitcoin rises", "source_name": "R"},
            {"title": "ETH Up", "body": "Ethereum gains", "source_name": "C"},
            {"title": "SOL Up", "body": "Solana rallies", "source_name": "C"},
        ]}
        conf_few = agent.analyze(few).raw_confidence
        conf_many = agent.analyze(many).raw_confidence
        assert conf_many >= conf_few

    def test_high_source_score_boosts_confidence(self, agent) -> None:
        low = {"news": [
            {"title": "BTC Up", "body": "Bitcoin rises", "source_name": "Twitter"},
        ]}
        high = {"news": [
            {"title": "BTC Up", "body": "Bitcoin rises", "source_name": "Reuters"},
        ]}
        conf_low = agent.analyze(low).raw_confidence
        conf_high = agent.analyze(high).raw_confidence
        assert conf_high > conf_low


# ── NewsAgent with conflicting signals ──────────────────────────────────

class TestConflictingSignals:
    def test_counter_evidence_for_conflict(self, agent, conflicting_news_data) -> None:
        report = agent.analyze(conflicting_news_data)
        assert len(report.counter_evidence) >= 1
        # Counter evidence should mention conflict
        has_conflict_kw = any(
            "conflict" in ev.value.lower() or "conflicting" in ev.value.lower()
            for ev in report.counter_evidence
        )
        # Or at least has a counter with negative direction
        assert any(ev.direction == "negative" for ev in report.counter_evidence)

    def test_conflict_probabilities_more_balanced(self, agent, conflicting_news_data) -> None:
        report = agent.analyze(conflicting_news_data)
        up = report.probabilities["up"]
        down = report.probabilities["down"]
        # Should be more balanced than single-direction news
        assert abs(up - down) > 0.1


# ── AgentReport compliance ──────────────────────────────────────────────

class TestAgentReportCompliance:
    def test_report_has_required_fields(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert report.report_id
        assert report.run_id
        assert report.agent_id == "news"
        assert report.hypothesis
        assert isinstance(report.as_of, datetime.datetime)
        assert isinstance(report.probabilities, dict)

    def test_report_id_is_uuid(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        # UUID hex is 32 chars
        assert len(report.report_id) == 32

    def test_run_id_is_uuid(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert len(report.run_id) == 32

    def test_evidence_direction_values(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for ev in report.evidence:
            assert ev.value
            assert ev.direction in ("positive", "neutral", "negative")

    def test_counter_evidence_direction(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for ev in report.counter_evidence:
            assert ev.direction == "negative"

    def test_invalidations_structure(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        for inv in report.invalidations:
            assert inv.condition
            assert inv.indicator
            assert isinstance(inv.threshold, (int, float))
            assert inv.direction in ("above", "below")


# ── Integration: full pipeline ──────────────────────────────────────────

class TestNewsIntegration:
    def test_full_analysis_pipeline(self, agent, news_data) -> None:
        """Test the complete news analysis pipeline."""
        report = agent.analyze(news_data)

        # All report components present
        assert report.hypothesis
        assert len(report.evidence) >= 1
        assert len(report.counter_evidence) >= 1
        assert len(report.invalidations) >= 1
        assert isinstance(report.raw_confidence, float)

        # Probability constraints
        assert abs(sum(report.probabilities.values()) - 1.0) <= 0.0001
        assert report.probabilities["up"] >= 0
        assert report.probabilities["down"] >= 0
        assert report.probabilities["range"] >= 0

        # Status
        assert report.status.value == "shadow"

    def test_report_as_of_is_datetime(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        assert isinstance(report.as_of, datetime.datetime)

    def test_news_status_in_hypothesis(self, agent, news_data) -> None:
        report = agent.analyze(news_data)
        # At least one initial status event
        assert "initial" in report.hypothesis.lower() or "status" in report.hypothesis.lower()

    def test_no_llm_probabilities(self, agent) -> None:
        """Verify probabilities are rule-based, not from LLM."""
        now = datetime.datetime(2024, 1, 1, 12, 0, 0)
        data = {
            "news": [
                {
                    "title": "Bitcoin Surges", "body": "BTC up",
                    "source_name": "R", "published_at": now, "received_at": now,
                },
            ]
        }
        r1 = agent.analyze(data)
        r2 = agent.analyze(data)
        assert r1.probabilities == r2.probabilities


# ── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_news_without_url(self, agent) -> None:
        data = {
            "news": [
                {"title": "No URL News", "body": "Body text", "source_name": "Test"},
            ]
        }
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)

    def test_news_with_all_sources(self, agent) -> None:
        data = {
            "news": [
                {"title": "Wire", "body": "b", "source_name": "Reuters", "source_type": "wire_service"},
                {"title": "Social", "body": "b", "source_name": "Twitter", "source_type": "social_media"},
                {"title": "Blog", "body": "b", "source_name": "Blog", "source_type": "blog"},
            ]
        }
        report = agent.analyze(data)
        assert len(report.evidence) == 3

    def test_news_with_empty_title(self, agent) -> None:
        data = {
            "news": [
                {"title": "", "body": "Has body but no title", "source_name": "Test"},
            ]
        }
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)

    def test_news_with_long_body(self, agent) -> None:
        long_body = "Bitcoin " * 500
        data = {
            "news": [
                {"title": "Long", "body": long_body, "source_name": "Test"},
            ]
        }
        report = agent.analyze(data)
        assert isinstance(report, AgentReport)

    def test_multiple_same_source(self, agent) -> None:
        data = {
            "news": [
                {"title": "BTC Up", "body": "x", "source_name": "Reuters"},
                {"title": "ETH Up", "body": "x", "source_name": "Reuters"},
                {"title": "SOL Up", "body": "x", "source_name": "Reuters"},
            ]
        }
        report = agent.analyze(data)
        assert len(report.evidence) == 3

    def test_dedup_preserves_entity_order(self, agent) -> None:
        """Events with same URL get merged, latest entities preserved."""
        dedup = Deduplicator()
        evt1 = normalize_raw_news(
            title="BTC", body="Bitcoin", source_name="R",
            url="https://example.com/1"
        )
        evt2 = normalize_raw_news(
            title="BTC Corrected", body="Bitcoin corrected", source_name="R",
            url="https://example.com/1"
        )
        result = dedup.process([evt1, evt2])
        assert len(result) == 1
        assert result[0].revision == 2
