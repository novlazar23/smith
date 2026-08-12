"""News Agent — ingests raw news, runs full pipeline, produces AgentReport."""

from __future__ import annotations

import datetime
import uuid

import numpy as np
from numpy.typing import NDArray
from packages.domain.news import (
    Deduplicator,
    EntityMatch,
    NewsCluster,
    NewsEvent,
    NewsStatus,
    resolve_entities,
    score_news_event,
)
from packages.domain.news.clustering import EventClusterer
from packages.schemas.agent_report import AgentReport

from .base import AgentConfig, AgentType, BaseAgent

SUPPORTED_KEYS = frozenset({"news"})


class NewsAgent(BaseAgent):
    """News-Agent — volstaendige Nachrichten-Pipeline.

    Pipeline: Ingestion -> Normalization -> Dedup -> Clustering ->
    Entity Resolution -> Source Scoring -> Impact Assessment.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        content_threshold: float = 0.85,
        cluster_threshold: float = 0.6,
    ) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="news",
                agent_type=AgentType.NEWS,
            )
        super().__init__(config)
        self._content_threshold = content_threshold
        self._cluster_threshold = cluster_threshold

    def analyze(
        self, data: dict[str, NDArray[np.float64]]
    ) -> AgentReport:
        """Analysiert Raw-News-Eingaben.

        Required keys:
            news (list[dict]) — list of raw news dicts with at least:
                title, body, source_name, source_type (optional),
                url (optional), published_at (optional),
                received_at (optional), language (optional)

        Returns:
            AgentReport mit Impact-Score, Sentiment und Evidenz.

        Raises:
            ValueError: Wenn erforderliche Fehlt.
        """
        if "news" not in data:
            raise ValueError("Missing required data keys: ['news']")

        raw_items = data["news"]
        if not isinstance(raw_items, list) or len(raw_items) == 0:
            raise ValueError("news must be a non-empty list of dicts")

        # ── Pipeline: Ingestion + Normalization ──
        events: list[NewsEvent] = self._ingest(raw_items)

        # ── Pipeline: Deduplication ──
        dedup = Deduplicator(content_similarity_threshold=self._content_threshold)
        events = dedup.process(events)

        # ── Pipeline: Clustering ──
        clusters = self._cluster(events)

        # ── Pipeline: Entity Resolution ──
        entity_matches: list[EntityMatch] = []
        for event in events:
            matches = resolve_entities(event.body + " " + event.title)
            entity_matches.extend(matches)

        # ── Pipeline: Source Scoring + Impact Assessment ──
        scored_events: list[tuple[NewsEvent, dict]] = []
        for event in events:
            scores = score_news_event(event)
            scored_events.append((event, scores))

        # Sort by composite score descending
        scored_events.sort(key=lambda x: -x[1]["composite"])

        # ── Build report components ──
        hypothesis = self._build_hypothesis(
            scored_events, clusters, entity_matches
        )
        probabilities = self._compute_probabilities(scored_events)
        evidence = self._build_evidence(scored_events)
        counter_evidence = self._build_counter_evidence(scored_events)
        invalidations = self._build_invalidations(scored_events, clusters)
        confidence = self._compute_confidence(scored_events)
        overall_status = self._determine_overall_status(scored_events)

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(),
            hypothesis=f"{hypothesis} overall_status={overall_status}",
            probabilities=probabilities,
            evidence=evidence,
            counter_evidence=counter_evidence,
            invalidations=invalidations,
            raw_confidence=confidence,
            status=self.config.status,
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _ingest(self, raw_items: list) -> list[NewsEvent]:
        """Ingest raw news dicts into NewsEvents."""
        from packages.domain.news import normalize_raw_news

        events: list[NewsEvent] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", ""))
            body = str(item.get("body", ""))
            source_name = str(item.get("source_name", "unknown"))
            source_type = str(item.get("source_type", "unknown"))
            url = str(item.get("url", ""))
            language = str(item.get("language", "en"))
            published_at = item.get("published_at")
            received_at = item.get("received_at")
            raw_entities = item.get("entities")

            event = normalize_raw_news(
                title=title,
                body=body,
                source_name=source_name,
                source_type=source_type,
                url=url,
                language=language,
                published_at=published_at,
                received_at=received_at,
                raw_entities=raw_entities,
            )
            events.append(event)

        return events

    def _cluster(self, events: list[NewsEvent]) -> list[NewsCluster]:
        """Run clustering on events."""
        if not self._cluster_threshold:
            return []
        clusterer = EventClusterer(similarity_threshold=self._cluster_threshold)
        return clusterer.cluster(events)

    def _build_hypothesis(
        self,
        scored: list[tuple[NewsEvent, dict]],
        clusters: list[NewsCluster],
        entities: list[EntityMatch],
    ) -> str:
        """Build hypothesis string."""
        if not scored:
            return "No news events to analyze."

        event, scores = scored[0]
        instrument_str = (
            ", ".join(sorted({m.entity for m in entities}))
            if entities
            else "no specific instruments"
        )
        cluster_str = f", {len(clusters)} cluster(s)" if clusters else ""

        return (
            f"News impact analysis: {len(scored)} event(s), "
            f"status={event.status.value}, "
            f"source_score={scores['source']:.2f}, "
            f"recency={scores['recency']:.2f}, "
            f"composite={scores['composite']:.2f}. "
            f"Affected: {instrument_str}{cluster_str}"
        )

    def _compute_probabilities(
        self, scored: list[tuple[NewsEvent, dict]]
    ) -> dict[str, float]:
        """Compute up/down/range from news sentiment and impact scores.

        Rule-based: positive news increases up probability, negative increases down.
        """
        if not scored:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        up_weight = 0.0
        down_weight = 0.0

        for event, scores in scored:
            weight = scores["composite"]

            # Sentiment from status and title
            status_signal = self._status_sentiment(event.status)
            # Title word-based sentiment
            title_signal = self._title_sentiment(event.title)

            # Combined signal
            combined = (status_signal + title_signal) / 2.0

            if combined > 0:
                up_weight += weight * abs(combined)
            elif combined < 0:
                down_weight += weight * abs(combined)
            else:
                # Neutral -> range
                pass

        if up_weight == 0 and down_weight == 0:
            return {"up": 0.33, "down": 0.33, "range": 0.34}

        # Normalize to probabilities
        total = up_weight + down_weight + 0.34  # +range buffer
        up_prob = round(up_weight / total, 4)
        down_prob = round(down_weight / total, 4)
        range_prob = round(1.0 - up_prob - down_prob, 4)

        return {"up": up_prob, "down": down_prob, "range": range_prob}

    def _build_evidence(self, scored: list[tuple[NewsEvent, dict]]) -> list:
        """Evidence from top news items with source scores."""
        evidence: list = []

        for i, (event, scores) in enumerate(scored[:5]):
            evidence.append(
                self._make_evidence(
                    f"news_{i}",
                    f"{event.source_name} ({event.status.value}), "
                    f"score={scores['composite']:.2f}, "
                    f"entities={len(event.entities)}",
                    "positive" if scores["composite"] > 0.5 else "neutral",
                    scores["composite"],
                )
            )

        if not evidence:
            evidence.append(
                self._make_evidence(
                    "no_news",
                    "no news events to analyze",
                    "neutral",
                    0.0,
                )
            )

        return evidence

    def _build_counter_evidence(
        self, scored: list[tuple[NewsEvent, dict]]
    ) -> list:
        """Counter-evidence: conflicting news signals."""
        counter: list = []

        if len(scored) < 2:
            counter.append(
                self._make_evidence(
                    "counter_single_event",
                    "only one news event, insufficient for counter-evidence",
                    "negative",
                    0.1,
                )
            )
            return counter

        # Find events with opposite signals
        up_events = [
            e for e, s in scored if self._status_sentiment(e.status) > 0
        ]
        down_events = [
            e for e, s in scored if self._status_sentiment(e.status) < 0
        ]

        if up_events and down_events:
            best_up = max(up_events, key=lambda e: e.revision)
            best_down = max(down_events, key=lambda e: e.revision)
            counter.append(
                self._make_evidence(
                    "counter_conflict",
                    f"conflicting signals: {best_up.status.value} "
                    f"from {best_up.source_name} vs "
                    f"{best_down.status.value} from {best_down.source_name}",
                    "negative",
                    0.6,
                )
            )
        elif scored:
            # All signals same direction — use lowest-scored event as weak counter
            worst = min(scored, key=lambda x: x[1]["composite"])
            counter.append(
                self._make_evidence(
                    "counter_weak_signal",
                    f"weakest signal: {worst[0].source_name} "
                    f"({worst[1]['composite']:.2f})",
                    "negative",
                    0.3,
                )
            )
        else:
            counter.append(
                self._make_evidence(
                    "counter_no_events",
                    "no news events for counter-evidence",
                    "negative",
                    0.1,
                )
            )

        return counter

    def _build_invalidations(
        self,
        scored: list[tuple[NewsEvent, dict]],
        clusters: list[NewsCluster],
    ) -> list:
        """Invalidation conditions."""
        invalidations: list = []

        if scored:
            invalidations.append(
                self._make_invalidations(
                    condition="News becomes stale beyond 24h",
                    indicator="recency",
                    threshold=0.0,
                    direction="below",
                )
            )
            invalidations.append(
                self._make_invalidations(
                    condition="Low source reliability undermines signal",
                    indicator="source_reliability",
                    threshold=0.3,
                    direction="below",
                )
            )

        invalidations.append(
            self._make_invalidations(
                condition="No news events received",
                indicator="event_count",
                threshold=1.0,
                direction="below",
            )
        )

        return invalidations

    def _compute_confidence(self, scored: list[tuple[NewsEvent, dict]]) -> float:
        """Compute raw confidence from event scores."""
        if not scored:
            return 0.1

        avg_composite = sum(s["composite"] for _, s in scored) / len(scored)
        count_factor = min(1.0, len(scored) / 3.0)
        confidence = 0.2 + 0.4 * avg_composite + 0.2 * count_factor
        return round(min(0.9, confidence), 4)

    def _determine_overall_status(self, scored: list[tuple[NewsEvent, dict]]) -> str:
        """Determine the overall news status for the report."""
        if not scored:
            return "initial"

        status_priority = [
            NewsStatus.RETRACTION,
            NewsStatus.CORRECTION,
            NewsStatus.CONFIRMATION,
            NewsStatus.UPDATE,
            NewsStatus.INITIAL,
            NewsStatus.RUMOR,
        ]

        for status in status_priority:
            for event, _ in scored:
                if event.status == status:
                    return status.value

        return "initial"

    def _status_sentiment(self, status: NewsStatus) -> float:
        """Map news status to sentiment (-1 to 1)."""
        sentiment_map = {
            NewsStatus.CONFIRMATION: 0.7,
            NewsStatus.UPDATE: 0.3,
            NewsStatus.INITIAL: 0.1,
            NewsStatus.RUMOR: -0.1,
            NewsStatus.CORRECTION: -0.3,
            NewsStatus.RETRACTION: -0.7,
        }
        return sentiment_map.get(status, 0.0)

    def _title_sentiment(self, title: str) -> float:
        """Compute sentiment from title keywords."""
        words = title.lower().split()

        positive = {
            "surge", "rally", "boom", "up", "gain", "profit",
            "bullish", "record", "high", "growth", "approval",
            "launch", "partnership", "adoption", "upgrade", "halt",
            "halting",
        }
        negative = {
            "crash", "dump", "down", "loss", "bearish",
            "hack", "exploit", "scam", "fraud", "ban", "crackdown",
            "fine", "penalty", "sue", "lawsuit",
            "retraced", "correction", "retracing", "drop", "slump",
            "halt",
        }

        pos_count = sum(1 for w in words if w in positive)
        neg_count = sum(1 for w in words if w in negative)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)
