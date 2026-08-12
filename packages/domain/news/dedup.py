"""Deduplication — URL hash + content hashing with revision tracking."""

from __future__ import annotations

from packages.domain.news.models import NewsEvent, NewsStatus


class Deduplicator:
    """Deduplicates news events by URL hash and content similarity.

    Merges duplicates, tracks revisions for corrections/retractions.
    """

    def __init__(self, content_similarity_threshold: float = 0.85) -> None:
        self._known_events: dict[str, NewsEvent] = {}
        self._content_hashes: dict[int, str] = {}
        self._threshold = content_similarity_threshold

    def process(self, events: list[NewsEvent]) -> list[NewsEvent]:
        """Process a batch of events, returning deduplicated events.

        Events with the same URL hash are merged (revision bumped).
        Events with similar content to existing events are deduped.
        Only the latest version of each URL is returned.
        """
        processed: list[NewsEvent] = []

        for event in events:
            merged = self._try_merge_url(event)
            if merged is not None:
                if merged.url_hash:
                    processed = [
                        e for e in processed
                        if e.url_hash != merged.url_hash
                    ]
                processed.append(merged)
                continue

            skip = self._should_skip_content(event)
            if not skip:
                self._store_event(event)
                processed.append(event)

        return processed

    def _try_merge_url(self, event: NewsEvent) -> NewsEvent | None:
        """Try to merge with an existing event sharing the same URL hash."""
        if not event.url_hash:
            return None

        existing = self._known_events.get(event.url_hash)
        if existing is None:
            return None

        revision = existing.revision + 1
        status = existing.status

        if event.status in (
            NewsStatus.CORRECTION,
            NewsStatus.RETRACTION,
            NewsStatus.CONFIRMATION,
            NewsStatus.UPDATE,
        ):
            status = event.status

        merged = NewsEvent(
            id=existing.id,
            title=event.title,
            body=event.body,
            source_name=event.source_name,
            source_type=event.source_type,
            url_hash=event.url_hash,
            published_at=event.published_at,
            received_at=event.received_at,
            entities=event.entities or existing.entities,
            instruments=event.instruments or existing.instruments,
            language=event.language,
            revision=revision,
            status=status,
        )

        self._known_events[event.url_hash] = merged
        chash = self._content_hash(event)
        self._content_hashes[chash] = event.url_hash

        return merged

    def _should_skip_content(self, event: NewsEvent) -> bool:
        """Check if event should be skipped due to content similarity.

        Returns:
            True if event is a duplicate (should be skipped).
            False if event is unique (should be kept).
        """
        event_chash = self._content_hash(event)

        # Quick check: same content hash means duplicate
        if event_chash in self._content_hashes:
            existing_hash = self._content_hashes[event_chash]
            existing = self._known_events.get(existing_hash)
            if existing is not None:
                return True

        # Fuzzy content check
        for _existing_hash, existing in self._known_events.items():
            similarity = self._content_similarity(event, existing)
            if similarity >= self._threshold:
                return True

        return False

    def _store_event(self, event: NewsEvent) -> None:
        """Store event in dedup index."""
        key = event.url_hash if event.url_hash else event.id
        self._known_events[key] = event
        self._content_hashes[self._content_hash(event)] = key

    def _content_hash(self, event: NewsEvent) -> int:
        """Fingerprint content for quick comparison."""
        normalized = f"{event.title.lower()}|{event.body.lower()[:500]}"
        return hash(normalized)

    def _content_similarity(self, a: NewsEvent, b: NewsEvent) -> float:
        """Compute content similarity (Jaccard on word sets)."""
        words_a = set(a.title.lower().split() + a.body.lower().split()[:100])
        words_b = set(b.title.lower().split() + b.body.lower().split()[:100])

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0
