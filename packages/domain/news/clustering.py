"""Clustering — groups similar news events by title similarity and entity overlap."""

from __future__ import annotations

from packages.domain.news.models import NewsCluster, NewsEvent


class EventClusterer:
    """Clusters similar news events using title similarity + entity overlap."""

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        self._threshold = similarity_threshold

    def cluster(self, events: list[NewsEvent]) -> list[NewsCluster]:
        """Cluster events into groups of similar news.

        Returns NewsCluster objects with representative event.
        """
        if not events:
            return []

        clusters: list[NewsCluster] = []
        assigned: set[str] = set()

        for i, event in enumerate(events):
            if event.id in assigned:
                continue

            cluster = [event]
            assigned.add(event.id)

            for j in range(i + 1, len(events)):
                other = events[j]
                if other.id in assigned:
                    continue

                if self._are_similar(event, other):
                    cluster.append(other)
                    assigned.add(other.id)

            # Find representative (highest source score)
            representative = max(cluster, key=lambda e: e.revision * 0.5 + len(e.entities))
            clusters.append(
                NewsCluster(
                    cluster_id=f"cluster_{cluster[0].id[:6]}",
                    event_ids=[e.id for e in cluster],
                    representative=representative.id,
                )
            )

        return clusters

    def _are_similar(self, a: NewsEvent, b: NewsEvent) -> bool:
        """Check if two events are similar enough for clustering."""
        # Entity overlap
        entity_sim = self._entity_overlap(a.entities, b.entities)
        # Title similarity (word overlap)
        title_sim = self._title_overlap(a.title, b.title)

        # Either high entity overlap or high title similarity
        return entity_sim >= 0.5 or title_sim >= self._threshold

    def _entity_overlap(self, entities_a: list[str], entities_b: list[str]) -> float:
        """Jaccard similarity on entity lists."""
        set_a = set(entities_a)
        set_b = set(entities_b)

        if not set_a and not set_b:
            return 0.0  # Neutral if both have no entities — do not auto-match
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def _title_overlap(self, title_a: str, title_b: str) -> float:
        """Word-overlap similarity between titles."""
        words_a = set(title_a.lower().split())
        words_b = set(title_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0
