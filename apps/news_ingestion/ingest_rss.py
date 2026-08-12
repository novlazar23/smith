"""RSS-Parsing und Deduplikation für News-Ingestion.

Bereitgestellt:
    - rss_fetch(url) -> list[dict] — RSS-Feed holen und parsen
    - deduplicate(items) -> list[dict] — Dedup via url_hash + title_similarity
    - url_hash(url) -> str — SHA256 Hash der URL
    - ingest_feed(source_config) -> list[NewsRawItem] — Feed mit Quelle annotieren
"""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from apps.news_ingestion.config import SourceConfig

logger = logging.getLogger(__name__)

# RSS/Atom Namespace-Mapper für portable Parser
_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


class _HTMLTextExtractor(HTMLParser):
    """Minimaler HTML-Parser, der nur Text extrahiert."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str | None) -> str:
    """Entfernt HTML-Tags und gibt reinen Text zurück."""
    if not html:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def url_hash(url: str) -> str:
    """Erzeugt einen deterministischen SHA256-Hash einer URL.

    Args:
        url: Die zu hashende URL.

    Returns:
        Hexadezimaler SHA256-Hash der URL.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _parse_rss(xml_text: str, source_url: str) -> list[dict[str, Any]]:
    """Parser für RSS 2.0 und Atom XML-Feeds.

    Args:
        xml_text: Roh-XML-Inhalt des Feeds.
        source_url: URL der Quelle für Dedup.

    Returns:
        Liste von Artikel-Dictionaries mit title, link, description, published, source_url.
    """
    items: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("Failed to parse RSS XML: %s", exc)
        return items

    # RSS 2.0: //item
    for item in root.findall(".//item"):
        title = _text_element(item, "title")
        link = _text_element(item, "link")
        published = _text_element(item, "pubDate")
        content = _text_element(item, "content:encoded") or _text_element(item, "description")

        items.append({
            "title": title.strip() if title else "",
            "link": link.strip() if link else source_url,
            "description": _strip_html(content) if content else "",
            "published_at": _parse_date(published),
            "source_url": source_url,
        })

    # Atom: //entry
    if not items:
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title = _text_element(entry, "title", ns="atom")
            link_el = entry.find("atom:link", _NAMESPACES)
            link = link_el.get("href", "") if link_el is not None else ""
            summary = _text_element(entry, "summary", ns="atom")
            published = _text_element(entry, "published", ns="atom")

            items.append({
                "title": title.strip() if title else "",
                "link": link.strip() if link else source_url,
                "description": _strip_html(summary) if summary else "",
                "published_at": _parse_date(published),
                "source_url": source_url,
            })

    return items


def _text_element(parent: ET.Element, tag: str, ns: str = "") -> str | None:
    """Holt den Text eines untergeordneten Elements."""
    full_tag = f"{{{ns}}}{tag}" if ns else tag
    el = parent.find(full_tag)
    if el is not None and el.text:
        return el.text
    return None


def _parse_date(date_str: str | None) -> datetime | None:
    """Versucht, ein Datum aus verschiedenen Formaten zu parsen."""
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    logger.warning("Could not parse date: %s", date_str)
    return None


def rss_fetch(url: str) -> list[dict[str, Any]]:
    """RSS-Feed von einer URL holen und parsen.

    Args:
        url: URL des RSS-Feeds.

    Returns:
        Liste von Artikel-Dictionaries.

    Raises:
        RuntimeError: Wenn httpx nicht verfügbar ist.
    """
    if not HTTPX_AVAILABLE:
        raise RuntimeError("httpx required for RSS fetching")

    headers = {
        "User-Agent": "TradingOrchestra/1.0 (News-Ingestion-Service)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
    }

    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
    except Exception as exc:
        logger.error("Failed to fetch RSS feed %s: %s", url, exc)
        return []

    return _parse_rss(response.text, url)


def _title_similarity(title_a: str, title_b: str) -> float:
    """Berechnet die Ähnlichkeit zwischen zwei Titeln (Jaccard-Ähnlichkeit)."""
    if not title_a or not title_b:
        return 0.0
    words_a = set(title_a.lower().split())
    words_b = set(title_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupliziert Item-Liste via url_hash und Titel-Ähnlichkeit.

    Items mit identischem url_hash oder sehr ähnlichen Titeln werden
    als Duplikat entfernt (das erste Exemplar wird behalten).

    Args:
        items: Liste von Artikel-Dictionaries mit 'link' und 'title'.

    Returns:
        Deduplizierte Liste.
    """
    seen_hashes: set[str] = set()
    seen_titles: dict[str, str] = {}  # hash → title for similarity
    result: list[dict[str, Any]] = []

    for item in items:
        if hasattr(item, "get"):
            link = item.get("link", "")
            title = item.get("title", "")
        else:
            link = getattr(item, "source_url", "")
            title = getattr(item, "title", "")
        item_hash = url_hash(link)

        # Exakte Duplikate
        if item_hash in seen_hashes:
            logger.debug("Skipping duplicate: %s", link)
            continue

        # Titel-Ähnlichkeits-Dedup
        is_similar = False
        for _existing_hash, existing_title in seen_titles.items():
            if _title_similarity(title, existing_title) >= 0.7:
                is_similar = True
                logger.debug(
                    "Skipping similar: '%s' ≈ '%s'", title, existing_title
                )
                break

        if is_similar:
            continue

        seen_hashes.add(item_hash)
        seen_titles[item_hash] = title
        result.append(item)

    return result


@dataclass(frozen=True)
class NewsRawItem:
    """Rohes News-Item vor der Normalisierung.

    Felder:
        title: Titel des Artikels.
        body: Beschreibung / Body des Artikels.
        source_url: URL der Quelle.
        published_at: Veröffentlichungsdatum.
        source_name: Name der Feed-Quelle.
        source_type: Typ der Feed-Quelle.
    """

    title: str
    body: str
    source_url: str
    published_at: datetime | None
    source_name: str
    source_type: str

    @property
    def url_hash(self) -> str:
        """SHA256-Hash der Source-URL."""
        return url_hash(self.source_url)


def ingest_feed(source_config: SourceConfig) -> list[NewsRawItem]:
    """Holt und parst einen RSS-Feed für eine gegebene Quelle.

    Args:
        source_config: Konfiguration der Quelle.

    Returns:
        Liste von NewsRawItem-Objekten.
    """
    if not source_config.enabled:
        logger.info("Source '%s' is disabled, skipping", source_config.name)
        return []

    logger.info("Ingesting feed from '%s' (%s)", source_config.name, source_config.url)
    items = rss_fetch(source_config.url)

    raw_items: list[NewsRawItem] = []
    for item in items:
        raw_items.append(NewsRawItem(
            title=item.get("title", ""),
            body=item.get("description", ""),
            source_url=item.get("link", source_config.url),
            published_at=item.get("published_at"),
            source_name=source_config.name,
            source_type=source_config.feed_type,
        ))

    logger.info("Fetched %d items from '%s'", len(raw_items), source_config.name)
    return raw_items
