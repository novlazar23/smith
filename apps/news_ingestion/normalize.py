"""Normalisierung und Entity Resolution für News-Events.

Bereitgestellt:
    - normalize_item(raw_item) -> NewsEvent — Rohdaten → NewsEvent
    - extract_entities(text) -> list[str] — Entities extrahieren
    - extract_instruments(text) -> list[str] — Instrumente extrahieren
    - resolve_entities(entities) -> list[str] — Entity Resolution
    - calculate_event_identity(title, body, source) -> str — Deterministische ID
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import Any

from apps.news_ingestion.config import NewsConfig
from apps.news_ingestion.ingest_rss import NewsRawItem

logger = logging.getLogger(__name__)

# Bekannte Krypto-Instrumente (Symbol → kanonische Bezeichnung)
_KNOWN_INSTRUMENTS: dict[str, str] = {
    "BTC": "BTC",
    "BTCUSDT": "BTC",
    "BITCOIN": "BTC",
    "ETH": "ETH",
    "ETHUSDT": "ETH",
    "ETHEREUM": "ETH",
    "SOL": "SOL",
    "SOLUSDT": "SOL",
    "SOLANA": "SOL",
    "XRP": "XRP",
    "XRPUSDT": "XRP",
    "USD Coin": "USDC",
    "USDC": "USDC",
    "Tether": "USDT",
    "USDT": "USDT",
    "BNB": "BNB",
    "ADA": "ADA",
    "ADAUSDT": "ADA",
    "DOGE": "DOGE",
    "DOGEUSDT": "DOGE",
    "AVAX": "AVAX",
    "DOT": "DOT",
    "LINK": "LINK",
    "MATIC": "MATIC",
    "POLYGON": "MATIC",
    "XLM": "XLM",
    "ATOM": "ATOM",
    "TRX": "TRX",
    "SHIB": "SHIB",
    "UNI": "UNI",
    "NEAR": "NEAR",
    "APT": "APT",
    "ARB": "ARB",
    "OP": "OP",
}

# Organisationen für Entity Resolution
_KNOWN_ORGANIZATIONS: dict[str, str] = {
    "SEC": "U.S. Securities and Exchange Commission",
    "SECURITIES AND EXCHANGE COMMISSION": "U.S. Securities and Exchange Commission",
    "CFTC": "Commodity Futures Trading Commission",
    "ECB": "European Central Bank",
    "FED": "Federal Reserve",
    "FEDERAL RESERVE": "Federal Reserve",
    "CME": "Chicago Mercantile Exchange",
    "CME GROUP": "Chicago Mercantile Exchange",
    "BINANCE": "Binance",
    "COINBASE": "Coinbase",
    "COINBASE INC": "Coinbase",
    "FINRA": "Financial Industry Regulatory Authority",
    "FCA": "Financial Conduct Authority",
}

# Synonym-Resolutions-Mapping
_ENTITY_SYNONYMS: dict[str, str] = {
    **dict(_KNOWN_ORGANIZATIONS),
    "BITCOIN": "BTC",
    "ETHEREUM": "ETH",
    "GOOGLE": "ALPHABET",
    "ALPHABET": "GOOGLE",
}


def normalize_item(
    raw_item: NewsRawItem,
    config: NewsConfig | None = None,
) -> dict[str, Any]:
    """Konvertiert ein NewsRawItem in ein normales News-Event Dictionary.

    Extrahiert Entities, Instrumente und berechnet eine deterministische
    event_identity.

    Args:
        raw_item: Rohes News-Item.
        config: Optionaler NewsConfig für Instrument-Erkennung.

    Returns:
        NewsEvent-Dictionary.
    """
    text_for_extraction = f"{raw_item.title} {raw_item.body}".strip()

    entities = extract_entities(text_for_extraction)
    resolved_entities = resolve_entities(entities)
    instruments = extract_instruments(text_for_extraction)
    event_identity = calculate_event_identity(
        raw_item.title, raw_item.body, raw_item.source_name
    )

    now = datetime.now(UTC)
    return {
        "news_id": str(hashlib.sha256(event_identity.encode("utf-8")).hexdigest()[:16]),
        "event_identity": event_identity,
        "title": raw_item.title,
        "body": raw_item.body,
        "source_name": raw_item.source_name,
        "source_type": raw_item.source_type,
        "url_hash": raw_item.url_hash,
        "published_at": raw_item.published_at or now,
        "received_at": now,
        "entities": resolved_entities,
        "instruments": instruments,
        "language": "en",
        "revision": 1,
        "status": "INITIAL",
    }


def extract_entities(text: str) -> list[str]:
    """Extrahiert Entities (Organisationen, Personen) aus einem Text.

    Nutzt bekanntes Vokabular aus _KNOWN_ORGANIZATIONS.

    Args:
        text: Der zu analysierende Text.

    Returns:
        Liste von erkannten Entity-Namen.
    """
    if not text:
        return []

    text_upper = text.upper()
    entities: list[str] = []

    for entity_name in _KNOWN_ORGANIZATIONS:
        if entity_name in text_upper:
            canonical = _KNOWN_ORGANIZATIONS[entity_name]
            if canonical not in entities:
                entities.append(canonical)

    return entities


def extract_instruments(text: str) -> list[str]:
    """Extrahiert Krypto-Instrumente aus einem Text.

    Nutzt bekanntes Vokabular aus _KNOWN_INSTRUMENTS und sucht
    nach Symbolen als ganze Wörter.

    Args:
        text: Der zu analysierende Text.

    Returns:
        Liste kanonischer Instrument-Bezeichnungen.
    """
    if not text:
        return []

    text_upper = text.upper()
    found: dict[str, str] = {}

    # Längste Symbole zuerst (z.B. BTCUSDT vor BTC)
    for raw_symbol, canonical in sorted(
        _KNOWN_INSTRUMENTS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        pattern = rf"\b{re.escape(raw_symbol)}\b"
        if re.search(pattern, text_upper) and canonical not in found:
            found[canonical] = raw_symbol

    return sorted(found.keys())


def resolve_entities(entities: list[str]) -> list[str]:
    """Auflösen von Entity-Synonymen auf kanonische Namen.

    Ersetzt bekannte Varianten durch ihre kanonische Bezeichnung
    und entfernt Duplikate.

    Args:
        entities: Liste von Entity-Namen (möglicherweise synonym).

    Returns:
        Liste von kanonisierten Entity-Namen ohne Duplikate.
    """
    if not entities:
        return []

    resolved: dict[str, bool] = {}
    for entity in entities:
        key = entity.upper()
        if key in _ENTITY_SYNONYMS:
            canonical = _ENTITY_SYNONYMS[key]
            resolved[canonical] = True
        else:
            resolved[entity] = True

    return sorted(resolved.keys())


def calculate_event_identity(title: str, body: str, source: str) -> str:
    """Erzeugt eine deterministische event_identity.

    Kombiniert Titel, Body und Quelle zu einem eindeutigen Hash.
    Dieselbe Nachricht erzeugt immer dieselbe event_identity.

    Args:
        title: Titel der Nachricht.
        body: Body/Text der Nachricht.
        source: Name der Quelle.

    Returns:
        32-stelliger hexadezimaler Hash als event_identity.
    """
    combined = f"{source}|{title}|{body}".strip()
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]
