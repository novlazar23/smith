"""Entity resolution — maps news entities to known instruments."""

from __future__ import annotations

import re

from packages.domain.news.models import EntityMatch

# Canonical mapping: ticker -> full name / alias set
TICKER_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "btcusd", "xbt", "xbtusd", "satoshi"],
    "ETH": ["ethereum", "eth", "ethusd", "ether"],
    "SOL": ["solana", "sol", "solusd"],
    "XRP": ["ripple", "xrp", "xrpusd"],
    "ADA": ["cardano", "ada", "adausd"],
    "DOGE": ["dogecoin", "doge", "dogeusd"],
    "MATIC": ["polygon", "matic", "maticusd", "polygon"],
    "DOT": ["polkadot", "dot", "dotusd"],
    "AVAX": ["avalanche", "avax", "avaxusd"],
    "LINK": ["chainlink", "link", "linkusd"],
    "USDT": ["tether", "usdt", "tetherusd"],
    "USDC": ["usd coin", "usdc", "usdcusd"],
    "BNB": ["binance coin", "bnb", "bnbusd"],
    "SPY": ["spy", "spy etf", "spy index"],
    "AAPL": ["apple", "aapl", "aaplusd"],
    "TSLA": ["tesla", "tsla", "tslausd"],
    "NVDA": ["nvidia", "nvda", "nvdausd"],
}

# Reverse lookup: alias -> ticker
_ALIAS_TO_TICKER: dict[str, str] = {}
for _ticker, _aliases in TICKER_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_TICKER[_alias.lower()] = _ticker


def resolve_entities(text: str) -> list[EntityMatch]:
    """Resolve entities in news text to known instrument tickers.

    Returns EntityMatches sorted by confidence (highest first).
    """
    if not text:
        return []

    text_lower = text.lower()
    matches: list[EntityMatch] = []
    seen: set[str] = set()

    # Exact alias matches
    for alias, ticker in _ALIAS_TO_TICKER.items():
        if alias in text_lower and ticker not in seen:
            seen.add(ticker)
            matches.append(
                EntityMatch(entity=ticker, confidence=0.9, type="ticker")
            )

    # Pattern-based: $TICKER or TICKERUSD
    for ticker in TICKER_ALIASES:
        pattern = rf"(?:\$|(?<=\s)){ticker}(?:usd)?\b"
        if ticker not in seen and re.search(pattern, text_lower):
            seen.add(ticker)
            matches.append(
                EntityMatch(entity=ticker, confidence=0.85, type="ticker")
            )

    return sorted(matches, key=lambda m: -m.confidence)
