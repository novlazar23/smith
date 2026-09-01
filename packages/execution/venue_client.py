"""Venue-agnostic execution client for multi-exchange support.

Provides ``VenueExecutionClient`` which:

- Creates exchange-specific adapter instances via ``create_client(venue)``
- Configures venue-specific fees, spreads, and rate limits
- Exposes ``health_check()`` per venue
- Exposes ``get_fee_structure(venue)`` → ``dict``

Supported venues (configurable via ``AnalysisRequest.venues``):
    - ``"binance"`` → BinanceAdapter (futures or spot)
    - ``"dummy"`` → DummyAdapter (simulated market data)

Fee structures:
    - Binance: taker 0.0004 (0.04%), maker 0.0001 (0.01%)
    - Dummy:   taker 0.001  (0.1%),  maker 0.0005 (0.05%)

Instrument formats:
    - Binance: ``"BTCUSDT"``, ``"ETHUSDT"``, etc.
    - Dummy:   ``"BTC/USDT"``, ``"ETH/USDT"``, etc.
"""

from __future__ import annotations

import logging
from typing import Any

from packages.ingestion.adapter import (
    BinanceAdapter,
    ConnectionConfig,
    DummyAdapter,
    ExchangeAdapterBase,
    VenueFees,
)

logger = logging.getLogger(__name__)

# ── Default venue configurations ─────────────────────────────────────

DEFAULT_VENUE_CONFIGS: dict[str, dict[str, Any]] = {
    "binance": {
        "base_url": "https://fapi.binance.com",
        "api_key": "",
        "api_secret": "",
        "rate_limit_per_second": 10,
        "reconnect_delay": 1.0,
        "max_reconnect_attempts": 10,
        "heartbeat_interval": 30.0,
        "use_futures": True,
        "fees": {
            "taker_rate": 0.0004,
            "maker_rate": 0.0001,
            "spread_bps": 1.0,
        },
    },
    "dummy": {
        "base_price": 67500.0,
        "seed": 42,
        "rate_limit_per_second": 100,
        "fees": {
            "taker_rate": 0.001,
            "maker_rate": 0.0005,
            "spread_bps": 5.0,
        },
    },
}

# Registry of supported venues and their factory functions
VENUE_REGISTRY: dict[str, type[ExchangeAdapterBase]] = {
    "binance": BinanceAdapter,
    "dummy": DummyAdapter,
}


class VenueExecutionClient:
    """Venue-agnostic execution client.

    Manages a pool of exchange adapters, each identified by a venue string.
    Clients are lazily created on first access via ``create_client(venue)``.
    """

    def __init__(
        self,
        venues: list[str] | None = None,
        custom_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the venue execution client.

        Args:
            venues: List of venue identifiers (e.g. ``["binance", "dummy"]``).
                    Defaults to all known venues.
            custom_configs: Optional per-venue configuration overrides.
        """
        self._venues = venues or list(VENUE_REGISTRY.keys())
        self._custom_configs = custom_configs or {}
        self._clients: dict[str, ExchangeAdapterBase] = {}
        self._configs: dict[str, dict[str, Any]] = {}

        # Merge default configs with custom overrides
        for venue_id in self._venues:
            default = DEFAULT_VENUE_CONFIGS.get(venue_id, {}).copy()
            overrides = self._custom_configs.get(venue_id, {}).copy()
            default.update(overrides)
            self._configs[venue_id] = default

        logger.info(
            "VenueExecutionClient initialized with venues: %s", self._venues,
        )

    # -- client creation ------------------------------------------------

    def create_client(self, venue: str) -> ExchangeAdapterBase:
        """Create or return a cached exchange adapter for a venue.

        Args:
            venue: Venue identifier (e.g. ``"binance"``, ``"dummy"``).

        Returns:
            An ``ExchangeAdapterBase`` subclass instance for the venue.

        Raises:
            ValueError: If the venue is not in the registry or not configured.
        """
        if venue not in VENUE_REGISTRY:
            available = sorted(VENUE_REGISTRY.keys())
            raise ValueError(
                f"Unknown venue: '{venue}'. "
                f"Available venues: {available}"
            )

        if venue not in self._venues:
            raise ValueError(
                f"Venue '{venue}' is not in the active venue list: "
                f"{self._venues}"
            )

        if venue not in self._clients:
            client_class = VENUE_REGISTRY[venue]
            config = self._configs.get(venue, {})
            adapter = self._build_adapter(venue, client_class, config)
            self._clients[venue] = adapter
            logger.info("Created adapter for venue: %s", venue)

        return self._clients[venue]

    # -- health ---------------------------------------------------------

    async def health_check(self, venue: str) -> dict[str, Any]:
        """Health-check a specific venue's adapter.

        Args:
            venue: Venue identifier.

        Returns:
            Dict with health status for the venue.

        Raises:
            ValueError: If the venue is not available.
        """
        client = self.create_client(venue)
        return await client.health_check()

    async def health_check_all(self) -> dict[str, dict[str, Any]]:
        """Health-check all configured venues.

        Returns:
            Dict mapping venue names to their health status dicts.
        """
        results: dict[str, dict[str, Any]] = {}
        for venue in self._venues:
            try:
                results[venue] = await self.health_check(venue)
            except Exception as exc:
                results[venue] = {
                    "error": str(exc),
                    "connected": False,
                    "state": "error",
                }
        return results

    # -- fee structure --------------------------------------------------

    def get_fee_structure(self, venue: str) -> dict[str, Any]:
        """Return fee structure for a venue.

        Args:
            venue: Venue identifier.

        Returns:
            Dict with venue, taker rate, maker rate, spread_bps.
        """
        default = DEFAULT_VENUE_CONFIGS.get(venue, {})
        fees_cfg = default.get("fees", {})
        return {
            "venue": venue,
            "taker": fees_cfg.get("taker_rate", 0.0),
            "maker": fees_cfg.get("maker_rate", 0.0),
            "spread_bps": fees_cfg.get("spread_bps", 1.0),
        }

    def get_all_fee_structures(self) -> dict[str, dict[str, Any]]:
        """Return fee structures for all configured venues.

        Returns:
            Dict mapping venue names to their fee structure dicts.
        """
        return {v: self.get_fee_structure(v) for v in self._venues}

    # -- config ---------------------------------------------------------

    @property
    def venues(self) -> list[str]:
        """List of active venue identifiers."""
        return list(self._venues)

    def get_config(self, venue: str) -> dict[str, Any]:
        """Return raw configuration for a venue.

        Args:
            venue: Venue identifier.

        Returns:
            Deep-copied config dict.
        """
        import copy
        return copy.deepcopy(self._configs.get(venue, {}))

    # -- internal -------------------------------------------------------

    def _build_adapter(
        self,
        venue: str,
        client_class: type[ExchangeAdapterBase],
        config: dict[str, Any],
    ) -> ExchangeAdapterBase:
        """Build an adapter instance from venue config.

        Args:
            venue: Venue identifier.
            client_class: Adapter class (BinanceAdapter, DummyAdapter, etc.).
            config: Per-venue configuration dict.

        Returns:
            Configured adapter instance.
        """
        if venue == "binance":
            api_key = config.get("api_key", "")
            api_secret = config.get("api_secret", "")
            base_url = config.get("base_url", "https://fapi.binance.com")
            use_futures = config.get("use_futures", True)
            rate_limit = config.get("rate_limit_per_second", 10)
            reconnect_delay = config.get("reconnect_delay", 1.0)
            max_reconnect = config.get("max_reconnect_attempts", 10)
            heartbeat_interval = config.get("heartbeat_interval", 30.0)

            fees_cfg = config.get("fees", {})
            fees = VenueFees(
                taker_rate=fees_cfg.get("taker_rate", 0.0004),
                maker_rate=fees_cfg.get("maker_rate", 0.0001),
                spread_bps=fees_cfg.get("spread_bps", 1.0),
            )

            cc = ConnectionConfig(
                api_key=api_key,
                api_secret=api_secret,
                base_url=base_url,
                rate_limit_per_second=rate_limit,
                reconnect_delay=reconnect_delay,
                max_reconnect_attempts=max_reconnect,
                heartbeat_interval=heartbeat_interval,
                venue="BINANCE_FUTURES" if use_futures else "BINANCE_SPOT",
                fees=fees,
            )
            return BinanceAdapter(config=cc, use_futures=use_futures)

        elif venue == "dummy":
            base_price = config.get("base_price", 67500.0)
            seed = config.get("seed", 42)

            fees_cfg = config.get("fees", {})
            fees = VenueFees(
                taker_rate=fees_cfg.get("taker_rate", 0.001),
                maker_rate=fees_cfg.get("maker_rate", 0.0005),
                spread_bps=fees_cfg.get("spread_bps", 5.0),
            )

            cc = ConnectionConfig(
                api_key="",
                api_secret="",
                base_url="http://dummy.exchange/v1",
                rate_limit_per_second=config.get("rate_limit_per_second", 100),
                venue="DUMMY_EXCHANGE",
                fees=fees,
            )
            return DummyAdapter(config=cc, base_price=base_price, seed=seed)

        # Generic fallback: pass raw config as ConnectionConfig
        fees_cfg = config.get("fees", {})
        fees = VenueFees(
            taker_rate=fees_cfg.get("taker_rate", 0.0),
            maker_rate=fees_cfg.get("maker_rate", 0.0),
            spread_bps=fees_cfg.get("spread_bps", 1.0),
        )

        cc = ConnectionConfig(
            api_key=config.get("api_key", ""),
            api_secret=config.get("api_secret", ""),
            base_url=config.get("base_url", ""),
            rate_limit_per_second=config.get("rate_limit_per_second", 10),
            reconnect_delay=config.get("reconnect_delay", 1.0),
            max_reconnect_attempts=config.get("max_reconnect_attempts", 10),
            heartbeat_interval=config.get("heartbeat_interval", 30.0),
            venue=venue.upper(),
            fees=fees,
        )
        return client_class(config=cc)
