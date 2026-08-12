"""Orderbook Domain Models mit Rekonstruktion aus Snapshot + Deltas.

Enthält:
- FullOrderBook: Vollständiges Orderbook mit bid/ask Seiten
- OrderBookReconstructor: Rekonstruktion aus Snapshot + Delta-Updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class FullOrderBook:
    """Vollständiges Orderbook zu einem Zeitpunkt.

    Enthält alle Preis-Level auf der Bid- und Ask-Seite.
    """

    instrument: str
    venue: str
    sequence: int
    bids: list[PriceLevel] = field(default_factory=list)
    asks: list[PriceLevel] = field(default_factory=list)
    event_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Sortiere Bids absteigend, Asks aufsteigend."""
        object.__setattr__(self, "bids", sorted(
            self.bids, key=lambda x: x.price, reverse=True,
        ))
        object.__setattr__(self, "asks", sorted(
            self.asks, key=lambda x: x.price,
        ))

    @property
    def best_bid(self) -> float | None:
        """Bestes Angebot (höchster Bid-Preis)."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Bestes Verlangen (niedrigster Ask-Preis)."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        """Mittelpreis zwischen Best Bid und Best Ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        """Spread in Preis-Einheiten."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> float | None:
        """Spread in Prozent des Mid-Preises."""
        mid = self.mid_price
        sp = self.spread
        if mid is not None and sp is not None and mid > 0:
            return (sp / mid) * 100.0
        return None

    @property
    def bid_depth(self) -> float:
        """Gesamte Bid-Menge."""
        return sum(level.quantity for level in self.bids)

    @property
    def ask_depth(self) -> float:
        """Gesamte Ask-Menge."""
        return sum(level.quantity for level in self.asks)

    @property
    def imbalance(self) -> float:
        """Orderbook-Ungleichgewicht [-1.0, 1.0].
        > 0 bedeutet mehr Bid-Dominanz (bullish), < 0 mehr Ask-Dominanz (bearish).
        """
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


@dataclass(frozen=True)
class PriceLevel:
    """Einzelnes Preis-Level im Orderbook."""

    price: float
    quantity: float


class OrderBookReconstructor:
    """Rekonstruiert das vollständige Orderbook aus Snapshot + Delta-Updates.

    Binance und viele andere Exchanges senden einen initialen Snapshot
    und dann inkrementelle Delta-Updates. Dieser Reconstructor
    kombiniert sie zu konsistenten FullOrderBook-Instanzen.
    """

    def __init__(self, instrument: str, venue: str) -> None:
        self._instrument = instrument
        self._venue = venue
        self._current_book: FullOrderBook | None = None

    def apply_snapshot(
        self,
        snapshot_data: dict,
        event_time: datetime | None = None,
    ) -> FullOrderBook:
        """Wendet einen Orderbook-Snapshot an.

        Args:
            snapshot_data: Dict mit 'sequence', 'bids', 'asks'
                bids/asks: Liste von [price, quantity] Paaren
            event_time: Event-Zeitpunkt

        Returns:
            FullOrderBook nach dem Snapshot
        """
        sequence = snapshot_data.get("sequence", 0)
        raw_bids = snapshot_data.get("bids", [])
        raw_asks = snapshot_data.get("asks", [])

        bids = [
            PriceLevel(price=float(p), quantity=float(q))
            for p, q in raw_bids
        ]
        asks = [
            PriceLevel(price=float(p), quantity=float(q))
            for p, q in raw_asks
        ]

        # Sortieren: Bids absteigend (höchster Preis zuerst), Asks aufsteigend
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        self._current_book = FullOrderBook(
            instrument=self._instrument,
            venue=self._venue,
            sequence=sequence,
            bids=bids,
            asks=asks,
            event_time=event_time,
        )

        return self._current_book

    def apply_delta(
        self,
        delta_data: dict,
        event_time: datetime | None = None,
    ) -> FullOrderBook | None:
        """Wendet ein Delta-Update auf das aktuelle Orderbook an.

        Das Delta enthält 'bids' und 'asks' als Liste von
        [price, quantity] Paaren, wobei quantity=0 das Löschen
        des Levels bedeutet.

        Args:
            delta_data: Dict mit 'bids' und 'asks' Delta-Listen
            event_time: Event-Zeitpunkt

        Returns:
            Aktualisiertes FullOrderBook oder None wenn kein Snapshot vorhanden
        """
        if self._current_book is None:
            return None

        current_bids = {level.price: level.quantity for level in self._current_book.bids}
        current_asks = {level.price: level.quantity for level in self._current_book.asks}

        for p, q in delta_data.get("bids", []):
            price = float(p)
            quantity = float(q)
            if quantity == 0:
                current_bids.pop(price, None)
            else:
                current_bids[price] = quantity

        for p, q in delta_data.get("asks", []):
            price = float(p)
            quantity = float(q)
            if quantity == 0:
                current_asks.pop(price, None)
            else:
                current_asks[price] = quantity

        # sequence aus Delta übernehmen (falls vorhanden)
        sequence = delta_data.get("sequence", self._current_book.sequence + 1)

        # Umwandeln zurück in sortierte Listen
        bids = sorted(
            [PriceLevel(price=p, quantity=q) for p, q in current_bids.items()],
            key=lambda x: x.price,
            reverse=True,
        )
        asks = sorted(
            [PriceLevel(price=p, quantity=q) for p, q in current_asks.items()],
            key=lambda x: x.price,
        )

        self._current_book = FullOrderBook(
            instrument=self._instrument,
            venue=self._venue,
            sequence=sequence,
            bids=bids,
            asks=asks,
            event_time=event_time,
        )

        return self._current_book

    def get_current_book(self) -> FullOrderBook | None:
        """Gibt das aktuelle Orderbook zurück."""
        return self._current_book

    def reset(self) -> None:
        """Setzt den Reconstructor zurück."""
        self._current_book = None

    def verify_consistency(self) -> bool:
        """Prüft die Konsistenz des aktuellen Orderbooks.

        - Bids müssen absteigend sortiert sein
        - Asks müssen aufsteigend sortiert sein
        - Kein交叉 zwischen best bid und best ask

        Returns:
            True wenn konsistent
        """
        if self._current_book is None:
            return False

        book = self._current_book

        # Sortierung prüfen
        if book.bids:
            for i in range(1, len(book.bids)):
                if book.bids[i].price >= book.bids[i - 1].price:
                    return False

        if book.asks:
            for i in range(1, len(book.asks)):
                if book.asks[i].price <= book.asks[i - 1].price:
                    return False

        # Kein交叉
        return not (book.bids and book.asks) or book.bids[0].price < book.asks[0].price
