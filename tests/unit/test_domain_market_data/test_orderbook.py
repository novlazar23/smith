"""Tests für Orderbook Domain Models."""

from __future__ import annotations

from datetime import datetime

from packages.domain.market_data.orderbook import (
    FullOrderBook,
    OrderBookReconstructor,
    PriceLevel,
)


class TestFullOrderBook:
    def test_best_prices(self) -> None:
        book = FullOrderBook(
            instrument="BTC/USDT", venue="BINANCE", sequence=1,
            bids=[PriceLevel(99.0, 1.0), PriceLevel(98.0, 2.0)],
            asks=[PriceLevel(101.0, 1.5), PriceLevel(102.0, 0.5)],
        )
        assert book.best_bid == 99.0
        assert book.best_ask == 101.0
        assert book.mid_price == 100.0
        assert book.spread == 2.0
        assert book.spread_pct == 2.0

    def test_empty_book(self) -> None:
        book = FullOrderBook(
            instrument="X", venue="Y", sequence=1,
            bids=[], asks=[],
        )
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None
        assert book.spread_pct is None
        assert book.imbalance == 0.0

    def test_depth(self) -> None:
        book = FullOrderBook(
            instrument="X", venue="Y", sequence=1,
            bids=[PriceLevel(10.0, 5.0), PriceLevel(9.0, 3.0)],
            asks=[PriceLevel(11.0, 2.0), PriceLevel(12.0, 4.0)],
        )
        assert book.bid_depth == 8.0
        assert book.ask_depth == 6.0
        assert book.imbalance == 0.14285714285714285

    def test_sorting(self) -> None:
        book = FullOrderBook(
            instrument="X", venue="Y", sequence=1,
            bids=[PriceLevel(95.0, 1), PriceLevel(100.0, 1), PriceLevel(97.0, 1)],
            asks=[PriceLevel(110.0, 1), PriceLevel(105.0, 1), PriceLevel(108.0, 1)],
        )
        # Bids sollten absteigend sortiert sein (best first)
        bid_prices = [b.price for b in book.bids]
        assert bid_prices == sorted(bid_prices, reverse=True)
        # Asks sollten aufsteigend sortiert sein (best first)
        ask_prices = [a.price for a in book.asks]
        assert ask_prices == sorted(ask_prices)


class TestOrderBookReconstructor:
    def test_snapshot_apply(self) -> None:
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        snapshot = {
            "sequence": 100,
            "bids": [[99.0, 1.0], [98.0, 2.0]],
            "asks": [[101.0, 1.5], [102.0, 0.5]],
        }
        book = recon.apply_snapshot(snapshot, event_time=datetime.now())
        assert book is not None
        assert book.sequence == 100
        assert book.best_bid == 99.0
        assert recon.verify_consistency()

    def test_delta_apply(self) -> None:
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[99.0, 1.0]],
            "asks": [[101.0, 1.0]],
        })

        # Delete best bid, update
        delta = {
            "sequence": 101,
            "bids": [[99.0, 0], [98.0, 3.0]],
            "asks": [],
        }
        book = recon.apply_delta(delta)
        assert book is not None
        assert book.best_bid == 98.0
        assert len(book.bids) == 1  # 99.0 gelöscht

    def test_snapshot_and_delta_sequence(self) -> None:
        recon = OrderBookReconstructor("X", "Y")
        recon.apply_snapshot({"sequence": 10, "bids": [[10.0, 1.0]], "asks": [[12.0, 1.0]]})
        recon.apply_delta({"sequence": 11, "bids": [], "asks": []})
        book = recon.get_current_book()
        assert book is not None
        assert book.sequence == 11

    def test_reset(self) -> None:
        recon = OrderBookReconstructor("X", "Y")
        recon.apply_snapshot({"sequence": 1, "bids": [[10.0, 1.0]], "asks": [[12.0, 1.0]]})
        recon.reset()
        assert recon.get_current_book() is None

    def test_delta_without_snapshot_returns_none(self) -> None:
        recon = OrderBookReconstructor("X", "Y")
        result = recon.apply_delta({"sequence": 1, "bids": [], "asks": []})
        assert result is None

    def test_consistency_crossing(self) -> None:
        recon = OrderBookReconstructor("X", "Y")
        # Ungültiges Orderbook: best bid >= best ask
        recon.apply_snapshot({
            "sequence": 1,
            "bids": [[12.0, 1.0]],  # bid höher als ask
            "asks": [[10.0, 1.0]],
        })
        assert not recon.verify_consistency()
