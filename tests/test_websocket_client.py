"""
Tests for the WebSocket client and local orderbook management.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.client.websocket_client import LocalOrderbook, MarketWebSocket, UserWebSocket
from src.models import Orderbook


class TestLocalOrderbook:
    def test_snapshot_populates_book(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[{"price": "0.53", "size": "100"}, {"price": "0.52", "size": "200"}],
            asks=[{"price": "0.57", "size": "100"}, {"price": "0.58", "size": "50"}],
        )
        ob = lb.to_orderbook()
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2
        assert ob.best_bid == pytest.approx(0.53)
        assert ob.best_ask == pytest.approx(0.57)

    def test_midpoint_is_average_of_best_bid_ask(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.60", "size": "100"}],
        )
        assert lb.midpoint == pytest.approx(0.55)

    def test_zero_size_removes_level(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "100"}],
        )
        lb.apply_change(0.50, "BUY", 0.0)  # remove the bid
        ob = lb.to_orderbook()
        assert len(ob.bids) == 0

    def test_new_level_added(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "100"}],
        )
        lb.apply_change(0.51, "BUY", 50.0)  # new bid
        ob = lb.to_orderbook()
        assert len(ob.bids) == 2
        assert ob.best_bid == pytest.approx(0.51)

    def test_existing_level_updated(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[{"price": "0.50", "size": "100"}],
            asks=[],
        )
        lb.apply_change(0.50, "BUY", 200.0)  # update size
        ob = lb.to_orderbook()
        assert ob.bids[0].size == pytest.approx(200.0)

    def test_bids_sorted_descending(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[
                {"price": "0.48", "size": "100"},
                {"price": "0.52", "size": "100"},
                {"price": "0.50", "size": "100"},
            ],
            asks=[],
        )
        ob = lb.to_orderbook()
        prices = [b.price for b in ob.bids]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[],
            asks=[
                {"price": "0.58", "size": "100"},
                {"price": "0.54", "size": "100"},
                {"price": "0.56", "size": "100"},
            ],
        )
        ob = lb.to_orderbook()
        prices = [a.price for a in ob.asks]
        assert prices == sorted(prices)

    def test_empty_book_has_no_midpoint(self):
        lb = LocalOrderbook("tok1")
        assert lb.midpoint is None

    def test_snapshot_replaces_previous(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[{"price": "0.40", "size": "100"}],
            asks=[{"price": "0.50", "size": "100"}],
        )
        lb.apply_snapshot(
            bids=[{"price": "0.55", "size": "200"}],
            asks=[{"price": "0.65", "size": "200"}],
        )
        ob = lb.to_orderbook()
        assert len(ob.bids) == 1
        assert ob.best_bid == pytest.approx(0.55)

    def test_zero_size_in_snapshot_ignored(self):
        lb = LocalOrderbook("tok1")
        lb.apply_snapshot(
            bids=[{"price": "0.50", "size": "0"}],  # size=0, should not add
            asks=[{"price": "0.55", "size": "100"}],
        )
        ob = lb.to_orderbook()
        assert len(ob.bids) == 0


class TestMarketWebSocketHandlers:
    async def test_book_event_triggers_callback(self):
        received = []

        async def on_book(token_id, ob):
            received.append((token_id, ob))

        ws = MarketWebSocket(
            token_ids=["tok1"],
            on_book=on_book,
        )
        msg = {
            "event_type": "book",
            "asset_id": "tok1",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        }
        await ws._handle(msg)
        assert len(received) == 1
        assert received[0][0] == "tok1"

    async def test_price_change_event_triggers_callback(self):
        received = []

        async def on_change(token_id, ob):
            received.append((token_id, ob))

        ws = MarketWebSocket(
            token_ids=["tok1"],
            on_price_change=on_change,
        )
        # First give it a snapshot
        await ws._handle({
            "event_type": "book",
            "asset_id": "tok1",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })
        # Then a change
        await ws._handle({
            "event_type": "price_change",
            "asset_id": "tok1",
            "changes": [{"price": "0.51", "side": "BUY", "size": "50"}],
        })
        assert len(received) == 1
        assert received[0][0] == "tok1"

    async def test_unknown_token_ignored(self):
        received = []

        async def on_book(token_id, ob):
            received.append(token_id)

        ws = MarketWebSocket(token_ids=["tok1"], on_book=on_book)
        await ws._handle({
            "event_type": "book",
            "asset_id": "tok_unknown",  # not subscribed
            "bids": [],
            "asks": [],
        })
        assert received == []

    async def test_list_message_processed(self):
        """Messages can arrive as lists of events."""
        received = []

        async def on_book(token_id, ob):
            received.append(token_id)

        ws = MarketWebSocket(token_ids=["tok1", "tok2"], on_book=on_book)
        await ws._handle([
            {"event_type": "book", "asset_id": "tok1", "bids": [], "asks": []},
            {"event_type": "book", "asset_id": "tok2", "bids": [], "asks": []},
        ])
        assert "tok1" in received
        assert "tok2" in received

    def test_get_orderbook_returns_current_state(self):
        ws = MarketWebSocket(token_ids=["tok1"])
        ws._books["tok1"].apply_snapshot(
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "100"}],
        )
        ob = ws.get_orderbook("tok1")
        assert ob is not None
        assert ob.best_bid == pytest.approx(0.50)

    def test_get_orderbook_unknown_token_returns_none(self):
        ws = MarketWebSocket(token_ids=["tok1"])
        assert ws.get_orderbook("unknown") is None

    def test_update_tokens_adds_new_subscriptions(self):
        ws = MarketWebSocket(token_ids=["tok1"])
        ws.update_tokens(["tok1", "tok2"])
        assert "tok2" in ws._books
