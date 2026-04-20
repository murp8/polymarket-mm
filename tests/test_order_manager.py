"""
Tests for the order manager — placement, cancel/replace, fill handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from src.config import ExecutionConfig
from src.execution.order_manager import OrderManager
from src.models import Fill, MarketQuote, MarketSide, QuoteTarget, Side
from src.strategy.inventory import InventoryManager
from src.utils.metrics import MetricsCollector


@pytest.fixture
def mock_client():
    c = MagicMock()
    # place_limit_orders_batch receives a list of order specs and returns a
    # list of unique order IDs (one per spec).
    _counter = [0]
    def _batch_place(specs):
        ids = []
        for _ in specs:
            _counter[0] += 1
            ids.append(f"order_{_counter[0]:03d}")
        return ids
    c.place_limit_orders_batch = AsyncMock(side_effect=_batch_place)
    c.cancel_order = AsyncMock(return_value=True)
    c.cancel_orders = AsyncMock(return_value=2)
    c.cancel_all = AsyncMock(return_value=True)
    c.get_open_orders = AsyncMock(return_value=[])
    return c


@pytest.fixture
def inv(sample_market):
    i = InventoryManager(max_inventory_per_market=500.0)
    i.ensure_position(
        sample_market.condition_id,
        sample_market.yes_token.token_id,
        sample_market.no_token.token_id,
    )
    return i


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def exec_cfg():
    return ExecutionConfig(
        order_rate_limit=10,
        cancel_replace_rate_limit=60,
        max_order_retries=3,
        retry_delay_seconds=0.1,
    )


@pytest.fixture
def order_mgr(mock_client, inv, metrics, exec_cfg):
    return OrderManager(
        client=mock_client,
        inventory=inv,
        metrics=metrics,
        cfg=exec_cfg,
        order_rate_limit=10,
        cancel_replace_rate_limit=60,
        min_requote_interval=0.0,  # no delay for tests
    )


def make_quote(condition_id, yes_tid, bid_p=0.48, ask_p=0.52, size=50.0):
    return MarketQuote(
        market_condition_id=condition_id,
        yes_bid=QuoteTarget(
            market_condition_id=condition_id,
            token_id=yes_tid,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=bid_p,
            size=size,
        ),
        yes_ask=QuoteTarget(
            market_condition_id=condition_id,
            token_id=yes_tid,
            outcome=MarketSide.YES,
            side=Side.SELL,
            price=ask_p,
            size=size,
        ),
    )


class TestUpdateQuotes:
    async def test_places_bid_and_ask_on_first_call(self, order_mgr, sample_market, mock_client):
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)
        # One batch call containing both bid and ask
        assert mock_client.place_limit_orders_batch.call_count == 1
        batch_specs = mock_client.place_limit_orders_batch.call_args[0][0]
        assert len(batch_specs) == 2

    async def test_places_bid_with_correct_params(self, order_mgr, sample_market, mock_client):
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id, bid_p=0.48)
        await order_mgr.update_quotes(quote)
        batch_specs = mock_client.place_limit_orders_batch.call_args[0][0]
        buy_spec = next(s for s in batch_specs if s["side"] == Side.BUY)
        assert buy_spec["price"] == 0.48

    async def test_no_replace_when_price_unchanged(self, order_mgr, sample_market, mock_client):
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)
        call_count_after_first = mock_client.place_limit_orders_batch.call_count

        # Same quote again — no orders need replacing
        await order_mgr.update_quotes(quote)
        assert mock_client.place_limit_orders_batch.call_count == call_count_after_first

    async def test_cancels_and_replaces_on_price_change(
        self, order_mgr, sample_market, mock_client
    ):
        quote1 = make_quote(sample_market.condition_id, sample_market.yes_token.token_id, bid_p=0.48)
        await order_mgr.update_quotes(quote1)

        # Move bid by 1 cent (> 0.5 cent threshold) — should cancel old bid and place new
        quote2 = make_quote(sample_market.condition_id, sample_market.yes_token.token_id, bid_p=0.49)
        await order_mgr.update_quotes(quote2)

        assert mock_client.cancel_order.called

    async def test_order_id_registered_in_mapping(self, order_mgr, sample_market, mock_client):
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)
        state = order_mgr._state(sample_market.condition_id)
        # Both bid and ask should be registered and distinct
        assert state.bid_order_id is not None
        assert state.ask_order_id is not None
        assert state.bid_order_id != state.ask_order_id
        assert state.bid_order_id in order_mgr._order_to_market
        assert state.ask_order_id in order_mgr._order_to_market


class TestCancelMarket:
    async def test_cancels_both_orders(self, order_mgr, sample_market, mock_client):
        # Place orders first
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)

        await order_mgr.cancel_market(sample_market.condition_id)

        state = order_mgr._state(sample_market.condition_id)
        assert state.bid_order_id is None
        assert state.ask_order_id is None


class TestHandleFill:
    async def test_matched_fill_updates_inventory(self, order_mgr, sample_market, mock_client, inv):
        # Place an order
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)
        state = order_mgr._state(sample_market.condition_id)
        order_id = state.bid_order_id

        # Simulate fill
        await order_mgr.handle_fill({
            "id": order_id,
            "status": "MATCHED",
            "asset_id": sample_market.yes_token.token_id,
            "side": "BUY",
            "price": "0.48",
            "size_matched": "50",
        })

        pos = inv.get_position(sample_market.condition_id)
        assert pos is not None
        assert pos.yes_shares == pytest.approx(50.0)

    async def test_confirmed_clears_order_from_state(
        self, order_mgr, sample_market, mock_client
    ):
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)
        state = order_mgr._state(sample_market.condition_id)
        order_id = state.bid_order_id

        await order_mgr.handle_fill({"id": order_id, "status": "CONFIRMED"})
        state2 = order_mgr._state(sample_market.condition_id)
        assert state2.bid_order_id is None

    async def test_failed_order_clears_state_and_triggers_requote(
        self, order_mgr, sample_market, mock_client
    ):
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)
        state = order_mgr._state(sample_market.condition_id)
        order_id = state.ask_order_id

        await order_mgr.handle_fill({"id": order_id, "status": "FAILED"})
        state2 = order_mgr._state(sample_market.condition_id)
        assert state2.ask_order_id is None
        # last_action_ts should be reset to enable immediate requote
        assert state2.last_action_ts == 0.0

    async def test_unknown_order_id_ignored(self, order_mgr):
        """Fill for an order we don't know about should not crash."""
        await order_mgr.handle_fill({
            "id": "unknown_order",
            "status": "MATCHED",
            "asset_id": "tok_x",
            "side": "BUY",
            "price": "0.5",
            "size_matched": "100",
        })


class TestStaleOrderCleanup:
    async def test_stale_orders_cancelled(self, order_mgr, sample_market, mock_client):
        import time
        quote = make_quote(sample_market.condition_id, sample_market.yes_token.token_id)
        await order_mgr.update_quotes(quote)

        # Age the last action
        state = order_mgr._state(sample_market.condition_id)
        state.last_action_ts = time.monotonic() - 30  # 30 seconds ago

        await order_mgr.cleanup_stale_orders()
        # _cancel_batch uses cancel_orders (plural) for bulk cancellation
        assert mock_client.cancel_orders.called
