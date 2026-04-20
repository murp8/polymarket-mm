"""
Tests for PaperOrderManager — fill simulation, reward accumulation, balance tracking.
"""

from __future__ import annotations

import pytest

from src.execution.paper_exchange import PaperOrderManager
from src.models import (
    IncentiveParams,
    Market,
    MarketQuote,
    MarketSide,
    Orderbook,
    PriceLevel,
    QuoteTarget,
    Side,
    TokenInfo,
)
from src.strategy.inventory import InventoryManager
from src.utils.metrics import MetricsCollector


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def inv():
    return InventoryManager(max_inventory_per_market=500.0)


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def paper(inv, metrics):
    return PaperOrderManager(inventory=inv, metrics=metrics, initial_balance_usdc=1000.0)


@pytest.fixture
def market():
    return Market(
        condition_id="0xmarket_cid_001",
        question_id="0xqid",
        question="Test market?",
        yes_token=TokenInfo(token_id="yes_tok", outcome=MarketSide.YES, price=0.55),
        no_token=TokenInfo(token_id="no_tok", outcome=MarketSide.NO, price=0.45),
        incentive=IncentiveParams(
            min_incentive_size=20.0,
            max_incentive_spread=0.05,
            reward_epoch_amount=0.005,  # USDC/day
            in_game_multiplier=1.0,
        ),
        volume_24h=10_000.0,
        volume_total=100_000.0,
        liquidity=5_000.0,
        end_date_iso="2026-12-31",
        sport="",
        tags=[],
    )


def make_quote(cid: str, bid_price: float = 0.52, ask_price: float = 0.58, size: float = 25.0) -> MarketQuote:
    return MarketQuote(
        market_condition_id=cid,
        yes_bid=QuoteTarget(
            market_condition_id=cid,
            token_id="yes_tok",
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=bid_price,
            size=size,
        ),
        yes_ask=QuoteTarget(
            market_condition_id=cid,
            token_id="yes_tok",
            outcome=MarketSide.YES,
            side=Side.SELL,
            price=ask_price,
            size=size,
        ),
    )


def make_book(best_bid: float, best_ask: float) -> Orderbook:
    return Orderbook(
        token_id="yes_tok",
        bids=[PriceLevel(price=best_bid, size=100.0)],
        asks=[PriceLevel(price=best_ask, size=100.0)],
    )


# ─── Interface properties ─────────────────────────────────────────────────────


def test_geoblocked_always_false(paper):
    assert paper.geoblocked is False


@pytest.mark.asyncio
async def test_sync_open_orders_is_noop(paper):
    await paper.sync_open_orders()  # should not raise


@pytest.mark.asyncio
async def test_handle_fill_is_noop(paper):
    await paper.handle_fill({"id": "x", "status": "MATCHED"})  # should not raise


# ─── Order placement ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_quotes_places_bid_and_ask(paper, market):
    quote = make_quote(market.condition_id)
    await paper.update_quotes(quote)

    state = paper._states[market.condition_id]
    assert state.bid is not None
    assert state.ask is not None
    assert state.bid.price == pytest.approx(0.52)
    assert state.ask.price == pytest.approx(0.58)


@pytest.mark.asyncio
async def test_update_quotes_locks_usdc_for_buy(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52, size=25.0)
    await paper.update_quotes(quote)
    # locked = 0.52 * 25 = 13.0
    assert paper._locked_usdc == pytest.approx(13.0)


@pytest.mark.asyncio
async def test_update_quotes_refuses_buy_if_insufficient_balance(paper, market):
    # Drain most of the balance
    paper._balance = 5.0
    quote = make_quote(market.condition_id, bid_price=0.52, size=25.0)  # cost = 13.0
    await paper.update_quotes(quote)

    state = paper._states[market.condition_id]
    assert state.bid is None  # should not have been placed


@pytest.mark.asyncio
async def test_replace_order_when_price_changes(paper, market):
    quote1 = make_quote(market.condition_id, bid_price=0.52)
    await paper.update_quotes(quote1)
    first_order_id = paper._states[market.condition_id].bid.order_id

    quote2 = make_quote(market.condition_id, bid_price=0.54)  # > 0.005 change
    await paper.update_quotes(quote2)
    second_order_id = paper._states[market.condition_id].bid.order_id

    assert first_order_id != second_order_id
    assert paper._states[market.condition_id].bid.price == pytest.approx(0.54)


@pytest.mark.asyncio
async def test_no_replace_when_price_unchanged(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52)
    await paper.update_quotes(quote)
    first_order_id = paper._states[market.condition_id].bid.order_id

    await paper.update_quotes(quote)  # same price
    assert paper._states[market.condition_id].bid.order_id == first_order_id


# ─── Fill simulation ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_buy_fills_when_ask_le_bid_price(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52, ask_price=0.58, size=25.0)
    await paper.update_quotes(quote)

    initial_balance = paper._balance
    # best_ask = 0.51 ≤ bid = 0.52 → should fill
    book = make_book(best_bid=0.49, best_ask=0.51)
    paper.on_orderbook_update("yes_tok", book)

    assert paper._fill_count == 1
    # balance reduced by fill_price * size = 0.51 * 25 = 12.75
    assert paper._balance == pytest.approx(initial_balance - 12.75)
    # bid should be cleared
    assert paper._states[market.condition_id].bid is None


@pytest.mark.asyncio
async def test_buy_does_not_fill_when_ask_above_bid(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52, ask_price=0.58, size=25.0)
    await paper.update_quotes(quote)

    book = make_book(best_bid=0.50, best_ask=0.53)  # ask > bid
    paper.on_orderbook_update("yes_tok", book)

    assert paper._fill_count == 0
    assert paper._states[market.condition_id].bid is not None


@pytest.mark.asyncio
async def test_sell_fills_when_bid_ge_ask_price(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52, ask_price=0.58, size=25.0)
    await paper.update_quotes(quote)

    initial_balance = paper._balance
    # best_bid = 0.59 ≥ ask = 0.58 → should fill
    book = make_book(best_bid=0.59, best_ask=0.62)
    paper.on_orderbook_update("yes_tok", book)

    assert paper._fill_count == 1
    # balance increased by fill_price * size = 0.59 * 25 = 14.75
    assert paper._balance == pytest.approx(initial_balance + 14.75)
    assert paper._states[market.condition_id].ask is None


@pytest.mark.asyncio
async def test_both_sides_can_fill_same_tick(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52, ask_price=0.58, size=25.0)
    await paper.update_quotes(quote)

    # Extreme book: ask=0.50 (fills buy), bid=0.60 (fills sell)
    book = make_book(best_bid=0.60, best_ask=0.50)
    paper.on_orderbook_update("yes_tok", book)

    assert paper._fill_count == 2
    assert paper._states[market.condition_id].bid is None
    assert paper._states[market.condition_id].ask is None


# ─── Cancel ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_market_clears_orders(paper, market):
    quote = make_quote(market.condition_id)
    await paper.update_quotes(quote)

    await paper.cancel_market(market.condition_id)
    state = paper._states[market.condition_id]
    assert state.bid is None
    assert state.ask is None


@pytest.mark.asyncio
async def test_cancel_market_releases_locked_usdc(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52, size=25.0)
    await paper.update_quotes(quote)
    assert paper._locked_usdc > 0

    await paper.cancel_market(market.condition_id)
    assert paper._locked_usdc == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_cancel_all_markets(paper, market):
    quote = make_quote(market.condition_id)
    await paper.update_quotes(quote)
    await paper.cancel_all_markets()
    assert paper._locked_usdc == pytest.approx(0.0)


# ─── Reward accumulation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rewards_accumulate_for_qualifying_order(paper, market):
    paper.register_market(market)
    quote = make_quote(market.condition_id, bid_price=0.52, size=25.0)
    await paper.update_quotes(quote)

    mid = 0.55
    # Manually tick rewards with dt=3600 (1 hour)
    state = paper._states[market.condition_id]
    # Inject a past tick time so _tick_rewards computes a large dt
    import time
    state.last_reward_tick = time.monotonic() - 3600

    book = make_book(best_bid=0.50, best_ask=0.56)  # does not fill bid at 0.52
    paper.on_orderbook_update("yes_tok", book)

    reward = paper._reward_usdc.get(market.condition_id, 0.0)
    assert reward > 0.0


@pytest.mark.asyncio
async def test_rewards_zero_when_outside_spread(paper, market):
    paper.register_market(market)
    # Both orders far outside max_spread=0.05 from mid
    quote = make_quote(market.condition_id, bid_price=0.30, ask_price=0.80, size=25.0)
    await paper.update_quotes(quote)

    state = paper._states[market.condition_id]
    import time
    state.last_reward_tick = time.monotonic() - 3600

    # mid = (0.54 + 0.58) / 2 = 0.56; |0.30 - 0.56| = 0.26 >> 0.05; |0.80 - 0.56| = 0.24 >> 0.05
    book = make_book(best_bid=0.54, best_ask=0.58)
    paper.on_orderbook_update("yes_tok", book)

    reward = paper._reward_usdc.get(market.condition_id, 0.0)
    assert reward == pytest.approx(0.0)


def test_no_orderbook_update_without_token_registration(paper, market):
    """on_orderbook_update silently ignores unknown tokens."""
    book = make_book(best_bid=0.55, best_ask=0.57)
    paper.on_orderbook_update("unknown_token", book)  # should not raise
    assert paper._fill_count == 0


# ─── Summary ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_structure(paper, market):
    quote = make_quote(market.condition_id)
    await paper.update_quotes(quote)

    s = paper.summary()
    assert "balance_usdc" in s
    assert "locked_usdc" in s
    assert "open_orders" in s
    assert "fills" in s
    assert "realized_pnl" in s
    assert "reward_earned_usdc" in s
    assert "open_yes_orders" in s
    assert "open_no_orders" in s
    assert s["open_orders"] == 2  # one bid, one ask


@pytest.mark.asyncio
async def test_summary_fill_count_increments(paper, market):
    quote = make_quote(market.condition_id, bid_price=0.52, ask_price=0.58, size=25.0)
    await paper.update_quotes(quote)

    book = make_book(best_bid=0.60, best_ask=0.50)  # fills both
    paper.on_orderbook_update("yes_tok", book)

    assert paper.summary()["fills"] == 2


# ─── Cross-token fill guard ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_token_update_does_not_fill_yes_orders(paper, market):
    """
    A NO token orderbook update (prices ~0.45) must not fill a YES bid (at 0.52).
    Without the token_id guard this triggered false YES fills at NO prices,
    producing impossibly cheap buys and inflated realized PnL.
    """
    await paper.update_quotes(make_quote(market.condition_id, bid_price=0.52, ask_price=0.58))
    start_balance = paper._balance

    # NO token book: best_ask=0.40 — would cross YES bid (0.52) without the guard
    no_book = Orderbook(
        token_id="no_tok",
        bids=[PriceLevel(price=0.46, size=100.0)],
        asks=[PriceLevel(price=0.40, size=100.0)],
    )
    paper.on_orderbook_update("no_tok", no_book)

    assert paper.summary()["fills"] == 0
    assert paper._balance == start_balance  # no cash should have moved


@pytest.mark.asyncio
async def test_yes_token_update_does_not_fill_no_orders(paper, market):
    """
    A YES token orderbook update must not fill NO token orders.
    """
    no_quote = MarketQuote(
        market_condition_id=market.condition_id,
        yes_bid=QuoteTarget(
            market_condition_id=market.condition_id, token_id="yes_tok",
            outcome=MarketSide.YES, side=Side.BUY, price=0.52, size=0.0,
        ),
        yes_ask=QuoteTarget(
            market_condition_id=market.condition_id, token_id="yes_tok",
            outcome=MarketSide.YES, side=Side.SELL, price=0.58, size=0.0,
        ),
        no_bid=QuoteTarget(
            market_condition_id=market.condition_id, token_id="no_tok",
            outcome=MarketSide.NO, side=Side.BUY, price=0.44, size=25.0,
        ),
        no_ask=QuoteTarget(
            market_condition_id=market.condition_id, token_id="no_tok",
            outcome=MarketSide.NO, side=Side.SELL, price=0.50, size=25.0,
        ),
    )
    await paper.update_quotes(no_quote)
    start_balance = paper._balance

    # YES token book: best_ask=0.40 — should NOT fill NO bid (0.44) since tokens differ
    yes_book = make_book(best_bid=0.56, best_ask=0.40)
    paper.on_orderbook_update("yes_tok", yes_book)

    assert paper.summary()["fills"] == 0
    assert paper._balance == start_balance
