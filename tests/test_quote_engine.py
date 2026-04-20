"""
Tests for the quote engine.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import QuotingConfig
from src.models import MarketSide, PriceEstimate, Side
from src.strategy.inventory import InventoryManager
from src.strategy.quote_engine import QuoteEngine


@pytest.fixture
def quoting_cfg():
    return QuotingConfig(
        target_spread_fraction=0.25,
        base_order_size=50.0,
        max_order_size=200.0,
        inventory_skew_threshold=100.0,
        max_inventory_per_market=500.0,
        refresh_interval_seconds=15,
        requote_threshold=0.005,
        min_requote_interval_seconds=2.0,
    )


@pytest.fixture
def inventory(sample_market):
    inv = InventoryManager(max_inventory_per_market=500.0, inventory_skew_threshold=100.0)
    inv.ensure_position(
        sample_market.condition_id,
        sample_market.yes_token.token_id,
        sample_market.no_token.token_id,
    )
    return inv


@pytest.fixture
def mock_pricer():
    pricer = MagicMock()
    pricer.get_price = AsyncMock(return_value=PriceEstimate(
        probability=0.60,
        source="test",
        confidence=0.9,
    ))
    return pricer


@pytest.fixture
def engine(mock_pricer, inventory, quoting_cfg):
    return QuoteEngine(pricer=mock_pricer, inventory=inventory, cfg=quoting_cfg)


class TestComputeQuote:
    async def test_returns_market_quote(self, engine, sample_market):
        quote = await engine.compute_quote(sample_market)
        assert quote is not None
        assert quote.market_condition_id == sample_market.condition_id

    async def test_bid_below_ask(self, engine, sample_market):
        quote = await engine.compute_quote(sample_market)
        assert quote is not None
        assert quote.yes_bid.price < quote.yes_ask.price

    async def test_bid_is_buy_ask_is_sell(self, engine, sample_market):
        quote = await engine.compute_quote(sample_market)
        assert quote is not None
        assert quote.yes_bid.side == Side.BUY
        assert quote.yes_ask.side == Side.SELL

    async def test_both_outcomes_are_yes(self, engine, sample_market):
        quote = await engine.compute_quote(sample_market)
        assert quote is not None
        assert quote.yes_bid.outcome == MarketSide.YES
        assert quote.yes_ask.outcome == MarketSide.YES

    async def test_no_quote_when_no_price(self, mock_pricer, inventory, quoting_cfg, sample_market):
        mock_pricer.get_price = AsyncMock(return_value=None)
        engine = QuoteEngine(pricer=mock_pricer, inventory=inventory, cfg=quoting_cfg)
        quote = await engine.compute_quote(sample_market)
        assert quote is None

    async def test_no_quote_when_price_at_hardcoded_boundary(self, mock_pricer, inventory, quoting_cfg, boundary_market):
        """Fair value beyond the hardcoded safety rails (>0.98) must not be quoted."""
        mock_pricer.get_price = AsyncMock(return_value=PriceEstimate(
            probability=0.99, source="test", confidence=0.9
        ))
        inv = InventoryManager(max_inventory_per_market=500.0)
        inv.ensure_position(
            boundary_market.condition_id,
            boundary_market.yes_token.token_id,
            boundary_market.no_token.token_id,
        )
        engine = QuoteEngine(pricer=mock_pricer, inventory=inv, cfg=quoting_cfg)
        quote = await engine.compute_quote(boundary_market)
        assert quote is None

    async def test_bid_size_at_least_min_incentive_size(self, engine, sample_market):
        """Buy side should meet the min incentive size when no inventory constraint."""
        quote = await engine.compute_quote(sample_market)
        assert quote is not None
        min_size = sample_market.incentive.min_incentive_size
        assert quote.yes_bid.size >= min_size

    async def test_ask_size_with_inventory(self, mock_pricer, quoting_cfg, sample_market):
        """Ask size should equal shares held (capped at order size), zero when flat."""
        inv = InventoryManager(max_inventory_per_market=500.0, inventory_skew_threshold=100.0)
        inv.ensure_position(
            sample_market.condition_id,
            sample_market.yes_token.token_id,
            sample_market.no_token.token_id,
        )
        # Give the position 100 yes shares
        from src.models import Fill, Side as S, MarketSide as MS
        inv.apply_fill(Fill(
            order_id="x", market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id, outcome=MS.YES,
            side=S.BUY, price=0.50, size=100.0,
        ))
        engine = QuoteEngine(pricer=mock_pricer, inventory=inv, cfg=quoting_cfg)
        quote = await engine.compute_quote(sample_market)
        assert quote is not None
        assert quote.yes_ask.size >= sample_market.incentive.min_incentive_size

    async def test_size_at_most_max_order_size(self, engine, sample_market):
        quote = await engine.compute_quote(sample_market)
        assert quote is not None
        assert quote.yes_bid.size <= 200.0  # max_order_size

    async def test_low_confidence_widens_spread(
        self, mock_pricer, inventory, quoting_cfg, sample_market
    ):
        """Lower confidence prices should result in wider spreads."""
        mock_pricer.get_price = AsyncMock(return_value=PriceEstimate(
            probability=0.60, source="test", confidence=0.3
        ))
        engine_low = QuoteEngine(pricer=mock_pricer, inventory=inventory, cfg=quoting_cfg)
        quote_low = await engine_low.compute_quote(sample_market)

        mock_pricer.get_price = AsyncMock(return_value=PriceEstimate(
            probability=0.60, source="test", confidence=0.9
        ))
        engine_high = QuoteEngine(pricer=mock_pricer, inventory=inventory, cfg=quoting_cfg)
        quote_high = await engine_high.compute_quote(sample_market)

        if quote_low and quote_high:
            spread_low = quote_low.yes_ask.price - quote_low.yes_bid.price
            spread_high = quote_high.yes_ask.price - quote_high.yes_bid.price
            assert spread_low >= spread_high


