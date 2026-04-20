"""
Tests for the orderbook midpoint pricer.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.pricing.orderbook import OrderbookPricer


@pytest.fixture
def pricer():
    return OrderbookPricer(max_age_seconds=60.0)


class TestOrderbookPricer:
    def test_name(self, pricer):
        assert pricer.name == "orderbook"

    async def test_no_data_returns_gamma_price(self, pricer, sample_market):
        """Falls back to market.yes_token.price when no WS data."""
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert result.probability == pytest.approx(sample_market.yes_token.price)
        assert "gamma" in result.source

    async def test_fresh_midpoint_returned(self, pricer, sample_market):
        pricer.update_midpoint(sample_market.condition_id, 0.62)
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert result.probability == pytest.approx(0.62)
        assert result.source == "orderbook"

    async def test_stale_data_returns_none(self, pricer, sample_market):
        pricer.update_midpoint(sample_market.condition_id, 0.62)
        # Age the entry
        pricer._midpoints[sample_market.condition_id] = (0.62, time.monotonic() - 200)
        result = await pricer.get_price(sample_market)
        assert result is None

    async def test_confidence_decays_with_age(self, pricer, sample_market):
        # Fresh data
        pricer.update_midpoint(sample_market.condition_id, 0.62)
        fresh = await pricer.get_price(sample_market)

        # Slightly aged data (30 seconds)
        pricer._midpoints[sample_market.condition_id] = (0.62, time.monotonic() - 30)
        aged = await pricer.get_price(sample_market)

        assert fresh is not None
        assert aged is not None
        assert fresh.confidence >= aged.confidence

    async def test_is_healthy_always_true(self, pricer):
        assert await pricer.is_healthy() is True

    def test_update_midpoint_replaces_old(self, pricer, sample_market):
        pricer.update_midpoint(sample_market.condition_id, 0.60)
        pricer.update_midpoint(sample_market.condition_id, 0.65)
        mid, _ = pricer._midpoints[sample_market.condition_id]
        assert mid == pytest.approx(0.65)
