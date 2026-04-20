"""
Tests for the composite pricer (cross-validation, fallback logic).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models import PriceEstimate
from src.pricing.composite import CompositePricer
from src.pricing.odds_api import OddsApiPricer
from src.pricing.orderbook import OrderbookPricer


@pytest.fixture
def ob_pricer():
    p = MagicMock(spec=OrderbookPricer)
    p.name = "orderbook"
    p.get_price = AsyncMock(return_value=PriceEstimate(
        probability=0.55, source="orderbook", confidence=0.8
    ))
    p.is_healthy = AsyncMock(return_value=True)
    return p


@pytest.fixture
def odds_pricer():
    p = MagicMock(spec=OddsApiPricer)
    p.name = "odds_api"
    p.get_price = AsyncMock(return_value=PriceEstimate(
        probability=0.60, source="odds_api/pinnacle", confidence=0.9
    ))
    p.is_healthy = AsyncMock(return_value=True)
    return p


class TestCompositePricer:
    async def test_returns_primary_when_available(self, odds_pricer, ob_pricer, sample_market):
        pricer = CompositePricer(
            odds_pricer=odds_pricer, orderbook_pricer=ob_pricer, primary_source="odds_api"
        )
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert result.source == "odds_api/pinnacle"
        assert result.probability == pytest.approx(0.60)

    async def test_falls_back_to_orderbook(self, odds_pricer, ob_pricer, sample_market):
        odds_pricer.get_price = AsyncMock(return_value=None)
        pricer = CompositePricer(
            odds_pricer=odds_pricer, orderbook_pricer=ob_pricer, primary_source="odds_api"
        )
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert result.source == "orderbook"

    async def test_no_odds_pricer_uses_orderbook(self, ob_pricer, sample_market):
        pricer = CompositePricer(
            odds_pricer=None, orderbook_pricer=ob_pricer, primary_source="odds_api"
        )
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert result.source == "orderbook"

    async def test_large_divergence_triggers_blend(self, odds_pricer, ob_pricer, sample_market):
        # Set external at 0.70, orderbook at 0.50 → diff 0.20 > 0.08 threshold
        odds_pricer.get_price = AsyncMock(return_value=PriceEstimate(
            probability=0.70, source="odds_api/pinnacle", confidence=0.9
        ))
        ob_pricer.get_price = AsyncMock(return_value=PriceEstimate(
            probability=0.50, source="orderbook", confidence=0.8
        ))
        pricer = CompositePricer(
            odds_pricer=odds_pricer,
            orderbook_pricer=ob_pricer,
            primary_source="odds_api",
            cross_check_threshold=0.08,
        )
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert "blended" in result.source
        # Blended probability should be between the two
        assert 0.50 <= result.probability <= 0.70
        # Confidence should be reduced
        assert result.confidence < 0.9

    async def test_small_divergence_returns_primary(self, odds_pricer, ob_pricer, sample_market):
        # Set external at 0.61, orderbook at 0.60 → diff 0.01 < 0.08 threshold
        odds_pricer.get_price = AsyncMock(return_value=PriceEstimate(
            probability=0.61, source="odds_api/pinnacle", confidence=0.9
        ))
        ob_pricer.get_price = AsyncMock(return_value=PriceEstimate(
            probability=0.60, source="orderbook", confidence=0.8
        ))
        pricer = CompositePricer(
            odds_pricer=odds_pricer,
            orderbook_pricer=ob_pricer,
            primary_source="odds_api",
            cross_check_threshold=0.08,
        )
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert result.source == "odds_api/pinnacle"

    async def test_odds_pricer_exception_falls_back(self, odds_pricer, ob_pricer, sample_market):
        odds_pricer.get_price = AsyncMock(side_effect=Exception("API error"))
        pricer = CompositePricer(
            odds_pricer=odds_pricer, orderbook_pricer=ob_pricer, primary_source="odds_api"
        )
        result = await pricer.get_price(sample_market)
        assert result is not None
        assert result.source == "orderbook"

    async def test_is_healthy_true_when_any_source_healthy(self, odds_pricer, ob_pricer):
        odds_pricer.is_healthy = AsyncMock(return_value=True)
        pricer = CompositePricer(
            odds_pricer=odds_pricer, orderbook_pricer=ob_pricer
        )
        assert await pricer.is_healthy() is True

    async def test_is_healthy_falls_back_to_orderbook(self, odds_pricer, ob_pricer):
        odds_pricer.is_healthy = AsyncMock(return_value=False)
        ob_pricer.is_healthy = AsyncMock(return_value=True)
        pricer = CompositePricer(
            odds_pricer=odds_pricer, orderbook_pricer=ob_pricer
        )
        assert await pricer.is_healthy() is True
