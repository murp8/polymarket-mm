"""
Tests for the market selector.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import MarketSelectionConfig
from src.market_selector.selector import MarketSelector, _score_market
from src.models import IncentiveParams, Market, MarketSide, TokenInfo


@pytest.fixture
def selector_cfg():
    return MarketSelectionConfig(
        max_markets=10,
        min_reward_pool_usd=100.0,
        min_midpoint=0.06,
        max_midpoint=0.94,
        require_clob_enabled=True,
        refresh_interval_seconds=300,
    )


def make_market(condition_id: str, reward=0.005, spread=0.05, mid=0.50, vol=50000) -> Market:
    """
    Build a test Market.  ``reward`` mirrors the CLOB rewards_daily_rate field
    stored in IncentiveParams.reward_epoch_amount (USDC/day, typically 0.001–0.01).
    """
    return Market(
        condition_id=condition_id,
        question_id=f"q_{condition_id}",
        question=f"Test market {condition_id}",
        yes_token=TokenInfo(token_id=f"yes_{condition_id}", outcome=MarketSide.YES, price=mid),
        no_token=TokenInfo(token_id=f"no_{condition_id}", outcome=MarketSide.NO, price=1 - mid),
        incentive=IncentiveParams(
            min_incentive_size=25.0,
            max_incentive_spread=spread,
            reward_epoch_amount=reward,
        ),
        volume_24h=vol,
        tags=[],
    )


@pytest.fixture
def gamma_mock():
    mock = MagicMock()
    # rates 0.001, 0.002, 0.003, 0.004, 0.005 — market5 scores highest
    markets = [make_market(f"market{i}", reward=0.001 * i) for i in range(1, 6)]
    mock.get_incentivized_markets = AsyncMock(return_value=markets)
    return mock


@pytest.fixture
def selector(gamma_mock, selector_cfg):
    return MarketSelector(gamma_client=gamma_mock, cfg=selector_cfg)


class TestEligibility:
    def test_eligible_market(self, selector):
        m = make_market("m1", reward=0.005, spread=0.05, mid=0.5)
        assert selector._is_eligible(m)

    def test_zero_spread_ineligible(self, selector):
        m = make_market("m1", spread=0.0)
        assert not selector._is_eligible(m)

    def test_tiny_reward_still_eligible_if_incentive_params_set(self, selector):
        # CLOB daily rates are tiny (0.001 USDC/day); still eligible as long as
        # min_incentive_size and max_incentive_spread are set.
        m = make_market("m1", reward=0.0001)
        assert selector._is_eligible(m)

    def test_boundary_mid_ineligible_high(self, selector):
        m = make_market("m1", mid=0.96)  # above 0.94 max
        assert not selector._is_eligible(m)

    def test_boundary_mid_ineligible_low(self, selector):
        m = make_market("m1", mid=0.04)  # below 0.06 min
        assert not selector._is_eligible(m)

    def test_zero_min_size_ineligible(self, selector):
        m = make_market("m1")
        m.incentive.min_incentive_size = 0
        assert not selector._is_eligible(m)


class TestRefresh:
    async def test_refresh_populates_markets(self, selector):
        await selector.refresh()
        assert selector.market_count == 5

    async def test_refresh_filters_ineligible(self, gamma_mock, selector_cfg):
        # Include one ineligible (midpoint out of range)
        markets = [
            make_market("good", reward=0.005),
            make_market("bad", mid=0.97),  # above 0.94 max_midpoint
        ]
        gamma_mock.get_incentivized_markets = AsyncMock(return_value=markets)
        sel = MarketSelector(gamma_client=gamma_mock, cfg=selector_cfg)
        await sel.refresh()
        assert sel.market_count == 1
        assert sel.get_market("good") is not None
        assert sel.get_market("bad") is None

    async def test_refresh_handles_exception(self, selector, gamma_mock):
        gamma_mock.get_incentivized_markets = AsyncMock(side_effect=Exception("API down"))
        await selector.refresh()  # should not raise


class TestTopMarkets:
    async def test_returns_at_most_max_markets(self, selector):
        await selector.refresh()
        top = selector.top_markets(n=3)
        assert len(top) <= 3

    async def test_higher_reward_ranked_first(self, selector):
        await selector.refresh()
        top = selector.top_markets()
        # market5 has rate=0.005 > market1 with rate=0.001
        assert top[0].incentive.reward_epoch_amount >= top[-1].incentive.reward_epoch_amount


class TestScoring:
    def test_higher_reward_higher_score(self):
        m1 = make_market("m1", reward=0.001)
        m2 = make_market("m2", reward=0.005)
        assert _score_market(m2) > _score_market(m1)

    def test_central_midpoint_higher_score(self):
        m_central = make_market("m1", mid=0.50)
        m_extreme = make_market("m2", mid=0.15)
        assert _score_market(m_central) > _score_market(m_extreme)

    def test_wider_spread_higher_score(self):
        m_wide = make_market("m1", spread=0.10)
        m_narrow = make_market("m2", spread=0.01)
        assert _score_market(m_wide) > _score_market(m_narrow)


class TestTokenMapping:
    async def test_all_token_ids_includes_yes_and_no(self, selector):
        await selector.refresh()
        ids = selector.all_token_ids()
        assert len(ids) == selector.market_count * 2

    async def test_yes_token_to_market_maps_correctly(self, selector):
        await selector.refresh()
        mapping = selector.yes_token_to_market()
        for cid, market in selector._markets.items():
            assert market.yes_token.token_id in mapping
            assert mapping[market.yes_token.token_id].condition_id == cid

    async def test_update_market_price(self, selector):
        await selector.refresh()
        first = list(selector._markets.values())[0]
        selector.update_market_price(first.condition_id, 0.75)
        updated = selector.get_market(first.condition_id)
        assert updated.yes_token.price == pytest.approx(0.75)
        assert updated.no_token.price == pytest.approx(0.25)
