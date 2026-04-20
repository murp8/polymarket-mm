"""
Tests for the Gamma API client.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from src.client.gamma import GammaClient
from src.models import MarketSide


# ─── Sample API response ──────────────────────────────────────────────────────

SAMPLE_RAW_MARKET = {
    "conditionId": "0xdeadbeef",
    "questionID": "0xabcdef",
    "question": "Will Team A win?",
    "clobTokenIds": json.dumps(["tok_yes_001", "tok_no_002"]),
    "outcomePrices": json.dumps(["0.60", "0.40"]),
    "rewardsMinSize": "25.0",
    "rewardsMaxSpread": "5.0",   # Gamma returns in cents (5.0 = 0.05 probability)
    "rewardEpochAmount": "1000.0",
    "enableOrderBook": True,
    "volume24hr": "50000",
    "volume": "200000",
    "liquidity": "10000",
    "endDate": "2025-06-30",
    "tags": [{"slug": "nba", "label": "NBA"}],
}

SAMPLE_RAW_MARKET_LIST_CLOB_IDS = {
    **SAMPLE_RAW_MARKET,
    "conditionId": "0xmarket002",
    "clobTokenIds": ["tok_yes_003", "tok_no_004"],  # Already a list, not JSON string
    "outcomePrices": ["0.45", "0.55"],
}


class TestParseMarket:
    def test_parses_json_string_clob_ids(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET)
        assert result is not None
        assert result.yes_token.token_id == "tok_yes_001"
        assert result.no_token.token_id == "tok_no_002"

    def test_parses_list_clob_ids(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET_LIST_CLOB_IDS)
        assert result is not None
        assert result.yes_token.token_id == "tok_yes_003"

    def test_condition_id_parsed(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET)
        assert result is not None
        assert result.condition_id == "0xdeadbeef"

    def test_incentive_params_parsed(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET)
        assert result is not None
        assert result.incentive.min_incentive_size == pytest.approx(25.0)
        assert result.incentive.max_incentive_spread == pytest.approx(0.05)
        assert result.incentive.reward_epoch_amount == pytest.approx(1000.0)

    def test_outcome_prices_parsed(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET)
        assert result is not None
        assert result.yes_token.price == pytest.approx(0.60)
        assert result.no_token.price == pytest.approx(0.40)

    def test_yes_token_outcome(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET)
        assert result is not None
        assert result.yes_token.outcome == MarketSide.YES
        assert result.no_token.outcome == MarketSide.NO

    def test_tags_parsed(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET)
        assert result is not None
        assert "nba" in result.tags

    def test_volume_parsed(self):
        result = GammaClient.parse_market(SAMPLE_RAW_MARKET)
        assert result is not None
        assert result.volume_24h == pytest.approx(50000.0)
        assert result.volume_total == pytest.approx(200000.0)

    def test_missing_clob_ids_returns_none(self):
        bad = {**SAMPLE_RAW_MARKET, "clobTokenIds": None}
        result = GammaClient.parse_market(bad)
        assert result is None

    def test_only_one_clob_id_returns_none(self):
        bad = {**SAMPLE_RAW_MARKET, "clobTokenIds": json.dumps(["only_one"])}
        result = GammaClient.parse_market(bad)
        assert result is None

    def test_parse_error_returns_none(self):
        result = GammaClient.parse_market({"completely": "wrong"})
        assert result is None


class TestGetIncentivizedMarkets:
    async def test_filters_non_incentivized(self):
        """Markets without reward params should be excluded."""
        raw_markets = [
            SAMPLE_RAW_MARKET,
            {**SAMPLE_RAW_MARKET, "conditionId": "0xno_reward", "rewardsMinSize": None},
            {**SAMPLE_RAW_MARKET, "conditionId": "0xzero_reward", "rewardsMinSize": "0"},
        ]

        with respx.mock:
            respx.get("https://gamma-api.polymarket.com/markets").mock(
                return_value=httpx.Response(200, json=raw_markets)
            )
            async with GammaClient() as client:
                raw = await client.get_incentivized_markets_raw()

        assert len(raw) == 1
        assert raw[0]["conditionId"] == "0xdeadbeef"

    async def test_returns_parsed_markets(self):
        with respx.mock:
            respx.get("https://gamma-api.polymarket.com/markets").mock(
                return_value=httpx.Response(200, json=[SAMPLE_RAW_MARKET])
            )
            async with GammaClient() as client:
                markets = await client.get_incentivized_markets()

        assert len(markets) == 1
        assert markets[0].condition_id == "0xdeadbeef"

    async def test_pagination_fetches_all_pages(self):
        """Should fetch multiple pages until an empty page is returned."""
        page1 = [SAMPLE_RAW_MARKET] * 100
        page2 = [{**SAMPLE_RAW_MARKET, "conditionId": f"0x{i}"} for i in range(30)]

        call_count = 0

        def paginated_response(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=page1 if call_count == 1 else page2)

        with respx.mock:
            respx.get("https://gamma-api.polymarket.com/markets").mock(
                side_effect=paginated_response
            )
            async with GammaClient(max_page_size=100) as client:
                raw = await client._paginate("/markets")

        assert len(raw) == 130
