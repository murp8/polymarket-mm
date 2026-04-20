"""
Shared test fixtures for the Polymarket market maker test suite.
"""

from __future__ import annotations

import pytest

from src.models import (
    IncentiveParams,
    Market,
    MarketSide,
    Orderbook,
    PriceEstimate,
    PriceLevel,
    TokenInfo,
)


# ─── Market fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def incentive_params():
    return IncentiveParams(
        min_incentive_size=25.0,
        max_incentive_spread=0.05,
        reward_epoch_amount=1000.0,
        in_game_multiplier=1.0,
    )


@pytest.fixture
def yes_token():
    return TokenInfo(
        token_id="yes_token_abc123",
        outcome=MarketSide.YES,
        price=0.55,
    )


@pytest.fixture
def no_token():
    return TokenInfo(
        token_id="no_token_def456",
        outcome=MarketSide.NO,
        price=0.45,
    )


@pytest.fixture
def sample_market(yes_token, no_token, incentive_params):
    return Market(
        condition_id="0xdeadbeef1234567890abcdef",
        question_id="0xabcdef1234",
        question="Will the Golden State Warriors win the NBA championship?",
        yes_token=yes_token,
        no_token=no_token,
        incentive=incentive_params,
        volume_24h=50000.0,
        volume_total=200000.0,
        liquidity=10000.0,
        sport="nba",
        tags=["nba", "basketball"],
    )


@pytest.fixture
def epl_market(incentive_params):
    return Market(
        condition_id="0xepl_market_001",
        question_id="0xepl_001",
        question="Will Manchester City beat Arsenal?",
        yes_token=TokenInfo(token_id="yes_epl_001", outcome=MarketSide.YES, price=0.60),
        no_token=TokenInfo(token_id="no_epl_001", outcome=MarketSide.NO, price=0.40),
        incentive=incentive_params,
        volume_24h=80000.0,
        sport="soccer",
        tags=["epl", "soccer"],
    )


@pytest.fixture
def boundary_market(incentive_params):
    """A near-resolved market (high probability)."""
    return Market(
        condition_id="0xboundary_market",
        question_id="0xboundary_001",
        question="Will Team A win?",
        yes_token=TokenInfo(token_id="yes_boundary", outcome=MarketSide.YES, price=0.95),
        no_token=TokenInfo(token_id="no_boundary", outcome=MarketSide.NO, price=0.05),
        incentive=incentive_params,
        volume_24h=1000.0,
        tags=[],
    )


# ─── Orderbook fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_orderbook():
    return Orderbook(
        token_id="yes_token_abc123",
        bids=[
            PriceLevel(price=0.53, size=100.0),
            PriceLevel(price=0.52, size=200.0),
            PriceLevel(price=0.50, size=300.0),
        ],
        asks=[
            PriceLevel(price=0.57, size=100.0),
            PriceLevel(price=0.58, size=200.0),
            PriceLevel(price=0.60, size=300.0),
        ],
    )


# ─── Price fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def price_estimate_high_confidence():
    return PriceEstimate(
        probability=0.62,
        source="odds_api/pinnacle",
        confidence=0.9,
    )


@pytest.fixture
def price_estimate_low_confidence():
    return PriceEstimate(
        probability=0.55,
        source="orderbook",
        confidence=0.4,
    )
