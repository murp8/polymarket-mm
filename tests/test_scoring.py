"""
Tests for the reward scoring calculator.

The scoring formula: S(v, s) = ((v - s) / v)² × b

where v = max_spread, s = actual_spread, b = size
"""

from __future__ import annotations

import pytest

from src.strategy.scoring import (
    ScoreResult,
    expected_reward_fraction,
    optimal_quote_prices,
    order_score,
    score_two_sided_quote,
)


class TestOrderScore:
    def test_at_midpoint_gets_full_score(self):
        """Order exactly at midpoint (spread=0) gets maximum score."""
        score = order_score(max_spread=0.05, actual_spread=0.0, size=100.0)
        assert score == pytest.approx(100.0)

    def test_at_max_spread_gets_zero(self):
        """Order at exactly the max spread boundary gets zero score."""
        score = order_score(max_spread=0.05, actual_spread=0.05, size=100.0)
        assert score == pytest.approx(0.0)

    def test_beyond_max_spread_gets_zero(self):
        """Order outside the max spread gets zero."""
        score = order_score(max_spread=0.05, actual_spread=0.08, size=100.0)
        assert score == 0.0

    def test_half_spread_gets_quarter_score(self):
        """At half the max spread, score = ((0.5)/1)^2 × b = 0.25 × b."""
        score = order_score(max_spread=0.10, actual_spread=0.05, size=100.0)
        assert score == pytest.approx(25.0)

    def test_quadratic_decay(self):
        """Score decreases quadratically as spread increases."""
        s0 = order_score(max_spread=0.10, actual_spread=0.00, size=100.0)
        s1 = order_score(max_spread=0.10, actual_spread=0.02, size=100.0)
        s2 = order_score(max_spread=0.10, actual_spread=0.05, size=100.0)
        s3 = order_score(max_spread=0.10, actual_spread=0.08, size=100.0)
        assert s0 > s1 > s2 > s3
        # Check it's truly quadratic: (0.8)^2 at 0.02, (0.5)^2 at 0.05, (0.2)^2 at 0.08
        assert s1 == pytest.approx(100.0 * 0.64, rel=1e-4)
        assert s2 == pytest.approx(100.0 * 0.25, rel=1e-4)
        assert s3 == pytest.approx(100.0 * 0.04, rel=1e-4)

    def test_size_scales_linearly(self):
        """Score is linear in order size."""
        s100 = order_score(max_spread=0.10, actual_spread=0.02, size=100.0)
        s200 = order_score(max_spread=0.10, actual_spread=0.02, size=200.0)
        assert s200 == pytest.approx(s100 * 2)

    def test_zero_size_gets_zero(self):
        assert order_score(max_spread=0.10, actual_spread=0.0, size=0.0) == 0.0

    def test_zero_max_spread_gets_zero(self):
        assert order_score(max_spread=0.0, actual_spread=0.0, size=100.0) == 0.0

    def test_in_game_multiplier_scales_score(self):
        base = order_score(max_spread=0.10, actual_spread=0.02, size=100.0, in_game_multiplier=1.0)
        boosted = order_score(max_spread=0.10, actual_spread=0.02, size=100.0, in_game_multiplier=2.0)
        assert boosted == pytest.approx(base * 2)


class TestTwoSidedScoring:
    def test_balanced_two_sided_quote(self):
        """Symmetric two-sided quote should give equal Q_one and Q_two."""
        result = score_two_sided_quote(
            midpoint=0.50,
            yes_bid_price=0.47,
            yes_ask_price=0.53,
            yes_bid_size=100.0,
            yes_ask_size=100.0,
            max_spread=0.05,
            min_size=10.0,
        )
        assert result.q_one == pytest.approx(result.q_two, rel=1e-4)
        assert result.q_min > 0
        assert result.within_spread is True

    def test_single_sided_penalty(self):
        """One-sided quote scores at 1/3 of the two-sided rate."""
        # Only bid side
        bid_only = score_two_sided_quote(
            midpoint=0.50,
            yes_bid_price=0.48,
            yes_ask_price=0.80,  # outside max spread
            yes_bid_size=100.0,
            yes_ask_size=100.0,
            max_spread=0.05,
            min_size=10.0,
        )
        # Both sides
        both_sides = score_two_sided_quote(
            midpoint=0.50,
            yes_bid_price=0.48,
            yes_ask_price=0.52,
            yes_bid_size=100.0,
            yes_ask_size=100.0,
            max_spread=0.05,
            min_size=10.0,
        )
        # Single-sided should be penalised
        assert bid_only.q_min < both_sides.q_min

    def test_below_min_size_gets_zero(self):
        """Orders below min_incentive_size should not score."""
        result = score_two_sided_quote(
            midpoint=0.50,
            yes_bid_price=0.48,
            yes_ask_price=0.52,
            yes_bid_size=5.0,  # below min_size
            yes_ask_size=5.0,
            max_spread=0.05,
            min_size=25.0,
        )
        assert result.q_min == 0.0
        assert result.within_spread is False

    def test_near_boundary_market_strict_two_sided(self):
        """For mid < 0.10, Q_min = min(Q_one, Q_two) with no single-side relief."""
        result = score_two_sided_quote(
            midpoint=0.05,  # boundary market
            yes_bid_price=0.03,
            yes_ask_price=0.07,
            yes_bid_size=100.0,
            yes_ask_size=0.0,  # no ask
            max_spread=0.05,
            min_size=10.0,
        )
        # min(Q_one, Q_two) where Q_two = 0 → q_min = 0
        assert result.q_min == 0.0

    def test_tighter_spread_gives_higher_score(self):
        """Tighter quotes always score higher, all else equal."""
        loose = score_two_sided_quote(
            midpoint=0.50, yes_bid_price=0.46, yes_ask_price=0.54,
            yes_bid_size=100.0, yes_ask_size=100.0, max_spread=0.05, min_size=10.0,
        )
        tight = score_two_sided_quote(
            midpoint=0.50, yes_bid_price=0.49, yes_ask_price=0.51,
            yes_bid_size=100.0, yes_ask_size=100.0, max_spread=0.05, min_size=10.0,
        )
        assert tight.q_min > loose.q_min


class TestOptimalQuotePrices:
    def test_symmetric_at_zero_skew(self):
        """With no inventory skew, quotes should be symmetric around fair value."""
        bid, ask = optimal_quote_prices(
            fair_value=0.50,
            max_spread=0.10,
            target_spread_fraction=0.25,
            inventory_skew=0.0,
        )
        assert abs((ask + bid) / 2 - 0.50) < 0.001
        assert ask > bid

    def test_positive_skew_lowers_bid(self):
        """Long position lowers the bid price (less eager to accumulate more YES)."""
        bid_no_skew, ask_no_skew = optimal_quote_prices(0.50, 0.10, 0.25, 0.0)
        bid_long, ask_long = optimal_quote_prices(0.50, 0.10, 0.25, 0.8)
        assert bid_long < bid_no_skew

    def test_prices_clamped_to_valid_range(self):
        """Prices must always be in (0.01, 0.99)."""
        bid, ask = optimal_quote_prices(
            fair_value=0.02,
            max_spread=0.10,
            target_spread_fraction=0.9,
        )
        assert bid >= 0.01
        assert ask <= 0.99
        assert bid < ask

    def test_bid_always_less_than_ask(self):
        for fv in [0.05, 0.10, 0.50, 0.90, 0.95]:
            for fraction in [0.0, 0.25, 0.5, 0.9]:
                bid, ask = optimal_quote_prices(fv, 0.10, fraction)
                assert bid < ask, f"bid >= ask at fv={fv}, fraction={fraction}"

    def test_wider_fraction_gives_wider_spread(self):
        _, ask_tight = optimal_quote_prices(0.50, 0.10, 0.1)
        _, ask_wide = optimal_quote_prices(0.50, 0.10, 0.5)
        assert ask_wide > ask_tight


class TestExpectedRewardFraction:
    def test_sole_market_maker_gets_full_pool(self):
        fraction = expected_reward_fraction(
            q_min=100.0,
            total_q_min_in_market=100.0,
            reward_pool_usdc=1000.0,
        )
        assert fraction == pytest.approx(1000.0)

    def test_equal_competition_halves_pool(self):
        fraction = expected_reward_fraction(
            q_min=50.0,
            total_q_min_in_market=100.0,
            reward_pool_usdc=1000.0,
        )
        assert fraction == pytest.approx(500.0)

    def test_zero_total_returns_zero(self):
        fraction = expected_reward_fraction(0.0, 0.0, 1000.0)
        assert fraction == 0.0
