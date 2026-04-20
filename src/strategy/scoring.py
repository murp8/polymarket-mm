"""
Polymarket liquidity reward scoring calculator.

Formula (from docs.polymarket.com/market-makers/liquidity-rewards):

  S(v, s) = ((v - s) / v)² × b

where:
  v = max_incentive_spread  (maximum distance from midpoint eligible for rewards)
  s = actual spread         (distance from midpoint to our quote price)
  b = order size            (in shares, must be ≥ min_incentive_size)

Per-side scores are summed across all our orders on that side, then the
two-sided minimum is taken (with adjustments for near-boundary markets):

  Q_min =
    if 0.10 ≤ mid ≤ 0.90:
      max(min(Q_one, Q_two), max(Q_one, Q_two) / c)   [c = 3.0]
    else:
      min(Q_one, Q_two)

This module is pure computation — no I/O, no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Single-sided penalty denominator (from Polymarket docs)
_ONE_SIDED_PENALTY = 3.0
_MID_BOUNDARY_LO = 0.10
_MID_BOUNDARY_HI = 0.90


@dataclass
class ScoreResult:
    """Result of scoring a two-sided quote."""
    q_one: float       # Sum of scored orders on the YES-buy / NO-sell side
    q_two: float       # Sum of scored orders on the YES-sell / NO-buy side
    q_min: float       # The effective score used for reward normalisation
    within_spread: bool  # Whether any orders qualify at all


def order_score(
    max_spread: float,
    actual_spread: float,
    size: float,
    in_game_multiplier: float = 1.0,
) -> float:
    """
    Score a single resting order.

    Args:
        max_spread:         max_incentive_spread from market params (v)
        actual_spread:      |quote_price - midpoint|               (s)
        size:               order size in shares                   (b)
        in_game_multiplier: live match multiplier                  (b modifier)

    Returns 0 if the order is outside the eligible spread.
    """
    if max_spread <= 0 or actual_spread >= max_spread:
        return 0.0
    if size <= 0:
        return 0.0
    ratio = (max_spread - actual_spread) / max_spread
    return ratio * ratio * size * in_game_multiplier


def score_two_sided_quote(
    midpoint: float,
    yes_bid_price: float,
    yes_ask_price: float,
    yes_bid_size: float,
    yes_ask_size: float,
    max_spread: float,
    min_size: float,
    in_game_multiplier: float = 1.0,
) -> ScoreResult:
    """
    Score a two-sided market-making quote on the YES token.

    The Polymarket scoring treats:
      Q_one = scored YES bids  (offers to buy YES = offers to sell the market going against)
      Q_two = scored YES asks  (offers to sell YES)

    Args:
        midpoint:       YES token midpoint price (0–1)
        yes_bid_price:  our bid price for YES
        yes_ask_price:  our ask price for YES
        yes_bid_size:   our bid size (shares)
        yes_ask_size:   our ask size (shares)
        max_spread:     max_incentive_spread from market params
        min_size:       min_incentive_size threshold (orders below this score 0)
        in_game_multiplier: live match multiplier

    Returns:
        ScoreResult with per-side and combined scores.
    """
    # Distance from midpoint (always positive)
    bid_spread = abs(midpoint - yes_bid_price)
    ask_spread = abs(yes_ask_price - midpoint)

    # Enforce minimum size
    bid_size = yes_bid_size if yes_bid_size >= min_size else 0.0
    ask_size = yes_ask_size if yes_ask_size >= min_size else 0.0

    q_one = order_score(max_spread, bid_spread, bid_size, in_game_multiplier)
    q_two = order_score(max_spread, ask_spread, ask_size, in_game_multiplier)

    # Compute Q_min with two-sided penalty
    if _MID_BOUNDARY_LO <= midpoint <= _MID_BOUNDARY_HI:
        # Normal regime: single-sided orders score at reduced rate
        paired = min(q_one, q_two)
        single = max(q_one, q_two) / _ONE_SIDED_PENALTY
        q_min = max(paired, single)
    else:
        # Near-boundary: strict two-sided requirement
        q_min = min(q_one, q_two)

    return ScoreResult(
        q_one=q_one,
        q_two=q_two,
        q_min=q_min,
        within_spread=(q_one > 0 or q_two > 0),
    )


def optimal_quote_prices(
    fair_value: float,
    max_spread: float,
    target_spread_fraction: float,
    inventory_skew: float = 0.0,
) -> tuple[float, float]:
    """
    Compute optimal bid/ask prices that maximise reward score subject to
    inventory and risk constraints.

    Args:
        fair_value:            probability estimate for YES (0–1)
        max_spread:            max_incentive_spread from market params
        target_spread_fraction: how far inside max_spread to quote (0.0 = at mid,
                               1.0 = at the edge of eligible spread)
        inventory_skew:        positive → we are long YES, widen ask side
                               negative → we are short YES, widen bid side
                               range: [-1, 1] as fraction of max_spread to add

    Returns:
        (bid_price, ask_price) for YES token, snapped to 4 decimal places.
    """
    # Half-spread: how far we quote from fair value
    half_spread = max_spread * target_spread_fraction

    # Apply inventory skew: if long YES, push ask further out (less eager to sell more)
    skew_offset = max_spread * inventory_skew * 0.3

    bid_price = fair_value - half_spread - max(0.0, skew_offset)
    ask_price = fair_value + half_spread + max(0.0, -skew_offset)

    # Clamp to valid range
    bid_price = max(0.01, min(0.99, bid_price))
    ask_price = max(0.01, min(0.99, ask_price))

    # Ensure bid < ask
    if bid_price >= ask_price:
        mid = (bid_price + ask_price) / 2
        bid_price = mid - 0.005
        ask_price = mid + 0.005

    return round(bid_price, 4), round(ask_price, 4)


def expected_reward_fraction(
    q_min: float,
    total_q_min_in_market: float,
    reward_pool_usdc: float,
) -> float:
    """
    Estimate our expected reward for one sampling period.

    Args:
        q_min:                  our q_min score
        total_q_min_in_market:  sum of all market makers' q_min (estimate)
        reward_pool_usdc:       total reward pool for this market

    Returns fraction of pool we would earn (0–1).
    """
    if total_q_min_in_market <= 0:
        return 0.0
    fraction = q_min / total_q_min_in_market
    return fraction * reward_pool_usdc
