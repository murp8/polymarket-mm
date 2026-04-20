"""
Orderbook midpoint pricer.

Uses the live Polymarket orderbook (maintained by the WebSocket client)
as the fair-value reference. This is the fallback pricer when no external
odds source is available or the market is not sport-related.

The midpoint of the YES token's orderbook IS the market's consensus
probability of the YES outcome resolving.
"""

from __future__ import annotations

import time
from typing import Optional

from src.models import Market, Orderbook, PriceEstimate
from src.pricing.base import BasePricer
from src.utils.logging import get_logger

log = get_logger(__name__)


class OrderbookPricer(BasePricer):
    """
    Derives fair value from the midpoint of the YES token orderbook.

    Relies on an external component to keep the orderbook up-to-date
    (either the WebSocket client or the REST client). Call
    `update_midpoint()` whenever a new orderbook snapshot arrives.
    """

    @property
    def name(self) -> str:
        return "orderbook"

    def __init__(self, max_age_seconds: float = 120.0) -> None:
        self._max_age = max_age_seconds
        # condition_id → (midpoint, timestamp)
        self._midpoints: dict[str, tuple[float, float]] = {}
        # condition_id → Orderbook (full depth, for join-at-best pricing)
        self._orderbooks: dict[str, Orderbook] = {}

    async def is_healthy(self) -> bool:
        return True  # always available as long as we have orderbook data

    def update_midpoint(self, condition_id: str, midpoint: float) -> None:
        """Record a fresh midpoint for a market. Called by the WS handler."""
        self._midpoints[condition_id] = (midpoint, time.monotonic())

    def update_orderbook(self, condition_id: str, orderbook: Orderbook) -> None:
        """Store the full orderbook for join-at-best pricing."""
        self._orderbooks[condition_id] = orderbook

    def get_orderbook(self, condition_id: str) -> Optional[Orderbook]:
        """Return the most recent orderbook for a market, or None."""
        return self._orderbooks.get(condition_id)

    async def get_price(self, market: Market) -> Optional[PriceEstimate]:
        entry = self._midpoints.get(market.condition_id)
        if entry is None:
            # Use price from the market object itself (Gamma API gives this)
            if market.yes_token.price > 0:
                return PriceEstimate(
                    probability=market.yes_token.price,
                    source="orderbook/gamma",
                    confidence=0.6,
                )
            return None

        midpoint, ts = entry
        age = time.monotonic() - ts
        if age > self._max_age:
            log.debug("orderbook_stale", condition_id=market.condition_id[:12], age=age)
            return None

        # Confidence decays with age
        confidence = max(0.3, 1.0 - age / self._max_age)
        return PriceEstimate(
            probability=midpoint,
            source="orderbook",
            confidence=confidence,
        )
