"""
Composite pricer: tries sources in priority order, returns first valid estimate.

Priority:
  1. OddsApiPricer (Pinnacle odds) — for markets where sport is recognisable
  2. OrderbookPricer (WebSocket midpoint or Gamma price)

If the primary returns a price, the orderbook midpoint is used as a sanity
check. If the two disagree by more than `cross_check_threshold`, confidence
is downgraded. This prevents stale external odds from sending us off a cliff
when a market has already moved.
"""

from __future__ import annotations

from typing import Optional

from src.models import Market, PriceEstimate
from src.pricing.base import BasePricer
from src.pricing.odds_api import OddsApiPricer
from src.pricing.orderbook import OrderbookPricer
from src.utils.logging import get_logger

log = get_logger(__name__)


class CompositePricer(BasePricer):
    """
    Combines multiple pricers with fallback logic and cross-validation.
    """

    @property
    def name(self) -> str:
        return "composite"

    def __init__(
        self,
        odds_pricer: Optional[OddsApiPricer],
        orderbook_pricer: OrderbookPricer,
        primary_source: str = "odds_api",
        cross_check_threshold: float = 0.08,
    ) -> None:
        self._odds = odds_pricer
        self._ob = orderbook_pricer
        self._primary = primary_source
        self._xcheck = cross_check_threshold

    async def is_healthy(self) -> bool:
        if self._odds is not None and await self._odds.is_healthy():
            return True
        return await self._ob.is_healthy()

    async def get_price(self, market: Market) -> Optional[PriceEstimate]:
        primary_estimate: Optional[PriceEstimate] = None
        ob_estimate: Optional[PriceEstimate] = None

        # Always get orderbook estimate (cheap, no I/O if WS is running)
        ob_estimate = await self._ob.get_price(market)

        # Try primary source
        if self._primary == "odds_api" and self._odds is not None:
            try:
                primary_estimate = await self._odds.get_price(market)
            except Exception as e:
                log.warning("odds_pricer_error", market=market.condition_id[:12], error=str(e))

        if primary_estimate is None:
            # Fall back to orderbook
            if ob_estimate is not None:
                log.debug(
                    "pricing_fallback_to_orderbook",
                    market=market.condition_id[:12],
                )
            return ob_estimate

        # Cross-check: if external odds diverge from live market by too much,
        # blend the two estimates and reduce confidence
        if ob_estimate is not None:
            diff = abs(primary_estimate.probability - ob_estimate.probability)
            if diff > self._xcheck:
                log.warning(
                    "price_divergence",
                    market=market.condition_id[:12],
                    external=round(primary_estimate.probability, 4),
                    orderbook=round(ob_estimate.probability, 4),
                    diff=round(diff, 4),
                )
                # Blend: weight by confidence
                w_ext = primary_estimate.confidence
                w_ob = ob_estimate.confidence
                blended = (
                    primary_estimate.probability * w_ext + ob_estimate.probability * w_ob
                ) / (w_ext + w_ob + 1e-9)
                return PriceEstimate(
                    probability=blended,
                    source=f"composite_blended({primary_estimate.source}+{ob_estimate.source})",
                    confidence=min(primary_estimate.confidence, ob_estimate.confidence) * 0.7,
                )

        return primary_estimate
