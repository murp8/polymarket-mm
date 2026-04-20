"""
Abstract base class for all price sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.models import Market, PriceEstimate


class BasePricer(ABC):
    """
    A price source returns a probability estimate for the YES outcome
    of a given Polymarket market.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def get_price(self, market: Market) -> Optional[PriceEstimate]:
        """
        Return a PriceEstimate for the YES outcome, or None if unavailable.
        Implementations must never raise; return None on any failure.
        """
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Return True if this source is currently reachable and returning data."""
        ...
