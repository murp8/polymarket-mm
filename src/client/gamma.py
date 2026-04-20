"""
Gamma API client — market discovery, metadata, and incentivized market discovery.

The Gamma API (https://gamma-api.polymarket.com) is Polymarket's read-only
market catalogue. No authentication required. We use it to:
  - Discover markets that offer liquidity rewards
  - Resolve min_incentive_size, max_incentive_spread, reward pool per market
  - Map Polymarket markets to sport/team names for Pinnacle odds matching
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from urllib.parse import urljoin

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models import IncentiveParams, Market, MarketSide, TokenInfo
from src.utils.logging import get_logger

log = get_logger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"


class GammaClient:
    """
    Async HTTP client for the Polymarket Gamma API.

    Usage:
        async with GammaClient() as client:
            markets = await client.get_incentivized_markets()
    """

    def __init__(
        self,
        base_url: str = _GAMMA_BASE,
        timeout: float = 20.0,
        max_page_size: int = 100,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_page_size = max_page_size
        self._http: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GammaClient":
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("GammaClient must be used as an async context manager")
        return self._http

    # ── Internal helpers ─────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, path: str, params: dict | None = None) -> Any:
        response = await self.http.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def _paginate(self, path: str, params: dict | None = None) -> list[dict]:
        """Fetch all pages of a paginated endpoint."""
        params = params or {}
        params["limit"] = self._max_page_size
        results: list[dict] = []
        offset = 0
        while True:
            params["offset"] = offset
            page: list[dict] = await self._get(path, params)
            if not page:
                break
            results.extend(page)
            if len(page) < self._max_page_size:
                break
            offset += len(page)
            await asyncio.sleep(0.1)  # gentle rate limiting
        return results

    # ── Public API ───────────────────────────────────────────────────────────

    async def get_markets_raw(
        self,
        active: bool = True,
        clob_enabled: bool = True,
        tag_id: Optional[str] = None,
        order_by: str = "volume_24hr",
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Return raw market dicts from the Gamma API."""
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "enableOrderBook": str(clob_enabled).lower(),
            "order": order_by,
        }
        if tag_id:
            params["tag_id"] = tag_id
        if limit:
            params["limit"] = limit
            return await self._get("/markets", params)
        return await self._paginate("/markets", params)

    async def get_incentivized_markets_raw(self, max_empty_pages: int = 3) -> list[dict]:
        """
        Return all markets that have active reward programmes.
        These have non-null rewardsMinSize and rewardsMaxSpread fields.

        Markets are fetched in descending volume order. Incentivized markets
        are all high-volume, so we stop early once we see `max_empty_pages`
        consecutive pages with no incentivized markets.
        """
        params: dict[str, Any] = {
            "active": "true",
            "enableOrderBook": "true",
            "order": "volume_24hr",
            "limit": self._max_page_size,
            "offset": 0,
        }

        incentivized: list[dict] = []
        total_fetched = 0
        consecutive_empty = 0

        while True:
            page: list[dict] = await self._get("/markets", params)
            if not page:
                break

            total_fetched += len(page)
            page_incentivized = [
                m for m in page
                if m.get("rewardsMinSize") is not None
                and m.get("rewardsMaxSpread") is not None
                and float(m.get("rewardsMinSize", 0)) > 0
            ]
            incentivized.extend(page_incentivized)

            if page_incentivized:
                consecutive_empty = 0
            else:
                consecutive_empty += 1

            if consecutive_empty >= max_empty_pages:
                log.info(
                    "incentivized_search_early_stop",
                    total_fetched=total_fetched,
                    consecutive_empty_pages=consecutive_empty,
                )
                break

            if len(page) < self._max_page_size:
                break

            params["offset"] += len(page)
            await asyncio.sleep(0.05)

        log.info(
            "incentivized_markets_found",
            total_fetched=total_fetched,
            incentivized=len(incentivized),
        )
        return incentivized

    async def get_market_by_condition_id(self, condition_id: str) -> Optional[dict]:
        """Fetch a single market by its condition ID."""
        try:
            return await self._get(f"/markets/{condition_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    # ── Parsing ──────────────────────────────────────────────────────────────

    @staticmethod
    def parse_market(raw: dict) -> Optional[Market]:
        """
        Convert a raw Gamma API market dict into a typed Market object.
        Returns None if the market is missing required fields.
        """
        try:
            condition_id = raw.get("conditionId") or raw.get("condition_id", "")
            question_id = raw.get("questionID") or raw.get("question_id", "")
            question = raw.get("question", "")

            # Token IDs come as a JSON string or list
            clob_ids = raw.get("clobTokenIds")
            if isinstance(clob_ids, str):
                import json
                clob_ids = json.loads(clob_ids)
            if not clob_ids or len(clob_ids) < 2:
                return None

            yes_token_id, no_token_id = clob_ids[0], clob_ids[1]

            # Outcome prices
            outcome_prices = raw.get("outcomePrices")
            if isinstance(outcome_prices, str):
                import json
                outcome_prices = json.loads(outcome_prices)
            yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
            no_price = float(outcome_prices[1]) if outcome_prices and len(outcome_prices) > 1 else 1.0 - yes_price

            # Incentive parameters
            rewards_min_size = raw.get("rewardsMinSize") or raw.get("min_incentive_size") or 0
            # rewardsMaxSpread from Gamma is in cents (e.g. 3.5 = 0.035 probability),
            # matching the CLOB API's rewards.max_spread field — divide by 100.
            rewards_max_spread_raw = float(raw.get("rewardsMaxSpread") or raw.get("max_incentive_spread") or 0)
            rewards_max_spread = rewards_max_spread_raw / 100.0
            reward_epoch = raw.get("rewardEpochAmount") or raw.get("reward_epoch_amount") or 0

            incentive = IncentiveParams(
                min_incentive_size=float(rewards_min_size),
                max_incentive_spread=rewards_max_spread,
                reward_epoch_amount=float(reward_epoch),
                in_game_multiplier=float(raw.get("inGameMultiplier", 1.0)),
            )

            # Tags
            tags_raw = raw.get("tags") or []
            tags = [
                t.get("slug", t.get("label", "")).lower()
                for t in tags_raw
                if isinstance(t, dict)
            ]
            if not tags and isinstance(tags_raw, list):
                tags = [str(t).lower() for t in tags_raw]

            return Market(
                condition_id=condition_id,
                question_id=question_id,
                question=question,
                yes_token=TokenInfo(
                    token_id=yes_token_id,
                    outcome=MarketSide.YES,
                    price=yes_price,
                ),
                no_token=TokenInfo(
                    token_id=no_token_id,
                    outcome=MarketSide.NO,
                    price=no_price,
                ),
                incentive=incentive,
                volume_24h=float(raw.get("volume24hr", raw.get("volume_24hr", 0)) or 0),
                volume_total=float(raw.get("volume", 0) or 0),
                liquidity=float(raw.get("liquidity", 0) or 0),
                end_date_iso=str(raw.get("endDate", raw.get("end_date_iso", "")) or ""),
                sport=str(raw.get("sport", "") or ""),
                tags=tags,
            )
        except Exception as e:
            log.warning(
                "market_parse_failed",
                condition_id=raw.get("conditionId", "unknown"),
                error=str(e),
            )
            return None

    async def get_incentivized_markets(self) -> list[Market]:
        """
        High-level method: return parsed Market objects for all incentivized markets.
        """
        raw_markets = await self.get_incentivized_markets_raw()
        markets: list[Market] = []
        for raw in raw_markets:
            m = self.parse_market(raw)
            if m is not None:
                markets.append(m)
        log.info("incentivized_markets_parsed", count=len(markets))
        return markets
