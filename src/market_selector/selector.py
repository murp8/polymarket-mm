"""
Market selector: picks the best incentivized markets to quote.

Primary data source: CLOB API ``get_sampling_markets`` — this is the
authoritative list of markets with active liquidity reward programmes.

Secondary / enrichment: Gamma API — adds volume_24h and tags for better
scoring and sport matching when the odds pricer is enabled.

Scoring heuristic:
  1. Reward rate (higher daily USDC reward = more opportunity)
  2. Volume / liquidity (active markets → more fills → faster position turnover)
  3. Max incentive spread (wider = easier to qualify)
  4. Midpoint centrality (markets near 0.5 = less directional risk)
  5. Exclude near-resolved markets (outside [min_mid, max_mid])
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from src.client.gamma import GammaClient
from src.config import MarketSelectionConfig
from src.models import IncentiveParams, Market, MarketSide, TokenInfo
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.client.polymarket import PolymarketClient

log = get_logger(__name__)


def _score_market(market: Market) -> float:
    """
    Compute a scalar priority score for a market. Higher = prefer to quote.
    """
    incentive = market.incentive

    # Reward rate: normalise daily reward rate to [0, 10]
    # rewards_daily_rate from CLOB is in USDC/day per order (typically 0.001–0.01)
    # We scale by 0.001 so a rate of 0.01 → score 10; legacy large values are capped.
    reward_score = min(incentive.reward_epoch_amount / 0.001, 10.0)

    # Volume component (log-scale, capped)
    vol_score = min(max(1.0, market.volume_24h) ** 0.25, 5.0)

    # Spread generosity: wider max_spread = easier to get full score
    spread_score = min(incentive.max_incentive_spread / 0.05, 3.0)

    # Midpoint centrality: markets near 0.5 are safer (less directional risk)
    mid = market.yes_token.price
    centrality = 1.0 - abs(mid - 0.5) * 2  # 1.0 at 0.50, 0.0 at 0.10/0.90

    return reward_score * 2 + vol_score + spread_score + centrality


def _parse_clob_market(raw: dict) -> Optional[Market]:
    """
    Convert a CLOB API sampling-market dict into a typed Market object.

    CLOB market structure::

        {
          "condition_id": "0x...",
          "question_id": "0x...",
          "question": "...",
          "end_date_iso": "...",
          "tokens": [
            {"token_id": "...", "outcome": "Yes", "price": 0.65},
            {"token_id": "...", "outcome": "No",  "price": 0.35}
          ],
          "rewards": {
            "min_size": 20,
            "max_spread": 3.5,
            "rates": [{"asset_address": "0x...", "rewards_daily_rate": 0.001}]
          },
          "tags": ["Sports", "NFL"]
        }
    """
    try:
        condition_id = raw.get("condition_id", "")
        if not condition_id:
            return None

        tokens = raw.get("tokens") or []
        if len(tokens) < 2:
            return None

        # Normalise to YES first, NO second (outcome field should be "Yes"/"No")
        tok0, tok1 = tokens[0], tokens[1]
        if tok1.get("outcome", "").lower() == "yes":
            tok0, tok1 = tok1, tok0

        yes_price = float(tok0.get("price", 0.5))
        no_price = float(tok1.get("price", 1.0 - yes_price))

        rewards_raw = raw.get("rewards") or {}
        min_size = float(rewards_raw.get("min_size", 0))
        max_spread_cents = float(rewards_raw.get("max_spread", 0))
        # max_spread in CLOB is expressed in *cents* (e.g. 3.5 = 0.035 probability)
        max_spread = max_spread_cents / 100.0

        # Sum daily rates across all reward assets
        daily_rate = sum(
            float(r.get("rewards_daily_rate", 0))
            for r in (rewards_raw.get("rates") or [])
        )

        incentive = IncentiveParams(
            min_incentive_size=min_size,
            max_incentive_spread=max_spread,
            reward_epoch_amount=daily_rate,  # USDC/day; used for scoring
            in_game_multiplier=1.0,
        )

        tags_raw = raw.get("tags") or []
        tags = [str(t).lower() for t in tags_raw]

        return Market(
            condition_id=condition_id,
            question_id=raw.get("question_id", ""),
            question=raw.get("question", ""),
            yes_token=TokenInfo(
                token_id=tok0.get("token_id", ""),
                outcome=MarketSide.YES,
                price=yes_price,
            ),
            no_token=TokenInfo(
                token_id=tok1.get("token_id", ""),
                outcome=MarketSide.NO,
                price=no_price,
            ),
            incentive=incentive,
            volume_24h=0.0,       # enriched from Gamma below
            volume_total=0.0,
            liquidity=0.0,
            end_date_iso=str(raw.get("end_date_iso", "") or ""),
            sport=str(raw.get("sport", "") or ""),
            tags=tags,
        )
    except Exception as e:
        log.warning(
            "clob_market_parse_failed",
            condition_id=raw.get("condition_id", "unknown"),
            error=str(e),
        )
        return None


class MarketSelector:
    """
    Maintains a ranked list of markets to quote and handles periodic refresh.
    """

    def __init__(
        self,
        gamma_client: GammaClient,
        cfg: MarketSelectionConfig,
        clob_client: Optional["PolymarketClient"] = None,
    ) -> None:
        self._gamma = gamma_client
        self._cfg = cfg
        self._clob = clob_client
        self._markets: dict[str, Market] = {}  # condition_id → Market
        self._last_refresh: float = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        """
        Fetch and re-rank incentivized markets.

        Strategy:
          1. Try CLOB ``get_sampling_markets`` (authoritative, fast)
          2. If CLOB unavailable, fall back to Gamma scan
          3. Enrich CLOB markets with Gamma volume data (best-effort)
        """
        async with self._lock:
            log.debug("market_selector_refreshing")

            markets = await self._fetch_from_clob()
            if not markets:
                log.warning(
                    "clob_market_fetch_empty",
                    msg="Falling back to Gamma incentivized scan",
                )
                markets = await self._fetch_from_gamma()

            new_markets: dict[str, Market] = {}
            for m in markets:
                if self._is_eligible(m):
                    new_markets[m.condition_id] = m

            # Best-effort Gamma volume enrichment (don't fail on error)
            if new_markets:
                await self._enrich_with_gamma_volume(new_markets)

            added = set(new_markets) - set(self._markets)
            removed = set(self._markets) - set(new_markets)
            self._markets = new_markets
            self._last_refresh = time.monotonic()

            log.debug(
                "market_selector_refreshed",
                total=len(new_markets),
                added=len(added),
                removed=len(removed),
            )

    async def _fetch_from_clob(self) -> list[Market]:
        """Fetch all sampling markets from CLOB API and parse into Market objects."""
        if self._clob is None:
            return []
        try:
            raw_list = await self._clob.get_sampling_markets_all()
        except Exception as e:
            log.error("clob_sampling_markets_failed", error=str(e))
            return []

        markets: list[Market] = []
        for raw in raw_list:
            m = _parse_clob_market(raw)
            if m is not None:
                markets.append(m)

        log.debug("clob_markets_parsed", raw=len(raw_list), parsed=len(markets))
        return markets

    async def _fetch_from_gamma(self) -> list[Market]:
        """Fallback: fetch incentivized markets via Gamma API scan."""
        try:
            return await self._gamma.get_incentivized_markets()
        except Exception as e:
            log.error("gamma_market_fetch_failed", error=str(e))
            return []

    async def _enrich_with_gamma_volume(
        self, markets: dict[str, Market]
    ) -> None:
        """
        Pull volume_24h from Gamma for the top markets (for better scoring).
        Only enriches up to 200 markets to keep startup fast.
        """
        condition_ids = list(markets.keys())[:200]
        try:
            raw_list = await self._gamma.get_markets_raw(
                active=True,
                clob_enabled=True,
                limit=len(condition_ids),
            )
            gamma_vol: dict[str, float] = {}
            for raw in raw_list:
                cid = raw.get("conditionId") or raw.get("condition_id", "")
                vol = float(raw.get("volume24hr", raw.get("volume_24hr", 0)) or 0)
                if cid:
                    gamma_vol[cid] = vol

            enriched = 0
            for cid, m in markets.items():
                if cid in gamma_vol:
                    m.volume_24h = gamma_vol[cid]
                    enriched += 1

            log.debug("gamma_volume_enriched", enriched=enriched, total=len(markets))
        except Exception as e:
            log.debug("gamma_volume_enrichment_skipped", error=str(e))

    def _is_eligible(self, market: Market) -> bool:
        """Filter markets we should quote."""
        incentive = market.incentive

        # Must have reward programme with both parameters set
        if incentive.min_incentive_size <= 0 or incentive.max_incentive_spread <= 0:
            return False

        # Midpoint must be in quotable range
        mid = market.yes_token.price
        if not (self._cfg.min_midpoint <= mid <= self._cfg.max_midpoint):
            return False

        return True

    def top_markets(self, n: Optional[int] = None) -> list[Market]:
        """
        Return the top N markets ranked by priority score.
        n defaults to max_markets from config.
        """
        limit = n or self._cfg.max_markets
        ranked = sorted(self._markets.values(), key=_score_market, reverse=True)
        return ranked[:limit]

    def get_market(self, condition_id: str) -> Optional[Market]:
        return self._markets.get(condition_id)

    def update_market_price(self, condition_id: str, yes_mid: float) -> None:
        """Update the cached midpoint for a market (from WebSocket)."""
        m = self._markets.get(condition_id)
        if m:
            m.yes_token.price = yes_mid
            m.no_token.price = 1.0 - yes_mid

    def remove_market(self, condition_id: str) -> None:
        """Remove a market (e.g., when it resolves)."""
        self._markets.pop(condition_id, None)

    def all_token_ids(self) -> list[str]:
        """Return all YES and NO token IDs for WebSocket subscription."""
        ids = []
        for m in self._markets.values():
            ids.append(m.yes_token.token_id)
            ids.append(m.no_token.token_id)
        return ids

    def yes_token_to_market(self) -> dict[str, Market]:
        """Map YES token_id → Market for fast lookup on orderbook updates."""
        return {m.yes_token.token_id: m for m in self._markets.values()}

    def no_token_to_market(self) -> dict[str, Market]:
        """Map NO token_id → Market for fast lookup on orderbook updates."""
        return {m.no_token.token_id: m for m in self._markets.values() if m.no_token.token_id}

    def needs_refresh(self) -> bool:
        elapsed = time.monotonic() - self._last_refresh
        return elapsed >= self._cfg.refresh_interval_seconds

    @property
    def market_count(self) -> int:
        return len(self._markets)
