"""
Quote engine: produces two-sided quotes for a market.

Strategy (based on poly-maker):
  1. Join at best bid/ask — place orders one tick better than the current
     best price that has sufficient size. This maximises reward score and
     fill rate simultaneously.
  2. Quote both YES and NO tokens — doubles the reward-earning surface on
     every market without requiring additional capital.
  3. Never sell below avg cost — hard floor prevents locking in losses.
  4. Take-profit floor — ask minimum = avg_cost × (1 + take_profit_pct).
  5. Liquidity ratio guard — skip buys if bid depth < ratio × ask depth,
     which signals the market is selling off.
  6. Stop-loss + risk-off cooldown — if unrealised PnL% < threshold, sell
     at market price and pause all buying on that market for risk_off_hours.
"""

from __future__ import annotations

import math
from typing import Optional

from src.config import QuotingConfig
from src.models import Market, MarketQuote, MarketSide, Orderbook, PriceLevel, QuoteTarget, Side
from src.pricing.composite import CompositePricer
from src.pricing.orderbook import OrderbookPricer
from src.risk.risk_manager import RiskManager
from src.strategy.inventory import InventoryManager
from src.utils.logging import get_logger

log = get_logger(__name__)

TICK = 0.01   # Polymarket minimum price increment
_MIN_PRICE = 0.01
_MAX_PRICE = 0.99


def _snap(price: float) -> float:
    """Snap a price to the nearest Polymarket tick (0.01)."""
    return round(round(price / TICK) * TICK, 4)


def _find_best_with_size(
    levels: list[PriceLevel],
    min_size: float,
    descending: bool,
) -> tuple[Optional[float], float]:
    """
    Find the best price level that has at least min_size available.
    Returns (price, size) or (None, 0) if no qualifying level exists.
    descending=True for bids, False for asks.
    """
    lst = levels if not descending else list(reversed(levels)) if not levels else levels
    # bids are already sorted descending, asks ascending
    for lvl in levels:
        if lvl.size >= min_size:
            return lvl.price, lvl.size
    # Fall back to best price regardless of size
    return (levels[0].price, levels[0].size) if levels else (None, 0.0)


def _liquidity_within_pct(levels: list[PriceLevel], mid: float, pct: float) -> float:
    """Sum size of all levels within `pct` fraction of mid."""
    lo, hi = mid * (1 - pct), mid * (1 + pct)
    return sum(lvl.size for lvl in levels if lo <= lvl.price <= hi)


class QuoteEngine:
    """Computes optimal two-sided quotes for a given market."""

    def __init__(
        self,
        pricer: CompositePricer,
        inventory: InventoryManager,
        cfg: QuotingConfig,
        ob_pricer: Optional[OrderbookPricer] = None,
        risk: Optional[RiskManager] = None,
    ) -> None:
        self._pricer = pricer
        self._inventory = inventory
        self._cfg = cfg
        self._ob_pricer = ob_pricer
        self._risk = risk

    async def compute_quote(self, market: Market) -> Optional[MarketQuote]:
        """
        Compute the desired two-sided quote for a market.
        Returns None if we should not quote this market right now.
        """
        cid = market.condition_id
        min_size = max(market.incentive.min_incentive_size, 1.0)

        # ── 1. Get orderbook depth ───────────────────────────────────────────
        orderbook: Optional[Orderbook] = None
        if self._ob_pricer:
            orderbook = self._ob_pricer.get_orderbook(cid)

        # ── 2. Get fair value (reference; also used as fallback mid) ─────────
        estimate = await self._pricer.get_price(market)

        if orderbook is None and estimate is None:
            log.debug("no_price_or_book", market=cid[:12])
            return None

        # Determine mid price
        if orderbook and orderbook.midpoint is not None:
            mid = orderbook.midpoint
        elif estimate:
            mid = estimate.probability
        else:
            return None

        if not (_MIN_PRICE < mid < _MAX_PRICE):
            return None

        # ── 3. Derive join-at-best prices ────────────────────────────────────
        if orderbook and orderbook.bids and orderbook.asks:
            min_liq = self._cfg.min_liquidity_size

            best_bid_p, best_bid_sz = _find_best_with_size(orderbook.bids, min_liq, descending=True)
            best_ask_p, best_ask_sz = _find_best_with_size(orderbook.asks, min_liq, descending=False)
            top_bid = orderbook.bids[0].price   # very top regardless of size
            top_ask = orderbook.asks[0].price

            if best_bid_p is None or best_ask_p is None:
                return None

            # Place one tick better if the level has enough size
            bid_price = _snap(best_bid_p + TICK) if best_bid_sz >= min_size * 1.5 else best_bid_p
            ask_price = _snap(best_ask_p - TICK) if best_ask_sz >= min_size * 1.5 else best_ask_p

            # Don't cross the spread
            if bid_price >= best_ask_p:
                bid_price = top_bid
            if ask_price <= best_bid_p:
                ask_price = top_ask
            if bid_price >= ask_price:
                bid_price, ask_price = top_bid, top_ask

            # Liquidity ratio: don't buy into a selling market
            bid_liq = _liquidity_within_pct(orderbook.bids, mid, 0.10)
            ask_liq = _liquidity_within_pct(orderbook.asks, mid, 0.10)
            liq_ratio = bid_liq / ask_liq if ask_liq > 0 else 0.0
        else:
            # No live orderbook — fall back to spread-around-mid
            if estimate is None:
                return None
            max_spread = market.incentive.max_incentive_spread
            if max_spread <= 0:
                return None
            half = max_spread * self._cfg.target_spread_fraction
            bid_price = _snap(mid - half)
            ask_price = _snap(mid + half)
            top_bid, top_ask = bid_price, ask_price
            liq_ratio = 1.0  # unknown — assume OK

        # ── 4. Price bounds ──────────────────────────────────────────────────
        bid_price = max(_MIN_PRICE, min(bid_price, _MAX_PRICE - TICK))
        ask_price = max(_MIN_PRICE + TICK, min(ask_price, _MAX_PRICE))

        # ── 5. Avg cost protection + take-profit floor ───────────────────────
        pos = self._inventory.get_position(cid)
        avg_yes_cost = pos.avg_yes_cost if pos and pos.yes_shares > 0 else 0.0
        avg_no_cost = pos.avg_no_cost if pos and pos.no_shares > 0 else 0.0
        yes_shares = pos.yes_shares if pos else 0.0
        no_shares = pos.no_shares if pos else 0.0

        stop_loss_triggered = False

        # Compute current spread once — used by both YES and NO stop-loss checks.
        # Must be defined before either block to avoid NameError when we hold
        # NO shares but have no YES position.
        spread_now = (
            (orderbook.best_ask - orderbook.best_bid)
            if orderbook and orderbook.best_ask and orderbook.best_bid else 1.0
        )

        if avg_yes_cost > 0 and yes_shares > 0:
            pnl_pct = (mid - avg_yes_cost) / avg_yes_cost * 100

            # Stop-loss: exit at market, freeze buys for risk_off_hours
            if pnl_pct < self._cfg.stop_loss_pct and spread_now <= 0.05:
                log.warning(
                    "stop_loss_triggered",
                    market=cid[:12],
                    pnl_pct=round(pnl_pct, 2),
                    avg_cost=round(avg_yes_cost, 4),
                    mid=round(mid, 4),
                )
                if self._risk:
                    self._risk.set_risk_off_buy(cid, self._cfg.risk_off_hours)
                # Exit at best bid (aggressive market sell)
                ask_price = _snap(top_bid)
                stop_loss_triggered = True

            if not stop_loss_triggered:
                # Take-profit floor: never ask below avg_cost × (1 + take_profit_pct%)
                tp_price = _snap(avg_yes_cost * (1 + self._cfg.take_profit_pct / 100))
                ask_price = max(ask_price, tp_price)
                # Hard floor: never sell below avg cost
                ask_price = max(ask_price, _snap(avg_yes_cost))

        # ── 6. Buy gating ────────────────────────────────────────────────────
        buy_allowed = not stop_loss_triggered
        if buy_allowed and liq_ratio < self._cfg.liquidity_ratio_min and liq_ratio > 0:
            log.debug("liquidity_ratio_too_low", market=cid[:12], ratio=round(liq_ratio, 3))
            buy_allowed = False
        if buy_allowed and self._risk and not self._risk.is_buy_allowed(cid):
            buy_allowed = False

        # ── 7. Sizing ────────────────────────────────────────────────────────
        base_size = max(self._cfg.base_order_size, min_size)
        order_size = min(base_size, self._cfg.max_order_size)

        # Progressive: build to max_inventory in order_size steps
        max_inv = self._cfg.max_inventory_per_market
        remaining_yes = max(0.0, max_inv - yes_shares)
        remaining_no = max(0.0, max_inv - no_shares)

        bid_size = min(order_size, remaining_yes) if buy_allowed else 0.0
        ask_size = min(order_size, yes_shares)

        # Inventory skew: taper bid size proportionally once net position
        # exceeds inventory_skew_threshold to reduce one-sided accumulation.
        # skew > 0 = long YES, skew < 0 = long NO (short YES).
        skew = self._inventory.inventory_skew(cid)
        if skew > 0 and bid_size > 0:
            bid_size *= max(0.0, 1.0 - skew)

        # Enforce min_size (round up to min if close, else zero)
        if 0 < bid_size < min_size:
            bid_size = min_size if bid_size >= min_size * 0.7 else 0.0
        if 0 < ask_size < min_size:
            ask_size = 0.0  # not enough to fill minimum

        # ── 8. Build YES QuoteTargets ─────────────────────────────────────────
        yes_bid = QuoteTarget(
            market_condition_id=cid,
            token_id=market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=bid_price,
            size=bid_size,
        ) if bid_size >= min_size else None

        yes_ask = QuoteTarget(
            market_condition_id=cid,
            token_id=market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.SELL,
            price=ask_price,
            size=ask_size,
        ) if ask_size >= min_size else None

        # Need at least one side to produce a quote
        if yes_bid is None and yes_ask is None:
            return None

        # Substitute a zero-size placeholder for the missing side so
        # OrderManager can cancel any stale order on that side
        if yes_bid is None:
            yes_bid = QuoteTarget(
                market_condition_id=cid, token_id=market.yes_token.token_id,
                outcome=MarketSide.YES, side=Side.BUY, price=bid_price, size=0.0,
            )
        if yes_ask is None:
            yes_ask = QuoteTarget(
                market_condition_id=cid, token_id=market.yes_token.token_id,
                outcome=MarketSide.YES, side=Side.SELL, price=ask_price, size=0.0,
            )

        # ── 9. NO token quotes (mirror of YES orderbook) ──────────────────────
        no_bid: Optional[QuoteTarget] = None
        no_ask: Optional[QuoteTarget] = None

        if self._cfg.quote_no_token and market.no_token.token_id:
            # NO prices are the complement of YES prices
            # NO best_bid = 1 - YES best_ask  (buying NO = someone selling YES)
            # NO best_ask = 1 - YES best_bid
            no_bid_price = _snap(1.0 - ask_price)
            no_ask_price = _snap(1.0 - bid_price)

            # Avg cost protection for NO position
            if avg_no_cost > 0 and no_shares > 0:
                no_pnl_pct = ((1.0 - mid) - avg_no_cost) / avg_no_cost * 100
                if no_pnl_pct < self._cfg.stop_loss_pct and spread_now <= 0.05:
                    if self._risk:
                        self._risk.set_risk_off_buy(cid + ":no", self._cfg.risk_off_hours)
                    no_ask_price = _snap(1.0 - top_ask)  # aggressive exit
                else:
                    tp_no = _snap(avg_no_cost * (1 + self._cfg.take_profit_pct / 100))
                    no_ask_price = max(no_ask_price, tp_no)
                    no_ask_price = max(no_ask_price, _snap(avg_no_cost))

            no_buy_allowed = buy_allowed and (
                not self._risk or self._risk.is_buy_allowed(cid + ":no")
            )

            no_bid_sz = min(order_size, remaining_no) if no_buy_allowed else 0.0
            no_ask_sz = min(order_size, no_shares)

            # Long NO (short YES, skew < 0): taper NO bids to slow accumulation
            if skew < 0 and no_bid_sz > 0:
                no_bid_sz *= max(0.0, 1.0 + skew)  # skew is negative, so this reduces

            if 0 < no_bid_sz < min_size:
                no_bid_sz = min_size if no_bid_sz >= min_size * 0.7 else 0.0
            if 0 < no_ask_sz < min_size:
                no_ask_sz = 0.0

            if no_bid_sz >= min_size and _MIN_PRICE < no_bid_price < _MAX_PRICE:
                no_bid = QuoteTarget(
                    market_condition_id=cid,
                    token_id=market.no_token.token_id,
                    outcome=MarketSide.NO,
                    side=Side.BUY,
                    price=no_bid_price,
                    size=no_bid_sz,
                )
            if no_ask_sz >= min_size and _MIN_PRICE < no_ask_price < _MAX_PRICE:
                no_ask = QuoteTarget(
                    market_condition_id=cid,
                    token_id=market.no_token.token_id,
                    outcome=MarketSide.NO,
                    side=Side.SELL,
                    price=no_ask_price,
                    size=no_ask_sz,
                )

        log.debug(
            "quote_computed",
            market=cid[:12],
            mid=round(mid, 4),
            yes_bid=bid_price,
            yes_ask=ask_price,
            bid_sz=round(bid_size, 1),
            ask_sz=round(ask_size, 1),
            liq_ratio=round(liq_ratio, 2),
            buy_ok=buy_allowed,
            stop_loss=stop_loss_triggered,
        )

        return MarketQuote(
            market_condition_id=cid,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            fair_value=mid,
            half_spread=(_snap(ask_price) - _snap(bid_price)) / 2,
        )
