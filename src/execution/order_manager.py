"""
Order manager: full lifecycle management of resting limit orders.

Responsibilities:
  - Translate QuoteTarget objects into actual CLOB orders
  - Track live orders per market (condition_id → {bid_order_id, ask_order_id})
  - Cancel and replace quotes when the quote engine signals a requote
  - Handle fills from the user WebSocket and update InventoryManager
  - Enforce per-market order deduplication (at most one bid, one ask at a time)
  - Clean up stale orders (pending >15 seconds with no confirmation, à la poly-maker)

Design decisions (mirrored from poly-maker):
  - Only cancel+replace if price changes >0.5 cents OR size changes >10%
  - Per-market asyncio.Lock prevents concurrent trading on the same market
  - Stale pending orders (>15 seconds) are forcibly cancelled
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from src.client.polymarket import GeoBlockedError, InsufficientBalanceError, PolymarketClient
from src.config import ExecutionConfig
from src.execution.rate_limiter import RateLimiter
from src.models import Fill, MarketQuote, MarketSide, Order, OrderStatus, QuoteTarget, Side
from src.strategy.inventory import InventoryManager
from src.utils.logging import get_logger
from src.utils.metrics import MetricsCollector

log = get_logger(__name__)

# Max time (seconds) to wait for order confirmation before treating as stale
_STALE_ORDER_SECONDS = 15.0


@dataclass
class MarketOrderState:
    """Tracks the live bid and ask order IDs for one market (YES and NO tokens)."""
    condition_id: str
    # YES token orders
    bid_order_id: Optional[str] = None
    ask_order_id: Optional[str] = None
    bid_quote: Optional[QuoteTarget] = None
    ask_quote: Optional[QuoteTarget] = None
    # NO token orders
    no_bid_order_id: Optional[str] = None
    no_ask_order_id: Optional[str] = None
    no_bid_quote: Optional[QuoteTarget] = None
    no_ask_quote: Optional[QuoteTarget] = None
    # Timestamp of last place/cancel action (for rate-limiting requotes)
    last_action_ts: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def pending_order_ids(self) -> list[str]:
        ids = []
        for oid in (self.bid_order_id, self.ask_order_id, self.no_bid_order_id, self.no_ask_order_id):
            if oid:
                ids.append(oid)
        return ids


class OrderManager:
    """
    Manages the full lifecycle of resting orders across all active markets.
    """

    def __init__(
        self,
        client: PolymarketClient,
        inventory: InventoryManager,
        metrics: MetricsCollector,
        cfg: ExecutionConfig,
        order_rate_limit: int = 5,
        cancel_replace_rate_limit: int = 30,
        min_requote_interval: float = 2.0,
    ) -> None:
        self._client = client
        self._inventory = inventory
        self._metrics = metrics
        self._cfg = cfg
        self._min_requote_interval = min_requote_interval
        # Per-second order placement limiter
        self._place_limiter = RateLimiter(max_calls=order_rate_limit, window_seconds=1.0)
        # Per-minute cancel+replace limiter
        self._cr_limiter = RateLimiter(max_calls=cancel_replace_rate_limit, window_seconds=60.0)
        # condition_id → MarketOrderState
        self._states: dict[str, MarketOrderState] = {}
        # order_id → condition_id (for fill routing)
        self._order_to_market: dict[str, str] = {}
        # Set of order_ids currently being cancelled (prevents double-cancel)
        self._cancelling: set[str] = set()
        # Set to True on first geoblock 403 so we stop trying
        self._geoblocked: bool = False

    def _state(self, condition_id: str) -> MarketOrderState:
        if condition_id not in self._states:
            self._states[condition_id] = MarketOrderState(condition_id=condition_id)
        return self._states[condition_id]

    # ── Main entry points ─────────────────────────────────────────────────────

    @property
    def geoblocked(self) -> bool:
        return self._geoblocked

    async def update_quotes(self, quote: MarketQuote) -> None:
        """
        Ensure the exchange has exactly one bid and one ask for this market,
        matching the desired quote. Cancel stale orders then batch-place all
        new orders in a single API call (up to 4: YES bid/ask + NO bid/ask).
        """
        if self._geoblocked:
            return
        state = self._state(quote.market_condition_id)

        async with state.lock:
            # Rate-limit requote frequency
            elapsed = time.monotonic() - state.last_action_ts
            if elapsed < self._min_requote_interval:
                return

            # Build the full list of sides to manage
            sides = [
                (Side.BUY,  quote.yes_bid,  state.bid_order_id,     state.bid_quote,     False),
                (Side.SELL, quote.yes_ask,  state.ask_order_id,     state.ask_quote,     False),
            ]
            if quote.no_bid is not None:
                sides.append((Side.BUY,  quote.no_bid, state.no_bid_order_id, state.no_bid_quote, True))
            if quote.no_ask is not None:
                sides.append((Side.SELL, quote.no_ask, state.no_ask_order_id, state.no_ask_quote, True))

            # Phase 1: cancel orders that need replacing or zeroing, collect placements
            to_place: list[tuple[Side, QuoteTarget, bool]] = []  # (side, target, is_no)
            for side, target, cur_oid, cur_quote, is_no in sides:
                if target is None:
                    continue
                if target.size <= 0:
                    if cur_oid:
                        await self._cr_limiter.acquire()
                        ok = await self._client.cancel_order(cur_oid)
                        if ok:
                            self._order_to_market.pop(cur_oid, None)
                            self._clear_side_state(state, side, is_no)
                            self._metrics.record_order_cancelled(state.condition_id)
                    continue

                needs_place = cur_oid is None
                needs_replace = False
                if not needs_place and cur_quote is not None:
                    price_diff = abs(target.price - cur_quote.price)
                    size_ratio = abs(target.size - cur_quote.size) / max(cur_quote.size, 1e-6)
                    if price_diff >= 0.005 or size_ratio >= 0.10:
                        needs_replace = True

                if needs_replace and cur_oid:
                    await self._cr_limiter.acquire()
                    ok = await self._client.cancel_order(cur_oid)
                    if ok:
                        self._order_to_market.pop(cur_oid, None)
                        self._clear_side_state(state, side, is_no)
                        self._metrics.record_order_cancelled(state.condition_id)
                    needs_place = True

                if needs_place:
                    to_place.append((side, target, is_no))

            if not to_place:
                return

            # Phase 2: batch-place all new orders in one API call
            await self._place_limiter.acquire()
            batch_specs = [
                {
                    "token_id": target.token_id,
                    "side": side,
                    "price": target.price,
                    "size": target.size,
                    "time_in_force": "GTC",
                }
                for side, target, _ in to_place
            ]
            try:
                order_ids = await self._client.place_limit_orders_batch(batch_specs)
            except GeoBlockedError:
                self._geoblocked = True
                log.error(
                    "GEOBLOCKED",
                    msg="Trading is restricted in your region. "
                        "Connect via a VPN (Polygon-supported jurisdiction) and restart.",
                )
                return
            except InsufficientBalanceError:
                self._geoblocked = True
                log.error(
                    "NO_BALANCE",
                    msg="Wallet has no USDC. Deposit USDC on Polygon to this address and restart.",
                    wallet=self._client._wallet_address,
                )
                return

            for (side, target, is_no), order_id in zip(to_place, order_ids):
                if order_id:
                    self._order_to_market[order_id] = state.condition_id
                    self._set_side_state(state, side, is_no, order_id, target)
                    state.last_action_ts = time.monotonic()
                    self._metrics.record_order_placed(state.condition_id)

    async def cancel_market(self, condition_id: str) -> None:
        """Cancel all open orders for a market (e.g., on risk event)."""
        state = self._state(condition_id)
        async with state.lock:
            ids = state.pending_order_ids()
            if ids:
                await self._cancel_batch(ids)
            state.bid_order_id = None
            state.ask_order_id = None
            state.bid_quote = None
            state.ask_quote = None
            state.no_bid_order_id = None
            state.no_ask_order_id = None
            state.no_bid_quote = None
            state.no_ask_quote = None

    async def cancel_all_markets(self) -> None:
        """Cancel every open order across all markets."""
        log.warning("cancel_all_markets")
        await self._client.cancel_all()
        for state in self._states.values():
            state.bid_order_id = None
            state.ask_order_id = None
            state.bid_quote = None
            state.ask_quote = None
            state.no_bid_order_id = None
            state.no_ask_order_id = None
            state.no_bid_quote = None
            state.no_ask_quote = None
        self._order_to_market.clear()

    # ── Fill handling ─────────────────────────────────────────────────────────

    async def handle_fill(self, trade_event: dict) -> None:
        """
        Process a trade event from the user WebSocket.
        Status flow: MATCHED → CONFIRMED/MINED → (optional FAILED)
        """
        order_id = trade_event.get("id", "")
        status = trade_event.get("status", "")
        size_matched = float(trade_event.get("size_matched", trade_event.get("size", 0)))
        price = float(trade_event.get("price", 0))
        token_id = trade_event.get("asset_id", "")
        side_str = trade_event.get("side", "BUY")
        side = Side.BUY if side_str == "BUY" else Side.SELL

        condition_id = self._order_to_market.get(order_id)
        if not condition_id:
            return

        state = self._state(condition_id)

        if status == "MATCHED" and size_matched > 0 and price > 0:
            # Determine outcome (YES or NO) from token_id
            pos = self._inventory.get_position(condition_id)
            if pos:
                outcome = (
                    MarketSide.YES if token_id == pos.yes_token_id else MarketSide.NO
                )
            else:
                outcome = MarketSide.YES

            # Snapshot realized PnL before the fill so we can compute the delta
            pnl_before = self._inventory.realized_pnl(condition_id)

            fill = Fill(
                order_id=order_id,
                market_condition_id=condition_id,
                token_id=token_id,
                outcome=outcome,
                side=side,
                price=price,
                size=size_matched,
            )
            self._inventory.apply_fill(fill)

            pnl_delta = self._inventory.realized_pnl(condition_id) - pnl_before
            self._metrics.record_fill(
                condition_id,
                volume_usdc=price * size_matched,
                realized_pnl=pnl_delta,
            )
            log.info(
                "fill_received",
                order_id=order_id,
                status=status,
                token=token_id[:12] + "…",
                side=side,
                price=price,
                size=size_matched,
            )

        elif status in ("CONFIRMED", "MINED"):
            # Order fully settled — clear from tracking
            self._cleanup_order(order_id, state)

        elif status == "FAILED":
            log.warning("order_failed", order_id=order_id, condition_id=condition_id[:12])
            self._cleanup_order(order_id, state)
            # Immediately requote
            state.last_action_ts = 0.0

    def _cleanup_order(self, order_id: str, state: MarketOrderState) -> None:
        if state.bid_order_id == order_id:
            state.bid_order_id = None
            state.bid_quote = None
        elif state.ask_order_id == order_id:
            state.ask_order_id = None
            state.ask_quote = None
        elif state.no_bid_order_id == order_id:
            state.no_bid_order_id = None
            state.no_bid_quote = None
        elif state.no_ask_order_id == order_id:
            state.no_ask_order_id = None
            state.no_ask_quote = None
        self._order_to_market.pop(order_id, None)

    # ── Stale order cleanup ───────────────────────────────────────────────────

    async def cleanup_stale_orders(self) -> None:
        """
        Cancel orders that have been 'placed' but unconfirmed for too long.
        This mirrors poly-maker's remove_from_pending() logic.
        """
        now = time.monotonic()
        for condition_id, state in list(self._states.items()):
            if now - state.last_action_ts > _STALE_ORDER_SECONDS:
                stale = []
                for oid in (
                    state.bid_order_id, state.ask_order_id,
                    state.no_bid_order_id, state.no_ask_order_id,
                ):
                    if oid:
                        stale.append(oid)
                if stale:
                    log.warning(
                        "stale_orders_detected",
                        condition_id=condition_id[:12],
                        order_ids=stale,
                    )
                    await self._cancel_batch(stale)
                    state.bid_order_id = None
                    state.ask_order_id = None
                    state.no_bid_order_id = None
                    state.no_ask_order_id = None

    # ── Sync on startup ───────────────────────────────────────────────────────

    async def sync_open_orders(self) -> None:
        """
        Cancel all pre-existing open orders on startup and start fresh.

        This avoids the complexity of resolving YES vs NO token IDs before the
        market selector has loaded (which would be required to correctly assign
        orders to bid/ask/no_bid/no_ask slots). A clean slate is safer: the
        first quoting cycle re-places all orders within seconds.

        Fill routing (_order_to_market) is still populated so any MATCHED
        events that arrive for orders filled during the restart window are
        correctly attributed to their market.
        """
        orders = await self._client.get_open_orders()
        if orders:
            # Populate routing map first so in-flight fills can still be attributed
            for order in orders:
                self._order_to_market[order.order_id] = order.market_condition_id
            # Cancel everything — the quoting loop re-places fresh orders immediately
            await self._client.cancel_all()
            log.info("startup_orders_cancelled", count=len(orders))
        else:
            log.info("open_orders_synced", count=0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _clear_side_state(self, state: MarketOrderState, side: Side, is_no: bool) -> None:
        if side == Side.BUY:
            if is_no:
                state.no_bid_order_id = None
                state.no_bid_quote = None
            else:
                state.bid_order_id = None
                state.bid_quote = None
        else:
            if is_no:
                state.no_ask_order_id = None
                state.no_ask_quote = None
            else:
                state.ask_order_id = None
                state.ask_quote = None

    def _set_side_state(
        self, state: MarketOrderState, side: Side, is_no: bool,
        order_id: str, target: QuoteTarget,
    ) -> None:
        if side == Side.BUY:
            if is_no:
                state.no_bid_order_id = order_id
                state.no_bid_quote = target
            else:
                state.bid_order_id = order_id
                state.bid_quote = target
        else:
            if is_no:
                state.no_ask_order_id = order_id
                state.no_ask_quote = target
            else:
                state.ask_order_id = order_id
                state.ask_quote = target

    def total_usdc_in_orders(self) -> float:
        """
        Return total USDC currently locked in open BUY orders.
        Used by RiskManager to enforce the max_total_usdc_in_orders cap.
        Only BUY orders lock USDC (SELL orders lock shares, not cash).
        """
        total = 0.0
        for state in self._states.values():
            for order_id, quote in (
                (state.bid_order_id, state.bid_quote),
                (state.no_bid_order_id, state.no_bid_quote),
            ):
                if order_id and quote:
                    total += quote.price * quote.size
        return total

    async def _cancel_batch(self, order_ids: list[str]) -> None:
        """Cancel multiple orders, deduplicating in-flight cancels."""
        to_cancel = [oid for oid in order_ids if oid not in self._cancelling]
        if not to_cancel:
            return
        self._cancelling.update(to_cancel)
        try:
            await self._client.cancel_orders(to_cancel)
            for oid in to_cancel:
                self._order_to_market.pop(oid, None)
        finally:
            self._cancelling -= set(to_cancel)
