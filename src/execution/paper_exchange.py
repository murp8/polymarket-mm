"""
Paper trading engine.

Mirrors the OrderManager interface exactly so it can be hot-swapped into the
bot without changing any other component. Uses real WebSocket orderbook data to
simulate fills and accumulates estimated Polymarket liquidity rewards.

Fill simulation rules:
  BUY  limit order fills when orderbook's best ask ≤ our bid price
  SELL limit order fills when orderbook's best bid ≥ our ask price

Reward simulation:
  Each qualifying resting order earns a continuous reward stream based on
  the quadratic Polymarket scoring formula:
      Q = ((max_spread - spread_from_mid) / max_spread)² × size
  We integrate Q over time and multiply by the market's rewards_daily_rate
  to get estimated USDC earned per day.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.models import Fill, Market, MarketQuote, MarketSide, Orderbook, QuoteTarget, Side
from src.strategy.inventory import InventoryManager
from src.utils.logging import get_logger
from src.utils.metrics import MetricsCollector

log = get_logger(__name__)


@dataclass
class PaperOrder:
    order_id: str
    condition_id: str
    token_id: str
    side: Side
    price: float
    size: float
    placed_at: float = field(default_factory=time.monotonic)


@dataclass
class PaperMarketState:
    condition_id: str
    bid: Optional[PaperOrder] = None
    ask: Optional[PaperOrder] = None
    no_bid: Optional[PaperOrder] = None
    no_ask: Optional[PaperOrder] = None
    # Accumulated reward score (unit-less — multiply by rate for USDC)
    reward_score_seconds: float = 0.0
    last_reward_tick: float = field(default_factory=time.monotonic)


class PaperOrderManager:
    """
    Drop-in replacement for OrderManager in paper-trading mode.

    All public methods match OrderManager exactly so the bot wiring
    requires zero changes.
    """

    def __init__(
        self,
        inventory: InventoryManager,
        metrics: MetricsCollector,
        initial_balance_usdc: float = 1_000.0,
        cash_reserve_fraction: float = 0.05,
        maker_fee_bps: int = 0,
    ) -> None:
        self._inventory = inventory
        self._metrics = metrics
        self._balance = initial_balance_usdc
        self._initial_balance = initial_balance_usdc  # fixed reference for reserve calc
        self._cash_reserve = initial_balance_usdc * cash_reserve_fraction
        self._maker_fee_rate = maker_fee_bps / 10_000.0
        self._locked_usdc: float = 0.0  # USDC reserved for open BUY orders
        # condition_id → state
        self._states: dict[str, PaperMarketState] = {}
        # token_id → condition_id (for orderbook routing)
        self._token_to_cid: dict[str, str] = {}
        # Reward accumulation: condition_id → total estimated USDC
        self._reward_usdc: dict[str, float] = {}
        # Market metadata for reward calc
        self._markets: dict[str, Market] = {}
        self._fill_count = 0
        self._start_time = time.monotonic()
        # Latest YES midpoint per market (from orderbook updates), for unrealized PnL
        self._current_mids: dict[str, float] = {}

    # ── OrderManager interface ────────────────────────────────────────────────

    @property
    def geoblocked(self) -> bool:
        return False

    async def sync_open_orders(self) -> None:
        pass  # Nothing to sync — paper orders don't persist

    async def update_quotes(self, quote: MarketQuote) -> None:
        cid = quote.market_condition_id
        state = self._get_state(cid)

        # Register token → condition mapping for fill routing
        self._token_to_cid[quote.yes_bid.token_id] = cid
        self._token_to_cid[quote.yes_ask.token_id] = cid

        self._place_or_replace(state, Side.BUY, quote.yes_bid, is_no=False)
        self._place_or_replace(state, Side.SELL, quote.yes_ask, is_no=False)

        # NO token orders
        if quote.no_bid is not None:
            if quote.no_bid.size > 0:
                self._token_to_cid[quote.no_bid.token_id] = cid
                self._place_or_replace(state, Side.BUY, quote.no_bid, is_no=True)
            else:
                self._cancel_paper_order(state.no_bid)
                state.no_bid = None
        if quote.no_ask is not None:
            if quote.no_ask.size > 0:
                self._token_to_cid[quote.no_ask.token_id] = cid
                self._place_or_replace(state, Side.SELL, quote.no_ask, is_no=True)
            else:
                self._cancel_paper_order(state.no_ask)
                state.no_ask = None

    async def handle_fill(self, trade_event: dict) -> None:
        pass  # Paper fills are generated from orderbook data, not WS events

    async def cancel_market(self, condition_id: str) -> None:
        state = self._states.get(condition_id)
        if state:
            self._cancel_paper_order(state.bid)
            self._cancel_paper_order(state.ask)
            self._cancel_paper_order(state.no_bid)
            self._cancel_paper_order(state.no_ask)
            state.bid = None
            state.ask = None
            state.no_bid = None
            state.no_ask = None

    async def cancel_all_markets(self) -> None:
        for cid in list(self._states):
            await self.cancel_market(cid)

    async def cleanup_stale_orders(self) -> None:
        pass  # Paper orders don't go stale

    # ── Feed from the market WebSocket ───────────────────────────────────────

    def on_orderbook_update(self, token_id: str, orderbook: Orderbook) -> None:
        """
        Called whenever the bot receives an orderbook snapshot or update.
        Checks if any paper orders should fill against the live book.
        Works for both YES and NO token orderbook updates.
        """
        cid = self._token_to_cid.get(token_id)
        if cid is None:
            return
        state = self._states.get(cid)
        if state is None:
            return

        best_bid = orderbook.best_bid
        best_ask = orderbook.best_ask

        # Only fill orders whose token_id matches this orderbook update.
        # Without this guard, a NO token update (prices ~0.27 for a YES=0.73 market)
        # would incorrectly trigger YES bid fills at 0.27 — free money that doesn't exist.
        if state.bid and state.bid.token_id == token_id:
            if best_ask is not None and best_ask <= state.bid.price:
                self._execute_fill(state.bid, fill_price=best_ask)
                state.bid = None

        if state.ask and state.ask.token_id == token_id:
            if best_bid is not None and best_bid >= state.ask.price:
                self._execute_fill(state.ask, fill_price=best_bid)
                state.ask = None

        if state.no_bid and state.no_bid.token_id == token_id:
            if best_ask is not None and best_ask <= state.no_bid.price:
                self._execute_fill(state.no_bid, fill_price=best_ask)
                state.no_bid = None

        if state.no_ask and state.no_ask.token_id == token_id:
            if best_bid is not None and best_bid >= state.no_ask.price:
                self._execute_fill(state.no_ask, fill_price=best_bid)
                state.no_ask = None

        # Tick reward accumulation (use YES token updates only to avoid double-counting)
        market = self._markets.get(cid)
        if market and orderbook.midpoint is not None:
            # Only tick rewards when we get a YES token update
            # (NO token updates arrive interleaved; midpoint here is NO token mid)
            # We identify YES vs NO by checking if the token_id is a registered YES token
            yes_token = market.yes_token.token_id
            if token_id == yes_token:
                self._current_mids[cid] = orderbook.midpoint
                self._tick_rewards(state, market, orderbook.midpoint)

    def register_market(self, market: Market) -> None:
        """Call once per market so we have metadata for reward calculations."""
        self._markets[market.condition_id] = market

    # ── Summary / reporting ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a snapshot of paper trading state for logging."""
        open_yes = sum(
            (1 if s.bid else 0) + (1 if s.ask else 0)
            for s in self._states.values()
        )
        open_no = sum(
            (1 if s.no_bid else 0) + (1 if s.no_ask else 0)
            for s in self._states.values()
        )
        total_reward = sum(self._reward_usdc.values())
        realized = self._inventory.total_realized_pnl()
        # Mark-to-market unrealized PnL across all open positions
        unrealized = sum(
            self._inventory.unrealized_pnl(cid, mid)
            for cid, mid in self._current_mids.items()
        )
        return {
            "balance_usdc": round(self._balance, 2),
            "locked_usdc": round(self._locked_usdc, 2),
            "open_orders": open_yes + open_no,
            "open_yes_orders": open_yes,
            "open_no_orders": open_no,
            "fills": self._fill_count,
            "realized_pnl": round(realized, 4),
            "unrealized_pnl": round(unrealized, 4),
            # Upper-bound estimate: assumes bot is sole LP on all markets.
            # Real earnings will be a fraction of this based on competition.
            "reward_earned_usdc": round(total_reward, 4),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_state(self, condition_id: str) -> PaperMarketState:
        if condition_id not in self._states:
            self._states[condition_id] = PaperMarketState(condition_id=condition_id)
        return self._states[condition_id]

    def _place_or_replace(
        self,
        state: PaperMarketState,
        side: Side,
        target: QuoteTarget,
        is_no: bool = False,
    ) -> None:
        if target.size <= 0:
            return

        current = (
            (state.no_bid if is_no else state.bid) if side == Side.BUY
            else (state.no_ask if is_no else state.ask)
        )

        needs_place = current is None
        needs_replace = False

        if current is not None:
            price_diff = abs(target.price - current.price)
            size_ratio = abs(target.size - current.size) / max(current.size, 1e-6)
            if price_diff >= 0.005 or size_ratio >= 0.10:
                needs_replace = True

        if needs_replace:
            self._cancel_paper_order(current)
            if side == Side.BUY:
                if is_no:
                    state.no_bid = None
                else:
                    state.bid = None
            else:
                if is_no:
                    state.no_ask = None
                else:
                    state.ask = None
            needs_place = True

        if needs_place:
            # Check we have available balance for BUY orders, keeping cash_reserve free
            cost = target.price * target.size
            available = self._balance - self._locked_usdc - self._cash_reserve
            if side == Side.BUY and cost > available:
                log.debug(
                    "paper_insufficient_balance",
                    needed=round(cost, 2),
                    available=round(available, 2),
                    reserve=round(self._cash_reserve, 2),
                )
                return

            order = PaperOrder(
                order_id=str(uuid.uuid4())[:8],
                condition_id=state.condition_id,
                token_id=target.token_id,
                side=side,
                price=target.price,
                size=target.size,
            )
            if side == Side.BUY:
                self._locked_usdc += cost
                if is_no:
                    state.no_bid = order
                else:
                    state.bid = order
            else:
                if is_no:
                    state.no_ask = order
                else:
                    state.ask = order

            self._metrics.record_order_placed(state.condition_id)
            log.debug(
                "paper_order_placed",
                side=side,
                token="NO" if is_no else "YES",
                price=target.price,
                size=target.size,
                market=state.condition_id[:10] + "…",
            )

    def _cancel_paper_order(self, order: Optional[PaperOrder]) -> None:
        if order is None:
            return
        if order.side == Side.BUY:
            self._locked_usdc = max(0.0, self._locked_usdc - order.price * order.size)

    def _execute_fill(self, order: PaperOrder, fill_price: float) -> None:
        """Simulate an order fill and update balance + inventory."""
        size = order.size

        fee = fill_price * size * self._maker_fee_rate
        if order.side == Side.BUY:
            cost = fill_price * size + fee
            self._locked_usdc = max(0.0, self._locked_usdc - order.price * size)
            self._balance -= cost
        else:
            proceeds = fill_price * size - fee
            self._balance += proceeds

        # Infer outcome from which token this order is for
        pos = self._inventory.get_position(order.condition_id)
        if pos and order.token_id == pos.no_token_id:
            outcome = MarketSide.NO
        else:
            outcome = MarketSide.YES
        fill = Fill(
            order_id=order.order_id,
            market_condition_id=order.condition_id,
            token_id=order.token_id,
            outcome=outcome,
            side=order.side,
            price=fill_price,
            size=size,
        )
        pnl_before = self._inventory.realized_pnl(order.condition_id)
        self._inventory.apply_fill(fill)
        pnl_delta = self._inventory.realized_pnl(order.condition_id) - pnl_before
        self._metrics.record_fill(
            order.condition_id,
            volume_usdc=fill_price * size,
            realized_pnl=pnl_delta,
        )
        self._fill_count += 1

        log.info(
            "paper_fill",
            side=order.side,
            price=round(fill_price, 4),
            size=size,
            market=order.condition_id[:10] + "…",
            balance=round(self._balance, 2),
        )

    def _tick_rewards(
        self,
        state: PaperMarketState,
        market: Market,
        mid: float,
    ) -> None:
        """
        Accumulate estimated reward earned since the last tick for this market.

        For each qualifying resting order, reward earned per second is:
            (rewards_daily_rate / 86400) × Q_score
        where Q = ((max_spread - |price - mid|) / max_spread)² × size
        and the order only qualifies if |price - mid| < max_spread.
        """
        now = time.monotonic()
        dt = min(now - state.last_reward_tick, 60.0)  # cap at 60s to prevent first-tick explosion
        state.last_reward_tick = now

        if dt <= 0:
            return

        incentive = market.incentive
        v = incentive.max_incentive_spread
        daily_rate = incentive.reward_epoch_amount  # USDC/day for this market
        per_second_rate = daily_rate / 86_400

        if v <= 0 or per_second_rate <= 0:
            return

        cid = state.condition_id

        # Q score is a proportional metric: as sole LP, your share of the pool is
        # your_Q / total_Q = 100%. So you earn `per_second_rate × dt` per market —
        # the size in Q only matters when splitting the pool among competing LPs.
        # We still use the spread-efficiency factor (without size) to reflect that
        # tighter quotes earn a larger share of the pool.
        yes_mid = mid
        no_mid = 1.0 - mid
        efficiency_sum = 0.0
        n_qualifying = 0
        for order, order_mid in (
            (state.bid, yes_mid), (state.ask, yes_mid),
            (state.no_bid, no_mid), (state.no_ask, no_mid),
        ):
            if order is None:
                continue
            spread = abs(order.price - order_mid)
            if spread >= v:
                continue  # outside qualifying spread
            efficiency_sum += ((v - spread) / v) ** 2
            n_qualifying += 1

        if n_qualifying == 0:
            return

        # Average spread efficiency across all qualifying sides (0–1)
        avg_efficiency = efficiency_sum / n_qualifying
        # Max reward as sole LP: full pool × average spread efficiency
        earned = per_second_rate * dt * avg_efficiency
        self._reward_usdc[cid] = self._reward_usdc.get(cid, 0.0) + earned
