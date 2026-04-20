"""
Polymarket Market Maker — main entry point.

Architecture:
  ┌────────────────────────────────────────────────────────────────┐
  │  MarketSelector  ←── Gamma API (periodic refresh)             │
  │       ↓                                                        │
  │  CompositePricer ←── Odds API (Pinnacle)                      │
  │       ↓             ←── Orderbook midpoint (WebSocket)        │
  │  QuoteEngine                                                   │
  │       ↓                                                        │
  │  RiskManager   ←── InventoryManager                           │
  │       ↓                                                        │
  │  OrderManager  ──→ CLOB API                                   │
  │       ↑                                                        │
  │  UserWebSocket ─────── fill events ──────────────────────────│
  │  MarketWebSocket ───── orderbook events ─────────────────────│
  └────────────────────────────────────────────────────────────────┘

Concurrency model:
  - MarketWebSocket and UserWebSocket each run in their own asyncio tasks
  - A periodic refresh task re-ranks markets every 5 minutes
  - A quoting loop fires every `refresh_interval_seconds` (15s default)
    and also on each WebSocket orderbook event
  - A stale-order cleanup task runs every 15 seconds
  - A metrics summary task runs every 60 seconds
  - A state persistence task saves position state every 30 seconds

Signal handling:
  - SIGINT / SIGTERM: cancel all orders, persist state, exit cleanly
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

from src.client.gamma import GammaClient
from src.client.polymarket import PolymarketClient
from src.client.websocket_client import MarketWebSocket, UserWebSocket
from src.config import Config, load_config
from src.market_selector.selector import MarketSelector
from src.models import Market, Orderbook
from src.pricing.composite import CompositePricer
from src.pricing.odds_api import OddsApiPricer
from src.pricing.orderbook import OrderbookPricer
from src.execution.order_manager import OrderManager
from src.execution.paper_exchange import PaperOrderManager
from src.risk.risk_manager import RiskManager
from src.strategy.inventory import InventoryManager
from src.strategy.quote_engine import QuoteEngine
from src.utils.dashboard import render_paper_dashboard
from src.utils.logging import get_logger, print_banner, setup_logging
from src.utils.metrics import MetricsCollector

log = get_logger(__name__)


class Bot:
    """
    The top-level bot object that owns all components and drives the event loop.
    """

    def __init__(self, cfg: Config, paper_trade: bool = False) -> None:
        self.cfg = cfg
        self._paper_trade = paper_trade
        creds = cfg.credentials

        # ── Clients ────────────────────────────────────────────────────────
        self.clob = PolymarketClient(
            private_key=creds.polymarket_private_key,
            wallet_address=creds.polymarket_wallet_address,
            clob_host=cfg.exchange.clob_host,
            chain_id=cfg.exchange.chain_id,
            signature_type=creds.polymarket_signature_type or cfg.exchange.signature_type,
            funder_address=creds.polymarket_funder_address or None,
        )
        self.gamma = GammaClient(base_url=cfg.exchange.gamma_host)

        # ── Core components ────────────────────────────────────────────────
        self.metrics = MetricsCollector()
        # Paper mode: no state file — always start clean, never persist fake positions
        state_file = None if paper_trade else cfg.metrics.state_file
        self.inventory = InventoryManager(
            max_inventory_per_market=cfg.quoting.max_inventory_per_market,
            inventory_skew_threshold=cfg.quoting.inventory_skew_threshold,
            state_file=state_file,
        )
        self.ob_pricer = OrderbookPricer(
            max_age_seconds=cfg.pricing.max_price_age_seconds
        )
        self.odds_pricer: Optional[OddsApiPricer] = None
        if creds.has_odds_api_key and cfg.pricing.primary_source == "odds_api":
            self.odds_pricer = OddsApiPricer(
                api_key=creds.odds_api_key,
                base_url=cfg.pricing.odds_api.base_url,
                bookmakers=cfg.pricing.odds_api.bookmakers,
                fallback_bookmakers=cfg.pricing.odds_api.fallback_bookmakers,
                regions=cfg.pricing.odds_api.regions,
                cache_ttl_seconds=cfg.pricing.odds_api.cache_ttl_seconds,
                vig_method=cfg.pricing.vig_removal,
            )
        self.composite_pricer = CompositePricer(
            odds_pricer=self.odds_pricer,
            orderbook_pricer=self.ob_pricer,
            primary_source=cfg.pricing.primary_source,
        )
        self.risk = RiskManager(
            cfg=cfg.risk,
            inventory=self.inventory,
            metrics=self.metrics,
        )
        if paper_trade:
            self.order_manager: OrderManager | PaperOrderManager = PaperOrderManager(
                inventory=self.inventory,
                metrics=self.metrics,
                initial_balance_usdc=cfg.execution.initial_paper_balance_usdc,
                cash_reserve_fraction=cfg.execution.cash_reserve_fraction,
                maker_fee_bps=cfg.execution.paper_maker_fee_bps,
            )
        else:
            self.order_manager: OrderManager | PaperOrderManager = OrderManager(
                client=self.clob,
                inventory=self.inventory,
                metrics=self.metrics,
                cfg=cfg.execution,
                order_rate_limit=cfg.execution.order_rate_limit,
                cancel_replace_rate_limit=cfg.execution.cancel_replace_rate_limit,
                min_requote_interval=cfg.quoting.min_requote_interval_seconds,
            )
        self.quote_engine = QuoteEngine(
            pricer=self.composite_pricer,
            inventory=self.inventory,
            cfg=cfg.quoting,
            ob_pricer=self.ob_pricer,
            risk=self.risk,
        )
        self.selector: Optional[MarketSelector] = None
        self.market_ws: Optional[MarketWebSocket] = None
        self.user_ws: Optional[UserWebSocket] = None

        self._shutdown = asyncio.Event()
        # Queued condition_ids that need requoting (from WS events)
        self._requote_queue: asyncio.Queue[str] = asyncio.Queue()
        # Track currently quoted market quotes
        self._current_quotes: dict[str, Market] = {}
        # Set of condition_ids currently in the active quoting universe (top N)
        self._active_market_ids: set[str] = set()
        # Bot start time for runtime display
        self._start_time: float = 0.0

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        wallet = self.cfg.credentials.polymarket_wallet_address
        dry_run = getattr(self, "_dry_run", False)

        if self._paper_trade:
            # Paper mode: no real CLOB connection needed, but we still need
            # the client for market data (orderbook WS, market selector)
            await self.clob.connect()
            log.info("ready", wallet=wallet[:10] + "…", mode="PAPER TRADE")
        else:
            # Init CLOB connection
            await self.clob.connect()

            # Cancel any pre-existing orders and build fill-routing map
            await self.order_manager.sync_open_orders()

            mode = "DRY RUN" if dry_run else "LIVE"

            if not dry_run:
                balance = await self.clob.get_usdc_balance()
                if balance == 0:
                    log.error(
                        "NO_BALANCE",
                        msg=f"Wallet {wallet[:10]}… has 0 USDC on Polygon. "
                            "Deposit USDC (bridged to Polygon) before running live.",
                    )
                    sys.exit(1)
                log.info("balance", usdc=round(balance, 2))

            log.info("ready", wallet=wallet[:10] + "…", mode=mode)

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown("signal")))

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        import time as _time
        self._start_time = _time.monotonic()
        await self.start()

        async with GammaClient(base_url=self.cfg.exchange.gamma_host) as gamma_ctx:
            self.selector = MarketSelector(
                gamma_ctx,
                self.cfg.market_selection,
                clob_client=self.clob,
            )
            # Initial market load
            await self.selector.refresh()
            top = self.selector.top_markets()
            self._active_market_ids = {m.condition_id for m in top}
            log.info(
                "markets_loaded",
                eligible=self.selector.market_count,
                quoting=len(top),
            )

            # Register markets with paper manager (needed for reward simulation)
            if self._paper_trade:
                pm = self.order_manager  # type: ignore[assignment]
                for m in self.selector.top_markets():
                    pm.register_market(m)
                log.info(
                    "paper_trade_ready",
                    markets=len(top),
                    balance=pm._balance,
                )

            # Build market→position state
            await self._init_positions()

            async with self._maybe_odds_pricer():
                tasks = [
                    asyncio.create_task(self._market_ws_loop(), name="market_ws"),
                    asyncio.create_task(self._quoting_loop(), name="quoting"),
                    asyncio.create_task(self._requote_event_loop(), name="requote_events"),
                    asyncio.create_task(self._market_refresh_loop(), name="market_refresh"),
                    asyncio.create_task(self._cleanup_loop(), name="cleanup"),
                    asyncio.create_task(self._metrics_loop(), name="metrics"),
                    asyncio.create_task(self._persist_loop(), name="persist"),
                ]
                # User WebSocket (fill events) — not needed in paper mode
                if not self._paper_trade:
                    tasks.append(asyncio.create_task(self._user_ws_loop(), name="user_ws"))
                log.debug("all_tasks_started", count=len(tasks))

                # Wait until shutdown is signalled
                await self._shutdown.wait()
                log.info("shutdown_initiated")

                # Cancel all tasks
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        await self._graceful_shutdown()

    # ── Async context for optional odds pricer ────────────────────────────────

    class _NullContext:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass

    def _maybe_odds_pricer(self):
        if self.odds_pricer is not None:
            return self.odds_pricer
        return self._NullContext()

    # ── WebSocket loops ───────────────────────────────────────────────────────

    async def _market_ws_loop(self) -> None:
        """Run the market orderbook WebSocket, reconnecting on failure."""
        while not self._shutdown.is_set():
            token_ids = self.selector.all_token_ids() if self.selector else []
            if not token_ids:
                await asyncio.sleep(5)
                continue

            self.market_ws = MarketWebSocket(
                token_ids=token_ids,
                on_book=self._on_book,
                on_price_change=self._on_price_change,
                ws_url=self.cfg.exchange.ws_market_host,
            )
            try:
                await self.market_ws.run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("market_ws_loop_error", error=str(e))
                await asyncio.sleep(2)

    async def _user_ws_loop(self) -> None:
        """Run the authenticated user WebSocket for fill events."""
        # API creds are stored on the client instance after connect()
        creds = getattr(self.clob.raw, "_creds", None) or getattr(self.clob.raw, "creds", None)
        if creds is None:
            # Try the attribute name used by py-clob-client
            creds = getattr(self.clob.raw, "api_creds", None)
        if creds is None:
            log.warning("user_ws_no_creds_skipping", msg="Could not retrieve API creds for user WS")
            return

        self.user_ws = UserWebSocket(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
            on_trade=self.order_manager.handle_fill,
            ws_url=self.cfg.exchange.ws_user_host,
        )
        try:
            await self.user_ws.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("user_ws_loop_error", error=str(e))

    # ── WebSocket callbacks ───────────────────────────────────────────────────

    async def _on_book(self, token_id: str, orderbook: Orderbook) -> None:
        """Full orderbook snapshot received."""
        await self._handle_orderbook_update(token_id, orderbook)

    async def _on_price_change(self, token_id: str, orderbook: Orderbook) -> None:
        """Incremental price change received."""
        await self._handle_orderbook_update(token_id, orderbook)

    async def _handle_orderbook_update(
        self, token_id: str, orderbook: Orderbook
    ) -> None:
        if self.selector is None:
            return

        # Check YES token map first, then NO token map
        yes_map = self.selector.yes_token_to_market()
        market = yes_map.get(token_id)
        is_no_token = False
        if market is None:
            no_map = self.selector.no_token_to_market()
            market = no_map.get(token_id)
            is_no_token = True

        # Fallback: market may have dropped from selector but we still hold a
        # position — use the cached Market object so stop-loss can still fire.
        if market is None:
            for m in self._current_quotes.values():
                pos = self.inventory.get_position(m.condition_id)
                if pos and (pos.yes_shares > 0 or pos.no_shares > 0):
                    if m.yes_token.token_id == token_id:
                        market, is_no_token = m, False
                        break
                    if m.no_token.token_id == token_id:
                        market, is_no_token = m, True
                        break

        if market is None:
            return

        # Paper trading: simulate fills against the live orderbook
        if self._paper_trade:
            self.order_manager.on_orderbook_update(token_id, orderbook)  # type: ignore[union-attr]

        mid = orderbook.midpoint
        if mid is not None and not is_no_token:
            # Only update pricing/risk state from the YES token orderbook
            self.ob_pricer.update_midpoint(market.condition_id, mid)
            self.ob_pricer.update_orderbook(market.condition_id, orderbook)
            self.selector.update_market_price(market.condition_id, mid)
            if market.condition_id in self._active_market_ids:
                self.risk.record_mid_price(market.condition_id, mid)
                self.metrics.record_mid_price(market.condition_id, mid)

        # Enqueue for requoting on any orderbook update:
        # - active markets (normal quoting)
        # - markets not in top-N but where we hold an open position (stop-loss monitoring)
        cid = market.condition_id
        pos = self.inventory.get_position(cid)
        has_position = pos is not None and (pos.yes_shares > 0 or pos.no_shares > 0)
        if cid in self._active_market_ids or has_position:
            try:
                self._requote_queue.put_nowait(cid)
            except asyncio.QueueFull:
                pass

    # ── Quoting loops ─────────────────────────────────────────────────────────

    async def _quoting_loop(self) -> None:
        """
        Periodic quoting loop: re-quote all active markets on a timer.
        This runs in addition to event-driven requotes from WebSocket.
        """
        interval = self.cfg.quoting.refresh_interval_seconds
        while not self._shutdown.is_set():
            await asyncio.sleep(interval)
            if self.order_manager.geoblocked:
                log.error("halted_geoblocked", msg="Use a VPN and restart.")
                await self.shutdown("geoblock")
                return
            if self.selector is None:
                continue
            top = self.selector.top_markets()
            self._active_market_ids = {m.condition_id for m in top}
            for market in top:
                if self._shutdown.is_set():
                    break
                await self._quote_market(market)

    async def _requote_event_loop(self) -> None:
        """
        Event-driven requoting: process markets queued by WebSocket callbacks.
        Uses a deduplication window so the same market isn't requoted twice
        in one cycle.
        """
        processed: set[str] = set()
        while not self._shutdown.is_set():
            try:
                condition_id = await asyncio.wait_for(
                    self._requote_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                processed.clear()
                continue

            if condition_id in processed:
                self._requote_queue.task_done()
                continue
            processed.add(condition_id)

            if self.selector:
                market = (
                    self.selector.get_market(condition_id)
                    or self._current_quotes.get(condition_id)
                )
                if market:
                    await self._quote_market(market)
            self._requote_queue.task_done()

    async def _quote_market(self, market: Market) -> None:
        """Quote a single market end-to-end."""
        # Cache market object so it can be used for stop-loss monitoring even
        # after the market drops out of the selector's top-N or _markets list.
        self._current_quotes[market.condition_id] = market

        can_quote, reason = self.risk.can_quote(market.condition_id)
        if not can_quote:
            log.debug("quote_blocked", market=market.condition_id[:12], reason=reason)
            return

        # Ensure position tracking is initialized
        self.inventory.ensure_position(
            market.condition_id,
            market.yes_token.token_id,
            market.no_token.token_id,
        )

        # Keep paper exchange metadata fresh so reward/mid tracking is always live.
        # _market_refresh_loop also does this but runs every 5 min; doing it here
        # ensures markets that enter the top-N between refreshes are registered immediately.
        if self._paper_trade:
            self.order_manager.register_market(market)  # type: ignore[union-attr]

        quote = await self.quote_engine.compute_quote(market)
        if quote is None:
            return

        await self.order_manager.update_quotes(quote)

    # ── Refresh, cleanup, metrics ─────────────────────────────────────────────

    async def _market_refresh_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(self.cfg.market_selection.refresh_interval_seconds)
            if self.selector and self.selector.needs_refresh():
                await self.selector.refresh()
                top = self.selector.top_markets()
                self._active_market_ids = {m.condition_id for m in top}
                # Update WebSocket subscriptions — include tokens for any
                # markets where we currently hold an open position so that
                # stop-loss monitoring continues even if the market dropped
                # out of the eligible set (e.g., midpoint went extreme).
                if self.market_ws:
                    held_tokens = {
                        tid
                        for pos in self.inventory._positions.values()
                        if pos.yes_shares > 0 or pos.no_shares > 0
                        for tid in (pos.yes_token_id, pos.no_token_id)
                        if tid
                    }
                    self.market_ws.update_tokens(
                        list(set(self.selector.all_token_ids()) | held_tokens)
                    )
                # Keep paper manager aware of any new markets
                if self._paper_trade:
                    pm = self.order_manager  # type: ignore[assignment]
                    for m in top:
                        pm.register_market(m)

    async def _cleanup_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(15)
            await self.order_manager.cleanup_stale_orders()

    async def _metrics_loop(self) -> None:
        import time as _time
        interval = (
            self.cfg.metrics.paper_dashboard_interval_seconds
            if self._paper_trade
            else self.cfg.metrics.summary_interval_seconds
        )
        while not self._shutdown.is_set():
            await asyncio.sleep(interval)
            if self._paper_trade:
                # Rich dashboard replaces the noisy per-component log blocks
                s = self.order_manager.summary()  # type: ignore[union-attr]
                ms = self.metrics.snapshot()
                render_paper_dashboard(
                    runtime_seconds=_time.monotonic() - self._start_time,
                    initial_balance=self.cfg.execution.initial_paper_balance_usdc,
                    summary=s,
                    positions=self.inventory.all_open_positions(),
                    risk=self.risk.status(),
                    fills_total=ms.get("fills", 0),
                    fill_volume_usdc=ms.get("fill_volume_usdc", 0.0),
                )
            else:
                # Sync USDC-in-orders into RiskManager so max_total_usdc_in_orders
                # circuit breaker fires correctly in live trading.
                if hasattr(self.order_manager, "total_usdc_in_orders"):
                    self.risk.update_usdc_in_orders(
                        self.order_manager.total_usdc_in_orders()  # type: ignore[union-attr]
                    )
                self.metrics.log_summary()
                log.info("risk_status", **self.risk.status())

    async def _persist_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(30)
            self.inventory.save()

    # ── Init helpers ──────────────────────────────────────────────────────────

    async def _init_positions(self) -> None:
        """Sync positions from CLOB API and set up inventory tracking."""
        if self.selector is None:
            return

        # Paper mode: inventory starts empty — never load real CLOB positions.
        # Importing real on-chain balances would corrupt the paper simulation.
        if self._paper_trade:
            return

        api_positions = await self.clob.get_positions()
        for market in self.selector.top_markets():
            yes_shares = api_positions.get(market.yes_token.token_id, 0.0)
            no_shares = api_positions.get(market.no_token.token_id, 0.0)
            self.inventory.sync_from_api(
                market.condition_id,
                market.yes_token.token_id,
                market.no_token.token_id,
                yes_shares,
                no_shares,
            )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self, reason: str = "unknown") -> None:
        log.info("shutdown_requested", reason=reason)
        self.risk.trip_global_manual("shutdown")
        self._shutdown.set()

    async def _graceful_shutdown(self) -> None:
        log.info("graceful_shutdown_start")
        await self.order_manager.cancel_all_markets()
        self.inventory.save()
        self.metrics.log_summary()
        if self._paper_trade:
            log.info("paper_final_summary", **self.order_manager.summary())  # type: ignore[union-attr]
        log.info("graceful_shutdown_complete")


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for the `polymarket-mm` CLI command."""
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Liquidity Rewards Market Maker")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute quotes but do not place any orders",
    )
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        help="Simulate trading with real orderbook data — no real orders placed",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.log_level:
        cfg.logging.level = args.log_level

    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    setup_logging(cfg.logging)
    print_banner()

    if not cfg.credentials.has_polymarket_creds:
        print("ERROR: Missing credentials. Set POLYMARKET_PRIVATE_KEY and "
              "POLYMARKET_WALLET_ADDRESS in .env", file=sys.stderr)
        sys.exit(1)

    if not cfg.credentials.has_odds_api_key:
        log.warning("pricing", source="orderbook only (no ODDS_API_KEY set)")

    bot = Bot(cfg, paper_trade=args.paper_trade)
    bot._dry_run = args.dry_run  # type: ignore[attr-defined]

    if args.dry_run:
        log.info("mode", value="DRY RUN — no orders will be placed")
        async def noop_update(quote):
            log.info(
                "quote",
                market=quote.market_condition_id[:10] + "…",
                bid=quote.yes_bid.price,
                ask=quote.yes_ask.price,
            )
        bot.order_manager.update_quotes = noop_update

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
