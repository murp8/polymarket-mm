"""
WebSocket client for real-time Polymarket CLOB data.

Two channels:
  - Market channel: orderbook snapshots + incremental price_change events
  - User channel:   authenticated fill/order status events

Architecture:
  - Each channel runs in its own asyncio task
  - Reconnects with exponential backoff on disconnect or error
  - Delivers parsed events to registered callbacks
  - Maintains a local orderbook via SortedDict for O(log n) updates

WebSocket message formats (from Polymarket docs):

Market subscription:
  {"assets_ids": ["<token_id>", ...], "type": "market"}

User subscription:
  {"auth": {"apiKey": "...", "secret": "...", "passphrase": "..."}, "type": "user"}

Book event (full snapshot):
  {"event_type": "book", "asset_id": "<token_id>",
   "bids": [{"price": "0.50", "size": "100"}, ...],
   "asks": [...], "timestamp": "..."}

Price change event (incremental):
  {"event_type": "price_change", "asset_id": "<token_id>",
   "changes": [{"price": "0.51", "side": "BUY", "size": "50"}, ...]}

User trade event:
  {"event_type": "trade", "id": "...", "status": "MATCHED|CONFIRMED|MINED|FAILED",
   "asset_id": "...", "side": "BUY|SELL", "price": "...", "size": "...",
   "maker_address": "..."}
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Awaitable, Callable, Optional

import websockets
from sortedcontainers import SortedDict

from src.models import Orderbook, PriceLevel
from src.utils.logging import get_logger

log = get_logger(__name__)

# Reconnect delays: 1, 2, 4, 8 … capped at 60 seconds
_BACKOFF_BASE = 1.0
_BACKOFF_MAX = 60.0
_PING_INTERVAL = 5  # seconds

# Callbacks
BookCallback = Callable[[str, Orderbook], Awaitable[None]]
PriceChangeCallback = Callable[[str, Orderbook], Awaitable[None]]
UserTradeCallback = Callable[[dict], Awaitable[None]]


class LocalOrderbook:
    """
    Maintains a local, incrementally-updated view of a single token's
    order book using SortedDict for O(log n) insertions and deletions.
    """

    def __init__(self, token_id: str) -> None:
        self.token_id = token_id
        # bids keyed by -price so highest bid is first
        self._bids: SortedDict = SortedDict()
        # asks keyed by +price so lowest ask is first
        self._asks: SortedDict = SortedDict()
        self.last_update = time.monotonic()

    def apply_snapshot(self, bids: list[dict], asks: list[dict]) -> None:
        self._bids.clear()
        self._asks.clear()
        for b in bids:
            p, s = float(b["price"]), float(b["size"])
            if s > 0:
                self._bids[-p] = s
        for a in asks:
            p, s = float(a["price"]), float(a["size"])
            if s > 0:
                self._asks[p] = s
        self.last_update = time.monotonic()

    def apply_change(self, price: float, side: str, size: float) -> None:
        """Apply a single price-level update. size=0 removes the level."""
        if side == "BUY":
            key = -price
            if size == 0:
                self._bids.pop(key, None)
            else:
                self._bids[key] = size
        else:
            key = price
            if size == 0:
                self._asks.pop(key, None)
            else:
                self._asks[key] = size
        self.last_update = time.monotonic()

    def to_orderbook(self) -> Orderbook:
        bids = [PriceLevel(price=-k, size=v) for k, v in self._bids.items()]
        asks = [PriceLevel(price=k, size=v) for k, v in self._asks.items()]
        return Orderbook(token_id=self.token_id, bids=bids, asks=asks)

    @property
    def midpoint(self) -> Optional[float]:
        ob = self.to_orderbook()
        return ob.midpoint


class MarketWebSocket:
    """
    Subscribes to real-time orderbook updates for a set of tokens.

    Callbacks are invoked with the token_id and the updated Orderbook.
    """

    _WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(
        self,
        token_ids: list[str],
        on_book: Optional[BookCallback] = None,
        on_price_change: Optional[PriceChangeCallback] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self._token_ids = list(token_ids)
        self._on_book = on_book
        self._on_price_change = on_price_change
        self._url = ws_url or self._WS_URL
        self._books: dict[str, LocalOrderbook] = {
            tid: LocalOrderbook(tid) for tid in token_ids
        }
        self._running = False
        # Active WS connection reference — set while _connect_once is running.
        # Allows update_tokens() to immediately re-subscribe on the live connection
        # rather than waiting for the next reconnect.
        self._active_ws: Optional[object] = None

    def update_tokens(self, token_ids: list[str]) -> None:
        """
        Update the token subscription list.

        Sends a fresh subscription message on the existing connection immediately
        so new tokens start receiving events right away. Falls back to the next
        reconnect if the connection is currently closed.
        """
        self._token_ids = list(token_ids)
        for tid in token_ids:
            if tid not in self._books:
                self._books[tid] = LocalOrderbook(tid)

        # Re-subscribe on the live connection so new tokens start streaming
        # without waiting for the next reconnect (which could be hours away).
        if self._active_ws is not None:
            asyncio.ensure_future(self._send_subscription(self._active_ws))

    def get_orderbook(self, token_id: str) -> Optional[Orderbook]:
        lb = self._books.get(token_id)
        return lb.to_orderbook() if lb else None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        lb = self._books.get(token_id)
        return lb.midpoint if lb else None

    async def run(self) -> None:
        """Run forever, reconnecting on errors."""
        self._running = True
        backoff = _BACKOFF_BASE
        while self._running:
            try:
                await self._connect_once()
                backoff = _BACKOFF_BASE  # reset on clean exit
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(
                    "market_ws_disconnected",
                    error=str(e),
                    reconnect_in=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    async def stop(self) -> None:
        self._running = False

    async def _send_subscription(self, ws) -> None:
        """Send (or re-send) the market subscription message on an open connection."""
        try:
            sub_msg = json.dumps({
                "assets_ids": self._token_ids,
                "type": "market",
            })
            await ws.send(sub_msg)
            log.debug("market_ws_subscribed", tokens=len(self._token_ids))
        except Exception as e:
            log.warning("market_ws_resubscribe_failed", error=str(e))

    async def _connect_once(self) -> None:
        async with websockets.connect(
            self._url,
            ping_interval=_PING_INTERVAL,
            ping_timeout=30,
        ) as ws:
            self._active_ws = ws
            try:
                log.debug("market_ws_connected", tokens=len(self._token_ids))
                await self._send_subscription(ws)

                async for raw_msg in ws:
                    if not self._running:
                        break
                    try:
                        await self._handle(json.loads(raw_msg))
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        log.warning("market_ws_handle_error", error=str(e))
            finally:
                self._active_ws = None

    async def _handle(self, msg: dict | list) -> None:
        # Messages can arrive as a single dict or a list of dicts
        if isinstance(msg, list):
            for item in msg:
                await self._handle(item)
            return

        event_type = msg.get("event_type", "")
        token_id = msg.get("asset_id", "")

        if not token_id or token_id not in self._books:
            return

        lb = self._books[token_id]

        if event_type == "book":
            lb.apply_snapshot(msg.get("bids", []), msg.get("asks", []))
            if self._on_book:
                await self._on_book(token_id, lb.to_orderbook())

        elif event_type == "price_change":
            for change in msg.get("changes", []):
                lb.apply_change(
                    float(change["price"]),
                    change["side"],
                    float(change["size"]),
                )
            if self._on_price_change:
                await self._on_price_change(token_id, lb.to_orderbook())


class UserWebSocket:
    """
    Authenticated WebSocket for real-time fill and order-status events.
    """

    _WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        on_trade: Optional[UserTradeCallback] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._on_trade = on_trade
        self._url = ws_url or self._WS_URL
        self._running = False

    async def run(self) -> None:
        """Run forever, reconnecting on errors."""
        self._running = True
        backoff = _BACKOFF_BASE
        while self._running:
            try:
                await self._connect_once()
                backoff = _BACKOFF_BASE
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(
                    "user_ws_disconnected",
                    error=str(e),
                    reconnect_in=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    async def stop(self) -> None:
        self._running = False

    async def _connect_once(self) -> None:
        async with websockets.connect(
            self._url,
            ping_interval=_PING_INTERVAL,
            ping_timeout=30,
        ) as ws:
            log.debug("user_ws_connected")
            sub_msg = json.dumps({
                "auth": {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "passphrase": self._api_passphrase,
                },
                "type": "user",
            })
            await ws.send(sub_msg)

            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    await self._handle(json.loads(raw_msg))
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    log.warning("user_ws_handle_error", error=str(e))

    async def _handle(self, msg: dict | list) -> None:
        if isinstance(msg, list):
            for item in msg:
                await self._handle(item)
            return

        if msg.get("event_type") == "trade" and self._on_trade:
            await self._on_trade(msg)
