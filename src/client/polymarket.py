"""
Polymarket CLOB client wrapper.

Wraps py-clob-client with:
  - Async-friendly interface (runs blocking calls in thread executor)
  - Auto-derivation of API credentials from wallet private key
  - Typed return values
  - Structured logging
  - Retry on transient failures
"""

from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any, Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    BookParams,
    OpenOrderParams,
    OrderArgs,
    PostOrdersArgs,
    TradeParams,
)
from py_clob_client.order_builder.constants import BUY, SELL
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models import (
    Fill,
    Market,
    MarketSide,
    Order,
    OrderStatus,
    Orderbook,
    PriceLevel,
    Side,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# CLOB tick size (prices must be multiples of this)
TICK_SIZE = 0.01

_GEOBLOCK_MSG = "Trading restricted in your region"
_NO_BALANCE_MSG = "not enough balance"


class GeoBlockedError(RuntimeError):
    """Raised when the CLOB API returns a geoblock 403."""


class InsufficientBalanceError(RuntimeError):
    """Raised when the wallet has no USDC to place orders."""


class PolymarketClient:
    """
    Async wrapper around py-clob-client.

    All blocking py-clob-client calls are dispatched to a thread executor
    so the asyncio event loop is never blocked.
    """

    def __init__(
        self,
        private_key: str,
        wallet_address: str,
        clob_host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        signature_type: int = 0,
        funder_address: Optional[str] = None,
    ) -> None:
        self._private_key = private_key
        self._wallet_address = wallet_address
        self._clob_host = clob_host
        self._chain_id = chain_id
        self._signature_type = signature_type
        self._funder_address = funder_address or ""
        self._client: Optional[ClobClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialise the CLOB client and derive/create API credentials."""
        self._loop = asyncio.get_event_loop()
        await self._run(self._init_client)
        log.debug("clob_client_connected", host=self._clob_host)

    def _init_client(self) -> None:
        kwargs: dict[str, Any] = {
            "host": self._clob_host,
            "key": self._private_key,
            "chain_id": self._chain_id,
            "signature_type": self._signature_type,
        }
        if self._funder_address:
            kwargs["funder"] = self._funder_address

        self._client = ClobClient(**kwargs)
        creds = self._client.create_or_derive_api_creds()
        self._client.set_api_creds(creds)
        # Expose creds for the user WebSocket
        self._client.api_creds = creds
        log.debug(
            "api_creds_set",
            api_key=creds.api_key[:8] + "…",
        )

    @property
    def raw(self) -> ClobClient:
        if self._client is None:
            raise RuntimeError("PolymarketClient.connect() must be called before use")
        return self._client

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _run(self, fn, *args, **kwargs):
        """Run a blocking function in the default thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

    @staticmethod
    def _snap_price(price: float) -> float:
        """Round price to nearest tick size."""
        return round(round(price / TICK_SIZE) * TICK_SIZE, 4)

    # ── Market data ──────────────────────────────────────────────────────────

    async def get_orderbook(self, token_id: str) -> Orderbook:
        raw = await self._run(self.raw.get_order_book, token_id)
        bids = sorted(
            [PriceLevel(float(b.price), float(b.size)) for b in (raw.bids or [])],
            key=lambda x: -x.price,
        )
        asks = sorted(
            [PriceLevel(float(a.price), float(a.size)) for a in (raw.asks or [])],
            key=lambda x: x.price,
        )
        return Orderbook(token_id=token_id, bids=bids, asks=asks)

    async def get_orderbooks(self, token_ids: list[str]) -> dict[str, Orderbook]:
        params = [BookParams(token_id=tid) for tid in token_ids]
        raw_books = await self._run(self.raw.get_order_books, params)
        result: dict[str, Orderbook] = {}
        for raw in raw_books:
            tid = raw.asset_id
            bids = sorted(
                [PriceLevel(float(b.price), float(b.size)) for b in (raw.bids or [])],
                key=lambda x: -x.price,
            )
            asks = sorted(
                [PriceLevel(float(a.price), float(a.size)) for a in (raw.asks or [])],
                key=lambda x: x.price,
            )
            result[tid] = Orderbook(token_id=tid, bids=bids, asks=asks)
        return result

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        try:
            resp = await self._run(self.raw.get_midpoint, token_id)
            return float(resp.get("mid", 0) or 0) or None
        except Exception:
            return None

    async def get_last_trade_price(self, token_id: str) -> Optional[float]:
        try:
            resp = await self._run(self.raw.get_last_trade_price, token_id)
            return float(resp.get("price", 0) or 0) or None
        except Exception:
            return None

    async def get_usdc_balance(self) -> float:
        """Return available USDC balance (in USDC, not micro-USDC)."""
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = await self._run(self.raw.get_balance_allowance, params)
            # balance is returned as a string in micro-USDC (6 decimals)
            raw = resp.get("balance", "0") or "0"
            return float(raw) / 1_000_000
        except Exception as e:
            log.warning("get_usdc_balance_failed", error=str(e))
            return 0.0

    # ── Order management ─────────────────────────────────────────────────────

    async def place_limit_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
        time_in_force: str = "GTC",
    ) -> Optional[str]:
        """
        Place a GTC limit order. Returns the order_id on success, None on failure.
        Price is snapped to tick size.
        """
        snapped = self._snap_price(price)
        clob_side = BUY if side == Side.BUY else SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=snapped,
            size=size,
            side=clob_side,
        )

        try:
            signed = await self._run(self.raw.create_order, order_args)
            resp = await self._run(self.raw.post_order, signed, time_in_force)
            order_id = resp.get("orderID") or resp.get("id")
            if order_id:
                log.info(
                    "order_placed",
                    order_id=order_id,
                    token_id=token_id[:12] + "…",
                    side=side,
                    price=snapped,
                    size=size,
                )
                return order_id
            log.warning("order_place_no_id", resp=resp)
            return None
        except Exception as e:
            err = str(e)
            if _GEOBLOCK_MSG in err:
                raise GeoBlockedError(err) from e
            if _NO_BALANCE_MSG in err:
                raise InsufficientBalanceError(err) from e
            log.error("order_place_failed", token_id=token_id[:12] + "…", side=side, price=snapped, size=size, error=err)
            return None

    async def place_limit_orders_batch(
        self,
        orders: list[dict],  # [{"token_id", "side", "price", "size", "time_in_force"}, ...]
    ) -> list[Optional[str]]:
        """
        Place up to 15 orders in a single batch API call.
        Returns a list of order_ids (None for any that failed).
        Docs recommend batching for lower latency: post both sides + NO token in one call.
        """
        if not orders:
            return []

        # Sign each order individually (signing is local, no I/O)
        signed_args: list[PostOrdersArgs] = []
        for o in orders:
            snapped = self._snap_price(o["price"])
            clob_side = BUY if o["side"] == Side.BUY else SELL
            order_args = OrderArgs(
                token_id=o["token_id"],
                price=snapped,
                size=o["size"],
                side=clob_side,
            )
            try:
                signed = await self._run(self.raw.create_order, order_args)
                tif = o.get("time_in_force", "GTC")
                signed_args.append(PostOrdersArgs(order=signed, orderType=tif))
            except Exception as e:
                err = str(e)
                if _GEOBLOCK_MSG in err:
                    raise GeoBlockedError(err) from e
                log.error("batch_sign_failed", token=o["token_id"][:12] + "…", error=err)
                signed_args.append(None)  # placeholder so index alignment is preserved

        # Submit all signed orders in one HTTP request
        valid_args = [a for a in signed_args if a is not None]
        if not valid_args:
            return [None] * len(orders)

        try:
            resp = await self._run(self.raw.post_orders, valid_args)
            # Normalise response: API returns a list, but defensively handle a
            # wrapped-dict response (e.g. {"orders": [...]} or {"data": [...]})
            if isinstance(resp, list):
                resp_list = resp
            elif isinstance(resp, dict):
                resp_list = (
                    resp.get("orders")
                    or resp.get("data")
                    or resp.get("results")
                    or []
                )
                if not resp_list:
                    log.warning("batch_place_unexpected_response", resp_keys=list(resp.keys()))
            else:
                log.warning("batch_place_unexpected_response_type", resp_type=type(resp).__name__)
                resp_list = []

            ids_iter = iter(resp_list)
            order_ids: list[Optional[str]] = []
            for arg in signed_args:
                if arg is None:
                    order_ids.append(None)
                else:
                    entry = next(ids_iter, {})
                    oid = (entry.get("orderID") or entry.get("id")) if isinstance(entry, dict) else None
                    order_ids.append(oid)
                    if oid:
                        log.info("batch_order_placed", order_id=oid)
                    else:
                        log.warning("batch_order_no_id", entry=entry)
            return order_ids
        except Exception as e:
            err = str(e)
            if _GEOBLOCK_MSG in err:
                raise GeoBlockedError(err) from e
            if _NO_BALANCE_MSG in err:
                raise InsufficientBalanceError(err) from e
            log.error("batch_place_failed", count=len(valid_args), error=err)
            return [None] * len(orders)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID. Returns True on success."""
        try:
            await self._run(self.raw.cancel, order_id)
            log.info("order_cancelled", order_id=order_id)
            return True
        except Exception as e:
            log.warning("order_cancel_failed", order_id=order_id, error=str(e))
            return False

    async def cancel_orders(self, order_ids: list[str]) -> int:
        """
        Cancel multiple orders in a single batch API call.
        Returns the count of cancellations (best-effort from response).
        """
        if not order_ids:
            return 0
        try:
            resp = await self._run(self.raw.cancel_orders, order_ids)
            # SDK returns {"canceled": [...], "not_canceled": [...]} or similar
            if isinstance(resp, dict):
                canceled = resp.get("canceled", order_ids)
                return len(canceled)
            return len(order_ids)
        except Exception as e:
            log.warning("cancel_orders_batch_failed", count=len(order_ids), error=str(e))
            # Fallback: cancel individually
            tasks = [self.cancel_order(oid) for oid in order_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return sum(1 for r in results if r is True)

    async def cancel_all(self) -> bool:
        """Cancel every open order. Use with care."""
        try:
            await self._run(self.raw.cancel_all)
            log.info("all_orders_cancelled")
            return True
        except Exception as e:
            log.error("cancel_all_failed", error=str(e))
            return False

    async def get_open_orders(
        self,
        market_condition_id: Optional[str] = None,
        token_id: Optional[str] = None,
    ) -> list[Order]:
        """Fetch open orders, optionally filtered by market or token."""
        params = OpenOrderParams(
            market=market_condition_id,
            asset_id=token_id,
        )
        try:
            raw_orders = await self._run(self.raw.get_orders, params)
        except Exception as e:
            log.warning("get_open_orders_failed", error=str(e))
            return []

        orders: list[Order] = []
        for raw in raw_orders:
            try:
                orders.append(
                    Order(
                        order_id=raw.get("id", ""),
                        market_condition_id=raw.get("market", ""),
                        token_id=raw.get("asset_id", ""),
                        outcome=MarketSide.YES,  # resolved from token later
                        side=Side.BUY if raw.get("side", "BUY") == "BUY" else Side.SELL,
                        price=float(raw.get("price", 0)),
                        size=float(raw.get("original_size", raw.get("size", 0))),
                        filled_size=float(raw.get("size_matched", 0)),
                        status=OrderStatus.OPEN,
                    )
                )
            except Exception:
                pass
        return orders

    async def get_trades(
        self, market_condition_id: Optional[str] = None
    ) -> list[dict]:
        """Fetch recent trade history."""
        params = TradeParams(market=market_condition_id)
        try:
            return await self._run(self.raw.get_trades, params) or []
        except Exception as e:
            log.warning("get_trades_failed", error=str(e))
            return []

    async def get_positions(self) -> dict[str, float]:
        """
        Return token_id → net_position mapping.
        Falls back to scanning recent trades if get_positions isn't available.
        Positive = long.
        """
        # Try the positions endpoint (available in newer py-clob-client versions)
        for method_name in ("get_positions", "get_all_positions"):
            method = getattr(self.raw, method_name, None)
            if method is None:
                continue
            try:
                raw = await self._run(method) or []
                return {
                    p.get("asset", p.get("token_id", "")): float(p.get("size", 0))
                    for p in raw
                    if float(p.get("size", 0)) != 0
                }
            except Exception as e:
                log.debug("get_positions_method_failed", method=method_name, error=str(e))

        log.debug("get_positions_not_available", msg="Position sync will be filled by WebSocket fills")
        return {}

    # ── Reward markets ───────────────────────────────────────────────────────

    async def get_sampling_markets_all(self) -> list[dict]:
        """
        Fetch ALL sampling/reward markets via cursor-based pagination.

        Uses the CLOB API's get_sampling_markets endpoint which is the
        authoritative source for markets with active liquidity rewards.
        Each market dict contains a `rewards` sub-object with:
          - min_size (minimum order size to qualify)
          - max_spread (maximum spread to qualify, in cents)
          - rates[].rewards_daily_rate (USDC/day reward rate)
        """
        all_markets: list[dict] = []
        cursor = ""
        page = 0

        while True:
            try:
                resp = await self._run(
                    self.raw.get_sampling_markets,
                    *(  # positional cursor arg if non-empty
                        [cursor] if cursor else []
                    ),
                )
            except Exception as e:
                log.warning("get_sampling_markets_failed", page=page, error=str(e))
                break

            data = resp.get("data") or []
            all_markets.extend(data)
            next_cursor = resp.get("next_cursor", "")
            limit = resp.get("limit", 0)
            page += 1

            log.debug(
                "sampling_markets_page",
                page=page,
                fetched=len(data),
                total=len(all_markets),
                cursor=next_cursor[:12] + "…" if next_cursor else "none",
            )

            # Stop when the page is smaller than the limit (last page) or no cursor
            if not next_cursor or next_cursor == cursor or len(data) < limit:
                break
            cursor = next_cursor

        log.debug("sampling_markets_fetched", total=len(all_markets))
        return all_markets

    async def get_earning_rewards_markets(self) -> list[dict]:
        """Return markets where the user is currently earning rewards."""
        try:
            return await self._run(self.raw.get_earning_rewards_markets) or []
        except Exception as e:
            log.warning("get_earning_rewards_markets_failed", error=str(e))
            return []
