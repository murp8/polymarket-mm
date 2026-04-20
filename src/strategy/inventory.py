"""
Inventory (position) manager.

Tracks net YES/NO share positions per market, computes inventory skew
for quote adjustment, and provides mark-to-market PnL.

Position accounting:
  - A BUY of YES at price p for n shares costs p*n USDC and gives +n YES shares
  - A SELL of YES at price p for n shares yields p*n USDC and gives -n YES shares
  - YES + NO = $1 at settlement (they're complementary tokens)

Inventory skew:
  - If we hold many YES shares, we're exposed to YES going to zero
  - We should widen our YES ask (make it more expensive to buy from us)
    and tighten our YES bid (be more eager to buy YES at a lower price to average in)
  - Skew = net_yes_position / max_inventory_per_market (clamped to [-1, 1])
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.models import Fill, MarketSide, Position, Side
from src.utils.logging import get_logger

log = get_logger(__name__)


class InventoryManager:
    """
    Maintains real-time position state and provides skew signals.

    Persists state to a JSON file so positions survive restarts.
    """

    def __init__(
        self,
        max_inventory_per_market: float = 500.0,
        inventory_skew_threshold: float = 100.0,
        state_file: Optional[str] = None,
    ) -> None:
        self._max_inventory = max_inventory_per_market
        self._skew_threshold = inventory_skew_threshold
        self._state_file = Path(state_file) if state_file else None
        # condition_id → Position
        self._positions: dict[str, Position] = {}
        if self._state_file:
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._state_file and self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                for cid, raw in data.items():
                    self._positions[cid] = Position(
                        market_condition_id=cid,
                        yes_token_id=raw.get("yes_token_id", ""),
                        no_token_id=raw.get("no_token_id", ""),
                        yes_shares=float(raw.get("yes_shares", 0)),
                        no_shares=float(raw.get("no_shares", 0)),
                        avg_yes_cost=float(raw.get("avg_yes_cost", 0)),
                        avg_no_cost=float(raw.get("avg_no_cost", 0)),
                        realized_pnl=float(raw.get("realized_pnl", 0)),
                    )
                log.info("inventory_loaded", markets=len(self._positions))
            except Exception as e:
                log.warning("inventory_load_failed", error=str(e))

    def save(self) -> None:
        if not self._state_file:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            cid: {
                "yes_token_id": pos.yes_token_id,
                "no_token_id": pos.no_token_id,
                "yes_shares": pos.yes_shares,
                "no_shares": pos.no_shares,
                "avg_yes_cost": pos.avg_yes_cost,
                "avg_no_cost": pos.avg_no_cost,
                "realized_pnl": pos.realized_pnl,
            }
            for cid, pos in self._positions.items()
        }
        self._state_file.write_text(json.dumps(data, indent=2))

    # ── Position access ───────────────────────────────────────────────────────

    def get_position(self, condition_id: str) -> Optional[Position]:
        return self._positions.get(condition_id)

    def ensure_position(
        self, condition_id: str, yes_token_id: str, no_token_id: str
    ) -> Position:
        if condition_id not in self._positions:
            self._positions[condition_id] = Position(
                market_condition_id=condition_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )
        return self._positions[condition_id]

    # ── Position updates ──────────────────────────────────────────────────────

    def apply_fill(self, fill: Fill) -> None:
        """
        Update position from a confirmed fill event.
        Uses average-cost accounting for buys, FIFO realized PnL for sells.
        """
        pos = self._positions.get(fill.market_condition_id)
        if pos is None:
            log.warning("apply_fill_unknown_market", condition_id=fill.market_condition_id)
            return

        is_yes = fill.token_id == pos.yes_token_id
        shares = fill.size

        if is_yes:
            if fill.side == Side.BUY:
                # Average in
                total_cost = pos.yes_shares * pos.avg_yes_cost + shares * fill.price
                pos.yes_shares += shares
                pos.avg_yes_cost = total_cost / pos.yes_shares if pos.yes_shares > 0 else 0
            else:  # SELL
                # Realise PnL
                pnl = shares * (fill.price - pos.avg_yes_cost)
                pos.realized_pnl += pnl
                pos.yes_shares = max(0.0, pos.yes_shares - shares)
        else:
            # NO token: price is (1 - yes_price) effectively
            if fill.side == Side.BUY:
                total_cost = pos.no_shares * pos.avg_no_cost + shares * fill.price
                pos.no_shares += shares
                pos.avg_no_cost = total_cost / pos.no_shares if pos.no_shares > 0 else 0
            else:
                pnl = shares * (fill.price - pos.avg_no_cost)
                pos.realized_pnl += pnl
                pos.no_shares = max(0.0, pos.no_shares - shares)

        log.info(
            "position_updated",
            condition_id=fill.market_condition_id[:12],
            token="YES" if is_yes else "NO",
            side=fill.side,
            shares=shares,
            price=fill.price,
            yes_net=round(pos.yes_shares, 2),
            no_net=round(pos.no_shares, 2),
            realized_pnl=round(pos.realized_pnl, 4),
        )

    def sync_from_api(
        self, condition_id: str, yes_token_id: str, no_token_id: str,
        yes_shares: float, no_shares: float
    ) -> None:
        """
        Overwrite position from the CLOB API (used at startup and periodic sync).
        Preserves realized PnL and average costs.
        """
        pos = self.ensure_position(condition_id, yes_token_id, no_token_id)
        pos.yes_shares = yes_shares
        pos.no_shares = no_shares

    # ── Skew calculation ──────────────────────────────────────────────────────

    def inventory_skew(self, condition_id: str) -> float:
        """
        Return a skew signal in [-1, 1].

        Positive = we are net long YES → widen ask, tighten bid.
        Negative = we are net short YES → widen bid, tighten ask.
        """
        pos = self._positions.get(condition_id)
        if pos is None:
            return 0.0
        net = pos.net_yes_position
        if abs(net) < self._skew_threshold:
            return 0.0
        return max(-1.0, min(1.0, net / self._max_inventory))

    def is_at_limit(self, condition_id: str, side: Side, outcome: MarketSide) -> bool:
        """Return True if we should not place any more orders on this side."""
        pos = self._positions.get(condition_id)
        if pos is None:
            return False
        if outcome == MarketSide.YES:
            if side == Side.BUY and pos.yes_shares >= self._max_inventory:
                return True
            if side == Side.SELL and pos.yes_shares <= 0:
                return True
        else:
            if side == Side.BUY and pos.no_shares >= self._max_inventory:
                return True
            if side == Side.SELL and pos.no_shares <= 0:
                return True
        return False

    # ── PnL queries ───────────────────────────────────────────────────────────

    def unrealized_pnl(self, condition_id: str, yes_mid: float) -> float:
        pos = self._positions.get(condition_id)
        if pos is None:
            return 0.0
        return pos.unrealized_pnl(yes_mid)

    def realized_pnl(self, condition_id: str) -> float:
        pos = self._positions.get(condition_id)
        return pos.realized_pnl if pos else 0.0

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    def all_open_positions(self) -> list[dict]:
        """Return all positions with non-zero shares, for dashboard display."""
        result = []
        for cid, pos in self._positions.items():
            if pos.yes_shares > 0:
                result.append({
                    "condition_id": cid,
                    "token": "YES",
                    "shares": pos.yes_shares,
                    "avg_cost": pos.avg_yes_cost,
                    "cost_basis": pos.yes_shares * pos.avg_yes_cost,
                })
            if pos.no_shares > 0:
                result.append({
                    "condition_id": cid,
                    "token": "NO",
                    "shares": pos.no_shares,
                    "avg_cost": pos.avg_no_cost,
                    "cost_basis": pos.no_shares * pos.avg_no_cost,
                })
        return result

    def total_exposure(self, yes_mids: dict[str, float]) -> float:
        """Total mark-to-market value of all open positions."""
        total = 0.0
        for cid, pos in self._positions.items():
            mid = yes_mids.get(cid, 0.5)
            total += pos.yes_shares * mid + pos.no_shares * (1 - mid)
        return total
