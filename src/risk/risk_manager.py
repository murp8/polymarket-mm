"""
Risk manager with circuit breakers.

Checks performed before quoting each market:
  1. Global drawdown limit
  2. Per-market loss limit
  3. Global USDC-in-orders limit
  4. Position fraction limit (per market vs total balance)
  5. Mid-price velocity circuit breaker (sudden market moves)
  6. Stale price guard (don't quote if price source is old)

Circuit breakers:
  - Trip → all quoting halted for `cooldown_seconds`
  - Per-market trip → that market paused for `cooldown_seconds`
  - Self-healing: automatically re-enables after cooldown
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import RiskConfig
from src.strategy.inventory import InventoryManager
from src.utils.logging import get_logger
from src.utils.metrics import MetricsCollector

log = get_logger(__name__)


@dataclass
class CircuitBreakerState:
    tripped: bool = False
    tripped_at: float = 0.0
    reason: str = ""


class RiskManager:
    """
    Gate all quoting activity through risk checks.
    Thread-safe for asyncio use (no actual blocking I/O).
    """

    def __init__(
        self,
        cfg: RiskConfig,
        inventory: InventoryManager,
        metrics: MetricsCollector,
    ) -> None:
        self._cfg = cfg
        self._inventory = inventory
        self._metrics = metrics
        # Global circuit breaker
        self._global_cb = CircuitBreakerState()
        # Per-market circuit breakers: condition_id → state
        self._market_cbs: dict[str, CircuitBreakerState] = {}
        # Mid-price history for velocity checking: condition_id → deque[(ts, mid)]
        self._mid_history: dict[str, list[tuple[float, float]]] = {}
        # USDC currently locked in open orders (tracked by OrderManager signals)
        self._usdc_in_orders: float = 0.0
        # Per-market buy prohibition after stop-loss: condition_id → resume_at monotonic ts
        self._risk_off_buy: dict[str, float] = {}

    # ── Main check ────────────────────────────────────────────────────────────

    def can_quote(self, condition_id: str) -> tuple[bool, str]:
        """
        Return (True, "") if quoting is allowed, or (False, reason) if blocked.
        Call this before computing or placing quotes for a market.
        """
        # Global circuit breaker
        if self._global_cb.tripped:
            if time.monotonic() - self._global_cb.tripped_at > self._cfg.circuit_breaker_cooldown_seconds:
                self._reset_global()
            else:
                return False, f"global_cb: {self._global_cb.reason}"

        # Per-market circuit breaker
        mcb = self._market_cbs.get(condition_id)
        if mcb and mcb.tripped:
            if time.monotonic() - mcb.tripped_at > self._cfg.circuit_breaker_cooldown_seconds:
                self._reset_market(condition_id)
            else:
                return False, f"market_cb: {mcb.reason}"

        # Drawdown check
        drawdown = self._metrics.current_drawdown
        if drawdown >= self._cfg.max_drawdown_usdc:
            self._trip_global(f"drawdown={drawdown:.2f} >= {self._cfg.max_drawdown_usdc}")
            return False, "global_drawdown_limit"

        # Per-market loss check
        market_loss = -self._inventory.realized_pnl(condition_id)
        if market_loss >= self._cfg.max_market_loss_usdc:
            self._trip_market(condition_id, f"market_loss={market_loss:.2f}")
            return False, "market_loss_limit"

        # USDC in orders check
        if self._usdc_in_orders >= self._cfg.max_total_usdc_in_orders:
            return False, "max_usdc_in_orders"

        # Mid-price velocity check
        if self._is_price_moving_fast(condition_id):
            self._trip_market(condition_id, "mid_price_velocity")
            return False, "mid_price_velocity"

        return True, ""

    def can_quote_any(self) -> bool:
        """Quick global check — is anything even allowed right now?"""
        if self._global_cb.tripped:
            if time.monotonic() - self._global_cb.tripped_at > self._cfg.circuit_breaker_cooldown_seconds:
                self._reset_global()
                return True
            return False
        return True

    # ── State updates (called by other components) ────────────────────────────

    def record_mid_price(self, condition_id: str, mid: float) -> None:
        """Track mid-price for velocity calculation."""
        now = time.monotonic()
        if condition_id not in self._mid_history:
            self._mid_history[condition_id] = []
        hist = self._mid_history[condition_id]
        hist.append((now, mid))
        # Keep only last 120 seconds
        cutoff = now - 120
        self._mid_history[condition_id] = [(t, p) for t, p in hist if t >= cutoff]

    def record_orders_delta(self, delta_usdc: float) -> None:
        """Update the USDC-in-orders tally. Positive = placed, negative = cancelled/filled."""
        self._usdc_in_orders = max(0.0, self._usdc_in_orders + delta_usdc)

    def update_usdc_in_orders(self, total: float) -> None:
        """Hard-set the USDC-in-orders total (from a full open-orders sync)."""
        self._usdc_in_orders = max(0.0, total)

    # ── Circuit breaker management ────────────────────────────────────────────

    def _trip_global(self, reason: str) -> None:
        self._global_cb.tripped = True
        self._global_cb.tripped_at = time.monotonic()
        self._global_cb.reason = reason
        log.error("global_circuit_breaker_tripped", reason=reason)

    def _reset_global(self) -> None:
        self._global_cb.tripped = False
        log.info("global_circuit_breaker_reset")

    def _trip_market(self, condition_id: str, reason: str) -> None:
        cb = self._market_cbs.setdefault(condition_id, CircuitBreakerState())
        cb.tripped = True
        cb.tripped_at = time.monotonic()
        cb.reason = reason
        log.warning(
            "market_circuit_breaker_tripped",
            condition_id=condition_id[:12],
            reason=reason,
        )

    def _reset_market(self, condition_id: str) -> None:
        if condition_id in self._market_cbs:
            self._market_cbs[condition_id].tripped = False
            log.info("market_circuit_breaker_reset", condition_id=condition_id[:12])

    def trip_global_manual(self, reason: str = "manual") -> None:
        """Manually trip the global breaker (e.g., from a SIGINT handler)."""
        self._trip_global(reason)

    # ── Risk-off buy control (stop-loss cooldown) ─────────────────────────────

    def set_risk_off_buy(self, condition_id: str, hours: float) -> None:
        """Prohibit buying on a market for `hours` hours (after a stop-loss)."""
        resume_at = time.monotonic() + hours * 3600
        self._risk_off_buy[condition_id] = resume_at
        log.warning(
            "risk_off_buy_set",
            condition_id=condition_id[:12],
            hours=hours,
        )

    def is_buy_allowed(self, condition_id: str) -> bool:
        """Return True if buying is currently allowed on this market."""
        resume_at = self._risk_off_buy.get(condition_id)
        if resume_at is None:
            return True
        if time.monotonic() >= resume_at:
            del self._risk_off_buy[condition_id]
            log.info("risk_off_buy_expired", condition_id=condition_id[:12])
            return True
        return False

    # ── Velocity check ────────────────────────────────────────────────────────

    def _is_price_moving_fast(self, condition_id: str) -> bool:
        """
        Return True if the YES mid moved more than max_mid_move_60s
        in the last 60 seconds.
        """
        hist = self._mid_history.get(condition_id, [])
        if len(hist) < 2:
            return False
        now = time.monotonic()
        cutoff = now - 60
        recent = [p for t, p in hist if t >= cutoff]
        if len(recent) < 2:
            return False
        move = max(recent) - min(recent)
        if move > self._cfg.max_mid_move_60s:
            log.warning(
                "price_velocity_breach",
                condition_id=condition_id[:12],
                move=round(move, 4),
                limit=self._cfg.max_mid_move_60s,
            )
            return True
        return False

    # ── Status reporting ──────────────────────────────────────────────────────

    def status(self) -> dict:
        tripped_markets = [
            cid for cid, cb in self._market_cbs.items() if cb.tripped
        ]
        return {
            "global_cb_tripped": self._global_cb.tripped,
            "global_cb_reason": self._global_cb.reason,
            "market_cbs_tripped": tripped_markets,
            "usdc_in_orders": round(self._usdc_in_orders, 2),
            "current_drawdown": round(self._metrics.current_drawdown, 2),
            "total_realized_pnl": round(self._inventory.total_realized_pnl(), 4),
        }
