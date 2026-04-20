"""
In-process performance metrics and periodic summary reporting.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class MarketMetrics:
    condition_id: str
    orders_placed: int = 0
    orders_cancelled: int = 0
    orders_filled: int = 0
    fill_volume_usdc: float = 0.0
    realized_pnl_usdc: float = 0.0
    reward_score_sum: float = 0.0
    reward_samples: int = 0
    # Rolling window of mid-prices for volatility estimation
    mid_prices: Deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=120)
    )

    @property
    def avg_reward_score(self) -> float:
        if self.reward_samples == 0:
            return 0.0
        return self.reward_score_sum / self.reward_samples


class MetricsCollector:
    """Thread-safe (asyncio) metrics store with periodic summary logging."""

    def __init__(self) -> None:
        self._markets: dict[str, MarketMetrics] = {}
        self._global_orders_placed = 0
        self._global_orders_cancelled = 0
        self._global_fills = 0
        self._global_fill_volume_usdc = 0.0
        self._global_realized_pnl = 0.0
        self._start_time = time.monotonic()
        # (timestamp, pnl) tuples for drawdown tracking
        self._pnl_history: Deque[tuple[float, float]] = deque(maxlen=1440)

    def _market(self, condition_id: str) -> MarketMetrics:
        if condition_id not in self._markets:
            self._markets[condition_id] = MarketMetrics(condition_id=condition_id)
        return self._markets[condition_id]

    # ── Event recording ──────────────────────────────────────────────────────

    def record_order_placed(self, condition_id: str) -> None:
        self._market(condition_id).orders_placed += 1
        self._global_orders_placed += 1

    def record_order_cancelled(self, condition_id: str) -> None:
        self._market(condition_id).orders_cancelled += 1
        self._global_orders_cancelled += 1

    def record_fill(
        self,
        condition_id: str,
        volume_usdc: float,
        realized_pnl: float,
    ) -> None:
        m = self._market(condition_id)
        m.orders_filled += 1
        m.fill_volume_usdc += volume_usdc
        m.realized_pnl_usdc += realized_pnl
        self._global_fills += 1
        self._global_fill_volume_usdc += volume_usdc
        self._global_realized_pnl += realized_pnl
        self._pnl_history.append((time.monotonic(), self._global_realized_pnl))

    def record_reward_score(self, condition_id: str, score: float) -> None:
        m = self._market(condition_id)
        m.reward_score_sum += score
        m.reward_samples += 1

    def record_mid_price(self, condition_id: str, mid: float) -> None:
        self._market(condition_id).mid_prices.append((time.monotonic(), mid))

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def peak_pnl(self) -> float:
        if not self._pnl_history:
            return 0.0
        return max(p for _, p in self._pnl_history)

    @property
    def current_drawdown(self) -> float:
        return self.peak_pnl - self._global_realized_pnl

    def runtime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    # ── Reporting ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a lightweight dict of global metrics for dashboard display."""
        return {
            "fills": self._global_fills,
            "fill_volume_usdc": self._global_fill_volume_usdc,
            "orders_placed": self._global_orders_placed,
            "orders_cancelled": self._global_orders_cancelled,
            "realized_pnl_usdc": self._global_realized_pnl,
        }

    def log_summary(self) -> None:
        runtime_h = self.runtime_seconds() / 3600
        log.info(
            "performance_summary",
            runtime_hours=round(runtime_h, 2),
            markets_active=len(self._markets),
            orders_placed=self._global_orders_placed,
            orders_cancelled=self._global_orders_cancelled,
            fills=self._global_fills,
            fill_volume_usdc=round(self._global_fill_volume_usdc, 2),
            realized_pnl_usdc=round(self._global_realized_pnl, 4),
            peak_pnl_usdc=round(self.peak_pnl, 4),
            current_drawdown_usdc=round(self.current_drawdown, 4),
        )
        # Only log per-market summaries for markets that have had fills
        filled_markets = [
            (cid, m) for cid, m in self._markets.items() if m.orders_filled > 0
        ]
        for cid, m in sorted(
            filled_markets, key=lambda kv: kv[1].fill_volume_usdc, reverse=True
        )[:10]:
            log.info(
                "market_summary",
                condition_id=cid[:12] + "…",
                fills=m.orders_filled,
                volume_usdc=round(m.fill_volume_usdc, 2),
                realized_pnl=round(m.realized_pnl_usdc, 4),
                avg_reward_score=round(m.avg_reward_score, 6),
            )
