"""
Token-bucket rate limiter for CLOB API calls.

Polymarket's CLOB enforces rate limits on order placement and cancellation.
This limiter ensures we never exceed those limits, avoiding 429 errors that
waste retry budget and slow us down.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimiter:
    """
    Sliding-window rate limiter.

    Allows at most `max_calls` calls in any `window_seconds` rolling window.
    Callers await `acquire()` before each API call; they block until a slot
    is available rather than failing.
    """

    def __init__(self, max_calls: int, window_seconds: float = 1.0) -> None:
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a rate-limit slot is available."""
        async with self._lock:
            now = time.monotonic()
            # Evict timestamps outside the window
            while self._timestamps and self._timestamps[0] < now - self._window:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_calls:
                # Wait until the oldest timestamp falls out of the window
                sleep_for = self._window - (now - self._timestamps[0]) + 0.001
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                # Re-evict after sleeping
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] < now - self._window:
                    self._timestamps.popleft()

            self._timestamps.append(time.monotonic())

    @property
    def available_slots(self) -> int:
        now = time.monotonic()
        recent = sum(1 for ts in self._timestamps if ts >= now - self._window)
        return max(0, self._max_calls - recent)
