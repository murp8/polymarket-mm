"""
Tests for the token-bucket rate limiter.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.execution.rate_limiter import RateLimiter


class TestRateLimiter:
    async def test_allows_up_to_limit(self):
        limiter = RateLimiter(max_calls=5, window_seconds=1.0)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        # Should complete essentially immediately
        assert elapsed < 0.5

    async def test_blocks_on_limit_exceeded(self):
        limiter = RateLimiter(max_calls=2, window_seconds=0.2)
        t0 = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        # Third call should block until window passes
        await limiter.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.1  # At least some blocking occurred

    async def test_available_slots_decrements(self):
        limiter = RateLimiter(max_calls=3, window_seconds=60.0)
        assert limiter.available_slots == 3
        await limiter.acquire()
        assert limiter.available_slots == 2
        await limiter.acquire()
        assert limiter.available_slots == 1

    async def test_slots_recover_after_window(self):
        limiter = RateLimiter(max_calls=2, window_seconds=0.1)
        await limiter.acquire()
        await limiter.acquire()
        await asyncio.sleep(0.15)  # Wait for window to expire
        assert limiter.available_slots >= 1

    async def test_concurrent_calls_respect_limit(self):
        """Multiple concurrent callers should all eventually get through."""
        limiter = RateLimiter(max_calls=3, window_seconds=0.5)
        results = []

        async def worker():
            await limiter.acquire()
            results.append(time.monotonic())

        await asyncio.gather(*[worker() for _ in range(6)])
        assert len(results) == 6
