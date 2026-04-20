"""
Tests for the risk manager and circuit breakers.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.config import RiskConfig
from src.risk.risk_manager import RiskManager
from src.strategy.inventory import InventoryManager
from src.utils.metrics import MetricsCollector


@pytest.fixture
def risk_cfg():
    return RiskConfig(
        max_total_usdc_in_orders=5000.0,
        max_drawdown_usdc=500.0,
        max_market_loss_usdc=100.0,
        circuit_breaker_cooldown_seconds=300,
        max_mid_move_60s=0.15,
        max_position_fraction=0.20,
    )


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def inv():
    return InventoryManager(max_inventory_per_market=500.0)


@pytest.fixture
def risk(risk_cfg, inv, metrics):
    return RiskManager(cfg=risk_cfg, inventory=inv, metrics=metrics)


class TestCanQuote:
    def test_clean_state_allows_quoting(self, risk):
        ok, reason = risk.can_quote("0xmarket1")
        assert ok is True
        assert reason == ""

    def test_global_circuit_breaker_blocks_all(self, risk):
        risk.trip_global_manual("test_trip")
        ok, reason = risk.can_quote("0xmarket1")
        assert ok is False
        assert "global_cb" in reason

    def test_market_circuit_breaker_blocks_market(self, risk):
        risk._trip_market("0xmarket1", "test_reason")
        ok, reason = risk.can_quote("0xmarket1")
        assert ok is False
        ok2, _ = risk.can_quote("0xmarket2")  # different market unaffected
        assert ok2 is True

    def test_max_usdc_in_orders_blocks(self, risk):
        risk.update_usdc_in_orders(6000.0)  # above 5000 limit
        ok, reason = risk.can_quote("0xmarket1")
        assert ok is False
        assert "usdc" in reason

    def test_drawdown_limit_trips_global_cb(self, risk, metrics):
        # Simulate a peak then a large loss exceeding max_drawdown_usdc=500
        # First record a profitable fill to establish a peak
        metrics.record_fill("0xmarket1", volume_usdc=100, realized_pnl=10.0)
        # Then record losses totalling > 510 from peak
        for _ in range(6):
            metrics.record_fill("0xmarket1", volume_usdc=100, realized_pnl=-100.0)
        # peak=10, current=10-600=-590, drawdown=10-(-590)=600 > 500
        assert metrics.current_drawdown >= 500
        ok, _ = risk.can_quote("0xmarket1")
        assert ok is False

    def test_circuit_breaker_resets_after_cooldown(self, risk):
        risk.trip_global_manual("test_trip")
        # Manually backdate the trip time
        risk._global_cb.tripped_at = time.monotonic() - 400  # past cooldown
        ok, _ = risk.can_quote("0xmarket1")
        assert ok is True
        assert risk._global_cb.tripped is False


class TestPriceVelocity:
    def test_slow_move_ok(self, risk):
        now = time.monotonic()
        # Move 5% over 60 seconds — below 15% limit
        for i in range(10):
            risk._mid_history.setdefault("0xm", []).append(
                (now - 60 + i * 6, 0.50 + i * 0.005)
            )
        assert not risk._is_price_moving_fast("0xm")

    def test_fast_move_triggers_cb(self, risk):
        now = time.monotonic()
        # Move 20% within the last 60 seconds — above 15% limit
        # Use 55s to stay safely within the 60s window
        risk._mid_history["0xm"] = [
            (now - 55, 0.30),
            (now - 1, 0.50),
        ]
        assert risk._is_price_moving_fast("0xm")

    def test_no_history_no_trigger(self, risk):
        assert not risk._is_price_moving_fast("0xunknown")

    def test_record_mid_price_stores_history(self, risk):
        risk.record_mid_price("0xm", 0.55)
        assert "0xm" in risk._mid_history
        assert len(risk._mid_history["0xm"]) == 1

    def test_old_history_pruned(self, risk):
        now = time.monotonic()
        risk._mid_history["0xm"] = [
            (now - 200, 0.50),  # old — should be pruned
            (now - 10, 0.50),   # recent
        ]
        risk.record_mid_price("0xm", 0.51)
        # After a check, old entries should be gone
        risk._is_price_moving_fast("0xm")
        recent = risk._mid_history["0xm"]
        assert all(t >= now - 120 for t, _ in recent)


class TestStatus:
    def test_status_returns_dict(self, risk):
        status = risk.status()
        assert "global_cb_tripped" in status
        assert "usdc_in_orders" in status
        assert "current_drawdown" in status

    def test_status_reflects_tripped_markets(self, risk):
        risk._trip_market("0xm1", "test")
        status = risk.status()
        assert "0xm1" in status["market_cbs_tripped"]
