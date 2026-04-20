"""
Tests for the inventory (position) manager.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.models import Fill, MarketSide, Side
from src.strategy.inventory import InventoryManager


@pytest.fixture
def inv():
    return InventoryManager(
        max_inventory_per_market=500.0,
        inventory_skew_threshold=100.0,
    )


@pytest.fixture
def inv_with_position(inv, sample_market):
    inv.ensure_position(
        sample_market.condition_id,
        sample_market.yes_token.token_id,
        sample_market.no_token.token_id,
    )
    return inv


class TestPositionTracking:
    def test_ensure_creates_position(self, inv, sample_market):
        pos = inv.ensure_position(
            sample_market.condition_id,
            sample_market.yes_token.token_id,
            sample_market.no_token.token_id,
        )
        assert pos.yes_shares == 0.0
        assert pos.no_shares == 0.0

    def test_buy_increases_yes_shares(self, inv_with_position, sample_market):
        fill = Fill(
            order_id="order1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.55,
            size=100.0,
        )
        inv_with_position.apply_fill(fill)
        pos = inv_with_position.get_position(sample_market.condition_id)
        assert pos.yes_shares == pytest.approx(100.0)

    def test_sell_decreases_yes_shares(self, inv_with_position, sample_market):
        # Buy first
        inv_with_position.apply_fill(Fill(
            order_id="o1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.55,
            size=100.0,
        ))
        # Then sell half
        inv_with_position.apply_fill(Fill(
            order_id="o2",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.SELL,
            price=0.60,
            size=50.0,
        ))
        pos = inv_with_position.get_position(sample_market.condition_id)
        assert pos.yes_shares == pytest.approx(50.0)

    def test_average_cost_buy_in(self, inv_with_position, sample_market):
        """Average cost tracks correctly across multiple buys."""
        inv_with_position.apply_fill(Fill(
            order_id="o1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.50,
            size=100.0,
        ))
        inv_with_position.apply_fill(Fill(
            order_id="o2",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.60,
            size=100.0,
        ))
        pos = inv_with_position.get_position(sample_market.condition_id)
        assert pos.avg_yes_cost == pytest.approx(0.55)
        assert pos.yes_shares == pytest.approx(200.0)

    def test_realized_pnl_on_sell(self, inv_with_position, sample_market):
        """Realised PnL = (sell_price - avg_cost) × shares."""
        inv_with_position.apply_fill(Fill(
            order_id="o1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.50,
            size=100.0,
        ))
        inv_with_position.apply_fill(Fill(
            order_id="o2",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.SELL,
            price=0.60,
            size=100.0,
        ))
        pos = inv_with_position.get_position(sample_market.condition_id)
        assert pos.realized_pnl == pytest.approx(10.0)  # 0.10 × 100 shares

    def test_sell_cannot_go_negative(self, inv_with_position, sample_market):
        """Position shouldn't go below zero."""
        inv_with_position.apply_fill(Fill(
            order_id="o1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.SELL,
            price=0.60,
            size=50.0,
        ))
        pos = inv_with_position.get_position(sample_market.condition_id)
        assert pos.yes_shares >= 0.0

    def test_fill_unknown_market_warns(self, inv, caplog):
        """Filling an unknown market logs a warning but doesn't crash."""
        import logging
        fill = Fill(
            order_id="o1",
            market_condition_id="unknown_market",
            token_id="token_x",
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.5,
            size=100.0,
        )
        # Should not raise
        inv.apply_fill(fill)


class TestInventorySkew:
    def test_zero_skew_at_no_position(self, inv_with_position, sample_market):
        skew = inv_with_position.inventory_skew(sample_market.condition_id)
        assert skew == 0.0

    def test_zero_skew_below_threshold(self, inv_with_position, sample_market):
        """Small positions within threshold → no skew."""
        inv_with_position.apply_fill(Fill(
            order_id="o1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.55,
            size=50.0,  # below 100 threshold
        ))
        skew = inv_with_position.inventory_skew(sample_market.condition_id)
        assert skew == 0.0

    def test_positive_skew_when_long(self, inv_with_position, sample_market):
        """Long YES position → positive skew."""
        inv_with_position.apply_fill(Fill(
            order_id="o1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.55,
            size=200.0,  # above threshold
        ))
        skew = inv_with_position.inventory_skew(sample_market.condition_id)
        assert skew > 0.0
        assert -1.0 <= skew <= 1.0

    def test_skew_clamped_to_pm_one(self, inv_with_position, sample_market):
        """Skew is always in [-1, 1]."""
        inv_with_position.apply_fill(Fill(
            order_id="o1",
            market_condition_id=sample_market.condition_id,
            token_id=sample_market.yes_token.token_id,
            outcome=MarketSide.YES,
            side=Side.BUY,
            price=0.55,
            size=10000.0,  # far exceeds max
        ))
        skew = inv_with_position.inventory_skew(sample_market.condition_id)
        assert -1.0 <= skew <= 1.0


class TestPersistence:
    def test_save_and_load(self, sample_market):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            inv = InventoryManager(state_file=state_file)
            inv.ensure_position(
                sample_market.condition_id,
                sample_market.yes_token.token_id,
                sample_market.no_token.token_id,
            )
            inv.apply_fill(Fill(
                order_id="o1",
                market_condition_id=sample_market.condition_id,
                token_id=sample_market.yes_token.token_id,
                outcome=MarketSide.YES,
                side=Side.BUY,
                price=0.55,
                size=100.0,
            ))
            inv.save()

            # Load fresh
            inv2 = InventoryManager(state_file=state_file)
            pos = inv2.get_position(sample_market.condition_id)
            assert pos is not None
            assert pos.yes_shares == pytest.approx(100.0)
            assert pos.avg_yes_cost == pytest.approx(0.55)
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_save_creates_directory(self, sample_market, tmp_path):
        nested_path = tmp_path / "nested" / "dir" / "state.json"
        inv = InventoryManager(state_file=str(nested_path))
        inv.ensure_position(
            sample_market.condition_id,
            sample_market.yes_token.token_id,
            sample_market.no_token.token_id,
        )
        inv.save()
        assert nested_path.exists()
