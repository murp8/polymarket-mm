"""
Tests for vig removal methods.
"""

from __future__ import annotations

import pytest

from src.pricing.vig import remove_vig, remove_vig_basic, remove_vig_power, remove_vig_shin


class TestBasicVig:
    def test_sums_to_one(self):
        result = remove_vig_basic([0.55, 0.50])
        assert sum(result) == pytest.approx(1.0)

    def test_preserves_relative_order(self):
        result = remove_vig_basic([0.6, 0.5])
        assert result[0] > result[1]

    def test_fair_odds_unchanged(self):
        result = remove_vig_basic([0.5, 0.5])
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.5)

    def test_empty_returns_empty(self):
        assert remove_vig_basic([]) == []

    def test_single_returns_one(self):
        assert remove_vig_basic([0.7]) == [1.0]

    def test_zero_total_uniform(self):
        result = remove_vig_basic([0.0, 0.0])
        assert result == [0.5, 0.5]


class TestShinVig:
    def test_sums_to_one_binary(self):
        result = remove_vig_shin([0.55, 0.50])
        assert sum(result) == pytest.approx(1.0, abs=1e-6)

    def test_sums_to_one_ternary(self):
        result = remove_vig_shin([0.45, 0.35, 0.30])
        assert sum(result) == pytest.approx(1.0, abs=1e-6)

    def test_favourite_longshot_bias(self):
        """Shin method gives more probability to longshots vs basic."""
        raw = [0.70, 0.35]  # vig ~5%, favourite at 70%
        basic = remove_vig_basic(raw)
        shin = remove_vig_shin(raw)
        # Longshot (second team) should get relatively more in Shin
        assert shin[1] >= basic[1]

    def test_fair_market_unchanged(self):
        """No vig → output equals input (normalised)."""
        result = remove_vig_shin([0.5, 0.5])
        assert result[0] == pytest.approx(0.5, abs=0.01)
        assert result[1] == pytest.approx(0.5, abs=0.01)

    def test_preserves_order(self):
        result = remove_vig_shin([0.65, 0.40])
        assert result[0] > result[1]

    def test_three_outcomes(self):
        result = remove_vig_shin([0.40, 0.35, 0.30])
        assert sum(result) == pytest.approx(1.0, abs=1e-6)
        assert all(0 < p < 1 for p in result)

    def test_all_values_between_zero_and_one(self):
        for probs in [[0.55, 0.50], [0.8, 0.3], [0.4, 0.35, 0.3]]:
            result = remove_vig_shin(probs)
            assert all(0.0 <= p <= 1.0 for p in result), f"Out of range for {probs}: {result}"


class TestPowerVig:
    def test_sums_to_one(self):
        result = remove_vig_power([0.55, 0.50])
        assert sum(result) == pytest.approx(1.0, abs=1e-6)

    def test_preserves_order(self):
        result = remove_vig_power([0.60, 0.45])
        assert result[0] > result[1]

    def test_all_values_valid(self):
        result = remove_vig_power([0.55, 0.50])
        assert all(0 <= p <= 1 for p in result)

    def test_three_outcomes(self):
        result = remove_vig_power([0.50, 0.40, 0.30])
        assert sum(result) == pytest.approx(1.0, abs=1e-6)


class TestDispatch:
    def test_shin_dispatched(self):
        r1 = remove_vig([0.55, 0.50], "shin")
        r2 = remove_vig_shin([0.55, 0.50])
        assert r1 == pytest.approx(r2, abs=1e-9)

    def test_basic_dispatched(self):
        r1 = remove_vig([0.55, 0.50], "basic")
        r2 = remove_vig_basic([0.55, 0.50])
        assert r1 == pytest.approx(r2, abs=1e-9)

    def test_power_dispatched(self):
        r1 = remove_vig([0.55, 0.50], "power")
        r2 = remove_vig_power([0.55, 0.50])
        assert r1 == pytest.approx(r2, abs=1e-9)

    def test_unknown_falls_back_to_basic(self):
        r1 = remove_vig([0.55, 0.50], "unknown_method")
        r2 = remove_vig_basic([0.55, 0.50])
        assert r1 == pytest.approx(r2)
