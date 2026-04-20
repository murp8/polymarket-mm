"""
Tests for configuration loading.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    Config,
    Credentials,
    ExchangeConfig,
    QuotingConfig,
    RiskConfig,
    load_config,
)


class TestDefaults:
    def test_default_chain_id(self):
        cfg = ExchangeConfig()
        assert cfg.chain_id == 137

    def test_default_quoting_spread(self):
        cfg = QuotingConfig()
        assert 0.0 < cfg.target_spread_fraction < 1.0

    def test_default_risk_limits_positive(self):
        cfg = RiskConfig()
        assert cfg.max_drawdown_usdc > 0
        assert cfg.max_market_loss_usdc > 0
        assert cfg.max_total_usdc_in_orders > 0


class TestLoadConfig:
    def test_loads_yaml_file(self, tmp_path):
        cfg_data = {
            "quoting": {"base_order_size": 75.0},
            "risk": {"max_drawdown_usdc": 250.0},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.quoting.base_order_size == pytest.approx(75.0)
        assert cfg.risk.max_drawdown_usdc == pytest.approx(250.0)

    def test_loads_with_missing_file(self):
        # Should not raise, just use defaults
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg is not None
        assert cfg.exchange.chain_id == 137

    def test_config_returns_config_instance(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("{}")
        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, Config)


class TestCredentials:
    def test_private_key_gets_0x_prefix(self, monkeypatch):
        monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "abcdef1234")
        monkeypatch.setenv("POLYMARKET_WALLET_ADDRESS", "0xwallet")
        creds = Credentials()
        assert creds.polymarket_private_key.startswith("0x")

    def test_private_key_with_prefix_unchanged(self, monkeypatch):
        monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabcdef1234")
        creds = Credentials()
        assert creds.polymarket_private_key == "0xabcdef1234"

    def test_has_polymarket_creds_true(self, monkeypatch):
        monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xkey")
        monkeypatch.setenv("POLYMARKET_WALLET_ADDRESS", "0xwallet")
        creds = Credentials()
        assert creds.has_polymarket_creds is True

    def test_has_polymarket_creds_false_when_missing(self, monkeypatch):
        # Set to empty strings so env vars take precedence over .env file values
        monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "")
        monkeypatch.setenv("POLYMARKET_WALLET_ADDRESS", "")
        creds = Credentials()
        assert creds.has_polymarket_creds is False

    def test_has_odds_api_key(self, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "my_key")
        creds = Credentials()
        assert creds.has_odds_api_key is True

    def test_no_odds_api_key(self, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "")
        creds = Credentials()
        assert creds.has_odds_api_key is False
