"""
Configuration loading and validation for the Polymarket market maker.

Settings are loaded from config/config.yaml and can be overridden by
environment variables using the POLYMARKET_MM__ prefix with double-underscores
as nested separators, e.g.:
    POLYMARKET_MM__QUOTING__BASE_ORDER_SIZE=100
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Sub-configs ──────────────────────────────────────────────────────────────


class ExchangeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__EXCHANGE__", extra="ignore")

    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    ws_market_host: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_user_host: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    chain_id: int = 137
    signature_type: int = 0


class MarketSelectionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__MARKET_SELECTION__", extra="ignore")

    max_markets: int = 30
    min_reward_pool_usd: float = 500.0
    min_midpoint: float = 0.06
    max_midpoint: float = 0.94
    require_clob_enabled: bool = True
    refresh_interval_seconds: int = 300


class QuotingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__QUOTING__", extra="ignore")

    target_spread_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    base_order_size: float = Field(default=200.0, gt=0)
    max_order_size: float = Field(default=500.0, gt=0)
    inventory_skew_threshold: float = Field(default=400.0, gt=0)
    max_inventory_per_market: float = Field(default=2000.0, gt=0)
    refresh_interval_seconds: int = Field(default=15, gt=0)
    requote_threshold: float = Field(default=0.005, ge=0.0)
    min_requote_interval_seconds: float = Field(default=2.0, ge=0.0)
    # ── poly-maker-style controls ──────────────────────────────────────────────
    # Minimum shares required at the best price before we consider it real liquidity
    min_liquidity_size: float = Field(default=50.0, gt=0)
    # Don't buy if bid_liquidity / ask_liquidity is below this (market selling off)
    liquidity_ratio_min: float = Field(default=0.5, ge=0.0)
    # Minimum profit % above avg cost before we'll place a sell order
    take_profit_pct: float = Field(default=3.0, ge=0.0)
    # Sell at market and pause buying if unrealized PnL % drops below this (negative)
    stop_loss_pct: float = Field(default=-15.0)
    # Hours to pause buying on a market after a stop-loss fires
    risk_off_hours: float = Field(default=4.0, gt=0)
    # Quote both YES and NO tokens (doubles reward surface)
    quote_no_token: bool = True


class OddsApiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__PRICING__ODDS_API__", extra="ignore")

    base_url: str = "https://api.the-odds-api.com/v4"
    bookmakers: list[str] = ["pinnacle"]
    fallback_bookmakers: list[str] = ["draftkings", "fanduel", "betmgm"]
    cache_ttl_seconds: int = 30
    regions: list[str] = ["eu", "us"]


class PricingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__PRICING__", extra="ignore")

    primary_source: Literal["odds_api", "orderbook"] = "odds_api"
    fallback_to_orderbook: bool = True
    max_price_age_seconds: int = 120
    vig_removal: Literal["basic", "shin", "power"] = "shin"
    odds_api: OddsApiConfig = Field(default_factory=OddsApiConfig)


class ExecutionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__EXECUTION__", extra="ignore")

    order_rate_limit: int = 5
    cancel_replace_rate_limit: int = 30
    max_order_retries: int = 3
    retry_delay_seconds: float = 1.0
    initial_paper_balance_usdc: float = 20_000.0
    # Fraction of starting capital to keep permanently free (never locked in orders).
    # Prevents the bot from running at 100% committed and leaving no room to manoeuvre.
    cash_reserve_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    # Maker fee in basis points. Polymarket CLOB charges 0% for limit-order makers,
    # so this defaults to 0. Set to a non-zero value to model conservative fee drag.
    paper_maker_fee_bps: int = Field(default=0, ge=0)


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__RISK__", extra="ignore")

    max_total_usdc_in_orders: float = 12_000.0
    max_drawdown_usdc: float = 2_000.0
    max_market_loss_usdc: float = 400.0
    circuit_breaker_cooldown_seconds: int = 300
    max_mid_move_60s: float = 0.15
    max_position_fraction: float = 0.15


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__LOGGING__", extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    file: str = "logs/polymarket-mm.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class MetricsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_MM__METRICS__", extra="ignore")

    summary_interval_seconds: int = 60       # live mode: log interval
    paper_dashboard_interval_seconds: int = 15  # paper mode: dashboard refresh
    state_file: str = "data/state.json"


# ─── Credentials (not in config.yaml) ────────────────────────────────────────


class Credentials(BaseSettings):
    """Sensitive credentials — always from environment variables, never YAML."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    polymarket_private_key: str = Field(default="", alias="POLYMARKET_PRIVATE_KEY")
    polymarket_wallet_address: str = Field(default="", alias="POLYMARKET_WALLET_ADDRESS")
    polymarket_funder_address: str = Field(default="", alias="POLYMARKET_FUNDER_ADDRESS")
    polymarket_signature_type: int = Field(default=0, alias="POLYMARKET_SIGNATURE_TYPE")
    odds_api_key: str = Field(default="", alias="ODDS_API_KEY")

    @field_validator("polymarket_private_key")
    @classmethod
    def validate_private_key(cls, v: str) -> str:
        if v and not v.startswith("0x"):
            return "0x" + v
        return v

    @property
    def has_polymarket_creds(self) -> bool:
        return bool(self.polymarket_private_key and self.polymarket_wallet_address)

    @property
    def has_odds_api_key(self) -> bool:
        return bool(self.odds_api_key)


# ─── Root Config ──────────────────────────────────────────────────────────────


class Config:
    """Root configuration container. Load once at startup."""

    def __init__(
        self,
        exchange: ExchangeConfig | None = None,
        market_selection: MarketSelectionConfig | None = None,
        quoting: QuotingConfig | None = None,
        pricing: PricingConfig | None = None,
        execution: ExecutionConfig | None = None,
        risk: RiskConfig | None = None,
        logging: LoggingConfig | None = None,
        metrics: MetricsConfig | None = None,
        credentials: Credentials | None = None,
    ):
        self.exchange = exchange or ExchangeConfig()
        self.market_selection = market_selection or MarketSelectionConfig()
        self.quoting = quoting or QuotingConfig()
        self.pricing = pricing or PricingConfig()
        self.execution = execution or ExecutionConfig()
        self.risk = risk or RiskConfig()
        self.logging = logging or LoggingConfig()
        self.metrics = metrics or MetricsConfig()
        self.credentials = credentials or Credentials()


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Load configuration from YAML file, then apply env-var overrides.

    Priority (highest to lowest):
      1. Environment variables (POLYMARKET_MM__SECTION__KEY=value)
      2. YAML config file
      3. Pydantic defaults
    """
    if config_path is None:
        # Walk up from CWD to find config/config.yaml
        cwd = Path.cwd()
        candidates = [
            cwd / "config" / "config.yaml",
            cwd.parent / "config" / "config.yaml",
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        config_path = next((p for p in candidates if p.exists()), None)

    yaml_data: dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    def _override_env(section: str, data: dict) -> dict:
        """Inject env vars with POLYMARKET_MM__SECTION__ prefix into dict."""
        prefix = f"POLYMARKET_MM__{section.upper()}__"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                field = key[len(prefix):].lower()
                data[field] = value
        return data

    exchange_data = _override_env("exchange", yaml_data.get("exchange", {}))
    market_selection_data = _override_env(
        "market_selection", yaml_data.get("market_selection", {})
    )
    quoting_data = _override_env("quoting", yaml_data.get("quoting", {}))
    pricing_raw = yaml_data.get("pricing", {})
    pricing_data = _override_env("pricing", {k: v for k, v in pricing_raw.items() if k != "odds_api"})
    odds_api_data = _override_env("pricing_odds_api", pricing_raw.get("odds_api", {}))
    execution_data = _override_env("execution", yaml_data.get("execution", {}))
    risk_data = _override_env("risk", yaml_data.get("risk", {}))
    logging_data = _override_env("logging", yaml_data.get("logging", {}))
    metrics_data = _override_env("metrics", yaml_data.get("metrics", {}))

    return Config(
        exchange=ExchangeConfig(**exchange_data),
        market_selection=MarketSelectionConfig(**market_selection_data),
        quoting=QuotingConfig(**quoting_data),
        pricing=PricingConfig(
            **pricing_data,
            odds_api=OddsApiConfig(**odds_api_data),
        ),
        execution=ExecutionConfig(**execution_data),
        risk=RiskConfig(**risk_data),
        logging=LoggingConfig(**logging_data),
        metrics=MetricsConfig(**metrics_data),
        credentials=Credentials(),
    )
