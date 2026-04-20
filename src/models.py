"""
Shared data models for the Polymarket market maker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ─── Enums ────────────────────────────────────────────────────────────────────


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    GTC = "GTC"   # Good-Till-Cancel
    GTD = "GTD"   # Good-Till-Date
    FOK = "FOK"   # Fill-or-Kill


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    MATCHED = "MATCHED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class MarketSide(str, Enum):
    YES = "YES"
    NO = "NO"


# ─── Market Models ────────────────────────────────────────────────────────────


@dataclass
class TokenInfo:
    """One leg of a binary Polymarket market (YES or NO)."""
    token_id: str
    outcome: MarketSide
    price: float = 0.0  # Last known mid-price


@dataclass
class IncentiveParams:
    """Reward parameters fetched from the CLOB/Gamma API."""
    min_incentive_size: float   # Minimum order size to qualify for rewards
    max_incentive_spread: float  # Maximum spread (from mid) for reward eligibility
    reward_epoch_amount: float  # Total USDC rewards allocated for this epoch
    in_game_multiplier: float = 1.0  # Live/in-game multiplier (b in S(v,s) formula)


@dataclass
class Market:
    """Full market descriptor including both outcome tokens and reward params."""
    condition_id: str
    question_id: str
    question: str
    yes_token: TokenInfo
    no_token: TokenInfo
    incentive: IncentiveParams
    # Gamma metadata
    volume_24h: float = 0.0
    volume_total: float = 0.0
    liquidity: float = 0.0
    end_date_iso: str = ""
    sport: str = ""
    # Tags (e.g., "nba", "epl", "nfl")
    tags: list[str] = field(default_factory=list)
    # Computed fair-value probability (YES wins)
    fair_value: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def tokens(self) -> tuple[TokenInfo, TokenInfo]:
        return self.yes_token, self.no_token

    def token_by_id(self, token_id: str) -> Optional[TokenInfo]:
        if self.yes_token.token_id == token_id:
            return self.yes_token
        if self.no_token.token_id == token_id:
            return self.no_token
        return None


# ─── Orderbook ────────────────────────────────────────────────────────────────


@dataclass
class PriceLevel:
    price: float
    size: float


@dataclass
class Orderbook:
    token_id: str
    bids: list[PriceLevel]  # sorted descending by price
    asks: list[PriceLevel]  # sorted ascending by price
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return bb
        if ba is not None:
            return ba
        return None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None


# ─── Order Models ─────────────────────────────────────────────────────────────


@dataclass
class Order:
    """A single resting limit order we've placed."""
    order_id: str
    market_condition_id: str
    token_id: str
    outcome: MarketSide
    side: Side
    price: float
    size: float
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN)


@dataclass
class QuoteTarget:
    """
    Desired quote for one side of one market.
    The quote engine produces these; the order manager executes them.
    """
    market_condition_id: str
    token_id: str
    outcome: MarketSide
    side: Side
    price: float
    size: float
    # The reward score this quote would earn (for logging/analytics)
    expected_score: float = 0.0


@dataclass
class MarketQuote:
    """Both sides of a two-sided quote for one market (YES and optionally NO token)."""
    market_condition_id: str
    yes_bid: QuoteTarget    # Buy YES = long the outcome
    yes_ask: QuoteTarget    # Sell YES = short the outcome
    # Optional NO-token quotes (quoting both tokens earns double rewards)
    no_bid: Optional[QuoteTarget] = None   # Buy NO
    no_ask: Optional[QuoteTarget] = None   # Sell NO
    fair_value: float = 0.0
    half_spread: float = 0.0


# ─── Position / Inventory ─────────────────────────────────────────────────────


@dataclass
class Position:
    """Current position in a single market."""
    market_condition_id: str
    yes_token_id: str
    no_token_id: str
    # Net YES shares (positive = long YES, negative = effectively long NO)
    yes_shares: float = 0.0
    # Net NO shares
    no_shares: float = 0.0
    # Average cost basis for YES position (USDC per share)
    avg_yes_cost: float = 0.0
    avg_no_cost: float = 0.0
    # Realized PnL (USDC)
    realized_pnl: float = 0.0

    @property
    def net_yes_position(self) -> float:
        """YES position expressed as net directional exposure."""
        return self.yes_shares - self.no_shares

    def unrealized_pnl(self, yes_mid: float) -> float:
        """Compute mark-to-market PnL given current YES midpoint."""
        yes_pnl = self.yes_shares * (yes_mid - self.avg_yes_cost)
        no_pnl = self.no_shares * ((1.0 - yes_mid) - self.avg_no_cost)
        return yes_pnl + no_pnl


# ─── Price Source ─────────────────────────────────────────────────────────────


@dataclass
class PriceEstimate:
    """A fair-value estimate for a YES outcome probability."""
    probability: float       # 0–1
    source: str              # "odds_api", "orderbook", "last_trade", etc.
    confidence: float = 1.0  # 0–1, lower if interpolated or stale
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_odds: Optional[dict] = None

    @property
    def no_probability(self) -> float:
        return 1.0 - self.probability


# ─── Trade / Fill ─────────────────────────────────────────────────────────────


@dataclass
class Fill:
    """A partial or full fill of one of our orders."""
    order_id: str
    market_condition_id: str
    token_id: str
    outcome: MarketSide
    side: Side
    price: float
    size: float
    fee_rate_bps: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def fee_usdc(self) -> float:
        return self.price * self.size * self.fee_rate_bps / 10000.0

    @property
    def gross_usdc(self) -> float:
        return self.price * self.size
