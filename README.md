# Polymarket Market Maker

A production-quality liquidity rewards market maker for [Polymarket](https://polymarket.com).

Earns rewards by posting two-sided resting limit orders on incentivized prediction markets. Uses Pinnacle (via The Odds API) as the primary fair-value source for sports markets and falls back to live orderbook midpoints for everything else.

---

## How it works

Polymarket pays liquidity rewards using a quadratic scoring formula:

```
S(v, s) = ((v - s) / v)² × b
```

Where **v** = max allowed spread, **s** = your actual distance from the midpoint, and **b** = your order size. Being tight earns quadratically more than being wide. Two-sided quotes (both YES buy and YES sell) get the full score; one-sided quotes are penalized 3×.

This bot:

1. **Discovers** all incentivized markets via the Gamma API
2. **Ranks** them by expected reward, volume, and spread generosity
3. **Prices** each market using Pinnacle odds (via The Odds API) where available, cross-validated against the live orderbook midpoint
4. **Quotes** tight two-sided orders calibrated to maximize reward score while managing inventory risk
5. **Manages fills** in real-time via WebSocket, updating positions and requoting as needed
6. **Protects capital** with drawdown limits, per-market loss limits, price velocity circuit breakers, and position size limits

---

## Architecture

```
MarketSelector ← Gamma API (5-min refresh)
      ↓
CompositePricer ← The Odds API / Pinnacle (30s cache)
      ↓           ← Orderbook WebSocket midpoint (real-time)
QuoteEngine  (scoring-optimal bid/ask + inventory skew)
      ↓
RiskManager  (circuit breakers, limits)
      ↓
OrderManager → CLOB API (place / cancel+replace)
      ↑
UserWebSocket  (fill events → InventoryManager)
MarketWebSocket (orderbook events → requote trigger)
```

Each component is independently testable. The WebSocket layer maintains a local SortedDict-backed order book for O(log n) incremental updates, the same approach used by [poly-maker](https://github.com/warproxxx/poly-maker).

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A funded Polymarket wallet (Polygon/USDC) that has completed at least one trade through the UI
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 2. Install

```bash
git clone <repo>
cd polymarket-mm
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
POLYMARKET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY
POLYMARKET_WALLET_ADDRESS=0xYOUR_WALLET_ADDRESS

# Optional but strongly recommended — get a free key at https://the-odds-api.com/
# Without it, the bot falls back to orderbook midpoints only
ODDS_API_KEY=YOUR_KEY
```

### 4. Tune strategy parameters

Edit `config/config.yaml`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `quoting.target_spread_fraction` | `0.25` | How far inside the max allowed spread to quote. Lower = tighter = higher score but more adverse selection. |
| `quoting.base_order_size` | `50 USDC` | Default order size per side. |
| `quoting.max_order_size` | `200 USDC` | Cap per order. |
| `risk.max_total_usdc_in_orders` | `5000` | Total USDC to deploy across all resting orders. |
| `risk.max_drawdown_usdc` | `500` | Global halt threshold. |
| `market_selection.max_markets` | `20` | Number of markets to quote simultaneously. |

### 5. Run

```bash
# Production
polymarket-mm

# Dry run (compute quotes, no orders placed)
polymarket-mm --dry-run

# With custom config
polymarket-mm --config path/to/config.yaml

# Verbose logging
polymarket-mm --log-level DEBUG
```

---

## Pricing: Odds API vs Orderbook

For sports markets (NBA, EPL, NFL, etc.), the bot fetches Pinnacle odds via [The Odds API](https://the-odds-api.com/) and removes the bookmaker's vig using the [Shin method](https://www.sciencedirect.com/science/article/pii/S1441352913000983) — a more accurate conversion than simple normalization, especially for extreme probabilities.

If the Pinnacle price diverges from the live Polymarket orderbook midpoint by more than 8%, the bot blends the two estimates and reduces its confidence score, causing the quote engine to widen spreads automatically.

For non-sports markets, the bot uses the WebSocket-maintained orderbook midpoint as fair value.

**The Odds API plans:**
- Free: 500 requests/month (enough for testing)
- $30/month: 20,000 requests (~6 requests/minute, sufficient for 20 concurrent markets)

---

## Reward scoring

The bot maximizes the Polymarket reward formula score by:

1. **Quoting inside `max_incentive_spread`** — orders outside this window score zero
2. **Keeping `target_spread_fraction` low** — tighter quotes score quadratically higher
3. **Maintaining two-sided depth** — single-sided orders score at 1/3 rate
4. **Sizing above `min_incentive_size`** — sub-minimum orders score zero

The `scoring.py` module is a pure Python implementation of the formula, fully unit-tested against the published spec.

---

## Risk management

Four independent circuit breakers:

| Trigger | Action |
|---|---|
| Realized PnL drawdown > `max_drawdown_usdc` | Global halt, all orders cancelled |
| Per-market loss > `max_market_loss_usdc` | That market paused for `cooldown_seconds` |
| Mid-price moves > `max_mid_move_60s` in 60s | That market paused |
| Total USDC in orders > `max_total_usdc_in_orders` | New quotes blocked |

Circuit breakers self-heal after `circuit_breaker_cooldown_seconds` (default: 5 minutes). The global breaker is also tripped manually on SIGINT/SIGTERM, ensuring clean shutdown.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run a specific module
pytest tests/test_scoring.py -v
```

191 tests covering all core modules. The CLOB client and WebSocket connections are mocked; no live exchange calls are made during tests.

---

## Project structure

```
polymarket-mm/
├── config/
│   └── config.yaml          # All strategy and risk parameters
├── src/
│   ├── client/
│   │   ├── gamma.py          # Gamma API: market discovery
│   │   ├── polymarket.py     # CLOB API: orders, positions, balances
│   │   └── websocket_client.py  # Real-time orderbook + fill feeds
│   ├── pricing/
│   │   ├── odds_api.py       # Pinnacle odds via The Odds API
│   │   ├── orderbook.py      # Fallback: live orderbook midpoint
│   │   ├── composite.py      # Cross-validates and blends sources
│   │   └── vig.py            # Shin / basic / power vig removal
│   ├── strategy/
│   │   ├── quote_engine.py   # Bid/ask computation + requote logic
│   │   ├── scoring.py        # Reward formula implementation
│   │   └── inventory.py      # Position tracking + skew signals
│   ├── execution/
│   │   ├── order_manager.py  # Full order lifecycle (place / cancel / fill)
│   │   └── rate_limiter.py   # Token-bucket rate limiter
│   ├── risk/
│   │   └── risk_manager.py   # Circuit breakers + limit checks
│   ├── market_selector/
│   │   └── selector.py       # Ranks and filters incentivized markets
│   ├── utils/
│   │   ├── logging.py        # Structured JSON / console logging
│   │   └── metrics.py        # In-process PnL and performance tracking
│   ├── config.py             # Config loading with env-var overrides
│   ├── models.py             # Shared data models (Market, Order, Fill, …)
│   └── main.py               # Top-level Bot + CLI entry point
├── tests/                    # 191 pytest tests
├── .env.example
└── pyproject.toml
```

---

## Key references

- [Polymarket Liquidity Rewards Docs](https://docs.polymarket.com/market-makers/liquidity-rewards)
- [Kalshi's New Liquidity Incentives](https://fiftycentdollars.substack.com/p/kalshis-new-liquidity-incentives) — strategy context
- [py-clob-client](https://github.com/Polymarket/py-clob-client) — official Polymarket Python SDK
- [poly-maker](https://github.com/warproxxx/poly-maker) — reference implementation this bot was informed by
- [The Odds API](https://the-odds-api.com/) — Pinnacle odds aggregator
- [Shin (1993)](https://www.jstor.org/stable/2234631) — vig removal method
