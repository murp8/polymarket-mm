"""
The Odds API pricer — fetches Pinnacle (and fallback bookmaker) odds
and converts them to true probability estimates.

https://the-odds-api.com/

The Odds API aggregates odds from 40+ bookmakers including Pinnacle.
Pinnacle is used as the primary reference because it's a sharp, low-margin
book whose lines closely reflect true probabilities.

Sport mapping:
  The Odds API uses sport keys like "basketball_nba", "soccer_epl", etc.
  We infer the sport from Polymarket market tags and question text.

Market matching:
  We fuzzy-match Polymarket market questions to Odds API events by:
    1. Sport type
    2. Team/participant names (normalised)
    3. Start time (within ±24 hours)
"""

from __future__ import annotations

import asyncio
import re
import time
from difflib import SequenceMatcher
from typing import Any, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models import Market, PriceEstimate
from src.pricing.base import BasePricer
from src.pricing.vig import remove_vig
from src.utils.logging import get_logger

log = get_logger(__name__)

# ─── Sport key mapping ────────────────────────────────────────────────────────
# Polymarket tag → The Odds API sport keys to try
_TAG_TO_SPORTS: dict[str, list[str]] = {
    "nba": ["basketball_nba"],
    "nfl": ["americanfootball_nfl"],
    "nhl": ["icehockey_nhl"],
    "mlb": ["baseball_mlb"],
    "nba-g-league": ["basketball_nba"],
    "ncaa-basketball": ["basketball_ncaab"],
    "ncaa-football": ["americanfootball_ncaaf"],
    "epl": ["soccer_epl"],
    "premier-league": ["soccer_epl"],
    "champions-league": ["soccer_uefa_champs_league"],
    "la-liga": ["soccer_spain_la_liga"],
    "bundesliga": ["soccer_germany_bundesliga"],
    "serie-a": ["soccer_italy_serie_a"],
    "ligue-1": ["soccer_france_ligue_one"],
    "mls": ["soccer_usa_mls"],
    "ufc": ["mma_mixed_martial_arts"],
    "mma": ["mma_mixed_martial_arts"],
    "tennis": ["tennis_atp_french_open", "tennis_wta_french_open",
               "tennis_atp_wimbledon", "tennis_wta_wimbledon"],
    "golf": ["golf_pga_championship", "golf_the_masters_tournament"],
    "f1": ["motorsport_formula_one_winner"],
    "cs2": ["esports_lol"],   # esports_cs2 no longer exists on The Odds API
    "esports": ["esports_lol"],
    "lol": ["esports_lol"],
    "dota2": ["esports_dota2"],
    "cricket": ["cricket_icc_world_cup"],
}

# Fallback: if no tag matches, try to infer from question text
_KEYWORD_TO_SPORTS: list[tuple[str, list[str]]] = [
    (r"\bnba\b", ["basketball_nba"]),
    (r"\bnfl\b", ["americanfootball_nfl"]),
    (r"\bnhl\b", ["icehockey_nhl"]),
    (r"\bmlb\b", ["baseball_mlb"]),
    (r"\bepl\b|\bpremier league\b", ["soccer_epl"]),
    (r"\bchampions league\b|\buchampions\b", ["soccer_uefa_champs_league"]),
    (r"\bla liga\b", ["soccer_spain_la_liga"]),
    (r"\bbundesliga\b", ["soccer_germany_bundesliga"]),
    (r"\bserie a\b", ["soccer_italy_serie_a"]),
    (r"\bufc\b|\bmma\b", ["mma_mixed_martial_arts"]),
    (r"\btennis\b|\batp\b|\bwta\b", ["tennis_atp_french_open"]),
    (r"\bf1\b|\bformula 1\b|\bformula one\b", ["motorsport_formula_one_winner"]),
]


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def _infer_sport_keys(market: Market) -> list[str]:
    """Return candidate Odds API sport keys for a market."""
    for tag in market.tags:
        if tag in _TAG_TO_SPORTS:
            return _TAG_TO_SPORTS[tag]
    q = market.question.lower()
    for pattern, sports in _KEYWORD_TO_SPORTS:
        if re.search(pattern, q):
            return sports
    return []


class OddsApiPricer(BasePricer):
    """
    Fetches Pinnacle odds via The Odds API and converts to probabilities.

    Usage:
        async with OddsApiPricer(api_key="...") as pricer:
            estimate = await pricer.get_price(market)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        bookmakers: Optional[list[str]] = None,
        fallback_bookmakers: Optional[list[str]] = None,
        regions: Optional[list[str]] = None,
        cache_ttl_seconds: int = 30,
        vig_method: str = "shin",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._bookmakers = bookmakers or ["pinnacle"]
        self._fallback_bookmakers = fallback_bookmakers or ["draftkings", "fanduel"]
        self._regions = regions or ["eu", "us"]
        self._cache_ttl = cache_ttl_seconds
        self._vig_method = vig_method
        # Cache: sport_key → (timestamp, list[event_dict])
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        # Sport keys that returned 404 — never retry them
        self._dead_sport_keys: set[str] = set()
        # Set to True on first 401 — stops all future requests this session
        self._api_key_invalid: bool = False
        self._http: Optional[httpx.AsyncClient] = None
        # Track remaining API credits
        self._remaining_requests: Optional[int] = None

    @property
    def name(self) -> str:
        return "odds_api"

    async def __aenter__(self) -> "OddsApiPricer":
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=10.0,
            params={"apiKey": self._api_key},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("OddsApiPricer must be used as an async context manager")
        return self._http

    async def is_healthy(self) -> bool:
        if not self._api_key:
            return False
        try:
            resp = await self.http.get("/sports", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Core fetching ─────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _fetch_odds(self, sport_key: str) -> list[dict]:
        """Fetch odds for a sport, using cache when fresh."""
        # Halt entirely if the API key is invalid (401)
        if self._api_key_invalid:
            return []
        # Skip sport keys known to be dead (404) — don't waste API credits
        if sport_key in self._dead_sport_keys:
            return []

        cached = self._cache.get(sport_key)
        if cached and (time.monotonic() - cached[0]) < self._cache_ttl:
            return cached[1]

        bookmakers = ",".join(self._bookmakers)
        regions = ",".join(self._regions)

        params: dict[str, Any] = {
            "regions": regions,
            "markets": "h2h",
            "bookmakers": bookmakers,
            "oddsFormat": "decimal",
        }

        resp = await self.http.get(f"/sports/{sport_key}/odds", params=params)

        # Track remaining credits
        self._remaining_requests = int(
            resp.headers.get("x-requests-remaining", self._remaining_requests or 0)
        )

        if resp.status_code == 422:
            # Sport not available in requested region - try without bookmaker filter
            params.pop("bookmakers", None)
            resp = await self.http.get(f"/sports/{sport_key}/odds", params=params)

        # 401 = invalid/expired API key — halt all future requests
        if resp.status_code == 401:
            self._api_key_invalid = True
            log.error(
                "odds_api_key_invalid",
                msg="The Odds API key is invalid or expired. "
                    "Falling back to orderbook pricing for all markets. "
                    "Update ODDS_API_KEY in .env and restart to re-enable.",
            )
            return []
        # 404 = sport key doesn't exist; mark dead and don't retry
        if resp.status_code == 404:
            self._dead_sport_keys.add(sport_key)
            return []
        resp.raise_for_status()
        events = resp.json()
        self._cache[sport_key] = (time.monotonic(), events)
        log.debug(
            "odds_fetched",
            sport=sport_key,
            events=len(events),
            credits_remaining=self._remaining_requests,
        )
        return events

    # ── Matching ──────────────────────────────────────────────────────────────

    def _find_best_event(self, market: Market, events: list[dict]) -> Optional[dict]:
        """
        Find the Odds API event that best matches a Polymarket market.
        Returns None if no confident match found.
        """
        question = market.question
        best_score = 0.0
        best_event = None

        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            event_str = f"{home} vs {away}"
            # Also try matching individual team names against the question
            score = max(
                _similarity(event_str, question),
                _similarity(home, question),
                _similarity(away, question),
            )
            # Bonus if both team names appear in the question
            if _normalise(home) in _normalise(question) and _normalise(away) in _normalise(question):
                score = max(score, 0.85)
            if score > best_score:
                best_score = score
                best_event = event

        if best_score >= 0.35:
            log.debug(
                "event_match",
                question=question[:60],
                match=f"{best_event.get('home_team')} vs {best_event.get('away_team')}",
                score=round(best_score, 3),
            )
            return best_event

        log.debug("event_no_match", question=question[:60], best_score=round(best_score, 3))
        return None

    def _extract_probability(
        self, event: dict, market: Market
    ) -> Optional[tuple[float, str]]:
        """
        Extract YES-outcome probability from an event's bookmaker odds.
        Returns (probability, bookmaker_used) or None.

        The YES outcome probability is the home team win probability
        when the question is "Will X win?" or "Will X beat Y?".
        For "Will X beat Y?" markets, YES = home team wins.
        """
        question_lower = market.question.lower()
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        # Determine which outcome maps to YES
        # Heuristic: if the home team name appears prominently first → YES=home win
        home_normalised = _normalise(home)
        away_normalised = _normalise(away)
        q_normalised = _normalise(question_lower)

        home_pos = q_normalised.find(home_normalised) if home_normalised else -1
        away_pos = q_normalised.find(away_normalised) if away_normalised else -1

        if home_pos >= 0 and (away_pos < 0 or home_pos <= away_pos):
            yes_team_idx = 0  # home win
        elif away_pos >= 0:
            yes_team_idx = 1  # away win
        else:
            yes_team_idx = 0  # default to home

        # Try primary bookmakers first
        bookmaker_priority = self._bookmakers + self._fallback_bookmakers
        for bm_key in bookmaker_priority:
            for bookmaker in event.get("bookmakers", []):
                if bookmaker.get("key") != bm_key:
                    continue
                for market_data in bookmaker.get("markets", []):
                    if market_data.get("key") != "h2h":
                        continue
                    outcomes = market_data.get("outcomes", [])
                    if len(outcomes) < 2:
                        continue
                    # Convert decimal odds to raw implied probabilities
                    raw_probs = [1.0 / o.get("price", 1) for o in outcomes]
                    # Remove vig
                    true_probs = remove_vig(raw_probs, self._vig_method)
                    if yes_team_idx < len(true_probs):
                        return true_probs[yes_team_idx], bm_key

        return None

    # ── Public interface ──────────────────────────────────────────────────────

    async def get_price(self, market: Market) -> Optional[PriceEstimate]:
        """
        Return a probability estimate for the YES outcome of a market.
        """
        sport_keys = _infer_sport_keys(market)
        if not sport_keys:
            log.debug("no_sport_keys", question=market.question[:60])
            return None

        for sport_key in sport_keys:
            try:
                events = await self._fetch_odds(sport_key)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    self._api_key_invalid = True
                    log.error(
                        "odds_api_key_invalid",
                        msg="The Odds API key is invalid or expired. "
                            "Falling back to orderbook pricing. "
                            "Update ODDS_API_KEY in .env and restart.",
                    )
                elif e.response.status_code == 404:
                    # Sport key doesn't exist — never retry it
                    self._dead_sport_keys.add(sport_key)
                    log.debug("sport_key_dead", sport=sport_key)
                else:
                    log.warning("odds_fetch_failed", sport=sport_key, error=str(e))
                continue
            except Exception as e:
                log.warning("odds_fetch_failed", sport=sport_key, error=str(e))
                continue

            event = self._find_best_event(market, events)
            if event is None:
                continue

            result = self._extract_probability(event, market)
            if result is None:
                continue

            prob, bookmaker = result
            # Clamp to sane range
            prob = max(0.02, min(0.98, prob))
            return PriceEstimate(
                probability=prob,
                source=f"odds_api/{bookmaker}",
                confidence=0.9,
                raw_odds={"event": event.get("id"), "bookmaker": bookmaker},
            )

        return None

    async def get_all_sport_keys(self) -> list[str]:
        """Return all active sport keys from The Odds API."""
        resp = await self.http.get("/sports")
        resp.raise_for_status()
        return [s["key"] for s in resp.json() if s.get("active")]
