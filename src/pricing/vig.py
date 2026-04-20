"""
Vig (overround) removal methods for converting raw bookmaker odds to
true probability estimates.

Three methods implemented:
  - basic:  proportional normalization (remove vig by scaling)
  - shin:   Shin (1993) method — assumes insider trading model, more accurate
  - power:  power method — better for extreme probabilities

References:
  - Shin (1993) "Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims"
  - Clarke (2016) "Adjusting Bookmaker's Odds for the Favourite-Longshot Bias"
"""

from __future__ import annotations

import math
from typing import Sequence


def remove_vig_basic(raw_probs: Sequence[float]) -> list[float]:
    """
    Proportional normalization: divide each probability by the sum.
    Fast but slightly biased against longshots.
    """
    total = sum(raw_probs)
    if total <= 0:
        n = len(raw_probs)
        return [1.0 / n] * n if n > 0 else []
    return [p / total for p in raw_probs]


def remove_vig_shin(raw_probs: Sequence[float]) -> list[float]:
    """
    Shin method: iterative solver that estimates the fraction of insider
    trading (z) and adjusts accordingly.

    For a two-outcome market this has a closed-form solution:
      p_true = (sqrt(z² + 4(1-z)·q_i·S) - z) / (2(1-z))
    where S = sum of raw_probs, q_i = raw_prob_i / S

    For n>2 outcomes, we iterate until convergence.
    """
    probs = list(raw_probs)
    n = len(probs)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    S = sum(probs)
    if S <= 0:
        return [1.0 / n] * n

    # Normalise so they sum to S
    q = [p / S for p in probs]

    if n == 2:
        # Closed-form for binary markets
        # z* from: sum_i [ sqrt(z^2 + 4(1-z)*q_i*S) - z ] / (2(1-z)) = 1
        # Binary closed form:
        disc = S * S - 4 * S * (S - 1) * q[0] * q[1]
        z = (S - math.sqrt(max(disc, 0))) / (2 * S - 2) if S != 1 else 0.0
        z = max(0.0, min(z, 0.5))
        adjusted = []
        for qi in q:
            d = z * z + 4 * (1 - z) * qi * S
            val = (math.sqrt(max(d, 0)) - z) / (2 * (1 - z)) if z != 1 else qi
            adjusted.append(val)
        total = sum(adjusted)
        return [a / total for a in adjusted] if total > 0 else remove_vig_basic(probs)

    # General case: iterative
    z = (S - 1) / (S - n)  # initial estimate
    z = max(0.0, min(z, 0.99))
    for _ in range(100):
        adjusted = []
        for qi in q:
            d = z * z + 4 * (1 - z) * qi * S
            val = (math.sqrt(max(d, 0)) - z) / (2 * (1 - z) + 1e-12)
            adjusted.append(val)
        total = sum(adjusted)
        new_z = z - (total - 1) * 0.5
        new_z = max(0.0, min(new_z, 0.99))
        if abs(new_z - z) < 1e-9:
            break
        z = new_z

    total = sum(adjusted)
    return [a / total for a in adjusted] if total > 0 else remove_vig_basic(probs)


def remove_vig_power(raw_probs: Sequence[float]) -> list[float]:
    """
    Power method: find exponent k such that sum(p_i^(1/k)) == 1 after normalisation,
    where true_p_i = p_i^(1/k) / sum(p_j^(1/k)).
    Better than basic for extreme probabilities.
    """
    probs = list(raw_probs)
    S = sum(probs)
    if S <= 0 or len(probs) == 0:
        n = len(probs)
        return [1.0 / n] * n

    # Binary search for k in [1, 100]
    lo, hi = 1.0, 200.0
    for _ in range(100):
        k = (lo + hi) / 2
        scaled = [p ** (1.0 / k) for p in probs]
        total = sum(scaled)
        if total > 1.0:
            lo = k
        else:
            hi = k
        if hi - lo < 1e-9:
            break

    k = (lo + hi) / 2
    scaled = [p ** (1.0 / k) for p in probs]
    total = sum(scaled)
    return [s / total for s in scaled] if total > 0 else remove_vig_basic(probs)


def remove_vig(
    raw_probs: Sequence[float],
    method: str = "shin",
) -> list[float]:
    """Dispatch to the requested vig-removal method."""
    if method == "shin":
        return remove_vig_shin(raw_probs)
    if method == "power":
        return remove_vig_power(raw_probs)
    return remove_vig_basic(raw_probs)
