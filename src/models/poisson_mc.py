"""Poisson-based simulation helpers used by the value engine.

The original project relied heavily on vectorised NumPy routines.  To
keep the kata lightweight we provide small, dependency-friendly
implementations with optional NumPy acceleration when the library is
available.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

try:  # Optional NumPy acceleration.
    import numpy as _np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when numpy unavailable
    _np = None  # type: ignore


def _poisson_sample(lam: float, rng: random.Random) -> int:
    """Generate a single Poisson-distributed sample using the Knuth method."""

    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def expected_goals_from_strengths(
    home_attack: float,
    away_defence: float,
    away_attack: float,
    home_defence: float,
    league_avg: float,
    home_adv: float,
) -> Tuple[float, float]:
    """Blend attack/defence strengths into expected goal values.

    The formula mirrors a classic multiplicative Poisson model where
    league averages provide the baseline scoring rate.  Inputs are
    normalised so that values around ``1.0`` represent average strength.
    """

    base = max(float(league_avg), 1e-6)
    h_adv = max(float(home_adv), 0.0) or 1.0

    lam_home = base * max(home_attack, 0.0) * max(away_defence, 0.0) * h_adv
    lam_away = base * max(away_attack, 0.0) * max(home_defence, 0.0)

    return max(lam_home, 0.01), max(lam_away, 0.01)


def simulate_goals(lam_home: float, lam_away: float, n_sims: int = 20000) -> Tuple[List[int], List[int], List[int]]:
    """Monte-Carlo draw of score lines for the provided Poisson means."""

    n = max(int(n_sims), 0)
    if n == 0:
        return [], [], []

    if _np is not None:
        home = _np.random.poisson(lam_home, size=n).astype(int)  # type: ignore[attr-defined]
        away = _np.random.poisson(lam_away, size=n).astype(int)  # type: ignore[attr-defined]
        totals = (home + away).astype(int)
        return home.tolist(), away.tolist(), totals.tolist()

    rng = random.Random()
    home_samples: List[int] = []
    away_samples: List[int] = []
    totals: List[int] = []
    for _ in range(n):
        h = _poisson_sample(lam_home, rng)
        a = _poisson_sample(lam_away, rng)
        home_samples.append(h)
        away_samples.append(a)
        totals.append(h + a)
    return home_samples, away_samples, totals


def _ou_probabilities(totals: List[int], line: float) -> Tuple[float, float]:
    if not totals:
        return 0.0, 0.0
    over_count = 0
    under_count = 0
    for total in totals:
        if total > line:
            over_count += 1
        elif total < line:
            under_count += 1
        else:
            # Pushes do not contribute to either bucket; probabilities are
            # normalised after the loop.
            pass
    n = len(totals)
    return over_count / n, under_count / n


def monte_carlo_simulate(
    lam_home: float,
    lam_away: float,
    n_sims: int = 20000,
    over_line: float = 2.5,
) -> Dict[str, float]:
    """Return match outcome probabilities via Monte-Carlo sampling."""

    home, away, totals = simulate_goals(lam_home, lam_away, n_sims)
    n = len(home)
    if n == 0:
        return {
            "p_home": 0.0,
            "p_draw": 0.0,
            "p_away": 0.0,
            "p_over": 0.0,
            "p_under": 0.0,
        }

    wins = draws = 0
    for h, a in zip(home, away):
        if h > a:
            wins += 1
        elif h == a:
            draws += 1
    p_home = wins / n
    p_draw = draws / n
    p_away = 1.0 - p_home - p_draw

    p_over, p_under = _ou_probabilities(totals, over_line)

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over": p_over,
        "p_under": p_under,
    }


__all__ = [
    "expected_goals_from_strengths",
    "simulate_goals",
    "monte_carlo_simulate",
]
