"""Poisson-based goal simulation helpers used by the daily brief pipeline."""
from __future__ import annotations

from typing import Dict

import numpy as np


def _clamp_lambda(value: float) -> float:
    return max(1e-6, float(value))


def expected_goals_from_strengths(
    home_attack: float,
    away_defence: float,
    away_attack: float,
    home_defence: float,
    league_avg: float,
    home_adv: float,
) -> tuple[float, float]:
    """Convert strength ratios into Poisson goal expectations."""

    league = max(float(league_avg), 1e-6)
    adv = max(-0.5, min(0.5, float(home_adv)))

    home_base = max(float(home_attack), 0.01) * max(float(away_defence), 0.01)
    away_base = max(float(away_attack), 0.01) * max(float(home_defence), 0.01)

    lam_home = league * home_base * (1.0 + adv)
    lam_away = league * away_base * max(0.1, 1.0 - adv)

    return float(max(lam_home, 1e-4)), float(max(lam_away, 1e-4))


def simulate_goals(
    lam_home: float,
    lam_away: float,
    *,
    n_sims: int = 10000,
    seed: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Draw paired goal samples from independent Poisson processes."""

    if seed is not None:
        np.random.seed(int(seed))

    n = max(int(n_sims), 1)
    lam_h = _clamp_lambda(lam_home)
    lam_a = _clamp_lambda(lam_away)

    home_goals = np.random.poisson(lam_h, size=n)
    away_goals = np.random.poisson(lam_a, size=n)
    totals = home_goals + away_goals

    return home_goals.tolist(), away_goals.tolist(), totals.tolist()


def monte_carlo_simulate(
    lam_home: float,
    lam_away: float,
    *,
    n_sims: int = 10000,
    over_line: float = 2.5,
    seed: int | None = None,
) -> Dict[str, float]:
    """Return outcome probabilities from Monte Carlo Poisson draws."""

    home_goals, away_goals, totals = simulate_goals(lam_home, lam_away, n_sims=n_sims, seed=seed)
    h = np.asarray(home_goals)
    a = np.asarray(away_goals)
    t = np.asarray(totals)

    diff = h - a
    p_home = float(np.mean(diff > 0))
    p_draw = float(np.mean(diff == 0))
    p_away = float(np.mean(diff < 0))

    p_over = float(np.mean(t > over_line))
    p_under = float(np.mean(t < over_line))

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over": p_over,
        "p_under": p_under,
        "samples": int(len(t)),
    }
