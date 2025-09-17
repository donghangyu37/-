"""Asian handicap probability utilities previously provided by ``src.markets.asian_handicap``."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _asian_components(line: float) -> list[tuple[float, float]]:
    base = math.floor(float(line))
    frac = round(float(line) - base, 2)
    if abs(frac - 0.25) < 1e-9:
        return [(float(base), 0.5), (float(base) + 0.5, 0.5)]
    if abs(frac - 0.75) < 1e-9:
        return [(float(base) + 0.5, 0.5), (float(base) + 1.0, 0.5)]
    return [(float(line), 1.0)]


def _payout_distribution(payouts: np.ndarray) -> Dict[str, float]:
    buckets = {
        "full_win": float(np.mean(np.isclose(payouts, 1.0))),
        "half_win": float(np.mean(np.isclose(payouts, 0.5))),
        "push": float(np.mean(np.isclose(payouts, 0.0))),
        "half_loss": float(np.mean(np.isclose(payouts, -0.5))),
        "full_loss": float(np.mean(np.isclose(payouts, -1.0))),
    }
    total = sum(buckets.values())
    if total > 0:
        buckets = {k: v / total for k, v in buckets.items()}
    return buckets


def ah_probabilities_from_lams(
    lam_home: float,
    lam_away: float,
    *,
    h: float,
    n_sims: int = 10000,
    seed: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Approximate Asian handicap outcome probabilities by simulation."""

    if seed is not None:
        np.random.seed(int(seed))

    n = max(int(n_sims), 1)
    lam_h = max(float(lam_home), 1e-6)
    lam_a = max(float(lam_away), 1e-6)

    home_goals = np.random.poisson(lam_h, size=n)
    away_goals = np.random.poisson(lam_a, size=n)
    diff = home_goals - away_goals

    payouts_home = np.zeros_like(diff, dtype=float)
    for component, weight in _asian_components(float(h)):
        adjusted = diff + component
        payouts_home += weight * np.where(adjusted > 0, 1.0, np.where(adjusted < 0, -1.0, 0.0))

    payouts_away = -payouts_home

    return {
        "home": _payout_distribution(payouts_home),
        "away": _payout_distribution(payouts_away),
        "samples": int(n),
        "line": float(h),
    }


def _normalise(prob_map: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in prob_map.values())
    if total <= 0:
        return {k: 0.0 for k in prob_map}
    return {k: max(0.0, v) / total for k, v in prob_map.items()}


def _kelly_fraction(prob_map: Dict[str, float], odds: float) -> float | None:
    if odds is None or odds <= 1.0:
        return None
    probs = _normalise(prob_map)
    outcomes = [
        (probs.get("full_win", 0.0), odds - 1.0),
        (probs.get("half_win", 0.0), 0.5 * (odds - 1.0)),
        (probs.get("push", 0.0), 0.0),
        (probs.get("half_loss", 0.0), -0.5),
        (probs.get("full_loss", 0.0), -1.0),
    ]
    best_f = 0.0
    best_log = float("-inf")
    for f in np.linspace(0, 1, 1001):
        if f <= 0:
            continue
        growth = 0.0
        valid = True
        for prob, payout in outcomes:
            if prob <= 0:
                continue
            multiplier = 1.0 + f * payout
            if multiplier <= 0:
                valid = False
                break
            growth += prob * math.log(multiplier)
        if not valid:
            continue
        if growth > best_log + 1e-12:
            best_log = growth
            best_f = f
    if best_f <= 0:
        return 0.0
    return float(best_f)


def _ev(prob_map: Dict[str, float], odds: float) -> float | None:
    if odds is None or odds <= 1.0:
        return None
    probs = _normalise(prob_map)
    win = probs.get("full_win", 0.0)
    half_win = probs.get("half_win", 0.0)
    push = probs.get("push", 0.0)
    half_loss = probs.get("half_loss", 0.0)
    full_loss = probs.get("full_loss", 0.0)
    b = odds - 1.0
    return win * b + half_win * 0.5 * b - half_loss * 0.5 - full_loss


def ah_ev_kelly(
    probabilities: Dict[str, Dict[str, float]],
    *,
    odds_home: float | None,
    odds_away: float | None,
) -> Dict[str, Dict[str, float | None]]:
    """Compute EV and Kelly fractions for Asian handicap selections."""

    home_probs = probabilities.get("home", {}) if isinstance(probabilities, dict) else {}
    away_probs = probabilities.get("away", {}) if isinstance(probabilities, dict) else {}

    res_home = {
        "EV": _ev(home_probs, odds_home),
        "Kelly": _kelly_fraction(home_probs, odds_home),
    }
    res_away = {
        "EV": _ev(away_probs, odds_away),
        "Kelly": _kelly_fraction(away_probs, odds_away),
    }
    return {"home": res_home, "away": res_away}
