"""Asian handicap probability helpers."""
from __future__ import annotations

import math
from typing import Dict, List

from ..models.poisson_mc import simulate_goals


def _split_handicap(line: float) -> List[tuple[float, float]]:
    """Return a list of ``(sub_line, weight)`` pairs for ``line``."""

    line = float(line)
    half_steps = round(line * 2)
    if abs(line * 2 - half_steps) < 1e-9:
        return [(round(half_steps / 2.0, 2), 1.0)]

    lower = math.floor(line * 2.0) / 2.0
    upper = lower + 0.5
    return [(round(lower, 2), 0.5), (round(upper, 2), 0.5)]


def _subline_probabilities(diffs: List[int], line: float) -> tuple[float, float, float]:
    wins = pushes = losses = 0
    for diff in diffs:
        adj = diff + line
        if adj > 1e-9:
            wins += 1
        elif adj < -1e-9:
            losses += 1
        else:
            pushes += 1
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    return wins / n, pushes / n, losses / n


def ah_probabilities_from_lams(
    lam_home: float,
    lam_away: float,
    h: float,
    n_sims: int = 20000,
) -> Dict[str, Dict[str, object]]:
    """Estimate Asian handicap outcome probabilities via simulation."""

    home_goals, away_goals, _ = simulate_goals(lam_home, lam_away, n_sims)
    diffs_home = [hg - ag for hg, ag in zip(home_goals, away_goals)]
    if not diffs_home:
        return {
            "home": {"p_win": 0.0, "p_push": 0.0, "p_loss": 0.0, "components": []},
            "away": {"p_win": 0.0, "p_push": 0.0, "p_loss": 0.0, "components": []},
        }

    diffs_away = [-d for d in diffs_home]

    def _build_side(line: float, diffs: List[int]) -> Dict[str, object]:
        parts = []
        total_win = total_push = total_loss = 0.0
        for sub_line, weight in _split_handicap(line):
            p_win, p_push, p_loss = _subline_probabilities(diffs, sub_line)
            parts.append(
                {
                    "line": sub_line,
                    "weight": weight,
                    "p_win": p_win,
                    "p_push": p_push,
                    "p_loss": p_loss,
                }
            )
            total_win += weight * p_win
            total_push += weight * p_push
            total_loss += weight * p_loss
        return {
            "p_win": total_win,
            "p_push": total_push,
            "p_loss": total_loss,
            "components": parts,
        }

    home_info = _build_side(float(h), diffs_home)
    away_info = _build_side(-float(h), diffs_away)

    return {"home": home_info, "away": away_info}


def _kelly_fraction(p_win: float, p_push: float, p_loss: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    denom = b * (p_win + p_loss)
    if denom <= 0:
        return 0.0
    numer = p_win * b - p_loss
    frac = numer / denom
    return max(0.0, min(1.0, frac))


def ah_ev_kelly(
    probs: Dict[str, Dict[str, object]],
    odds_home: float,
    odds_away: float,
) -> Dict[str, Dict[str, float | None]]:
    """Compute EV and Kelly fractions for both sides of an AH market."""

    def _eval(side: str, odds: float) -> Dict[str, float | None]:
        info = probs.get(side, {}) if isinstance(probs, dict) else {}
        components = info.get("components") if isinstance(info, dict) else None
        if not isinstance(components, list) or not components:
            return {"EV": None, "Kelly": None}
        ev_total = 0.0
        kelly_total = 0.0
        b = float(odds) - 1.0
        if b <= 0:
            return {"EV": None, "Kelly": None}
        for comp in components:
            weight = float(comp.get("weight", 0.0))
            p_win = float(comp.get("p_win", 0.0))
            p_push = float(comp.get("p_push", 0.0))
            p_loss = float(comp.get("p_loss", 0.0))
            ev_total += weight * (p_win * b - p_loss)
            kelly_total += weight * _kelly_fraction(p_win, p_push, p_loss, float(odds))
        return {
            "EV": round(ev_total, 6),
            "Kelly": round(kelly_total, 6),
        }

    return {
        "home": _eval("home", float(odds_home)),
        "away": _eval("away", float(odds_away)),
    }


__all__ = ["ah_probabilities_from_lams", "ah_ev_kelly"]
