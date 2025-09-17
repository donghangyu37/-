"""Data transformation helpers used by the daily briefing pipeline.

Only a very small subset of the original project is required for the
unit tests in this kata.  The goal of this module is therefore to
provide light-weight, well-tested utilities that reproduce the public
APIs consumed by :mod:`daily_.daily_brief` without requiring the entire
historical code base.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Tuple, Dict

_Number = float | int


def _safe_get(mapping: Mapping[str, object] | None, *keys: str) -> object | None:
    """Safely traverse nested dictionaries.

    The helper mirrors the very defensive style that is used throughout
    the original project where API responses may omit fields depending on
    the competition.  Missing or malformed entries simply return
    ``None`` so that callers can decide how to handle them.
    """

    cur: object | None = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)  # type: ignore[call-arg]
    return cur


def _to_float(value: object, default: float | None = None) -> float | None:
    try:
        if isinstance(value, str):
            value = value.replace(",", ".")
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _weighted_average(values: Iterable[tuple[float, float]]) -> float | None:
    total_weight = 0.0
    acc = 0.0
    for value, weight in values:
        if weight <= 0:
            continue
        acc += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return acc / total_weight


def league_goal_averages(stats_list: Iterable[Mapping[str, object]]) -> Tuple[float, Dict[str, float]]:
    """Estimate league-wide scoring averages from team statistics.

    Parameters
    ----------
    stats_list:
        Iterable of ``team/statistics`` payloads returned by the
        Football API.  Each entry may be partially populated, therefore
        the routine is intentionally tolerant and falls back to sensible
        defaults whenever information is missing.

    Returns
    -------
    tuple
        ``(league_avg, meta)`` where ``league_avg`` represents the
        expected number of goals scored by a single team in a match
        (i.e. half of the total goals per game).  ``meta`` contains a few
        descriptive aggregates that are useful for debugging.
    """

    records = list(stats_list)

    home_scoring: list[tuple[float, float]] = []
    away_scoring: list[tuple[float, float]] = []
    home_conceded: list[tuple[float, float]] = []
    away_conceded: list[tuple[float, float]] = []

    for st in records:
        if not isinstance(st, Mapping):
            continue
        home_games = _to_float(_safe_get(st, "fixtures", "played", "home"), 0.0) or 0.0
        away_games = _to_float(_safe_get(st, "fixtures", "played", "away"), 0.0) or 0.0

        h_for = _to_float(_safe_get(st, "goals", "for", "average", "home"))
        h_against = _to_float(_safe_get(st, "goals", "against", "average", "home"))
        a_for = _to_float(_safe_get(st, "goals", "for", "average", "away"))
        a_against = _to_float(_safe_get(st, "goals", "against", "average", "away"))

        if h_for is not None and home_games > 0:
            home_scoring.append((h_for, home_games))
        if h_against is not None and home_games > 0:
            home_conceded.append((h_against, home_games))
        if a_for is not None and away_games > 0:
            away_scoring.append((a_for, away_games))
        if a_against is not None and away_games > 0:
            away_conceded.append((a_against, away_games))

    avg_home_for = _weighted_average(home_scoring) or 1.3
    avg_away_for = _weighted_average(away_scoring) or 1.1
    avg_home_against = _weighted_average(home_conceded) or avg_away_for
    avg_away_against = _weighted_average(away_conceded) or avg_home_for

    # League-average goals scored per team per match.  Using the mean of
    # the home/away attacking averages keeps the value close to the
    # widely quoted 1.3 goals per side in top leagues.
    league_avg = (avg_home_for + avg_away_for) / 2.0

    meta = {
        "avg_home_for": float(avg_home_for),
        "avg_away_for": float(avg_away_for),
        "avg_home_against": float(avg_home_against),
        "avg_away_against": float(avg_away_against),
        "samples": float(len(records)),
    }

    return float(league_avg), meta


__all__ = ["league_goal_averages"]
