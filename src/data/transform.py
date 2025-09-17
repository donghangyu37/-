"""Utility helpers recreated from the legacy ``src.data.transform`` module."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

_DEFAULT_LEAGUE_AVG = 2.6


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
            if not value:
                return None
        return float(value)
    except (TypeError, ValueError):
        return None


def league_goal_averages(stats_list: Iterable[Dict[str, Any]] | None) -> Tuple[float, Dict[str, Any]]:
    """Estimate league-level average total goals from team snapshots."""

    cleaned = [st for st in (stats_list or []) if isinstance(st, dict) and st]
    if not cleaned:
        return float(_DEFAULT_LEAGUE_AVG), {
            "teams": 0,
            "avg_total": float(_DEFAULT_LEAGUE_AVG),
            "home_avg": None,
            "away_avg": None,
            "used_totals_field": 0,
            "used_homeaway_field": 0,
        }

    total_per_match: list[float] = []
    home_match_totals: list[float] = []
    away_match_totals: list[float] = []
    used_totals_field = 0
    used_homeaway_field = 0

    for st in cleaned:
        goals = st.get("goals") or {}
        for_avg = (goals.get("for") or {}).get("average") or {}
        against_avg = (goals.get("against") or {}).get("average") or {}

        total_for = _to_float(for_avg.get("total"))
        total_against = _to_float(against_avg.get("total"))
        if total_for is not None and total_against is not None:
            used_totals_field += 1
            total_per_match.append(total_for + total_against)
        else:
            home_for = _to_float(for_avg.get("home"))
            home_against = _to_float(against_avg.get("home"))
            away_for = _to_float(for_avg.get("away"))
            away_against = _to_float(against_avg.get("away"))
            partials = [home_for, home_against, away_for, away_against]
            if any(val is not None for val in partials):
                used_homeaway_field += 1
                home_total = (home_for or 0.0) + (home_against or 0.0)
                away_total = (away_for or 0.0) + (away_against or 0.0)
                total_per_match.append(0.5 * (home_total + away_total))
                if home_for is not None or home_against is not None:
                    home_match_totals.append(home_total)
                if away_for is not None or away_against is not None:
                    away_match_totals.append(away_total)
            else:
                total_per_match.append(_DEFAULT_LEAGUE_AVG)

        if total_for is not None or total_against is not None:
            home_match_totals.append((
                (_to_float(for_avg.get("home")) or 0.0)
                + (_to_float(against_avg.get("home")) or 0.0)
            ))
            away_match_totals.append((
                (_to_float(for_avg.get("away")) or 0.0)
                + (_to_float(against_avg.get("away")) or 0.0)
            ))

    if not total_per_match:
        total_per_match.append(_DEFAULT_LEAGUE_AVG)

    league_avg = sum(total_per_match) / len(total_per_match)
    home_avg = sum(home_match_totals) / len(home_match_totals) if home_match_totals else None
    away_avg = sum(away_match_totals) / len(away_match_totals) if away_match_totals else None

    meta = {
        "teams": len(cleaned),
        "avg_total": float(league_avg),
        "home_avg": float(home_avg) if home_avg is not None else None,
        "away_avg": float(away_avg) if away_avg is not None else None,
        "used_totals_field": int(used_totals_field),
        "used_homeaway_field": int(used_homeaway_field),
    }
    return float(league_avg), meta
