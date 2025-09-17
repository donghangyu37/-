"""Compatibility wrapper exposing the odds/fixture helpers under the new ``src`` namespace."""
from __future__ import annotations

from daily_.football_api import (
    fixtures_by_date,
    list_teams_in_league,
    team_statistics,
    recent_fixtures_by_team,
    odds_by_fixture,
    find_league,
)

__all__ = [
    "fixtures_by_date",
    "list_teams_in_league",
    "team_statistics",
    "recent_fixtures_by_team",
    "odds_by_fixture",
    "find_league",
]
