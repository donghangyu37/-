"""Compatibility wrapper re-exporting the Football API utilities."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_SRC_PATH = Path(__file__).resolve().parents[2] / "src" / "api" / "football_api.py"
_SPEC = importlib.util.spec_from_file_location("src.api.football_api", _SRC_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, _MODULE)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

fixtures_by_date = _MODULE.fixtures_by_date
list_teams_in_league = _MODULE.list_teams_in_league
team_statistics = _MODULE.team_statistics
recent_fixtures_by_team = _MODULE.recent_fixtures_by_team
find_league = _MODULE.find_league
_MEMO = _MODULE._MEMO
_get_cached = _MODULE._get_cached

_SIDE_TOKENS = {
    "home": "home",
    "1": "home",
    "team1": "home",
    "h": "home",
    "away": "away",
    "2": "away",
    "team2": "away",
    "a": "away",
    "guest": "away",
    "visitor": "away",
}

_PICK_TOKENS = {"pk", "pick", "pick'em", "pickem", "p.k.", "p.k"}
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _median(arr: List[float]) -> float | None:
    arr = [x for x in (arr or []) if x]
    if not arr:
        return None
    arr.sort()
    n = len(arr)
    mid = n // 2
    if n % 2:
        return float(arr[mid])
    return float((arr[mid - 1] + arr[mid]) / 2)


def _clean_ah_map(ah_map: Dict[float, Dict[str, List[float]]]) -> Dict[float, Dict[str, List[float]]]:
    cleaned: Dict[float, Dict[str, List[float]]] = {}
    for line, sides in ah_map.items():
        try:
            key = float(line)
        except (TypeError, ValueError):
            continue
        home = [float(x) for x in sides.get("home", []) if x]
        away = [float(x) for x in sides.get("away", []) if x]
        cleaned[key] = {"home": home, "away": away}
    return cleaned


def _pack_ah_lines(raw: Dict[float, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for line, sides in raw.items():
        home = [x for x in sides.get("home", []) if x]
        away = [x for x in sides.get("away", []) if x]
        if not home or not away:
            continue
        oh = _median(home)
        oa = _median(away)
        if oh is None or oa is None:
            continue
        overround = 1.0 / oh + 1.0 / oa
        summary[str(float(line))] = {
            "home_median": float(oh),
            "away_median": float(oa),
            "home_cnt": int(len(home)),
            "away_cnt": int(len(away)),
            "overround": float(overround),
        }
    return summary


def _is_asian_handicap(name: str) -> bool:
    n = (name or "").lower()
    if "asian handicap" in n or ("handicap" in n and "european" not in n):
        forbid = [
            "1st half",
            "first half",
            "2nd half",
            "second half",
            "half time",
            "half-time",
            "halftime",
        ]
        if any(term in n for term in forbid):
            return False
        return "corner" not in n and "cards" not in n and "booking" not in n
    return False


def _parse_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    text_norm = text.lower().replace("−", "-")
    if text_norm in _PICK_TOKENS:
        return 0.0

    try:
        return float(text_norm.replace(",", "."))
    except ValueError:
        match = _NUM_RE.search(text_norm)
        if match:
            token = match.group(0).replace(",", ".")
            try:
                return float(token)
            except ValueError:
                return None
    return None


def _normalise_side(value) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = text.replace("home team", "home").replace("away team", "away")
    if "home" in text:
        return "home"
    if "away" in text or "visitor" in text or "guest" in text:
        return "away"

    tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
    for token in tokens:
        if token in _SIDE_TOKENS:
            return _SIDE_TOKENS[token]
    return None


def _extract_handicap(entry: Dict[str, Any]) -> float | None:
    line = _parse_float(entry.get("handicap"))
    if line is not None:
        return line
    return _parse_float(entry.get("value"))


def _rebuild_odds(fixture_id: int) -> Dict[str, object]:
    try:
        data = _get_cached("odds", {"fixture": fixture_id}, ttl_sec=180)
    except Exception:
        return {}

    ah_map: Dict[float, Dict[str, List[float]]] = {}
    for item in data.get("response", []) or []:
        for bookmaker in item.get("bookmakers", []) or []:
            for bet in bookmaker.get("bets", []) or []:
                if not _is_asian_handicap(bet.get("name", "")):
                    continue
                for val in bet.get("values", []) or []:
                    side = _normalise_side(val.get("value"))
                    odd = _parse_float(val.get("odd"))
                    line = _extract_handicap(val)
                    if side not in ("home", "away") or line is None or odd is None:
                        continue
                    if not (1.10 <= odd <= 50.0):
                        continue
                    line_val = float(line)
                    resolved = None
                    for cand in (line_val, -line_val):
                        if cand in ah_map:
                            resolved = cand
                            break
                    if resolved is None:
                        resolved = line_val if side == "home" else -line_val
                    bucket = ah_map.setdefault(resolved, {"home": [], "away": []})
                    bucket[side].append(float(odd))

    raw_ah = _clean_ah_map(ah_map)
    return {
        "_raw_ah_map": raw_ah,
        "ah_lines": _pack_ah_lines(raw_ah),
    }


def odds_by_fixture(fixture_id: int) -> Dict[str, object]:
    base = _MODULE.odds_by_fixture(fixture_id)
    if isinstance(base, dict) and base.get("_raw_ah_map"):
        return base
    rebuilt = _rebuild_odds(fixture_id)
    if isinstance(base, dict):
        merged = dict(base)
        merged.update(rebuilt)
        return merged
    return rebuilt


__all__ = [
    "fixtures_by_date",
    "list_teams_in_league",
    "team_statistics",
    "recent_fixtures_by_team",
    "find_league",
    "odds_by_fixture",
    "_MEMO",
    "_get_cached",
]
