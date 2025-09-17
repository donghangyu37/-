import importlib.util
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("FOOTBALL_API_KEY", "test-key")

try:  # pragma: no cover - exercised indirectly by import failure path
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = types.ModuleType("requests")

    class _DummySession:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            self.trust_env = False

        def mount(self, *_args, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            raise RuntimeError("Network access not available in tests")

    class _DummyHTTPAdapter:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            pass

    requests.Session = _DummySession
    adapters_module = types.ModuleType("requests.adapters")
    adapters_module.HTTPAdapter = _DummyHTTPAdapter
    requests.adapters = adapters_module
    requests.exceptions = types.SimpleNamespace(RequestException=Exception)
    sys.modules["requests"] = requests
    sys.modules["requests.adapters"] = adapters_module

try:  # pragma: no cover
    from urllib3.util.retry import Retry  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    urllib3_module = types.ModuleType("urllib3")
    urllib3_util_module = types.ModuleType("urllib3.util")
    urllib3_retry_module = types.ModuleType("urllib3.util.retry")

    class _DummyRetry:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            pass

    urllib3_retry_module.Retry = _DummyRetry
    sys.modules.setdefault("urllib3", urllib3_module)
    sys.modules["urllib3.util"] = urllib3_util_module
    sys.modules["urllib3.util.retry"] = urllib3_retry_module

try:  # pragma: no cover
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    dotenv_module = types.ModuleType("dotenv")

    def _dummy_load_dotenv(*_, **__):  # pragma: no cover - simple stub
        return False

    dotenv_module.load_dotenv = _dummy_load_dotenv
    sys.modules["dotenv"] = dotenv_module

_MODULE_PATH = Path(__file__).resolve().parents[1] / "daily_" / "daily_" / "football_api.py"
_SPEC = importlib.util.spec_from_file_location("daily_football_api", _MODULE_PATH)
football_api = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = football_api
assert _SPEC.loader is not None
_SPEC.loader.exec_module(football_api)


@pytest.fixture(autouse=True)
def _clear_cache():
    # Ensure per-test isolation for cached responses.
    football_api._MEMO.clear()
    yield
    football_api._MEMO.clear()


def test_first_half_asian_handicap_filtered(monkeypatch):
    fake_odds = {
        "response": [
            {
                "bookmakers": [
                    {
                        "bets": [
                            {
                                "name": "Asian Handicap",
                                "values": [
                                    {"value": "Home", "odd": "1.83", "handicap": "-0.5"},
                                    {"value": "Away", "odd": "2.00", "handicap": "0.5"},
                                ],
                            },
                            {
                                "name": "Asian Handicap - 1st Half",
                                "values": [
                                    {"value": "Home", "odd": "1.90", "handicap": "-0.25"},
                                    {"value": "Away", "odd": "1.90", "handicap": "0.25"},
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }

    def _fake_get_cached(endpoint, params, ttl_sec):  # pragma: no cover - exercised in test
        return fake_odds

    monkeypatch.setattr(football_api, "_get_cached", _fake_get_cached)
    monkeypatch.setattr(football_api._MODULE, "_get_cached", _fake_get_cached)

    result = football_api.odds_by_fixture(1234)

    assert result["_raw_ah_map"] == {
        -0.5: {
            "home": [1.83],
            "away": [2.0],
        }
    }


def test_asian_handicap_with_embedded_line(monkeypatch):
    fake_odds = {
        "response": [
            {
                "bookmakers": [
                    {
                        "bets": [
                            {
                                "name": "Asian Handicap",
                                "values": [
                                    {"value": "Home -0.25", "odd": "1.95", "handicap": None},
                                    {"value": "Away +0.25", "odd": "1.95", "handicap": None},
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }

    def _fake_get_cached(endpoint, params, ttl_sec):  # pragma: no cover - exercised in test
        return fake_odds

    monkeypatch.setattr(football_api, "_get_cached", _fake_get_cached)

    result = football_api.odds_by_fixture(5678)

    raw = result.get("_raw_ah_map")
    assert raw is not None
    assert set(raw.keys()) == {-0.25}
    assert raw[-0.25]["home"][0] == pytest.approx(1.95)
    assert raw[-0.25]["away"][0] == pytest.approx(1.95)

    lines = result.get("ah_lines")
    assert lines is not None
    entry = lines.get("-0.25")
    assert entry is not None
    assert entry["home_median"] == pytest.approx(1.95)
    assert entry["away_median"] == pytest.approx(1.95)
    assert entry["home_cnt"] == 1
    assert entry["away_cnt"] == 1


def test_odds_counts_and_updates(monkeypatch):
    fake_odds = {
        "response": [
            {
                "bookmakers": [
                    {
                        "update": "2025-09-17T12:30:00+00:00",
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "2.00"},
                                    {"value": "Draw", "odd": "3.40"},
                                    {"value": "Away", "odd": "3.60"},
                                ],
                            },
                            {
                                "name": "Over/Under",
                                "values": [
                                    {"value": "Over 2.5", "odd": "1.88", "handicap": "2.5"},
                                    {"value": "Under 2.5", "odd": "1.96", "handicap": "2.5"},
                                ],
                            },
                            {
                                "name": "Asian Handicap",
                                "values": [
                                    {"value": "Home", "odd": "1.92", "handicap": "-0.5"},
                                    {"value": "Away", "odd": "1.95", "handicap": "+0.5"},
                                ],
                            },
                        ],
                    },
                    {
                        "update": "2025-09-17T12:35:00+00:00",
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "1.95"},
                                    {"value": "Draw", "odd": "3.60"},
                                    {"value": "Away", "odd": "3.80"},
                                ],
                            },
                            {
                                "name": "Over/Under",
                                "values": [
                                    {"value": "Over 2.5", "odd": "1.90", "handicap": "2.5"},
                                    {"value": "Under 2.5", "odd": "1.94", "handicap": "2.5"},
                                ],
                            },
                            {
                                "name": "Asian Handicap",
                                "values": [
                                    {"value": "Home", "odd": "1.94", "handicap": "-0.5"},
                                    {"value": "Away", "odd": "1.98", "handicap": "+0.5"},
                                ],
                            },
                        ],
                    },
                ]
            }
        ]
    }

    def _fake_get_cached(endpoint, params, ttl_sec):  # pragma: no cover - exercised in test
        return fake_odds

    monkeypatch.setattr(football_api, "_get_cached", _fake_get_cached)
    monkeypatch.setattr(football_api._MODULE, "_get_cached", _fake_get_cached)

    result = football_api.odds_by_fixture(9999)

    assert result["1x2_home_cnt"] == 2
    assert result["1x2_draw_cnt"] == 2
    assert result["1x2_away_cnt"] == 2
    assert result["ou_main_over_cnt"] == 2
    assert result["ou_main_under_cnt"] == 2
    assert result["ah_home_cnt"] == 2
    assert result["ah_away_cnt"] == 2

    latest_ts = datetime.fromisoformat("2025-09-17T12:35:00+00:00").timestamp()
    assert result["1x2_last_update_ts"] == pytest.approx(latest_ts)
    assert result["ou_main_last_update_ts"] == pytest.approx(latest_ts)
    assert result["ah_last_update_ts"] == pytest.approx(latest_ts)

    line_entry = result["ou_lines"].get("2.5")
    assert line_entry is not None
    assert line_entry["last_update_ts"] == pytest.approx(latest_ts)

    ah_line_entry = next(iter(result["ah_lines"].values()))
    assert ah_line_entry["last_update_ts"] == pytest.approx(latest_ts)
