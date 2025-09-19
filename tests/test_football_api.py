import importlib.util
import os
import sys
import types
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

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "api" / "football_api.py"
_SPEC = importlib.util.spec_from_file_location("src_football_api", _MODULE_PATH)
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

    result = football_api.odds_by_fixture(1234)

    assert result["_raw_ah_map"] == {
        -0.5: {
            "home": [1.83],
            "away": [2.0],
        }
    }
