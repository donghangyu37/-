import os
import sys
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("FOOTBALL_API_KEY", "test-key")

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")

    def _load_dotenv(*_args, **_kwargs):  # type: ignore[return-value]
        return None

    dotenv_module.load_dotenv = _load_dotenv  # type: ignore[attr-defined]
    sys.modules["dotenv"] = dotenv_module

if "requests" not in sys.modules:
    requests_module = types.ModuleType("requests")

    class _DummySession:
        def __init__(self) -> None:
            self.trust_env = False

        def get(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("Network access not expected during tests")

        def mount(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

    class _DummyHTTPAdapter:
        def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

    requests_module.Session = _DummySession  # type: ignore[attr-defined]
    adapters_module = types.ModuleType("requests.adapters")
    adapters_module.HTTPAdapter = _DummyHTTPAdapter  # type: ignore[attr-defined]
    requests_module.adapters = adapters_module  # type: ignore[attr-defined]
    requests_module.exceptions = types.SimpleNamespace(RequestException=Exception)
    sys.modules["requests"] = requests_module
    sys.modules["requests.adapters"] = adapters_module

if "urllib3" not in sys.modules:
    retry_module = types.ModuleType("urllib3.util.retry")

    class _DummyRetry:  # pylint: disable=too-few-public-methods
        def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

    retry_module.Retry = _DummyRetry  # type: ignore[attr-defined]

    util_module = types.ModuleType("urllib3.util")
    util_module.retry = retry_module  # type: ignore[attr-defined]

    urllib3_module = types.ModuleType("urllib3")
    urllib3_module.util = util_module  # type: ignore[attr-defined]

    sys.modules["urllib3"] = urllib3_module
    sys.modules["urllib3.util"] = util_module
    sys.modules["urllib3.util.retry"] = retry_module

from daily_.daily_ import football_api


class OddsParsingTests(unittest.TestCase):
    def test_descriptive_totals_values_yield_lines(self) -> None:
        fixture_id = 4242
        mock_data = {
            "response": [
                {
                    "bookmakers": [
                        {
                            "bets": [
                                {
                                    "name": "Over/Under",
                                    "values": [
                                        {"value": "Over 2.5 Goals", "odd": "1.91", "handicap": None},
                                        {"value": "Under 2.5 Goals", "odd": "1.93", "handicap": None},
                                    ],
                                },
                                {
                                    "name": "Total Corners",
                                    "values": [
                                        {"value": "Over 10 Corners", "odd": "1.80", "handicap": ""},
                                        {"value": "Under 10 Corners", "odd": "1.95", "handicap": ""},
                                    ],
                                },
                            ],
                        },
                        {
                            "bets": [
                                {
                                    "name": "Over/Under",
                                    "values": [
                                        {"value": "Over 2.5 Goals", "odd": "1.90", "handicap": ""},
                                        {"value": "Under 2.5 Goals", "odd": "1.95", "handicap": ""},
                                    ],
                                },
                                {
                                    "name": "Total Corners",
                                    "values": [
                                        {"value": "Over 10 Corners", "odd": "1.82", "handicap": None},
                                        {"value": "Under 10 Corners", "odd": "1.98", "handicap": None},
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ]
        }

        with patch.object(football_api, "_get_cached", return_value=mock_data) as mock_get_cached:
            result = football_api.odds_by_fixture(fixture_id)

        mock_get_cached.assert_called_once_with("odds", {"fixture": fixture_id}, ttl_sec=180)

        self.assertIn("ou_main_line", result)
        self.assertEqual(result["ou_main_line"], 2.5)
        self.assertIn(2.5, result["_raw_ou_map"])
        self.assertCountEqual(result["_raw_ou_map"][2.5]["over"], [1.91, 1.9])
        self.assertCountEqual(result["_raw_ou_map"][2.5]["under"], [1.93, 1.95])

        self.assertIn("crn_main_line", result)
        self.assertEqual(result["crn_main_line"], 10.0)
        self.assertIn(10.0, result["_raw_crn_map"])
        self.assertCountEqual(result["_raw_crn_map"][10.0]["over"], [1.8, 1.82])
        self.assertCountEqual(result["_raw_crn_map"][10.0]["under"], [1.95, 1.98])


if __name__ == "__main__":
    unittest.main()
