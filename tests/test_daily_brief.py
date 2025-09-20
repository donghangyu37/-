
import importlib.util
import math
import os
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

from daily_.value_engine import LEAGUE_TIER_TOP

_DAILY_BRIEF_PATH = Path(__file__).resolve().parent.parent / "daily_" / "daily_brief"
if "numpy" not in sys.modules:
    class _RandomStub:
        @staticmethod
        def seed(_seed):
            return None

        @staticmethod
        def poisson(lam, size, *_, **__):
            try:
                count = int(size)
            except (TypeError, ValueError):
                count = 1
            return [lam] * max(count, 1)

    sys.modules["numpy"] = types.SimpleNamespace(random=_RandomStub())

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")

    def _load_dotenv_stub(*_args, **_kwargs):
        return None

    dotenv_stub.load_dotenv = _load_dotenv_stub
    sys.modules["dotenv"] = dotenv_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _SessionStub:
        def __init__(self):
            self.trust_env = False

        def get(self, *_args, **_kwargs):
            raise RuntimeError("requests Session stub used in tests")

        def mount(self, *_args, **_kwargs):
            return None

    requests_stub.Session = _SessionStub
    requests_stub.exceptions = types.SimpleNamespace(RequestException=Exception)
    sys.modules["requests"] = requests_stub

    adapters_stub = types.ModuleType("requests.adapters")

    class _HTTPAdapterStub:
        def __init__(self, *_args, **_kwargs):
            pass

    adapters_stub.HTTPAdapter = _HTTPAdapterStub
    sys.modules["requests.adapters"] = adapters_stub

if "urllib3" not in sys.modules:
    urllib3_stub = types.ModuleType("urllib3")
    urllib3_util_stub = types.ModuleType("urllib3.util")
    urllib3_retry_stub = types.ModuleType("urllib3.util.retry")

    class _RetryStub:
        def __init__(self, *_args, **_kwargs):
            pass

    urllib3_retry_stub.Retry = _RetryStub
    urllib3_util_stub.retry = urllib3_retry_stub
    urllib3_stub.util = urllib3_util_stub
    sys.modules["urllib3"] = urllib3_stub
    sys.modules["urllib3.util"] = urllib3_util_stub
    sys.modules["urllib3.util.retry"] = urllib3_retry_stub

os.environ.setdefault("FOOTBALL_API_KEY", "dummy-key")
_LOADER = SourceFileLoader("daily_brief_module", str(_DAILY_BRIEF_PATH))
_SPEC = importlib.util.spec_from_loader(_LOADER.name, _LOADER)
if _SPEC is None:  # pragma: no cover - defensive guard
    raise ImportError(f"Unable to build spec for daily_brief module at {_DAILY_BRIEF_PATH}")
_daily_brief_module = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(_daily_brief_module)
apply_ev_filter = getattr(_daily_brief_module, "apply_ev_filter")



def test_apply_ev_filter_populates_diagnostics_meta() -> None:
    flag_map: dict[str, str] = {}
    reason_map: dict[str, str] = {}
    vi_map: dict[str, float | None] = {}
    meta_map: dict[str, dict[str, object]] = {}

    ev, kelly = apply_ev_filter(
        key="ou_main_over",
        ev=0.08,
        kelly=0.07,
        market="ou",
        tier=LEAGUE_TIER_TOP,
        min_bookmakers=10,
        overround=1.08,
        update_age=4.0,
        flag_map=flag_map,
        reason_map=reason_map,
        vi_map=vi_map,
        meta_map=meta_map,
        odds=2.05,
        model_prob=0.58,
        consensus_prob=0.54,
        data_quality=0.9,
        sample_size=12,
    )

    assert ev is not None and kelly is not None
    assert flag_map.get("ou_main_over") in {"keep", "review"}
    assert "ou_main_over" not in reason_map
    assert vi_map.get("ou_main_over") is not None

    diagnostics = meta_map.get("diagnostics")
    assert isinstance(diagnostics, dict)
    entry = diagnostics.get("ou_main_over")
    assert isinstance(entry, dict)
    assert "quality" in entry and entry["quality"] is not None
    assert "ev_input" in entry and entry["ev_input"] is not None
    assert "ev_calibrated" in entry and entry["ev_calibrated"] is not None

    thresholds = entry.get("thresholds")
    assert isinstance(thresholds, dict)
    assert math.isclose(float(thresholds["keep_min"]), 0.02, rel_tol=1e-9)
    assert math.isclose(float(thresholds["keep_max"]), 0.06, rel_tol=1e-9)
    assert math.isclose(float(thresholds["drop"]), 0.12, rel_tol=1e-9)



def test_export_picks_applies_gate_levels(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rows = [
        {
            "date_utc": "2025-09-19",
            "kickoff_utc": "2025-09-19T12:00:00Z",
            "league": "Sample League",
            "league_tier": LEAGUE_TIER_TOP,
            "home": "Team A",
            "away": "Team B",
            "ou_main_line": 2.5,
            "odds_ou_main_over": 1.95,
            "ou_main_over_cnt": 8,
            "ou_main_under_cnt": 8,
            "ou_main_overround": 1.08,
            "ev_ou_main_over": 0.06,
            "kelly_ou_main_over": 0.055,
            "flag_ev_ou_main_over": "keep",
            "quality_ev_ou_main_over": 0.8,
            "ev_ou_main_under": None,
            "kelly_ou_main_under": None,
            "odds_1x2_home": 1.92,
            "1x2_home_cnt": 9,
            "1x2_draw_cnt": 9,
            "1x2_away_cnt": 9,
            "1x2_overround": 1.05,
            "ev_1x2_home": 0.033,
            "kelly_1x2_home": 0.032,
            "flag_ev_1x2_home": "keep",
            "quality_ev_1x2_home": 0.9,
            "ev_1x2_draw": None,
            "ev_1x2_away": None,
            "ah_line": -0.25,
            "odds_ah_home": 1.9,
            "odds_ah_away": 1.9,
            "ah_home_cnt": 10,
            "ah_away_cnt": 10,
            "ah_overround": 1.05,
            "ev_ah_home": 0.024,
            "kelly_ah_home": 0.028,
            "flag_ev_ah_home": "keep",
            "quality_ev_ah_home": 0.82,
            "ev_ah_away": None,
            "kelly_ah_away": None,
        }
    ]

    export_picks(rows, "2025-09-19")

    out_path = tmp_path / "out" / "picks_2025-09-19.csv"
    assert out_path.exists()

    with out_path.open(newline="", encoding="utf-8") as fh:
        data = list(csv.DictReader(fh))

    markets = {row["market"]: row for row in data}
    assert markets["OU-Over"]["gate_level"] == "strict"
    assert markets["1X2-Home"]["gate_level"] == "backoff1"
    assert markets["AH-Home"]["gate_level"] == "backoff2"
main
