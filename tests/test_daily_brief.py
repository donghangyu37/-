import csv
import importlib.util
from importlib.machinery import SourceFileLoader
import os
import sys
from collections import Counter
from pathlib import Path
import types

import pytest

os.environ.setdefault("FOOTBALL_API_KEY", "test-key")


class _RandomStub:
    def seed(self, _seed):
        return None

    def poisson(self, lam=1.0, size=1):
        try:
            lam_val = float(lam)
        except Exception:
            lam_val = 0.0
        count = int(size) if isinstance(size, int) else 1
        return [lam_val] * max(1, count)


_NUMPY_STUB = types.SimpleNamespace(
    random=_RandomStub(),
    isscalar=lambda obj: not isinstance(obj, (list, tuple, dict, set)),
    bool_=bool,
)
sys.modules.setdefault("numpy", _NUMPY_STUB)

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test stub
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

try:
    from urllib3.util.retry import Retry  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test stub
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

try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test stub
    dotenv_module = types.ModuleType("dotenv")

    def _dummy_load_dotenv(*_, **__):  # pragma: no cover - simple stub
        return False

    dotenv_module.load_dotenv = _dummy_load_dotenv
    sys.modules["dotenv"] = dotenv_module

_MODULE_PATH = Path(__file__).resolve().parents[1] / "daily_" / "daily_brief"
_LOADER = SourceFileLoader("daily_brief_module", str(_MODULE_PATH))
_SPEC = importlib.util.spec_from_loader(_LOADER.name, _LOADER)
daily_brief = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(daily_brief)


def test_export_picks_respects_row_thresholds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [
        {
            "date_utc": "2025-09-20",
            "kickoff_utc": "2025-09-20T20:00:00Z",
            "league": "Test League",
            "home": "Team A",
            "away": "Team B",
            "flag_ev_ou_main_over": "keep",
            "ev_ou_main_over": 0.02,
            "kelly_ou_main_over": 0.021,
            "score_ev_ou_main_over": 0.055,
            "quality_ev_ou_main_over": 0.76,
            "liquidity_ev_ou_main_over": 0.88,
            "kelly_full_ev_ou_main_over": 0.028,
            "threshold_score_keep_ev_ou_main_over": 0.05,
            "threshold_quality_keep_ev_ou_main_over": 0.75,
            "threshold_liquidity_keep_ev_ou_main_over": 0.85,
            "threshold_kelly_keep_ev_ou_main_over": 0.02,
            "ou_main_line": 2.5,
            "odds_ou_main_over": 1.95,
            "odds_ou_main_under": 1.95,
        }
    ]

    daily_brief.export_picks(rows, "2025-09-20")

    out_file = tmp_path / "out" / "picks_2025-09-20.csv"
    assert out_file.exists()

    with out_file.open("r", encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))

    assert len(exported) == 1
    record = exported[0]
    assert record["market"] == "OU-Over"
    assert pytest.approx(float(record["threshold_score_keep"])) == 0.05
    assert pytest.approx(float(record["threshold_quality_keep"])) == 0.75
    assert pytest.approx(float(record["threshold_liquidity_keep"])) == 0.85


def test_summarize_diagnostics_outputs_member_breakdown(capsys):
    diagnostics = {
        "ev_ou_main_over": {
            "seen": 10,
            "ev_inputs": 4,
            "tags": Counter({"invalid": 2, "keep": 2}),
            "reasons": {"invalid": Counter({"bookmakers": 2})},
            "ev_sum": {"keep": 0.04},
            "ev_count": {"keep": 2},
            "kelly_sum": {"keep": 0.03},
            "kelly_count": {"keep": 2},
        },
        "ev_ou_main_under": {
            "seen": 9,
            "ev_inputs": 3,
            "tags": Counter({"reject": 2, "low": 1}),
            "reasons": {"reject": Counter({"stale": 1, "overround": 1})},
            "ev_sum": {"low": 0.03},
            "ev_count": {"low": 1},
            "kelly_sum": {},
            "kelly_count": {},
        },
    }

    daily_brief.summarize_diagnostics(diagnostics)
    out = capsys.readouterr().out

    assert "OU主盘" in out
    assert "ev_ou_main_over" in out
    assert "ev_ou_main_under" in out
    assert "invalid(bookmakers:2" in out
    assert "reject(" in out

