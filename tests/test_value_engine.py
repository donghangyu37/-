import sys
from pathlib import Path

import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daily_"))

numpy_stub = types.ModuleType("numpy")
numpy_stub.random = types.SimpleNamespace(seed=lambda _=None: None)
numpy_stub.isscalar = lambda obj: isinstance(obj, (int, float, complex, bool))
sys.modules.setdefault("numpy", numpy_stub)

import value_engine  # noqa: E402


def test_ev_kelly_returns_zero_for_fair_even_money_odds():
    ev, kelly = value_engine.ev_kelly(0.5, 2.0)
    assert ev == 0.0
    assert kelly == 0.0


def test_ev_kelly_positive_value_bet():
    ev, kelly = value_engine.ev_kelly(0.55, 2.2)
    assert ev == 0.21
    assert kelly == 0.175


def test_ev_kelly_negative_value_returns_none():
    ev, kelly = value_engine.ev_kelly(0.4, 2.0)
    assert ev == -0.2
    assert kelly is None
