"""Re-export API helpers from the main src package for legacy tooling.

This thin wrapper ensures that ``src.api.football_api`` can be imported even
when the repository root is not yet on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from importlib import import_module

_impl = import_module("src.api.football_api")

_MEMO = _impl._MEMO  # noqa: F401
_get_cached = _impl._get_cached  # noqa: F401


def _make_proxy(name: str):
    target = getattr(_impl, name)

    if not callable(target):  # pragma: no cover - no proxy needed
        return target

    def wrapper(*args, **kwargs):
        original = _impl._get_cached
        try:
            _impl._get_cached = globals().get("_get_cached", original)
            return target(*args, **kwargs)
        finally:
            _impl._get_cached = original

    return wrapper


for _name in getattr(_impl, "__all__", []):
    globals()[_name] = _make_proxy(_name)


__all__ = list(getattr(_impl, "__all__", [])) + ["_MEMO", "_get_cached"]
