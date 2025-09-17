"""Project-wide configuration values.

This module centralises runtime tunables that are commonly shared by
multiple components.  Values are primarily sourced from environment
variables so that users can adjust behaviour without modifying code.
"""
from __future__ import annotations

import os

# Default home-advantage multiplier used by the goal simulation models.
# The value can be overridden via the ``HOME_ADV`` environment variable.
HOME_ADV: float = float(os.getenv("HOME_ADV", "1.10") or 1.10)

__all__ = ["HOME_ADV"]
