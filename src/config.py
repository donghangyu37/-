"""Lightweight configuration shim for the daily brief pipeline."""
from __future__ import annotations

import os

HOME_ADV = float(os.getenv("HOME_ADV", "0.12"))
