"""Put the repository root on sys.path so `pytest` works without installing.

Without this, only `python -m pytest` works, because that form adds the
working directory itself. A fresh clone should pass either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
