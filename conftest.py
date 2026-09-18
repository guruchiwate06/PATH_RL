"""
conftest.py (project root)
--------------------------
Ensures the workspace root is on sys.path so that the
``evacuation_simulation`` package is importable from any pytest invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives at RL_ENV/evacuation_simulation/conftest.py
# The package 'evacuation_simulation' lives at RL_ENV/evacuation_simulation/
# Its parent (RL_ENV/) is what needs to be on sys.path.
_ROOT = Path(__file__).parent.parent.resolve()  # → F:\Projects\RL_ENV
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
