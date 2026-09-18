"""
conftest.py
-----------
Pytest shared configuration for the evacuation_simulation test suite.

This file makes the project root available on sys.path so that tests
can import from ``evacuation_simulation`` regardless of how pytest is
invoked (i.e. whether from the project root or from the tests/ directory).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the project root (parent of the ``evacuation_simulation`` package)
# into sys.path so that all test modules can resolve imports correctly.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
