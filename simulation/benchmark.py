"""
benchmark.py
------------
Lightweight reference benchmark configuration for Stage 7 validation.

Responsibilities
~~~~~~~~~~~~~~~~
- Provide a named, reproducible metadata container (BenchmarkConfig) that
  identifies a known/reference exit configuration for a benchmark scenario.
- Support Stage 7 validation: same floor plan, different exit configurations,
  reference vs. alternative comparison.

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- This module does NOT introduce FloorPlan or ExitConfiguration as domain types.
  Those are Stage 8 architecture concerns.
- This module does NOT make any optimality claims about reference exits.
  Reference exits are KNOWN/REFERENCE configurations, not guaranteed-optimal ones.
- No regulatory constraints, candidate generation, RL, or optimization logic.

Design notes
~~~~~~~~~~~~
- BenchmarkConfig is a plain Python dataclass with no Pydantic dependency.
- It is intentionally minimal: Stage 8 will introduce the full architectural
  separation of FloorPlan, ExitConfiguration, and OccupantScenario.
- The ``reference_exits`` field stores exit cells as known reference data.
  The word "reference" means "used as a comparison baseline", NOT "optimal".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# GridCell = (row, col) — consistent with config.py
GridCell = tuple[int, int]


@dataclass
class BenchmarkConfig:
    """
    Metadata identifying a reference benchmark scenario.

    A BenchmarkConfig names a known scenario + exit configuration that
    serves as a reproducible comparison baseline.  It does NOT claim
    that the reference exits are optimal — they are simply the exits
    used in the benchmark, against which other configurations may be
    compared.

    Attributes
    ----------
    benchmark_id : str
        Unique identifier for this benchmark
        (e.g. ``"two_exits_v1"``, ``"bottleneck_cap1_v1"``).
    scenario_name : str
        Human-readable scenario label, matching ``SimulationConfig.scenario_name``.
    reference_exits : list[GridCell]
        The known/reference exit cell coordinates for this benchmark.
        NOT assumed to be optimal; provides a reproducible comparison baseline.
    seed : int
        Canonical random seed for reproducible baseline measurements.
    description : str
        Human-readable description of the benchmark purpose and setup.
    metadata : dict[str, Any]
        Optional additional metadata (e.g. grid size, agent count, capacity).
        Used for documentation and export labeling only.
    """

    benchmark_id: str
    scenario_name: str
    reference_exits: list[GridCell]
    seed: int
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def exit_labels(self) -> str:
        """Return a compact string representation of the reference exits."""
        return ", ".join(str(e) for e in self.reference_exits)

    def summary(self) -> str:
        """Return a one-line summary of the benchmark for display."""
        return (
            f"[{self.benchmark_id}] scenario={self.scenario_name!r} "
            f"exits={self.exit_labels()} seed={self.seed}"
        )
