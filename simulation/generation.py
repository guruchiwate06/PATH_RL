"""
generation.py
-------------
Stage 10: Realistic Floor-Plan & Exit Representation.
(Evolved from Stage 9: Candidate Exit Generation & Feasibility).

This module provides reusable mechanisms to:
1. Generate realistic CandidateExit locations from a FloorPlan's explicit
   exterior DoorOpenings (with documented legacy fallback).
2. Construct feasible ExitConfigurations carrying physical opening properties
   (coordinates and widths) from those candidates.

These components are purely structural and physical. They do NOT evaluate
evacuation performance, select "best" exits, or validate legal/building
code compliance. Those are responsibilities of future stages.
"""

from __future__ import annotations

import itertools
from typing import Iterator, Optional

from pydantic import BaseModel, Field

from evacuation_simulation.simulation.config import (
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridCell,
)


class CandidateExit(BaseModel):
    """
    A candidate emergency exit derived from an explicit exterior DoorOpening
    (or legacy boundary cell).

    This does NOT imply regulatory or safety compliance. It simply means
    an opening exists that is eligible to be evaluated as an emergency exit.
    
    Attributes
    ----------
    position:
        The (row, col) grid coordinates of the candidate exit.
    width:
        Physical width of the opening in meters (must be > 0.0, default: 1.0).
    door_id:
        Optional reference identifier from the originating DoorOpening.
    """
    position: GridCell = Field(
        ...,
        description="The (row, col) grid position of the candidate exit."
    )
    width: float = Field(
        default=1.0,
        gt=0.0,
        description="Physical width of the door opening in meters."
    )
    door_id: Optional[str] = Field(
        default=None,
        description="Optional identifier of the underlying DoorOpening."
    )


def generate_candidate_exits(
    floor_plan: FloorPlan,
    fallback_legacy_boundary: bool = True,
) -> list[CandidateExit]:
    """
    Generate candidate emergency exits for a given FloorPlan.

    Stage 10 Realistic Behavior:
    ----------------------------
    When explicit doors are defined on ``floor_plan.doors``:
    - Candidate exits are derived strictly from explicit exterior door openings
      (``d.exterior is True``).
    - Interior doors (``d.exterior is False``) connecting rooms to corridors
      are excluded from exterior exit candidates.
    - Physical opening properties (width, door_id) are preserved.
    - Candidates are returned in deterministic order sorted by grid position.

    Legacy Fallback (Backward Compatibility):
    -----------------------------------------
    When ``floor_plan.doors`` is empty:
    - If ``fallback_legacy_boundary`` is True (default), falls back to the
      Stage 9 simplified behavior: scans all boundary non-wall cells, assigning
      a default width of 1.0m. This ensures existing test scenarios and code
      continue to operate without interruption.
    - If ``fallback_legacy_boundary`` is False, returns an empty list.

    Parameters
    ----------
    floor_plan:
        The static building geometry.
    fallback_legacy_boundary:
        Whether to fall back to scanning all boundary non-wall cells when
        no explicit doors are defined (default: True).

    Returns
    -------
    list[CandidateExit]
        A deterministically ordered list of candidate exit locations.
    """
    candidates: list[CandidateExit] = []
    walls_set = set(map(tuple, floor_plan.walls))

    # 1. Realistic explicit doors representation
    if floor_plan.doors:
        # Filter for exterior openings that are not solid walls
        exterior_doors = [
            d for d in floor_plan.doors
            if d.exterior and (tuple(d.position) not in walls_set)
        ]
        # Sort deterministically by (row, col)
        sorted_doors = sorted(
            exterior_doors, key=lambda d: (d.position[0], d.position[1])
        )
        for d in sorted_doors:
            candidates.append(
                CandidateExit(
                    position=d.position,
                    width=d.width,
                    door_id=d.door_id,
                )
            )
        return candidates

    # 2. Legacy fallback for scenarios without explicit doors
    if not fallback_legacy_boundary:
        return []

    rows = floor_plan.grid.rows
    cols = floor_plan.grid.cols

    for r in range(rows):
        for c in range(cols):
            # Must be on the boundary
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                # Must not be a wall
                if (r, c) not in walls_set:
                    candidates.append(CandidateExit(position=(r, c), width=1.0))

    return candidates


def generate_exit_configurations(
    candidates: list[CandidateExit], 
    exit_count: int
) -> Iterator[ExitConfiguration]:
    """
    Generate valid ExitConfigurations from a pool of candidate exits.

    Parameters
    ----------
    candidates:
        List of possible CandidateExits.
    exit_count:
        Number of exits each generated configuration should contain.

    Yields
    ------
    ExitConfiguration
        A deterministic stream of unique exit configurations. Each configuration
        contains the exit coordinates and preserves the corresponding opening widths.

    Raises
    ------
    ValueError
        If exit_count < 1 or exit_count > len(candidates).
    """
    if exit_count < 1:
        raise ValueError(f"exit_count must be >= 1, got {exit_count}")
    if exit_count > len(candidates):
        raise ValueError(
            f"exit_count ({exit_count}) cannot exceed available "
            f"candidates ({len(candidates)})"
        )

    # Deduplicate candidates by position while preserving width and sorting deterministically
    unique_map: dict[tuple[int, int], CandidateExit] = {}
    for c in candidates:
        pos = (c.position[0], c.position[1])
        if pos not in unique_map:
            unique_map[pos] = c

    sorted_candidates = [unique_map[k] for k in sorted(unique_map.keys())]

    for i, combination in enumerate(itertools.combinations(sorted_candidates, exit_count)):
        config_id = f"config_{i+1}_exits_{exit_count}"
        exits = [c.position for c in combination]
        widths = [c.width for c in combination]
        yield ExitConfiguration(
            configuration_id=config_id,
            exits=exits,
            widths=widths,
        )
