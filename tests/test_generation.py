"""
test_generation.py
------------------
Tests for Stage 9 candidate exit generation and configuration construction.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.config import FloorPlan
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
    generate_exit_configurations,
)


def test_candidate_generation_boundary_only() -> None:
    """Candidates are only generated on the boundary of the grid."""
    fp = FloorPlan(grid={"rows": 5, "cols": 5}, walls=[])
    candidates = generate_candidate_exits(fp)
    
    # Grid is 5x5. Total boundary cells = 5 + 5 + 3 + 3 = 16
    assert len(candidates) == 16
    
    for c in candidates:
        r, col = c.position
        assert r == 0 or r == 4 or col == 0 or col == 4


def test_candidate_generation_excludes_walls() -> None:
    """Boundary cells that are walls are excluded from candidates."""
    # 3x3 grid. 8 boundary cells. Make 2 of them walls.
    fp = FloorPlan(grid={"rows": 3, "cols": 3}, walls=[(0, 0), (2, 2)])
    candidates = generate_candidate_exits(fp)
    
    assert len(candidates) == 6
    for c in candidates:
        assert c.position != (0, 0)
        assert c.position != (2, 2)


def test_candidate_generation_excludes_interior() -> None:
    """Interior traversable cells are not considered candidates (in Stage 9)."""
    fp = FloorPlan(grid={"rows": 3, "cols": 3}, walls=[])
    candidates = generate_candidate_exits(fp)
    
    for c in candidates:
        assert c.position != (1, 1)


def test_candidate_generation_is_deterministic() -> None:
    """Candidates are returned in a deterministic order."""
    fp = FloorPlan(grid={"rows": 4, "cols": 4}, walls=[])
    c1 = generate_candidate_exits(fp)
    c2 = generate_candidate_exits(fp)
    
    assert [c.position for c in c1] == [c.position for c in c2]


def test_exit_configuration_generation_validates_counts() -> None:
    """Invalid exit counts raise ValueError."""
    c = [CandidateExit(position=(0, 0)), CandidateExit(position=(0, 1))]
    
    with pytest.raises(ValueError, match="exit_count must be >= 1"):
        list(generate_exit_configurations(c, 0))
        
    with pytest.raises(ValueError, match="cannot exceed available"):
        list(generate_exit_configurations(c, 3))


def test_exit_configuration_generation_yields_correct_combinations() -> None:
    """Correct number of combinations is generated."""
    c = [
        CandidateExit(position=(0, 0)),
        CandidateExit(position=(0, 1)),
        CandidateExit(position=(0, 2)),
        CandidateExit(position=(0, 3)),
    ]
    
    # 4 choose 2 = 6
    configs = list(generate_exit_configurations(c, 2))
    assert len(configs) == 6
    
    # 4 choose 1 = 4
    configs_1 = list(generate_exit_configurations(c, 1))
    assert len(configs_1) == 4


def test_exit_configuration_generation_yields_unique_exits() -> None:
    """Each generated configuration has unique exit positions."""
    c = [
        CandidateExit(position=(0, 0)),
        CandidateExit(position=(0, 1)),
        CandidateExit(position=(0, 2)),
    ]
    
    for config in generate_exit_configurations(c, 2):
        assert len(config.exits) == 2
        assert config.exits[0] != config.exits[1]


def test_exit_configuration_generation_deterministic_ids() -> None:
    """Generated configurations have deterministic and unique IDs."""
    c = [
        CandidateExit(position=(0, 0)),
        CandidateExit(position=(0, 1)),
    ]
    
    configs = list(generate_exit_configurations(c, 2))
    assert len(configs) == 1
    assert configs[0].configuration_id == "config_1_exits_2"


def test_floor_plan_unmutated_by_generation() -> None:
    """Candidate generation does not mutate the FloorPlan."""
    fp = FloorPlan(grid={"rows": 5, "cols": 5}, walls=[(1, 1)])
    original_walls = list(fp.walls)
    
    _ = generate_candidate_exits(fp)
    
    assert list(fp.walls) == original_walls
    assert not hasattr(fp, "exits")
