"""
test_stage10_representation.py
------------------------------
Stage 10: Realistic Floor-Plan & Exit Representation.

Comprehensive verification of:
1. DoorOpening model (validation, positive width, coordinates).
2. Exterior vs Interior classification.
3. CandidateExit generation from explicit exterior openings.
4. Physical opening width retention.
5. ExitConfiguration integration and combinatorial construction.
6. Multi-configuration simulation execution on a shared FloorPlan without mutation.
7. Scenario JSON schema validation for updated basic_building and bottleneck.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from evacuation_simulation.simulation.config import (
    AgentConfig,
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    OccupantScenario,
    SimulationConfig,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


# ==============================================================================
# A. DoorOpening Model
# ==============================================================================


def test_door_opening_valid() -> None:
    """DoorOpening instantiates cleanly with valid position and width."""
    door = DoorOpening(position=(0, 4), width=1.2, exterior=True, door_id="d1")
    assert door.position == (0, 4)
    assert door.width == 1.2
    assert door.exterior is True
    assert door.door_id == "d1"


def test_door_opening_default_values() -> None:
    """DoorOpening applies sensible defaults (width=1.0, exterior=True)."""
    door = DoorOpening(position=(2, 3))
    assert door.position == (2, 3)
    assert door.width == 1.0
    assert door.exterior is True
    assert door.door_id is None


def test_door_opening_rejects_non_positive_width() -> None:
    """Opening width must be strictly greater than 0.0."""
    with pytest.raises(ValidationError):
        DoorOpening(position=(0, 0), width=0.0)

    with pytest.raises(ValidationError):
        DoorOpening(position=(0, 0), width=-1.5)


def test_door_opening_rejects_negative_coordinates() -> None:
    """Position coordinates must be non-negative."""
    with pytest.raises(ValidationError):
        DoorOpening(position=(-1, 4), width=1.0)

    with pytest.raises(ValidationError):
        DoorOpening(position=(2, -3), width=1.0)


def test_floor_plan_rejects_door_outside_grid() -> None:
    """DoorOpening located outside the grid bounds raises ValidationError."""
    with pytest.raises(ValidationError, match="outside the grid"):
        FloorPlan(
            grid={"rows": 5, "cols": 5},
            walls=[],
            doors=[DoorOpening(position=(5, 2), width=1.0)],
        )


def test_floor_plan_rejects_door_on_wall() -> None:
    """DoorOpening placed directly on a solid wall cell raises ValidationError."""
    with pytest.raises(ValidationError, match="cannot be placed on a solid wall"):
        FloorPlan(
            grid={"rows": 6, "cols": 6},
            walls=[(2, 2)],
            doors=[DoorOpening(position=(2, 2), width=1.0)],
        )


def test_floor_plan_rejects_duplicate_door_positions() -> None:
    """Duplicate DoorOpenings at the same grid coordinate raise ValidationError."""
    with pytest.raises(ValidationError, match="Duplicate door opening defined"):
        FloorPlan(
            grid={"rows": 6, "cols": 6},
            walls=[],
            doors=[
                DoorOpening(position=(0, 3), width=1.0, door_id="d1"),
                DoorOpening(position=(0, 3), width=1.5, door_id="d2"),
            ],
        )


# ==============================================================================
# B. Exterior vs Interior Distinction
# ==============================================================================


def test_exterior_vs_interior_openings_filtering() -> None:
    """Only exterior openings become candidate exits; interior openings are excluded."""
    fp = FloorPlan(
        grid={"rows": 8, "cols": 8},
        walls=[(4, c) for c in range(8) if c not in (3, 4)],  # wall at row 4 with doorway at (4,3),(4,4)
        doors=[
            # Exterior doors (to the outside)
            DoorOpening(position=(0, 3), width=1.0, exterior=True, door_id="ext_north"),
            DoorOpening(position=(7, 4), width=1.5, exterior=True, door_id="ext_south"),
            # Interior doors (between rooms / corridor)
            DoorOpening(position=(4, 3), width=1.2, exterior=False, door_id="int_door_1"),
            DoorOpening(position=(4, 4), width=1.2, exterior=False, door_id="int_door_2"),
        ],
    )

    candidates = generate_candidate_exits(fp)

    # Must only contain the 2 exterior doors
    assert len(candidates) == 2
    candidate_positions = [c.position for c in candidates]
    assert (0, 3) in candidate_positions
    assert (7, 4) in candidate_positions

    # Interior doors must NOT be candidate exits
    assert (4, 3) not in candidate_positions
    assert (4, 4) not in candidate_positions


def test_only_interior_openings_yields_no_candidates() -> None:
    """When a FloorPlan only defines interior doors, 0 candidate exits are produced."""
    fp = FloorPlan(
        grid={"rows": 5, "cols": 5},
        walls=[(2, 0), (2, 1), (2, 3), (2, 4)],
        doors=[
            DoorOpening(position=(2, 2), width=0.9, exterior=False, door_id="corridor_opening")
        ],
    )
    candidates = generate_candidate_exits(fp)
    assert len(candidates) == 0


# ==============================================================================
# C. Candidate Generation Rules & Backward Compatibility
# ==============================================================================


def test_candidate_generation_deterministic_order() -> None:
    """Candidate exits are sorted deterministically regardless of doors list input order."""
    doors_order_a = [
        DoorOpening(position=(7, 2), width=1.2, exterior=True),
        DoorOpening(position=(0, 4), width=1.0, exterior=True),
        DoorOpening(position=(3, 0), width=0.9, exterior=True),
    ]
    doors_order_b = [
        DoorOpening(position=(0, 4), width=1.0, exterior=True),
        DoorOpening(position=(3, 0), width=0.9, exterior=True),
        DoorOpening(position=(7, 2), width=1.2, exterior=True),
    ]

    fp_a = FloorPlan(grid={"rows": 8, "cols": 8}, walls=[], doors=doors_order_a)
    fp_b = FloorPlan(grid={"rows": 8, "cols": 8}, walls=[], doors=doors_order_b)

    candidates_a = generate_candidate_exits(fp_a)
    candidates_b = generate_candidate_exits(fp_b)

    assert [c.position for c in candidates_a] == [c.position for c in candidates_b]
    assert [c.position for c in candidates_a] == [(0, 4), (3, 0), (7, 2)]


def test_candidate_generation_does_not_mutate_floor_plan() -> None:
    """Generating candidate exits does not modify walls, doors, or add exits to FloorPlan."""
    fp = FloorPlan(
        grid={"rows": 6, "cols": 6},
        walls=[(1, 1), (1, 2)],
        doors=[DoorOpening(position=(0, 3), width=1.4, exterior=True)],
    )
    original_walls = list(fp.walls)
    original_doors = [d.model_dump() for d in fp.doors]

    _ = generate_candidate_exits(fp)

    assert list(fp.walls) == original_walls
    assert [d.model_dump() for d in fp.doors] == original_doors
    assert not hasattr(fp, "exits")


def test_legacy_fallback_when_doors_empty() -> None:
    """When doors is empty, generator falls back to Stage 9 boundary cell scanning."""
    fp = FloorPlan(grid={"rows": 4, "cols": 4}, walls=[(0, 0)])
    # 4x4 grid: 12 boundary cells. Minus 1 wall cell = 11 boundary cells.
    candidates = generate_candidate_exits(fp, fallback_legacy_boundary=True)
    assert len(candidates) == 11
    for c in candidates:
        assert c.width == 1.0  # Default legacy width
        assert c.position != (0, 0)


def test_legacy_fallback_disabled() -> None:
    """When fallback_legacy_boundary is False and doors is empty, return empty list."""
    fp = FloorPlan(grid={"rows": 4, "cols": 4}, walls=[])
    candidates = generate_candidate_exits(fp, fallback_legacy_boundary=False)
    assert len(candidates) == 0


# ==============================================================================
# D. Exit Width Retention
# ==============================================================================


def test_candidate_preserves_opening_width_and_metadata() -> None:
    """CandidateExit preserves the exact width and door_id from the originating DoorOpening."""
    fp = FloorPlan(
        grid={"rows": 10, "cols": 10},
        walls=[],
        doors=[
            DoorOpening(position=(0, 2), width=1.0, exterior=True, door_id="main_north"),
            DoorOpening(position=(0, 6), width=1.8, exterior=True, door_id="double_wide"),
            DoorOpening(position=(9, 5), width=2.4, exterior=True, door_id="loading_bay"),
        ],
    )

    candidates = generate_candidate_exits(fp)
    assert len(candidates) == 3

    c_map = {c.position: c for c in candidates}
    assert c_map[(0, 2)].width == 1.0
    assert c_map[(0, 2)].door_id == "main_north"

    assert c_map[(0, 6)].width == 1.8
    assert c_map[(0, 6)].door_id == "double_wide"

    assert c_map[(9, 5)].width == 2.4
    assert c_map[(9, 5)].door_id == "loading_bay"


# ==============================================================================
# E. Configuration Generation & Width Preservation
# ==============================================================================


def test_configuration_generation_preserves_widths() -> None:
    """Generated ExitConfiguration retains widths matching each exit in exits."""
    candidates = [
        CandidateExit(position=(0, 2), width=1.0, door_id="d1"),
        CandidateExit(position=(0, 5), width=1.5, door_id="d2"),
        CandidateExit(position=(9, 4), width=2.0, door_id="d3"),
    ]

    # 3 choose 2 = 3 configurations
    configs = list(generate_exit_configurations(candidates, exit_count=2))
    assert len(configs) == 3

    for cfg in configs:
        assert len(cfg.exits) == 2
        assert len(cfg.widths) == 2
        for exit_cell, width in zip(cfg.exits, cfg.widths):
            assert cfg.get_width(exit_cell) == width
            assert cfg.exit_widths[tuple(exit_cell)] == width

    # Check specific configuration values
    cfg1 = configs[0]
    assert cfg1.exits == [(0, 2), (0, 5)]
    assert cfg1.widths == [1.0, 1.5]

    cfg3 = configs[2]
    assert cfg3.exits == [(0, 5), (9, 4)]
    assert cfg3.widths == [1.5, 2.0]


def test_configuration_generation_bounds_validation() -> None:
    """ExitConfiguration generation raises ValueError on invalid counts."""
    c = [CandidateExit(position=(0, 1), width=1.0)]
    with pytest.raises(ValueError, match="exit_count must be >= 1"):
        list(generate_exit_configurations(c, 0))

    with pytest.raises(ValueError, match="cannot exceed available"):
        list(generate_exit_configurations(c, 2))


# ==============================================================================
# F. Simulation Execution with Shared FloorPlan & Multiple Configurations
# ==============================================================================


def test_simulation_execution_with_realistic_exit_configurations() -> None:
    """Verify that multiple ExitConfigurations can be evaluated on a shared FloorPlan."""
    # 1. Authoritative FloorPlan with walls and realistic exterior doors
    shared_floor_plan = FloorPlan(
        grid={"rows": 5, "cols": 11},
        walls=[(1, 5), (2, 5), (3, 5)],  # central pillar/wall
        doors=[
            DoorOpening(position=(0, 2), width=1.0, exterior=True, door_id="north_west"),
            DoorOpening(position=(0, 8), width=1.5, exterior=True, door_id="north_east"),
            DoorOpening(position=(4, 5), width=2.0, exterior=True, door_id="south_center"),
        ],
    )

    # 2. Extract realistic candidates
    candidates = generate_candidate_exits(shared_floor_plan)
    assert len(candidates) == 3

    # 3. Generate 1-exit configurations: 3 choose 1 = 3 configs
    configs = list(generate_exit_configurations(candidates, exit_count=1))
    assert len(configs) == 3

    # 4. Shared occupants
    occupants = OccupantScenario(
        agents=[
            AgentConfig(agent_id=f"a_{i}", row=2, col=1 + i) for i in range(3)
        ]
    )

    # 5. Run simulations for each configuration
    results = []
    for cfg in configs:
        sim_cfg = SimulationConfig(
            scenario_name=f"sim_{cfg.configuration_id}",
            floor_plan=shared_floor_plan,
            exit_configuration=cfg,
            occupants=occupants,
        )
        sim = Simulation(sim_cfg, movement_strategy=ShortestPathStrategy())
        res = sim.run()
        assert res.evacuated_count == 3  # All evacuated
        assert res.total_timesteps > 0
        results.append(res.total_timesteps)

    # Verify Environment has exit widths accessible
    env = Environment.from_floor_plan_and_exits(shared_floor_plan, configs[0])
    assert env.get_exit_width(0, 2) == 1.0

    # Ensure shared FloorPlan was completely unmutated
    assert len(shared_floor_plan.walls) == 3
    assert len(shared_floor_plan.doors) == 3
    assert not hasattr(shared_floor_plan, "exits")


# ==============================================================================
# G. Scenario JSON File Validation
# ==============================================================================


def test_basic_building_scenario_json() -> None:
    """Test loading and candidate generation for updated basic_building.json."""
    scenario_path = Path("scenarios/basic_building.json")
    assert scenario_path.exists(), "basic_building.json missing"

    cfg = SimulationConfig.from_json(scenario_path)
    fp = cfg.floor_plan

    # Check doors parsed
    assert len(fp.doors) == 6
    ext_doors = [d for d in fp.doors if d.exterior]
    int_doors = [d for d in fp.doors if not d.exterior]

    assert len(ext_doors) == 4
    assert len(int_doors) == 2

    # Candidates derived strictly from exterior doors
    candidates = generate_candidate_exits(fp)
    assert len(candidates) == 4

    candidate_coords = {c.position for c in candidates}
    assert (0, 4) in candidate_coords
    assert (0, 5) in candidate_coords
    assert (9, 4) in candidate_coords
    assert (9, 5) in candidate_coords

    # Interior doors not in candidates
    assert (4, 4) not in candidate_coords
    assert (4, 5) not in candidate_coords

    # Simulation runs cleanly with existing reference configuration
    sim = Simulation(cfg, ShortestPathStrategy())
    res = sim.run()
    assert res.evacuated_count == len(cfg.agents)


def test_bottleneck_scenario_json() -> None:
    """Test loading and candidate generation for updated bottleneck.json."""
    scenario_path = Path("scenarios/bottleneck.json")
    assert scenario_path.exists(), "bottleneck.json missing"

    cfg = SimulationConfig.from_json(scenario_path)
    fp = cfg.floor_plan

    # Check doors parsed: 3 exterior + 1 interior bottleneck
    assert len(fp.doors) == 4
    ext_doors = [d for d in fp.doors if d.exterior]
    int_doors = [d for d in fp.doors if not d.exterior]

    assert len(ext_doors) == 3
    assert len(int_doors) == 1

    candidates = generate_candidate_exits(fp)
    assert len(candidates) == 3

    candidate_coords = {c.position for c in candidates}
    assert (0, 2) in candidate_coords
    assert (9, 2) in candidate_coords
    assert (9, 1) in candidate_coords
    assert (4, 2) not in candidate_coords

    # 3 choose 2 = 3 configurations
    configs = list(generate_exit_configurations(candidates, exit_count=2))
    assert len(configs) == 3

    # Simulation runs cleanly with reference exit
    sim = Simulation(cfg, ShortestPathStrategy())
    res = sim.run()
    assert res.evacuated_count == len(cfg.agents)
