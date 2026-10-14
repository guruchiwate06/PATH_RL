"""
test_occupant_seeder.py
-----------------------
Unit tests for OccupantSeeder.
"""

import pytest
from evacuation_simulation.simulation.config import (
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridConfig,
)
from evacuation_simulation.simulation.occupancy_seeder import (
    OccupantSeeder,
    OccupantSeederConfig,
)


@pytest.fixture
def sample_floor_plan():
    # 10x10 grid with perimeter walls and 2 door openings
    walls = []
    for c in range(10):
        walls.append((0, c))
        walls.append((9, c))
    for r in range(10):
        walls.append((r, 0))
        walls.append((r, 9))

    # Remove 2 door cells
    walls.remove((0, 5))
    walls.remove((9, 5))

    doors = [
        DoorOpening(door_id="EXIT_NORTH", position=(0, 5), width=1.5, exterior=True),
        DoorOpening(door_id="EXIT_SOUTH", position=(9, 5), width=1.5, exterior=True),
    ]

    return FloorPlan(
        grid=GridConfig(rows=10, cols=10),
        walls=sorted(walls),
        doors=doors,
    )


def test_seeder_places_requested_count(sample_floor_plan):
    """Seeds exact number of requested occupants."""
    cfg = OccupantSeederConfig(total_occupants=15, seed=123)
    scenario = OccupantSeeder.seed(sample_floor_plan, config=cfg)
    assert len(scenario.agents) == 15


def test_seeder_never_places_on_walls_or_exits(sample_floor_plan):
    """Occupants must not spawn on walls or exit cells."""
    exit_cfg = ExitConfiguration(exits=[(0, 5), (9, 5)], widths=[1.5, 1.5])
    cfg = OccupantSeederConfig(total_occupants=20, seed=42, allow_exit_cells=False)
    scenario = OccupantSeeder.seed(sample_floor_plan, exit_config=exit_cfg, config=cfg)

    wall_set = set(map(tuple, sample_floor_plan.walls))
    exit_set = set(map(tuple, exit_cfg.exits))

    for agent in scenario.agents:
        pos = (agent.row, agent.col)
        assert pos not in wall_set, f"Agent {agent.agent_id} placed on wall {pos}"
        assert pos not in exit_set, f"Agent {agent.agent_id} placed on exit {pos}"
        assert 0 <= agent.row < sample_floor_plan.grid.rows
        assert 0 <= agent.col < sample_floor_plan.grid.cols


def test_seeder_deterministic_reproducibility(sample_floor_plan):
    """Same seed yields identical agent positions."""
    cfg1 = OccupantSeederConfig(total_occupants=10, seed=777)
    cfg2 = OccupantSeederConfig(total_occupants=10, seed=777)

    scen1 = OccupantSeeder.seed(sample_floor_plan, config=cfg1)
    scen2 = OccupantSeeder.seed(sample_floor_plan, config=cfg2)

    pos1 = [(a.row, a.col) for a in scen1.agents]
    pos2 = [(a.row, a.col) for a in scen2.agents]
    assert pos1 == pos2


def test_seeder_capacity_exceeded(sample_floor_plan):
    """Raises ValueError when requested occupants exceed available walkable capacity."""
    # Walkable cells in 10x10 with perimeter walls is 8x8 = 64 cells
    cfg = OccupantSeederConfig(total_occupants=100, max_occupants_per_cell=1)
    with pytest.raises(ValueError, match="walkable cells available"):
        OccupantSeeder.seed(sample_floor_plan, config=cfg)
