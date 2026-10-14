"""
test_rl_edge_cases.py
---------------------
Unit tests for edge-case guards in ExitPlacementEnv:
- Too-few-exits handling
- Narrow-door masking and rejection
- All-invalid candidate sets
- Dead-end and illegal action terminations
"""

import pytest
from evacuation_simulation.simulation.config import (
    DoorOpening,
    ExitConfiguration,
    FloorPlan,
    GridConfig,
    OccupantScenario,
    SimulationConfig,
    SimulationParameters,
)
from evacuation_simulation.simulation.generation import CandidateExit
from evacuation_simulation.simulation.rl_environment import ExitPlacementEnv


@pytest.fixture
def base_scenario():
    # 10x10 with 3 doors:
    # Door 1: wide exterior (1.5m)
    # Door 2: wide exterior (1.2m)
    # Door 3: narrow exterior (0.7m < 1.0m threshold)
    walls = []
    for c in range(10):
        walls.append((0, c))
        walls.append((9, c))
    for r in range(10):
        walls.append((r, 0))
        walls.append((r, 9))

    for pos in [(0, 3), (0, 7), (9, 5)]:
        if pos in walls:
            walls.remove(pos)

    doors = [
        DoorOpening(door_id="WIDE_1", position=(0, 3), width=1.5, exterior=True),
        DoorOpening(door_id="WIDE_2", position=(0, 7), width=1.2, exterior=True),
        DoorOpening(door_id="NARROW_1", position=(9, 5), width=0.7, exterior=True),
    ]

    fp = FloorPlan(
        grid=GridConfig(rows=10, cols=10),
        walls=sorted(walls),
        doors=doors,
    )

    return SimulationConfig(
        scenario_name="test_edge_cases",
        floor_plan=fp,
        exit_configuration=ExitConfiguration(exits=[(0, 3)], widths=[1.5]),
        occupants=OccupantScenario(agents=[]),
        parameters=SimulationParameters(),
    )


def test_narrow_door_is_masked_out(base_scenario):
    """Candidate narrower than 1.0m has action_mask=False."""
    env = ExitPlacementEnv()
    obs = env.reset(base_scenario, exit_count=2)

    # 3 candidates: indices 0 (1.5m), 1 (1.2m), 2 (0.7m)
    mask = env.action_mask()
    assert mask[0] is True
    assert mask[1] is True
    assert mask[2] is False, "Narrow door (0.7m) must be masked out as False"


def test_step_on_narrow_door_penalized(base_scenario):
    """Attempting to select a narrow door terminates with PENALTY_INVALID_ACTION."""
    env = ExitPlacementEnv()
    env.reset(base_scenario, exit_count=2)

    # Action 2 is the narrow door (0.7m)
    obs, reward, terminated, info = env.step(action=2)
    assert terminated is True
    assert reward == env.PENALTY_INVALID_ACTION
    assert info["reason"] == "invalid_action"


def test_all_invalid_candidates_rejected_on_reset():
    """If all candidate doors are narrow (<1.0m), reset() raises ValueError."""
    walls = [(0, 0), (0, 1), (0, 3), (0, 4)]
    doors = [
        DoorOpening(door_id="NARROW_A", position=(0, 2), width=0.8, exterior=True),
    ]
    fp = FloorPlan(grid=GridConfig(rows=5, cols=5), walls=walls, doors=doors)
    cfg = SimulationConfig(
        scenario_name="all_narrow",
        floor_plan=fp,
        exit_configuration=ExitConfiguration(exits=[(0, 2)], widths=[0.8]),
    )

    env = ExitPlacementEnv()
    with pytest.raises(ValueError, match="Cannot satisfy requested exit_count=1"):
        env.reset(cfg, exit_count=1)


def test_too_few_valid_candidates_raises_error(base_scenario):
    """If requested exit_count exceeds count of valid (non-narrow) exits, raises ValueError."""
    env = ExitPlacementEnv()
    # Scenario has 3 doors, but only 2 are wide (>=1.0m). Requesting 3 exits is impossible!
    with pytest.raises(ValueError, match="Cannot satisfy requested exit_count=3"):
        env.reset(base_scenario, exit_count=3)


def test_exit_count_less_than_one_raises_error(base_scenario):
    """exit_count < 1 raises ValueError."""
    env = ExitPlacementEnv()
    with pytest.raises(ValueError, match="must be at least 1"):
        env.reset(base_scenario, exit_count=0)
