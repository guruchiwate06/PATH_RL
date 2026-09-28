"""
test_stage8_architecture.py
---------------------------
Stage 8: Explicit FloorPlan + ExitConfiguration Architecture.

Tests that the architecture explicitly separates building geometry
from exit configurations and occupant scenarios, and that the same
FloorPlan can be reused independently across multiple simulations.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.config import (
    AgentConfig,
    ExitConfiguration,
    FloorPlan,
    OccupantScenario,
    SimulationConfig,
)
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


def test_floor_plan_isolation() -> None:
    """FloorPlan contains only grid and walls, no exits or agents."""
    fp = FloorPlan(
        grid={"rows": 5, "cols": 5},
        walls=[(1, 1), (2, 2)]
    )
    assert not hasattr(fp, "exits")
    assert not hasattr(fp, "agents")
    assert fp.grid.rows == 5


def test_exit_configuration_isolation() -> None:
    """ExitConfiguration contains exits and ID, no geometry."""
    ec = ExitConfiguration(
        configuration_id="candidate_1",
        exits=[(0, 0), (4, 4)]
    )
    assert ec.configuration_id == "candidate_1"
    assert len(ec.exits) == 2
    assert not hasattr(ec, "grid")
    assert not hasattr(ec, "walls")


def test_occupant_scenario_isolation() -> None:
    """OccupantScenario contains agents, no geometry or exits."""
    osc = OccupantScenario(
        agents=[AgentConfig(agent_id="a1", row=0, col=0)]
    )
    assert len(osc.agents) == 1
    assert not hasattr(osc, "grid")
    assert not hasattr(osc, "exits")


def test_environment_from_floor_plan_and_exits() -> None:
    """Environment can be built directly from separated domain objects."""
    fp = FloorPlan(grid={"rows": 5, "cols": 5}, walls=[(2, 2)])
    ec = ExitConfiguration(configuration_id="test", exits=[(4, 4)])
    
    env = Environment.from_floor_plan_and_exits(fp, ec)
    assert env.rows == 5
    assert env.is_wall(2, 2)
    assert env.is_exit(4, 4)


def test_same_floor_plan_multiple_exit_configurations() -> None:
    """The same FloorPlan object can be evaluated with different ExitConfigurations."""
    # 1. Create a single, shared FloorPlan
    shared_floor_plan = FloorPlan(
        grid={"rows": 3, "cols": 12},
        walls=[(0, c) for c in range(12)] + [(2, c) for c in range(12)]
    )
    
    # 2. Create shared occupants
    shared_occupants = OccupantScenario(
        agents=[
            AgentConfig(agent_id=f"a{i}", row=1, col=5+i) for i in range(6)
        ]
    )

    # 3. Create independent exit configurations
    ref_exits = ExitConfiguration(configuration_id="reference", exits=[(1, 11)])
    alt_exits = ExitConfiguration(configuration_id="alternative", exits=[(1, 0)])

    # 4. Construct simulations reusing the shared objects
    cfg_ref = SimulationConfig(
        scenario_name="ref_sim",
        floor_plan=shared_floor_plan,
        exit_configuration=ref_exits,
        occupants=shared_occupants,
    )
    cfg_alt = SimulationConfig(
        scenario_name="alt_sim",
        floor_plan=shared_floor_plan,
        exit_configuration=alt_exits,
        occupants=shared_occupants,
    )

    sim_ref = Simulation(cfg_ref, ShortestPathStrategy())
    sim_alt = Simulation(cfg_alt, ShortestPathStrategy())

    res_ref = sim_ref.run()
    res_alt = sim_alt.run()

    # Verify simulations produced different results
    assert res_ref.total_timesteps != res_alt.total_timesteps
    
    # Verify the shared FloorPlan was not mutated
    assert len(shared_floor_plan.walls) == 24
    assert not hasattr(shared_floor_plan, "exits")

def test_legacy_flat_schema_migration() -> None:
    """Ensure flat dictionary kwargs are migrated to nested domain objects."""
    flat_data = {
        "scenario_name": "flat",
        "grid": {"rows": 5, "cols": 5},
        "walls": [[1, 1]],
        "exits": [[4, 4]],
        "agents": [{"agent_id": "a1", "row": 2, "col": 2}],
    }
    cfg = SimulationConfig(**flat_data)
    
    # Nested objects should be populated
    assert cfg.floor_plan.grid.rows == 5
    assert (1, 1) in cfg.floor_plan.walls
    assert (4, 4) in cfg.exit_configuration.exits
    assert cfg.occupants.agents[0].agent_id == "a1"
    
    # Backward compatibility properties should work
    assert cfg.grid.rows == 5
    assert (1, 1) in cfg.walls
    assert (4, 4) in cfg.exits
    assert cfg.agents[0].agent_id == "a1"
