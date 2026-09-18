"""
test_simulation.py
------------------
Tests for the Simulation class, movement strategy protocol,
and the discrete-timestep loop.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.agent import Agent, AgentState
from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.pathfinding import NavigationGraph
from evacuation_simulation.simulation.metrics import SimulationResult
from evacuation_simulation.simulation.simulation import (
    MovementStrategy,
    NullMovementStrategy,
    Simulation,
)

import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trivial_config() -> SimulationConfig:
    """
    3×3 grid, agents start next to exit so they can reach it in one step
    if moved — but with NullStrategy they will never move.
    """
    return SimulationConfig(
        scenario_name="trivial",
        grid={"rows": 3, "cols": 3},
        exits=[[2, 2]],
        agents=[
            {"agent_id": "a0", "row": 0, "col": 0},
            {"agent_id": "a1", "row": 0, "col": 2},
        ],
        parameters={"max_timesteps": 10, "random_seed": 0},
    )


@pytest.fixture()
def on_exit_config() -> SimulationConfig:
    """
    Config where one agent starts directly on the exit cell.
    This agent should evacuate on the very first timestep.
    """
    return SimulationConfig(
        scenario_name="on_exit",
        grid={"rows": 3, "cols": 3},
        exits=[[0, 0]],
        agents=[
            {"agent_id": "a0", "row": 0, "col": 0},
            {"agent_id": "a1", "row": 2, "col": 2},
        ],
        parameters={"max_timesteps": 20, "random_seed": 1},
    )


# ---------------------------------------------------------------------------
# Simulation construction
# ---------------------------------------------------------------------------


class TestSimulationConstruction:
    def test_simulation_instantiates(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        assert isinstance(sim, Simulation)

    def test_environment_constructed(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        assert isinstance(sim.environment, Environment)

    def test_nav_graph_constructed(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        assert isinstance(sim.nav_graph, NavigationGraph)

    def test_agents_created_from_config(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        assert len(sim.agents) == 2
        agent_ids = {a.agent_id for a in sim.agents}
        assert agent_ids == {"a0", "a1"}

    def test_initial_timestep_is_zero(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        assert sim.state.timestep == 0

    def test_initial_not_terminated(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        assert sim.state.is_terminated is False

    def test_default_strategy_is_null(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        assert isinstance(sim._strategy, NullMovementStrategy)


# ---------------------------------------------------------------------------
# Null movement strategy
# ---------------------------------------------------------------------------


class TestNullStrategy:
    def test_agents_do_not_move(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        initial_positions = {a.agent_id: a.position for a in sim.agents}
        sim.step()
        for agent in sim.agents:
            assert agent.position == initial_positions[agent.agent_id]

    def test_null_strategy_satisfies_protocol(self) -> None:
        assert isinstance(NullMovementStrategy(), MovementStrategy)


# ---------------------------------------------------------------------------
# Timestep execution
# ---------------------------------------------------------------------------


class TestTimestepExecution:
    def test_step_increments_timestep(self, trivial_config: SimulationConfig) -> None:
        sim = Simulation(trivial_config)
        sim.step()
        assert sim.state.timestep == 1
        sim.step()
        assert sim.state.timestep == 2

    def test_step_returns_true_while_running(
        self, trivial_config: SimulationConfig
    ) -> None:
        sim = Simulation(trivial_config)
        result = sim.step()
        assert result is True  # Should continue (agents not evacuated yet)

    def test_step_returns_false_after_termination(
        self, trivial_config: SimulationConfig
    ) -> None:
        sim = Simulation(trivial_config)
        # Manually terminate
        sim.state.is_terminated = True
        result = sim.step()
        assert result is False

    def test_max_timesteps_terminates_simulation(
        self, trivial_config: SimulationConfig
    ) -> None:
        sim = Simulation(trivial_config)
        result = sim.run()
        assert sim.state.is_terminated is True
        # NullStrategy → no evacuations → hits max_timesteps (10)
        assert sim.state.timestep == 10


# ---------------------------------------------------------------------------
# Agent-on-exit evacuation detection
# ---------------------------------------------------------------------------


class TestEvacuationDetection:
    def test_agent_on_exit_evacuates_on_first_step(
        self, on_exit_config: SimulationConfig
    ) -> None:
        sim = Simulation(on_exit_config)
        sim.step()
        # a0 starts on exit (0,0) — must be evacuated after step 1
        a0 = next(a for a in sim.agents if a.agent_id == "a0")
        assert a0.state == AgentState.EVACUATED
        assert a0.evacuation_timestep == 1

    def test_agent_not_on_exit_stays_active(
        self, on_exit_config: SimulationConfig
    ) -> None:
        sim = Simulation(on_exit_config)
        sim.step()
        a1 = next(a for a in sim.agents if a.agent_id == "a1")
        assert a1.state == AgentState.MOVING

    def test_metrics_records_evacuation(
        self, on_exit_config: SimulationConfig
    ) -> None:
        sim = Simulation(on_exit_config)
        sim.step()
        assert sim.metrics.evacuated_count == 1

    def test_active_agents_filtered_correctly(
        self, on_exit_config: SimulationConfig
    ) -> None:
        sim = Simulation(on_exit_config)
        sim.step()
        assert len(sim.active_agents) == 1
        assert len(sim.evacuated_agents) == 1


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_run_returns_simulation_result(
        self, trivial_config: SimulationConfig
    ) -> None:
        sim = Simulation(trivial_config)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_result_total_agents_matches_config(
        self, trivial_config: SimulationConfig
    ) -> None:
        sim = Simulation(trivial_config)
        result = sim.run()
        assert result.total_agents == 2

    def test_result_scenario_name_matches(
        self, trivial_config: SimulationConfig
    ) -> None:
        sim = Simulation(trivial_config)
        result = sim.run()
        assert result.scenario_name == "trivial"

    def test_no_evacuations_with_null_strategy(
        self, trivial_config: SimulationConfig
    ) -> None:
        sim = Simulation(trivial_config)
        result = sim.run()
        assert result.evacuated_count == 0
        assert result.evacuation_rate == 0.0

    def test_reproducibility_with_seed(
        self, trivial_config: SimulationConfig
    ) -> None:
        """Two simulations with the same seed must produce identical results."""
        sim1 = Simulation(trivial_config)
        result1 = sim1.run()

        sim2 = Simulation(trivial_config)
        result2 = sim2.run()

        assert result1.total_timesteps == result2.total_timesteps
        assert result1.evacuated_count == result2.evacuated_count
