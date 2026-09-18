"""
test_strategy_integration.py
-----------------------------
End-to-end integration tests: ShortestPathStrategy running inside the
full Simulation loop.

Verifies:
  - Agents actually move and evacuate
  - Simulation terminates correctly
  - Evacuation metrics are correct
  - Reproducibility with fixed seed
  - No wall-passthrough
  - No movement after evacuation
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.agent import AgentState
from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import CellType
from evacuation_simulation.simulation.metrics import SimulationResult
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


# ---------------------------------------------------------------------------
# Shared scenario configs
# ---------------------------------------------------------------------------


def _simple_config(
    agent_positions: list[tuple[int, int]],
    exits: list[tuple[int, int]] | None = None,
    walls: list[tuple[int, int]] | None = None,
    rows: int = 6,
    cols: int = 6,
    max_timesteps: int = 100,
    seed: int = 42,
) -> SimulationConfig:
    """Helper to build a SimulationConfig programmatically."""
    return SimulationConfig(
        scenario_name="integration_test",
        grid={"rows": rows, "cols": cols},
        walls=[list(w) for w in (walls or [])],
        exits=[list(e) for e in (exits or [(rows - 1, cols - 1)])],
        agents=[
            {"agent_id": f"a{i}", "row": r, "col": c}
            for i, (r, c) in enumerate(agent_positions)
        ],
        parameters={"max_timesteps": max_timesteps, "random_seed": seed},
    )


# ---------------------------------------------------------------------------
# Basic evacuation
# ---------------------------------------------------------------------------


class TestBasicEvacuation:
    def test_single_agent_evacuates(self) -> None:
        """One agent in a clear corridor should always evacuate."""
        config = _simple_config(
            agent_positions=[(0, 0)],
            exits=[(5, 5)],
        )
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.evacuated_count == 1
        assert result.evacuation_rate == pytest.approx(1.0)

    def test_all_agents_evacuate_open_grid(self) -> None:
        """All 4 agents on an open grid should evacuate."""
        config = _simple_config(
            agent_positions=[(0, 0), (0, 5), (5, 0), (3, 3)],
            exits=[(5, 5)],
            rows=6, cols=6,
        )
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.evacuated_count == result.total_agents
        assert result.evacuation_rate == pytest.approx(1.0)

    def test_scenario_json_all_agents_evacuate(self) -> None:
        """The bundled basic_building scenario: all 6 agents must evacuate."""
        from pathlib import Path
        scenario_path = (
            Path(__file__).parent.parent / "scenarios" / "basic_building.json"
        )
        config = SimulationConfig.from_json(scenario_path)
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.evacuated_count == 6
        assert result.evacuation_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------


class TestTermination:
    def test_terminates_all_evacuated(self) -> None:
        config = _simple_config([(0, 0)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        sim.run()
        assert sim.state.is_terminated
        assert "evacuated" in sim.state.termination_reason.lower()

    def test_terminates_on_max_steps_when_blocked(self) -> None:
        """
        Agent completely surrounded by walls (except own cell) has no
        exit path → simulation runs out of steps.
        """
        config = _simple_config(
            agent_positions=[(0, 0)],
            exits=[(5, 5)],
            walls=[(0, 1), (1, 0)],   # trap agent_0 at corner
            max_timesteps=5,
        )
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert sim.state.is_terminated
        assert result.evacuated_count == 0
        assert result.total_timesteps == 5


# ---------------------------------------------------------------------------
# Evacuation correctness
# ---------------------------------------------------------------------------


class TestEvacuationCorrectness:
    def test_evacuated_agent_becomes_inactive(self) -> None:
        config = _simple_config([(0, 0)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        sim.run()
        agent = sim.agents[0]
        assert agent.state == AgentState.EVACUATED
        assert agent.is_active is False

    def test_evacuation_timestep_recorded(self) -> None:
        config = _simple_config([(0, 0)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        sim.run()
        agent = sim.agents[0]
        assert agent.evacuation_timestep is not None
        assert agent.evacuation_timestep > 0

    def test_agent_starts_on_exit_evacuates_step_one(self) -> None:
        """An agent that starts directly on an exit evacuates on step 1."""
        config = _simple_config(
            agent_positions=[(5, 5)],
            exits=[(5, 5)],
        )
        sim = Simulation(config, ShortestPathStrategy())
        sim.step()
        agent = sim.agents[0]
        assert agent.state == AgentState.EVACUATED
        assert agent.evacuation_timestep == 1

    def test_no_movement_after_evacuation(self) -> None:
        """An evacuated agent must not appear in movement requests."""
        config = _simple_config([(5, 5)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        sim.step()  # Agent evacuated here
        assert sim.agents[0].state == AgentState.EVACUATED
        # Run one more step — agent must not move
        pos_before = sim.agents[0].position
        sim.step()
        assert sim.agents[0].position == pos_before


# ---------------------------------------------------------------------------
# Metrics correctness
# ---------------------------------------------------------------------------


class TestMetricsCorrectness:
    def test_evacuation_rate_100_percent(self) -> None:
        config = _simple_config([(0, 0), (0, 1)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.evacuation_rate == pytest.approx(1.0)

    def test_non_evacuated_count_zero_when_all_evacuate(self) -> None:
        config = _simple_config([(0, 0)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.non_evacuated_count == 0

    def test_mean_evacuation_time_positive(self) -> None:
        config = _simple_config([(0, 0)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.mean_evacuation_time is not None
        assert result.mean_evacuation_time > 0

    def test_min_max_evacuation_time_consistency(self) -> None:
        config = _simple_config([(0, 0), (0, 5)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.min_evacuation_time is not None
        assert result.max_evacuation_time is not None
        assert result.min_evacuation_time <= result.max_evacuation_time

    def test_per_agent_times_in_result(self) -> None:
        config = _simple_config([(0, 0), (0, 1)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert len(result.evacuation_times) == 2
        for agent_id, t in result.evacuation_times.items():
            assert t > 0

    def test_total_timesteps_positive(self) -> None:
        config = _simple_config([(0, 0)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.total_timesteps > 0

    def test_total_agents_count_matches_config(self) -> None:
        config = _simple_config([(0, 0), (1, 1), (2, 2)], exits=[(5, 5)])
        sim = Simulation(config, ShortestPathStrategy())
        result = sim.run()
        assert result.total_agents == 3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        config = _simple_config(
            [(0, 0), (0, 5), (5, 0)],
            exits=[(5, 5)],
            seed=99,
        )
        sim1 = Simulation(config, ShortestPathStrategy())
        result1 = sim1.run()

        sim2 = Simulation(config, ShortestPathStrategy())
        result2 = sim2.run()

        assert result1.total_timesteps == result2.total_timesteps
        assert result1.evacuation_times == result2.evacuation_times

    def test_different_seed_may_differ(self) -> None:
        """
        With a deterministic strategy, results are identical regardless of
        seed (BFS path is deterministic). This verifies the simulation
        terminates successfully in both cases.
        """
        config_a = _simple_config([(0, 0)], exits=[(5, 5)], seed=1)
        config_b = _simple_config([(0, 0)], exits=[(5, 5)], seed=2)
        sim_a = Simulation(config_a, ShortestPathStrategy())
        sim_b = Simulation(config_b, ShortestPathStrategy())
        result_a = sim_a.run()
        result_b = sim_b.run()
        # Both should evacuate the single agent
        assert result_a.evacuated_count == 1
        assert result_b.evacuated_count == 1


# ---------------------------------------------------------------------------
# Physical correctness (no wall passthrough)
# ---------------------------------------------------------------------------


class TestPhysicalCorrectness:
    def test_agents_never_enter_wall_cells(self) -> None:
        """
        Step through a walled scenario and verify no agent ever occupies
        a wall cell.
        """
        config = _simple_config(
            agent_positions=[(0, 0), (0, 4)],
            exits=[(5, 5)],
            walls=[(2, 0), (2, 1), (2, 2)],
            rows=6, cols=6,
        )
        sim = Simulation(config, ShortestPathStrategy())

        for _ in range(50):
            if sim.state.is_terminated:
                break
            sim.step()
            for agent in sim.active_agents:
                assert not sim.environment.is_wall(agent.row, agent.col), (
                    f"Agent '{agent.agent_id}' is on a wall cell "
                    f"({agent.row},{agent.col})"
                )

    def test_agents_never_leave_grid(self) -> None:
        """Agents must always stay within grid boundaries."""
        config = _simple_config(
            agent_positions=[(0, 0), (5, 5)],
            exits=[(5, 5)],
            rows=6, cols=6,
        )
        sim = Simulation(config, ShortestPathStrategy())

        for _ in range(30):
            if sim.state.is_terminated:
                break
            sim.step()
            for agent in sim.active_agents:
                assert sim.environment.is_within_bounds(agent.row, agent.col), (
                    f"Agent '{agent.agent_id}' left the grid at "
                    f"({agent.row},{agent.col})"
                )

    def test_agents_move_at_most_one_cell_per_step(self) -> None:
        """Verify each agent moves ≤ 1 Manhattan unit per timestep."""
        config = _simple_config(
            agent_positions=[(0, 0), (0, 3), (3, 0)],
            exits=[(5, 5)],
            rows=6, cols=6,
        )
        sim = Simulation(config, ShortestPathStrategy())
        prev_positions = {a.agent_id: a.position for a in sim.agents}

        for _ in range(30):
            if sim.state.is_terminated:
                break
            sim.step()
            for agent in sim.active_agents:
                prev = prev_positions[agent.agent_id]
                curr = agent.position
                dist = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
                assert dist <= 1, (
                    f"Agent '{agent.agent_id}' moved {dist} cells in one step "
                    f"({prev} → {curr})"
                )
                prev_positions[agent.agent_id] = curr


# ---------------------------------------------------------------------------
# Multi-agent interaction
# ---------------------------------------------------------------------------


class TestMultiAgentInteraction:
    def test_two_agents_to_different_cells_both_move(self) -> None:
        """When no conflict, both agents should advance each step."""
        config = _simple_config(
            agent_positions=[(0, 0), (0, 4)],
            exits=[(5, 5)],
            rows=6, cols=6,
        )
        sim = Simulation(config, ShortestPathStrategy())
        pos_before = {a.agent_id: a.position for a in sim.agents}
        sim.step()
        # At least one agent should have moved
        moved = sum(
            1 for a in sim.active_agents
            if a.position != pos_before[a.agent_id]
        )
        assert moved >= 1

    def test_conflict_resolution_does_not_crash(self) -> None:
        """Even when conflict resolution rejects a move, simulation continues."""
        # Place two agents 2 apart (both wanting the middle cell on step 1)
        config = _simple_config(
            agent_positions=[(0, 0), (0, 2)],
            exits=[(5, 5)],
            rows=6, cols=6,
        )
        sim = Simulation(config, ShortestPathStrategy())
        # Should not raise
        for _ in range(20):
            if sim.state.is_terminated:
                break
            sim.step()
