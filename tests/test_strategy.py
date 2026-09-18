"""
test_strategy.py
----------------
Unit tests for ShortestPathStrategy.

Tests are written against the strategy in isolation (not through
the full simulation loop) to verify:
  - Exit selection (nearest reachable exit chosen)
  - Unreachable exit handled gracefully
  - Returned next step is adjacent to the agent's current position
  - Agent already on exit gets no movement request
  - Agent in an isolated region gets no movement request
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.agent import Agent
from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.pathfinding import NavigationGraph
from evacuation_simulation.simulation.strategy import ShortestPathStrategy

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(
    rows: int,
    cols: int,
    exits: list[list[int]],
    walls: list[list[int]] | None = None,
    agents_cfg: list[dict] | None = None,
) -> tuple[Environment, NavigationGraph]:
    config = SimulationConfig(
        scenario_name="test",
        grid={"rows": rows, "cols": cols},
        walls=walls or [],
        exits=exits,
        agents=agents_cfg or [],
    )
    env = Environment.from_config(config)
    nav = NavigationGraph.from_environment(env)
    return env, nav


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dict(self) -> None:
        env, nav = _make(5, 5, [[4, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)
        result = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        assert isinstance(result, dict)

    def test_returns_empty_dict_for_no_agents(self) -> None:
        env, nav = _make(5, 5, [[4, 4]])
        strategy = ShortestPathStrategy()
        result = strategy.move_agents([], env, nav, _rng(), timestep=1)
        assert result == {}


# ---------------------------------------------------------------------------
# Next step is adjacent
# ---------------------------------------------------------------------------


class TestNextStepAdjacency:
    def test_requested_cell_is_adjacent_to_agent(self) -> None:
        """The strategy must request exactly one cardinal step."""
        env, nav = _make(5, 5, [[4, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)
        result = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        assert "a0" in result
        next_r, next_c = result["a0"]
        row_delta = abs(next_r - agent.row)
        col_delta = abs(next_c - agent.col)
        assert row_delta + col_delta == 1, (
            f"Step ({next_r},{next_c}) is not adjacent to ({agent.row},{agent.col})"
        )

    def test_requested_cell_is_passable(self) -> None:
        env, nav = _make(5, 5, [[4, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 2, 2)
        result = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        if "a0" in result:
            r, c = result["a0"]
            assert env.is_passable(r, c)


# ---------------------------------------------------------------------------
# Exit selection — nearest exit chosen
# ---------------------------------------------------------------------------


class TestExitSelection:
    def test_nearest_of_two_exits_chosen(self) -> None:
        """
        Agent at (0,0). Exit A at (0,1) (distance 1), Exit B at (4,4) (distance 8).
        The strategy should move the agent toward exit A (the closer one).
        """
        env, nav = _make(5, 5, exits=[[0, 1], [4, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)
        result = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        # Next step toward (0,1) from (0,0) is (0,1)
        assert "a0" in result
        assert result["a0"] == (0, 1)

    def test_farther_exit_not_chosen_when_nearer_reachable(self) -> None:
        """Path to exit at (0,4) is 4 steps; path to exit at (0,1) is 1 step."""
        env, nav = _make(5, 5, exits=[[0, 1], [0, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)
        result = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        assert result["a0"] == (0, 1)  # one step to the nearest exit


# ---------------------------------------------------------------------------
# Unreachable exit
# ---------------------------------------------------------------------------


class TestUnreachableExit:
    def test_unreachable_exit_not_selected(self) -> None:
        """
        Full wall at row 2 (all cols) blocks the path.
        Exit is at (4,2) (below wall). Agent is at (0,0) (above wall).
        No path → agent gets no request.
        """
        env, nav = _make(
            5, 5,
            exits=[[4, 2]],
            walls=[[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
        )
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)
        result = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        assert "a0" not in result

    def test_no_requests_when_no_exit_reachable(self) -> None:
        """All agents in isolated region → empty dict."""
        env, nav = _make(
            5, 5,
            exits=[[4, 2]],
            walls=[[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
        )
        strategy = ShortestPathStrategy()
        agents = [Agent(f"a{i}", 0, i) for i in range(3)]
        result = strategy.move_agents(agents, env, nav, _rng(), timestep=1)
        assert result == {}


# ---------------------------------------------------------------------------
# Agent already on exit
# ---------------------------------------------------------------------------


class TestAgentOnExit:
    def test_agent_on_exit_not_in_requests(self) -> None:
        """Agent starting on an exit cell should get no movement request."""
        env, nav = _make(5, 5, exits=[[0, 0]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)
        result = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        assert "a0" not in result


# ---------------------------------------------------------------------------
# Path cache validation
# ---------------------------------------------------------------------------


class TestPathCache:
    def test_cache_used_on_second_call(self) -> None:
        """
        Calling move_agents twice for the same agent (without moving) should
        return the same next step both times (cache hit on second call).
        """
        env, nav = _make(5, 5, exits=[[4, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)

        result1 = strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        result2 = strategy.move_agents([agent], env, nav, _rng(), timestep=2)

        assert result1 == result2

    def test_cache_invalidated_when_agent_moves_off_path(self) -> None:
        """
        If the agent is moved to a position that doesn't match the cached
        path head, the strategy recomputes a valid path.
        """
        env, nav = _make(5, 5, exits=[[4, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)

        # Populate cache
        strategy.move_agents([agent], env, nav, _rng(), timestep=1)

        # Simulate conflict rejection — agent ended up at a different cell
        agent.set_position(0, 2)

        result = strategy.move_agents([agent], env, nav, _rng(), timestep=2)

        # Should still produce a valid adjacent step (recomputed)
        if "a0" in result:
            r, c = result["a0"]
            row_delta = abs(r - agent.row)
            col_delta = abs(c - agent.col)
            assert row_delta + col_delta == 1

    def test_clear_cache_for_agent(self) -> None:
        env, nav = _make(5, 5, exits=[[4, 4]])
        strategy = ShortestPathStrategy()
        agent = Agent("a0", 0, 0)
        strategy.move_agents([agent], env, nav, _rng(), timestep=1)
        assert len(strategy._path_cache) == 1
        strategy.clear_cache_for("a0")
        assert len(strategy._path_cache) == 0

    def test_clear_all_caches(self) -> None:
        env, nav = _make(5, 5, exits=[[4, 4]])
        strategy = ShortestPathStrategy()
        for i in range(3):
            a = Agent(f"a{i}", 0, i)
            strategy.move_agents([a], env, nav, _rng(), timestep=1)
        assert len(strategy._path_cache) == 3
        strategy.clear_all_caches()
        assert len(strategy._path_cache) == 0
