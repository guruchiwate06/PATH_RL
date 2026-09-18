"""
test_environment.py
-------------------
Tests for the Environment class and its grid-query helpers.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import CellType, Environment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config() -> SimulationConfig:
    """Minimal valid config: 5×5 grid, one exit, no walls, no agents."""
    return SimulationConfig(
        scenario_name="test",
        grid={"rows": 5, "cols": 5},
        exits=[[4, 4]],
    )


@pytest.fixture()
def config_with_walls() -> SimulationConfig:
    """Config with walls and exits for boundary-case testing."""
    return SimulationConfig(
        scenario_name="walls_test",
        grid={"rows": 5, "cols": 5},
        walls=[[1, 0], [1, 1], [1, 2]],
        exits=[[4, 4]],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEnvironmentConstruction:
    def test_from_config_returns_environment(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert isinstance(env, Environment)

    def test_shape_matches_config(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.rows == 5
        assert env.cols == 5
        assert env.shape == (5, 5)

    def test_exit_cells_registered(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert (4, 4) in env.exits

    def test_wall_cells_registered(self, config_with_walls: SimulationConfig) -> None:
        env = Environment.from_config(config_with_walls)
        assert (1, 0) in env.walls
        assert (1, 1) in env.walls
        assert (1, 2) in env.walls

    def test_no_walls_on_open_config(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert len(env.walls) == 0


# ---------------------------------------------------------------------------
# Cell type queries
# ---------------------------------------------------------------------------


class TestCellTypeQueries:
    def test_open_cell_type(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.get_cell_type(0, 0) == CellType.OPEN

    def test_exit_cell_type(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.get_cell_type(4, 4) == CellType.EXIT

    def test_wall_cell_type(self, config_with_walls: SimulationConfig) -> None:
        env = Environment.from_config(config_with_walls)
        assert env.get_cell_type(1, 0) == CellType.WALL

    def test_get_cell_type_out_of_bounds_raises(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        with pytest.raises(IndexError):
            env.get_cell_type(10, 10)


# ---------------------------------------------------------------------------
# Passability
# ---------------------------------------------------------------------------


class TestPassability:
    def test_open_cell_is_passable(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.is_passable(0, 0) is True

    def test_exit_cell_is_passable(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.is_passable(4, 4) is True

    def test_wall_cell_is_not_passable(self, config_with_walls: SimulationConfig) -> None:
        env = Environment.from_config(config_with_walls)
        assert env.is_passable(1, 0) is False

    def test_out_of_bounds_is_not_passable(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.is_passable(-1, 0) is False
        assert env.is_passable(0, 99) is False


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------


class TestBoundsChecking:
    def test_valid_cell_is_within_bounds(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.is_within_bounds(0, 0) is True
        assert env.is_within_bounds(4, 4) is True

    def test_negative_row_is_out_of_bounds(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.is_within_bounds(-1, 0) is False

    def test_over_boundary_col_is_out_of_bounds(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        assert env.is_within_bounds(0, 5) is False


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------


class TestNeighbours:
    def test_corner_cell_has_two_neighbours(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        # (0,0) → right (0,1) and down (1,0)
        neighbours = env.neighbours(0, 0)
        assert len(neighbours) == 2
        assert (0, 1) in neighbours
        assert (1, 0) in neighbours

    def test_centre_cell_has_four_neighbours(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        neighbours = env.neighbours(2, 2)
        assert len(neighbours) == 4

    def test_wall_cell_excluded_from_neighbours(self, config_with_walls: SimulationConfig) -> None:
        env = Environment.from_config(config_with_walls)
        # (0,0) is adjacent to wall (1,0); wall must not be in neighbours
        neighbours = env.neighbours(0, 0)
        assert (1, 0) not in neighbours


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


class TestIteration:
    def test_iter_passable_cells_excludes_walls(
        self, config_with_walls: SimulationConfig
    ) -> None:
        env = Environment.from_config(config_with_walls)
        passable = list(env.iter_passable_cells())
        for wall in env.walls:
            assert wall not in passable

    def test_iter_exit_cells_returns_exits(self, minimal_config: SimulationConfig) -> None:
        env = Environment.from_config(minimal_config)
        exits = list(env.iter_exit_cells())
        assert (4, 4) in exits

    def test_iter_wall_cells_returns_walls(
        self, config_with_walls: SimulationConfig
    ) -> None:
        env = Environment.from_config(config_with_walls)
        wall_cells = list(env.iter_wall_cells())
        assert (1, 0) in wall_cells
