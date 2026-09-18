"""
test_pathfinding.py
-------------------
Tests for the NavigationGraph — graph construction, node/edge queries,
connectivity analysis, and path-finding.
"""

from __future__ import annotations

import pytest

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.environment import Environment
from evacuation_simulation.simulation.pathfinding import NavigationGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def open_3x3_config() -> SimulationConfig:
    """3×3 all-open grid with one exit at (2,2)."""
    return SimulationConfig(
        scenario_name="open_3x3",
        grid={"rows": 3, "cols": 3},
        exits=[[2, 2]],
    )


@pytest.fixture()
def walled_config() -> SimulationConfig:
    """
    5×5 grid with a horizontal wall cutting the grid in two.
    Exit is in the lower half — agents in the upper half cannot reach it.
    The wall runs from (2,0) to (2,4) with no gap → graph is disconnected.
    """
    return SimulationConfig(
        scenario_name="walled",
        grid={"rows": 5, "cols": 5},
        walls=[[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
        exits=[[4, 2]],
    )


@pytest.fixture()
def nav_3x3(open_3x3_config: SimulationConfig) -> NavigationGraph:
    env = Environment.from_config(open_3x3_config)
    return NavigationGraph.from_environment(env)


@pytest.fixture()
def nav_walled(walled_config: SimulationConfig) -> NavigationGraph:
    env = Environment.from_config(walled_config)
    return NavigationGraph.from_environment(env)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_node_count_equals_passable_cells(
        self, open_3x3_config: SimulationConfig, nav_3x3: NavigationGraph
    ) -> None:
        env = Environment.from_config(open_3x3_config)
        passable = list(env.iter_passable_cells())
        assert nav_3x3.node_count == len(passable)

    def test_all_3x3_cells_are_nodes(self, nav_3x3: NavigationGraph) -> None:
        for r in range(3):
            for c in range(3):
                assert nav_3x3.has_node((r, c))

    def test_wall_cells_not_in_graph(
        self, walled_config: SimulationConfig, nav_walled: NavigationGraph
    ) -> None:
        for r, c in walled_config.walls:
            assert not nav_walled.has_node((r, c))

    def test_adjacent_open_cells_share_edge(self, nav_3x3: NavigationGraph) -> None:
        # (0,0) and (0,1) are adjacent — must have an edge
        assert nav_3x3.has_edge((0, 0), (0, 1))
        assert nav_3x3.has_edge((0, 1), (0, 0))  # undirected

    def test_non_adjacent_cells_have_no_edge(self, nav_3x3: NavigationGraph) -> None:
        # (0,0) and (2,2) are not adjacent
        assert not nav_3x3.has_edge((0, 0), (2, 2))

    def test_from_config_factory(self, open_3x3_config: SimulationConfig) -> None:
        nav = NavigationGraph.from_config(open_3x3_config)
        assert isinstance(nav, NavigationGraph)
        assert nav.node_count == 9  # 3×3 open grid


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------


class TestGraphNeighbours:
    def test_corner_has_two_neighbours(self, nav_3x3: NavigationGraph) -> None:
        neighbours = nav_3x3.neighbours_of((0, 0))
        assert len(neighbours) == 2
        assert (0, 1) in neighbours
        assert (1, 0) in neighbours

    def test_centre_has_four_neighbours(self, nav_3x3: NavigationGraph) -> None:
        neighbours = nav_3x3.neighbours_of((1, 1))
        assert len(neighbours) == 4

    def test_neighbours_of_nonexistent_node_raises(
        self, nav_3x3: NavigationGraph
    ) -> None:
        with pytest.raises(KeyError):
            nav_3x3.neighbours_of((99, 99))


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


class TestConnectivity:
    def test_open_grid_is_connected(self, nav_3x3: NavigationGraph) -> None:
        assert nav_3x3.is_connected() is True

    def test_fully_walled_grid_is_disconnected(
        self, nav_walled: NavigationGraph
    ) -> None:
        # The wall at row 2 splits the graph into two components
        assert nav_walled.is_connected() is False

    def test_exit_reachable_from_lower_half(
        self, nav_walled: NavigationGraph
    ) -> None:
        # (3, 2) is in the lower half — exit (4,2) is reachable
        assert nav_walled.all_exits_reachable_from((3, 2)) is True

    def test_exit_not_reachable_from_upper_half(
        self, nav_walled: NavigationGraph
    ) -> None:
        # (0, 0) is in the upper half — exit is behind the wall
        assert nav_walled.all_exits_reachable_from((0, 0)) is False


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------


class TestShortestPath:
    def test_path_from_corner_to_corner(self, nav_3x3: NavigationGraph) -> None:
        path = nav_3x3.shortest_path((0, 0), (2, 2))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    def test_path_length_on_open_grid(self, nav_3x3: NavigationGraph) -> None:
        # Manhattan distance from (0,0) to (2,2) is 4 steps → 5 cells
        path = nav_3x3.shortest_path((0, 0), (2, 2))
        assert path is not None
        assert len(path) == 5

    def test_path_is_none_when_blocked(
        self, nav_walled: NavigationGraph
    ) -> None:
        # Upper half cannot reach lower half exit
        path = nav_walled.shortest_path((0, 0), (4, 2))
        assert path is None

    def test_path_to_self_is_trivial(self, nav_3x3: NavigationGraph) -> None:
        path = nav_3x3.shortest_path((1, 1), (1, 1))
        assert path == [(1, 1)]

    def test_path_continuity(self, nav_3x3: NavigationGraph) -> None:
        """Each consecutive pair in the path must share an edge."""
        path = nav_3x3.shortest_path((0, 0), (2, 2))
        assert path is not None
        for i in range(len(path) - 1):
            assert nav_3x3.has_edge(path[i], path[i + 1]), (
                f"No edge between {path[i]} and {path[i+1]}"
            )
