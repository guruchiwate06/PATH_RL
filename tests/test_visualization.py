"""
tests/test_visualization.py
----------------------------
Comprehensive unit and integration test suite for Stage 4 visualization
and simulation inspection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated testing
import matplotlib.pyplot as plt
import pytest

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.agent import AgentState
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy
from visualize import SimulationVisualizer, render_grid, save_grid_snapshot


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def make_config(
    rows: int = 4,
    cols: int = 5,
    walls: list[list[int]] | None = None,
    exits: list[list[int]] | None = None,
    agents: list[dict] | None = None,
    capacity: int = 1,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        scenario_name="test_viz",
        grid={"rows": rows, "cols": cols},
        walls=walls or [],
        exits=exits or [[rows - 1, cols - 1]],
        agents=agents or [{"agent_id": "a0", "row": 0, "col": 0}],
        parameters={
            "max_timesteps": 50,
            "random_seed": seed,
            "default_cell_capacity": capacity,
        },
    )


# ---------------------------------------------------------------------------
# TestVisualizerInitialization
# ---------------------------------------------------------------------------


class TestVisualizerInitialization:
    def test_init_with_config(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg)
        assert viz.config == cfg
        assert viz.sim is not None
        assert viz.sim.state.timestep == 0
        assert not viz.is_playing
        assert viz.selected_agent_id is None

    def test_init_with_simulation(self) -> None:
        cfg = make_config()
        sim = Simulation(cfg, ShortestPathStrategy())
        viz = SimulationVisualizer(simulation=sim)
        assert viz.sim is sim
        assert viz.config == cfg

    def test_init_without_args_raises(self) -> None:
        with pytest.raises(ValueError, match="Either simulation or config must be provided"):
            SimulationVisualizer(simulation=None, config=None)

    def test_grid_dimensions_match_environment(self) -> None:
        cfg = make_config(rows=6, cols=8)
        viz = SimulationVisualizer(config=cfg)
        assert viz.sim.environment.rows == 6
        assert viz.sim.environment.cols == 8


# ---------------------------------------------------------------------------
# TestFigureAndLayout
# ---------------------------------------------------------------------------


class TestFigureAndLayout:
    def test_build_figure_creates_axes_and_buttons(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg)
        fig = viz.build_figure()
        assert fig is not None
        assert viz.ax_grid is not None
        assert viz.ax_hud is not None
        assert viz.ax_inspector is not None
        assert viz.ax_trace is not None
        assert "play" in viz._buttons
        assert "step" in viz._buttons
        assert "reset" in viz._buttons
        assert "paths" in viz._buttons
        assert "occ" in viz._buttons
        plt.close(fig)

    def test_render_frame_executes_cleanly(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg)
        viz.render_frame()
        assert viz.fig is not None
        plt.close(viz.fig)


# ---------------------------------------------------------------------------
# TestAgentRenderingAndStates
# ---------------------------------------------------------------------------


class TestAgentRenderingAndStates:
    def test_agent_positions_and_active_count(self) -> None:
        agents = [
            {"agent_id": "a0", "row": 0, "col": 0},
            {"agent_id": "a1", "row": 1, "col": 1},
        ]
        cfg = make_config(agents=agents)
        viz = SimulationVisualizer(config=cfg)
        assert len(viz.sim.active_agents) == 2
        assert viz.sim.active_agents[0].position == (0, 0)
        assert viz.sim.active_agents[1].position == (1, 1)

    def test_evacuated_agents_excluded_from_active(self) -> None:
        # Agent starts on exit cell (3, 4)
        agents = [{"agent_id": "a0", "row": 3, "col": 4}]
        cfg = make_config(rows=4, cols=5, exits=[[3, 4]], agents=agents)
        viz = SimulationVisualizer(config=cfg)
        viz.step()  # Agent evacuates at step 1
        assert len(viz.sim.active_agents) == 0
        assert len(viz.sim.evacuated_agents) == 1
        assert viz.sim.agents[0].state == AgentState.EVACUATED
        # Rendering with evacuated agent executes without error
        viz.render_frame()
        plt.close(viz.fig)

    def test_waiting_agents_detected_from_trace(self) -> None:
        # 2 agents competing for same destination with capacity 1
        agents = [
            {"agent_id": "a0", "row": 0, "col": 0},
            {"agent_id": "a1", "row": 0, "col": 1},
        ]
        cfg = make_config(rows=3, cols=2, exits=[[2, 0]], agents=agents, capacity=1)
        viz = SimulationVisualizer(config=cfg)
        viz.step()
        waiting_ids = viz._get_current_waiting_agent_ids()
        assert "a1" in waiting_ids
        assert "a0" not in waiting_ids
        plt.close(viz.fig)


# ---------------------------------------------------------------------------
# TestPlannedPaths
# ---------------------------------------------------------------------------


class TestPlannedPaths:
    def test_get_agent_planned_path(self) -> None:
        agents = [{"agent_id": "a0", "row": 0, "col": 0}]
        cfg = make_config(rows=3, cols=3, exits=[[2, 2]], agents=agents)
        viz = SimulationVisualizer(config=cfg)
        path = viz.sim.get_agent_planned_path("a0")
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert len(path) == 5  # Manhattan distance 4 + 1 start node

    def test_unreachable_agent_path_is_none(self) -> None:
        # Agent completely surrounded by walls
        walls = [[0, 1], [1, 0], [1, 1]]
        agents = [{"agent_id": "a0", "row": 0, "col": 0}]
        cfg = make_config(rows=4, cols=4, walls=walls, exits=[[3, 3]], agents=agents)
        viz = SimulationVisualizer(config=cfg)
        path = viz.sim.get_agent_planned_path("a0")
        assert path is None

    def test_toggle_paths(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg, show_paths=True)
        assert viz.show_paths is True
        viz.toggle_paths()
        assert viz.show_paths is False
        viz.toggle_paths(True)
        assert viz.show_paths is True
        plt.close(viz.fig)


# ---------------------------------------------------------------------------
# TestOccupancyOverlay
# ---------------------------------------------------------------------------


class TestOccupancyOverlay:
    def test_toggle_occupancy(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg, show_occupancy=False)
        assert viz.show_occupancy is False
        viz.toggle_occupancy()
        assert viz.show_occupancy is True
        viz.toggle_occupancy(False)
        assert viz.show_occupancy is False
        plt.close(viz.fig)

    def test_occupancy_rendering_with_agents(self) -> None:
        agents = [
            {"agent_id": "a0", "row": 0, "col": 0},
            {"agent_id": "a1", "row": 0, "col": 0},
        ]
        cfg = make_config(agents=agents, capacity=2)
        viz = SimulationVisualizer(config=cfg, show_occupancy=True)
        viz.render_frame()
        occ_map = viz.sim.occupancy.get_occupied_cells()
        assert occ_map.get((0, 0)) == 2
        plt.close(viz.fig)


# ---------------------------------------------------------------------------
# TestSimulationControls
# ---------------------------------------------------------------------------


class TestSimulationControls:
    def test_step_advances_exactly_one_timestep(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg)
        assert viz.sim.state.timestep == 0
        running = viz.step()
        assert viz.sim.state.timestep == 1
        assert isinstance(running, bool)
        plt.close(viz.fig)

    def test_play_and_pause_state(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg)
        viz.build_figure()
        assert not viz.is_playing
        viz.play()
        assert viz.is_playing
        viz.pause()
        assert not viz.is_playing
        viz.toggle_play()
        assert viz.is_playing
        viz.toggle_play()
        assert not viz.is_playing
        plt.close(viz.fig)

    def test_reset_restores_initial_state(self) -> None:
        cfg = make_config(rows=3, cols=3, exits=[[2, 2]], agents=[{"agent_id": "a0", "row": 0, "col": 0}])
        viz = SimulationVisualizer(config=cfg)
        viz.step()
        assert viz.sim.state.timestep == 1
        assert viz.sim.agents[0].position != (0, 0)
        viz.reset()
        assert viz.sim.state.timestep == 0
        assert viz.sim.agents[0].position == (0, 0)
        assert viz.sim.agents[0].state == AgentState.MOVING
        assert not viz.sim.state.is_terminated
        plt.close(viz.fig)


# ---------------------------------------------------------------------------
# TestAgentInspection
# ---------------------------------------------------------------------------


class TestAgentInspection:
    def test_get_agent_info_structure(self) -> None:
        agents = [{"agent_id": "a0", "row": 0, "col": 0}]
        cfg = make_config(rows=3, cols=3, exits=[[2, 2]], agents=agents)
        viz = SimulationVisualizer(config=cfg)
        info = viz.sim.get_agent_info("a0")
        assert info is not None
        assert info["agent_id"] == "a0"
        assert info["position"] == (0, 0)
        assert info["state"] == "MOVING"
        assert info["target_exit"] == (2, 2)
        assert info["path_length"] == 4
        assert info["waiting_steps"] == 0

    def test_select_agent(self) -> None:
        cfg = make_config()
        viz = SimulationVisualizer(config=cfg)
        viz.select_agent("a0")
        assert viz.selected_agent_id == "a0"
        viz.select_agent(None)
        assert viz.selected_agent_id is None
        plt.close(viz.fig)

    def test_click_event_selects_agent(self) -> None:
        cfg = make_config(agents=[{"agent_id": "a0", "row": 1, "col": 2}])
        viz = SimulationVisualizer(config=cfg)
        viz.build_figure()

        # Simulate click inside grid at (row 1, col 2)
        class MockEvent:
            inaxes = viz.ax_grid
            xdata = 2.4
            ydata = 1.3

        viz._on_click(MockEvent())
        assert viz.selected_agent_id == "a0"

        # Simulate click on empty cell (row 0, col 0)
        class MockEmptyEvent:
            inaxes = viz.ax_grid
            xdata = 0.5
            ydata = 0.5

        viz._on_click(MockEmptyEvent())
        assert viz.selected_agent_id is None
        plt.close(viz.fig)


# ---------------------------------------------------------------------------
# TestDeterministicReplay
# ---------------------------------------------------------------------------


class TestDeterministicReplay:
    def test_replay_after_reset_matches_original(self) -> None:
        cfg = make_config(
            rows=5, cols=5,
            walls=[[2, 0], [2, 1], [2, 3], [2, 4]],
            exits=[[4, 2]],
            agents=[{"agent_id": f"a{i}", "row": 0, "col": i} for i in range(5)],
            capacity=1,
            seed=42,
        )
        viz = SimulationVisualizer(config=cfg)

        # Run 5 steps and record agent positions
        for _ in range(5):
            viz.step()
        pos_first_run = {a.agent_id: a.position for a in viz.sim.agents}
        wait_first_run = viz.sim.metrics.compute_result(5).total_waiting_steps

        # Reset and run 5 steps again
        viz.reset()
        for _ in range(5):
            viz.step()
        pos_second_run = {a.agent_id: a.position for a in viz.sim.agents}
        wait_second_run = viz.sim.metrics.compute_result(5).total_waiting_steps

        assert pos_first_run == pos_second_run
        assert wait_first_run == wait_second_run
        plt.close(viz.fig)


# ---------------------------------------------------------------------------
# TestStandaloneSnapshotFunctions
# ---------------------------------------------------------------------------


class TestStandaloneSnapshotFunctions:
    def test_render_grid_and_save_snapshot(self) -> None:
        cfg = make_config()
        sim = Simulation(cfg, ShortestPathStrategy())
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "snapshot.png"
            render_grid(sim, title="Test Render", show=False, save_path=snap_path)
            assert snap_path.exists()
            assert snap_path.stat().st_size > 0

    def test_save_grid_snapshot_helper(self) -> None:
        cfg = make_config()
        sim = Simulation(cfg, ShortestPathStrategy())
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_grid_snapshot(sim, output_dir=tmpdir, prefix="test_step")
            assert saved.exists()
            assert saved.name == "test_step_0000.png"
            assert saved.stat().st_size > 0
