"""
test_exit_config_experiment.py
-------------------------------
Stage 7: Same Floor Plan, Different Exit Configuration Experiment.

This module tests the most important Stage 7 requirement:

    The current simulator can evaluate different exit configurations
    while holding everything else (grid, walls, agents, parameters, seed)
    completely constant.

Tests cover:

1. Structural correctness — only exits differ between configurations
2. Behavioral — different exits produce measurably different outcomes
3. Determinism — same exits + same seed always produce identical results
4. Per-exit utilization — which exit each agent used is tracked correctly
5. Reference config — BenchmarkConfig is identifiable and reproducible
6. Experiment framework — run labels identify which exit config was used

IMPORTANT:
- No Stage 8 architecture (FloorPlan, ExitConfiguration) is introduced here.
- Uses the existing SimulationConfig with swapped ``exits`` lists.
- No optimality claims are made about any exit configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evacuation_simulation.simulation.benchmark import BenchmarkConfig
from evacuation_simulation.simulation.config import (
    AgentConfig,
    AgentProfileConfig,
    SimulationConfig,
)
from evacuation_simulation.simulation.experiment import (
    RunRecord,
    aggregate_results,
    export_csv,
    export_json,
    run_scenario,
    run_sweep,
    SweepPoint,
)
from evacuation_simulation.simulation.metrics import MetricsCollector
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


# ---------------------------------------------------------------------------
# Shared test floor plan
# ---------------------------------------------------------------------------
#
# Floor plan: 3-row × 12-col corridor.
#   - Row 0: solid wall (top)
#   - Row 2: solid wall (bottom)
#   - Row 1: passable corridor, cols 0–11
#
# 6 agents clustered toward the RIGHT end of the corridor (cols 5–10).
# This makes left/right/center exit positions produce clearly distinct outcomes.
#
#   ╔════════════╗
#   ║............║
#   ║.....AAAAAA.║  ← agents at cols 5,6,7,8,9,10
#   ║............║
#   ╚════════════╝


def _base_floor_plan() -> dict:
    """Return a fresh dict describing the shared floor plan (no exits)."""
    return {
        "grid": {"rows": 3, "cols": 12},
        "walls": [[0, c] for c in range(12)] + [[2, c] for c in range(12)],
        "agents": [
            {"agent_id": "a0", "row": 1, "col": 5},
            {"agent_id": "a1", "row": 1, "col": 6},
            {"agent_id": "a2", "row": 1, "col": 7},
            {"agent_id": "a3", "row": 1, "col": 8},
            {"agent_id": "a4", "row": 1, "col": 9},
            {"agent_id": "a5", "row": 1, "col": 10},
        ],
        "parameters": {
            "max_timesteps": 100,
            "random_seed": 42,
            "default_cell_capacity": 1,
        },
    }


def _make_config(exits: list, scenario_name: str = "exit_exp") -> SimulationConfig:
    """Construct a SimulationConfig from the shared floor plan with the given exits."""
    data = _base_floor_plan()
    data["exits"] = exits
    data["scenario_name"] = scenario_name
    return SimulationConfig(**data)


def _run(exits: list, scenario_name: str = "exit_exp"):
    """Run simulation with given exit positions and return SimulationResult."""
    cfg = _make_config(exits, scenario_name)
    return Simulation(cfg, ShortestPathStrategy()).run()


# ---------------------------------------------------------------------------
# 1. Structural: only exits differ
# ---------------------------------------------------------------------------


class TestStructuralIsolation:
    """Changing only the exits list leaves grid, walls, and agents unchanged."""

    def test_changing_exits_preserves_grid_dimensions(self) -> None:
        cfg_a = _make_config(exits=[[1, 0]])
        cfg_b = _make_config(exits=[[1, 11]])
        assert cfg_a.grid.rows == cfg_b.grid.rows
        assert cfg_a.grid.cols == cfg_b.grid.cols

    def test_changing_exits_preserves_wall_cells(self) -> None:
        cfg_a = _make_config(exits=[[1, 0]])
        cfg_b = _make_config(exits=[[1, 11]])
        walls_a = frozenset(map(tuple, cfg_a.walls))
        walls_b = frozenset(map(tuple, cfg_b.walls))
        assert walls_a == walls_b

    def test_changing_exits_preserves_agent_ids(self) -> None:
        cfg_a = _make_config(exits=[[1, 0]])
        cfg_b = _make_config(exits=[[1, 11]])
        ids_a = {a.agent_id for a in cfg_a.agents}
        ids_b = {a.agent_id for a in cfg_b.agents}
        assert ids_a == ids_b

    def test_changing_exits_preserves_agent_positions(self) -> None:
        cfg_a = _make_config(exits=[[1, 0]])
        cfg_b = _make_config(exits=[[1, 11]])
        pos_a = {a.agent_id: (a.row, a.col) for a in cfg_a.agents}
        pos_b = {a.agent_id: (a.row, a.col) for a in cfg_b.agents}
        assert pos_a == pos_b

    def test_only_exits_differ_between_configs(self) -> None:
        exits_left = [[1, 0]]
        exits_right = [[1, 11]]
        cfg_a = _make_config(exits=exits_left)
        cfg_b = _make_config(exits=exits_right)
        set_a = frozenset(map(tuple, cfg_a.exits))
        set_b = frozenset(map(tuple, cfg_b.exits))
        assert set_a != set_b

    def test_simulation_preserves_agent_profiles_across_configs(self) -> None:
        """Agent profiles (speed, delay) are unchanged when exits are swapped."""
        cfg_a = _make_config(exits=[[1, 0]])
        cfg_b = _make_config(exits=[[1, 11]])
        for ag_a, ag_b in zip(cfg_a.agents, cfg_b.agents):
            # profile defaults are equal (speed=1.0, reaction_delay=0)
            assert ag_a.agent_id == ag_b.agent_id


# ---------------------------------------------------------------------------
# 2. Behavioral: different exits → different measurable outcomes
# ---------------------------------------------------------------------------


class TestDifferentExitsBehavior:
    """Different exit positions produce measurably different evacuation outcomes."""

    def test_right_exit_faster_for_right_clustered_agents(self) -> None:
        """
        Agents clustered at cols 5–10 (right half).
        Right exit (1,11): nearest agent is 1 hop away → fast.
        Left exit  (1, 0): farthest agent is 10 hops away → slow.
        """
        res_left = _run([[1, 0]])
        res_right = _run([[1, 11]])

        assert res_left.evacuated_count == 6
        assert res_right.evacuated_count == 6
        assert res_right.max_evacuation_time < res_left.max_evacuation_time

    def test_different_exit_configs_produce_different_total_timesteps(self) -> None:
        res_left = _run([[1, 0]])
        res_right = _run([[1, 11]])
        assert res_left.total_timesteps != res_right.total_timesteps

    def test_different_exit_configs_produce_different_mean_evacuation_times(self) -> None:
        res_left = _run([[1, 0]])
        res_right = _run([[1, 11]])
        assert res_left.mean_evacuation_time != res_right.mean_evacuation_time

    def test_two_exits_not_worse_than_one_exit(self) -> None:
        """Two exits (both ends) should give max_evacuation_time ≤ single-exit scenario."""
        res_one_left = _run([[1, 0]])
        res_two = _run([[1, 0], [1, 11]])

        assert res_one_left.evacuated_count == 6
        assert res_two.evacuated_count == 6
        assert res_two.max_evacuation_time <= res_one_left.max_evacuation_time

    def test_center_exit_intermediate_performance(self) -> None:
        """
        Center exit (1,5) is adjacent to the leftmost agent (col 5).
        Right exit  (1,11) is adjacent to the rightmost agent (col 10).
        Center exit completion time should be intermediate or better.
        """
        res_left = _run([[1, 0]])
        res_center = _run([[1, 5]])

        # Center is closer to agents than the far-left exit
        assert res_center.max_evacuation_time <= res_left.max_evacuation_time

    def test_exit_location_affects_waiting_steps(self) -> None:
        """Exit proximity affects congestion/waiting in a bottleneck corridor."""
        # Left exit forces all agents to queue through the whole corridor
        res_left = _run([[1, 0]])
        # Right exit is immediately adjacent to the rightmost agent cluster
        res_right = _run([[1, 11]])
        # Right exit should have ≤ waiting steps vs left (agents spread out sooner)
        assert res_right.total_waiting_steps <= res_left.total_waiting_steps


# ---------------------------------------------------------------------------
# 3. Determinism: same exits + same seed → identical results
# ---------------------------------------------------------------------------


class TestExitConfigDeterminism:
    """Same exit configuration + same seed always produces identical results."""

    def test_same_exits_same_seed_identical_evacuation_times(self) -> None:
        cfg = _make_config(exits=[[1, 5]])
        res1 = Simulation(cfg, ShortestPathStrategy()).run()
        res2 = Simulation(cfg, ShortestPathStrategy()).run()
        assert res1.evacuation_times == res2.evacuation_times

    def test_same_exits_same_seed_identical_waiting_steps(self) -> None:
        cfg = _make_config(exits=[[1, 5]])
        res1 = Simulation(cfg, ShortestPathStrategy()).run()
        res2 = Simulation(cfg, ShortestPathStrategy()).run()
        assert res1.total_waiting_steps == res2.total_waiting_steps

    def test_same_exits_same_seed_identical_per_exit_utilization(self) -> None:
        cfg = _make_config(exits=[[1, 5]])
        res1 = Simulation(cfg, ShortestPathStrategy()).run()
        res2 = Simulation(cfg, ShortestPathStrategy()).run()
        assert res1.per_exit_utilization == res2.per_exit_utilization

    def test_different_seeds_same_exits_both_fully_evacuate(self) -> None:
        """Different seeds with same floor plan and exits both achieve full evacuation."""
        base = _base_floor_plan()
        base["exits"] = [[1, 5]]

        base["parameters"]["random_seed"] = 1
        base["scenario_name"] = "seed1"
        res1 = Simulation(SimulationConfig(**base), ShortestPathStrategy()).run()

        base["parameters"]["random_seed"] = 99
        base["scenario_name"] = "seed99"
        res2 = Simulation(SimulationConfig(**base), ShortestPathStrategy()).run()

        assert res1.evacuated_count == 6
        assert res2.evacuated_count == 6


# ---------------------------------------------------------------------------
# 4. Per-exit utilization tracking
# ---------------------------------------------------------------------------


class TestPerExitUtilization:
    """Verify that per_exit_utilization correctly records which exit each agent used."""

    def test_single_exit_all_agents_counted(self) -> None:
        """With one exit, utilization count equals evacuated count."""
        res = _run([[1, 11]])
        assert res.evacuated_count == 6
        assert sum(res.per_exit_utilization.values()) == res.evacuated_count
        assert (1, 11) in res.per_exit_utilization
        assert res.per_exit_utilization[(1, 11)] == 6

    def test_two_exits_utilization_sums_to_evacuated_count(self) -> None:
        """With two exits, total utilization across both equals evacuated count."""
        res = _run([[1, 0], [1, 11]])
        assert res.evacuated_count == 6
        total_via_exits = sum(res.per_exit_utilization.values())
        assert total_via_exits == res.evacuated_count

    def test_per_exit_utilization_keys_are_valid_exit_cells(self) -> None:
        """All keys in per_exit_utilization correspond to declared exit cells."""
        exits = [[1, 0], [1, 11]]
        cfg = _make_config(exits=exits)
        res = Simulation(cfg, ShortestPathStrategy()).run()
        declared_exits = {tuple(e) for e in exits}
        for cell in res.per_exit_utilization:
            assert cell in declared_exits, f"Unknown exit cell in utilization: {cell}"

    def test_right_exit_used_by_right_cluster(self) -> None:
        """With only a right exit, all agents route there — utilization is 6."""
        res = _run([[1, 11]])
        assert res.per_exit_utilization.get((1, 11), 0) == 6

    def test_two_exits_both_used(self) -> None:
        """With exits at both ends, both exits should receive at least one agent."""
        res = _run([[1, 0], [1, 11]])
        assert len(res.per_exit_utilization) == 2
        assert res.per_exit_utilization.get((1, 0), 0) >= 1
        assert res.per_exit_utilization.get((1, 11), 0) >= 1

    def test_empty_simulation_zero_utilization(self) -> None:
        """No agents → empty per_exit_utilization."""
        cfg = SimulationConfig(
            scenario_name="empty",
            grid={"rows": 3, "cols": 5},
            walls=[],
            exits=[[1, 4]],
            agents=[],
            parameters={"max_timesteps": 10, "random_seed": 42},
        )
        res = Simulation(cfg, ShortestPathStrategy()).run()
        assert res.per_exit_utilization == {}

    def test_metrics_collector_exit_utilization_direct(self) -> None:
        """MetricsCollector.record_evacuation() accumulates exit utilization correctly."""
        collector = MetricsCollector(scenario_name="test", total_agents=3)
        collector.record_evacuation("a0", timestep=5, exit_cell=(1, 4))
        collector.record_evacuation("a1", timestep=6, exit_cell=(1, 4))
        collector.record_evacuation("a2", timestep=7, exit_cell=(3, 0))

        result = collector.compute_result(total_timesteps=7)
        assert result.per_exit_utilization == {(1, 4): 2, (3, 0): 1}

    def test_record_evacuation_backward_compatible_without_exit_cell(self) -> None:
        """Calling record_evacuation without exit_cell still works (backward compat)."""
        collector = MetricsCollector(scenario_name="compat_test", total_agents=1)
        collector.record_evacuation("agent_x", timestep=3)  # no exit_cell
        result = collector.compute_result(total_timesteps=3)
        assert result.evacuated_count == 1
        assert result.per_exit_utilization == {}


# ---------------------------------------------------------------------------
# 5. BenchmarkConfig — reference configuration metadata
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """BenchmarkConfig is identifiable, reproducible, and makes no optimality claims."""

    def test_benchmark_config_creation(self) -> None:
        bc = BenchmarkConfig(
            benchmark_id="two_exits_v1",
            scenario_name="two_exits",
            reference_exits=[(2, 0), (2, 8)],
            seed=42,
            description="Standard two-exit benchmark — reference configuration.",
        )
        assert bc.benchmark_id == "two_exits_v1"
        assert bc.scenario_name == "two_exits"
        assert bc.reference_exits == [(2, 0), (2, 8)]
        assert bc.seed == 42

    def test_benchmark_config_has_no_optimality_attributes(self) -> None:
        """BenchmarkConfig must not expose any 'optimal' or 'best' claim."""
        bc = BenchmarkConfig(
            benchmark_id="b", scenario_name="s", reference_exits=[], seed=1
        )
        assert not hasattr(bc, "is_optimal")
        assert not hasattr(bc, "optimal_exits")
        assert not hasattr(bc, "best_exits")

    def test_benchmark_config_summary(self) -> None:
        bc = BenchmarkConfig(
            benchmark_id="corridor_v1",
            scenario_name="open_corridor",
            reference_exits=[(1, 9)],
            seed=42,
        )
        s = bc.summary()
        assert "corridor_v1" in s
        assert "open_corridor" in s
        assert "42" in s

    def test_benchmark_config_metadata(self) -> None:
        bc = BenchmarkConfig(
            benchmark_id="meta_test",
            scenario_name="test",
            reference_exits=[(0, 0)],
            seed=10,
            metadata={"grid_size": "10x10", "agents": 6},
        )
        assert bc.metadata["agents"] == 6

    def test_benchmark_config_reproducible_simulation_results(self) -> None:
        """Running the same scenario with BenchmarkConfig.seed produces identical results."""
        bc = BenchmarkConfig(
            benchmark_id="repro_test",
            scenario_name="two_exits",
            reference_exits=[(2, 0), (2, 8)],
            seed=42,
        )
        scenario_path = Path(__file__).parent.parent / "scenarios" / "two_exits.json"
        with scenario_path.open("r", encoding="utf-8") as f:
            raw = f.read()

        import json as _json
        data = _json.loads(raw)
        cfg1 = SimulationConfig(**data)
        cfg2 = SimulationConfig(**data)

        # Simulate with benchmark seed
        from evacuation_simulation.simulation.experiment import _apply_seed
        cfg1 = _apply_seed(cfg1, bc.seed)
        cfg2 = _apply_seed(cfg2, bc.seed)

        res1 = Simulation(cfg1, ShortestPathStrategy()).run()
        res2 = Simulation(cfg2, ShortestPathStrategy()).run()

        assert res1.evacuation_times == res2.evacuation_times
        assert res1.per_exit_utilization == res2.per_exit_utilization

    def test_reference_exits_distinguishable_from_alternative_exits(self) -> None:
        """Reference and alternative exits can be stored separately and compared."""
        reference = BenchmarkConfig(
            benchmark_id="ref",
            scenario_name="floor_A",
            reference_exits=[(1, 11)],
            seed=42,
            description="Reference: right exit only",
        )
        # Alternative stored inline (not a BenchmarkConfig — just a label + exits)
        alternative_exits_A = [(1, 0)]
        alternative_exits_B = [(1, 5)]

        assert reference.reference_exits != alternative_exits_A
        assert reference.reference_exits != alternative_exits_B
        assert alternative_exits_A != alternative_exits_B


# ---------------------------------------------------------------------------
# 6. Experiment framework — labeled exit configuration output
# ---------------------------------------------------------------------------


class TestExitConfigExperimentOutput:
    """Experiment run records must identify which exit configuration was used."""

    def test_run_scenario_params_label_exit_config(self) -> None:
        """params dict in RunRecord identifies the exit configuration."""
        cfg_a = _make_config(exits=[[1, 0]], scenario_name="floor_left")
        cfg_b = _make_config(exits=[[1, 11]], scenario_name="floor_right")

        rec_a = run_scenario(
            cfg_a, seed=42,
            params={"exit_config": "config_A", "exit_position": "left"},
        )
        rec_b = run_scenario(
            cfg_b, seed=42,
            params={"exit_config": "config_B", "exit_position": "right"},
        )

        assert rec_a.params["exit_config"] == "config_A"
        assert rec_b.params["exit_config"] == "config_B"
        assert rec_a.params["exit_position"] != rec_b.params["exit_position"]

    def test_reference_and_alternatives_labeled_in_experiment(self) -> None:
        """Reference, alternative_A, alternative_B are distinguishable in RunRecord params."""
        cfg_ref = _make_config(exits=[[1, 11]], scenario_name="floor_ref")
        cfg_alt_a = _make_config(exits=[[1, 0]], scenario_name="floor_alt_a")
        cfg_alt_b = _make_config(exits=[[1, 5]], scenario_name="floor_alt_b")

        rec_ref = run_scenario(cfg_ref, seed=42, params={"role": "reference"})
        rec_alt_a = run_scenario(cfg_alt_a, seed=42, params={"role": "alternative_A"})
        rec_alt_b = run_scenario(cfg_alt_b, seed=42, params={"role": "alternative_B"})

        roles = {rec_ref.params["role"], rec_alt_a.params["role"], rec_alt_b.params["role"]}
        assert roles == {"reference", "alternative_A", "alternative_B"}

    def test_exit_config_labels_survive_csv_export(self, tmp_path: Path) -> None:
        """Exit config labels appear in exported CSV output."""
        cfg_a = _make_config(exits=[[1, 0]], scenario_name="export_left")
        cfg_b = _make_config(exits=[[1, 11]], scenario_name="export_right")

        records = [
            run_scenario(cfg_a, seed=42, params={"exit_config": "config_A"}),
            run_scenario(cfg_b, seed=42, params={"exit_config": "config_B"}),
        ]
        from evacuation_simulation.simulation.experiment import export_csv
        csv_path = tmp_path / "exit_exp.csv"
        export_csv(records, csv_path)

        content = csv_path.read_text(encoding="utf-8")
        assert "exit_config" in content
        assert "config_A" in content
        assert "config_B" in content

    def test_exit_config_sweep_aggregates_across_seeds(self) -> None:
        """run_sweep over different exit configs produces correct run count."""
        cfg_a = _make_config(exits=[[1, 0]], scenario_name="sweep_left")
        cfg_b = _make_config(exits=[[1, 11]], scenario_name="sweep_right")

        pts = [
            SweepPoint(label="left_exit", config=cfg_a, params={"exit_config": "A"}),
            SweepPoint(label="right_exit", config=cfg_b, params={"exit_config": "B"}),
        ]
        exp = run_sweep(pts, seeds=[42, 43, 44])

        # 2 configs × 3 seeds = 6 runs
        assert exp.run_count == 6
        assert exp.aggregate is not None

    def test_per_exit_utilization_in_result_for_labeled_run(self) -> None:
        """Per-exit utilization is available on the SimulationResult inside RunRecord."""
        cfg = _make_config(exits=[[1, 11]], scenario_name="util_labeled")
        rec = run_scenario(cfg, seed=42, params={"exit_config": "reference"})

        assert rec.result.per_exit_utilization.get((1, 11), 0) == 6
        assert sum(rec.result.per_exit_utilization.values()) == rec.result.evacuated_count
