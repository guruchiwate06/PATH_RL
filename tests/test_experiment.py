"""
test_experiment.py
------------------
Tests for simulation/experiment.py — Stage 5 scenario runner.

Test classes
~~~~~~~~~~~~
TestSingleRun
    Single-scenario execution returns a correct RunRecord.

TestRepeatedRuns
    Multi-seed execution: count, determinism, independence.

TestSweep
    Parameter sweeps: run count, parameter isolation.

TestAggregation
    Aggregate statistics: known values, None handling, empty guard.

TestExport
    CSV and JSON export: content, structure, parseability.

TestExistingRegression (implicit)
    All previous simulation tests must continue to pass; this file adds
    new tests only.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.experiment import (
    AggregateStats,
    ExperimentResult,
    RunRecord,
    SweepPoint,
    aggregate_results,
    export_csv,
    export_json,
    run_repeated,
    run_scenario,
    run_sweep,
)
from evacuation_simulation.simulation.metrics import SimulationResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _minimal_config(
    *,
    scenario_name: str = "test_scenario",
    rows: int = 6,
    cols: int = 6,
    agent_positions: Optional[list[tuple[int, int]]] = None,
    exits: Optional[list[tuple[int, int]]] = None,
    walls: Optional[list[tuple[int, int]]] = None,
    max_timesteps: int = 100,
    seed: Optional[int] = 42,
    capacity: int = 1,
) -> SimulationConfig:
    """
    Build a minimal SimulationConfig for use in tests.

    Defaults: 6×6 grid, one agent at (0,0), single exit at (5,5), seed=42.
    """
    positions = agent_positions or [(0, 0)]
    return SimulationConfig(
        scenario_name=scenario_name,
        grid={"rows": rows, "cols": cols},
        walls=[list(w) for w in (walls or [])],
        exits=[list(e) for e in (exits or [(rows - 1, cols - 1)])],
        agents=[
            {"agent_id": f"a{i}", "row": r, "col": c}
            for i, (r, c) in enumerate(positions)
        ],
        parameters={
            "max_timesteps": max_timesteps,
            "random_seed": seed,
            "default_cell_capacity": capacity,
        },
    )


def _synthetic_result(
    *,
    scenario_name: str = "synth",
    total_agents: int = 5,
    evacuated_count: int = 5,
    total_timesteps: int = 20,
    mean_evacuation_time: Optional[float] = 10.0,
    min_evacuation_time: Optional[int] = 8,
    max_evacuation_time: Optional[int] = 12,
    total_waiting_steps: int = 3,
    congested_cell_steps: int = 1,
    max_occupancy_ratio: float = 1.0,
) -> SimulationResult:
    """Create a synthetic SimulationResult for aggregation tests."""
    evac_rate = evacuated_count / total_agents if total_agents > 0 else 0.0
    return SimulationResult(
        scenario_name=scenario_name,
        total_agents=total_agents,
        evacuated_count=evacuated_count,
        non_evacuated_count=total_agents - evacuated_count,
        total_timesteps=total_timesteps,
        evacuation_rate=evac_rate,
        mean_evacuation_time=mean_evacuation_time,
        min_evacuation_time=min_evacuation_time,
        max_evacuation_time=max_evacuation_time,
        evacuation_times={},
        total_waiting_steps=total_waiting_steps,
        mean_waiting_steps=total_waiting_steps / total_agents if total_agents else 0.0,
        max_waiting_steps=total_waiting_steps,
        per_agent_waiting_steps={},
        max_cell_occupancy=1,
        max_occupancy_ratio=max_occupancy_ratio,
        congested_cell_steps=congested_cell_steps,
    )


def _synthetic_record(
    run_id: str = "r0",
    seed: Optional[int] = 42,
    params: Optional[dict] = None,
    **result_kwargs,
) -> RunRecord:
    """Wrap a synthetic SimulationResult in a RunRecord."""
    return RunRecord(
        run_id=run_id,
        scenario_name="synth",
        seed=seed,
        params=params or {},
        result=_synthetic_result(**result_kwargs),
    )


# ---------------------------------------------------------------------------
# Test 1 — Single run
# ---------------------------------------------------------------------------


class TestSingleRun:
    """A valid configuration produces a RunRecord with a SimulationResult."""

    def test_returns_run_record(self) -> None:
        config = _minimal_config()
        record = run_scenario(config)
        assert isinstance(record, RunRecord)

    def test_result_is_simulation_result(self) -> None:
        config = _minimal_config()
        record = run_scenario(config)
        assert isinstance(record.result, SimulationResult)

    def test_scenario_name_preserved(self) -> None:
        config = _minimal_config(scenario_name="my_test")
        record = run_scenario(config)
        assert record.scenario_name == "my_test"

    def test_seed_in_record_matches_config_seed(self) -> None:
        config = _minimal_config(seed=99)
        record = run_scenario(config)
        assert record.seed == 99

    def test_seed_override(self) -> None:
        """Providing seed= overrides the config seed; config is NOT mutated."""
        config = _minimal_config(seed=1)
        record = run_scenario(config, seed=77)
        assert record.seed == 77
        # Original config unchanged
        assert config.parameters.random_seed == 1

    def test_agent_evacuates_in_trivial_scenario(self) -> None:
        """Single agent adjacent to exit should evacuate within a few steps."""
        config = _minimal_config(
            rows=3, cols=3,
            agent_positions=[(0, 0)],
            exits=[(2, 2)],
            max_timesteps=50,
        )
        record = run_scenario(config)
        assert record.result.evacuated_count == 1
        assert record.result.evacuation_rate == pytest.approx(1.0)

    def test_empty_params_by_default(self) -> None:
        config = _minimal_config()
        record = run_scenario(config)
        assert record.params == {}

    def test_params_stored(self) -> None:
        config = _minimal_config()
        record = run_scenario(config, params={"agent_count": 1, "capacity": 1})
        assert record.params["agent_count"] == 1

    def test_custom_run_id(self) -> None:
        config = _minimal_config()
        record = run_scenario(config, run_id="custom_id_001")
        assert record.run_id == "custom_id_001"


# ---------------------------------------------------------------------------
# Test 2 — Repeated runs: count and types
# ---------------------------------------------------------------------------


class TestRepeatedRuns:
    """Five seeds produce five independent RunRecords."""

    def test_five_seeds_return_five_records(self) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2, 3, 4, 5])
        assert len(records) == 5

    def test_all_records_are_run_records(self) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[10, 20])
        assert all(isinstance(r, RunRecord) for r in records)

    def test_seeds_stored_in_records(self) -> None:
        seeds = [7, 8, 9]
        config = _minimal_config(seed=None)
        records = run_repeated(config, seeds=seeds)
        stored_seeds = [r.seed for r in records]
        assert stored_seeds == seeds

    def test_empty_seeds_raises(self) -> None:
        config = _minimal_config()
        with pytest.raises(ValueError, match="seeds must not be empty"):
            run_repeated(config, seeds=[])


# ---------------------------------------------------------------------------
# Test 3 — Determinism: same config + seed → same result
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same config and seed must produce identical SimulationResults."""

    def test_same_seed_same_result(self) -> None:
        config = _minimal_config(seed=42)
        r1 = run_scenario(config)
        r2 = run_scenario(config)
        assert r1.result.total_timesteps == r2.result.total_timesteps
        assert r1.result.evacuated_count == r2.result.evacuated_count
        assert r1.result.mean_evacuation_time == r2.result.mean_evacuation_time
        assert r1.result.total_waiting_steps == r2.result.total_waiting_steps

    def test_repeated_run_reproducible(self) -> None:
        """run_repeated with the same seed list produces reproducible results."""
        config = _minimal_config(seed=None)
        seeds = [1, 2, 3]
        records_a = run_repeated(config, seeds=seeds)
        records_b = run_repeated(config, seeds=seeds)
        for ra, rb in zip(records_a, records_b):
            assert ra.result.total_timesteps == rb.result.total_timesteps
            assert ra.result.evacuated_count == rb.result.evacuated_count


# ---------------------------------------------------------------------------
# Test 4 — Independence: different seed → independent simulation state
# ---------------------------------------------------------------------------


class TestIndependence:
    """Each run must use a fresh Simulation instance."""

    def test_run_ids_are_unique(self) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2, 3, 4, 5])
        run_ids = [r.run_id for r in records]
        assert len(set(run_ids)) == len(run_ids), "Run IDs must be unique."

    def test_results_are_independent_objects(self) -> None:
        """Each RunRecord holds a distinct SimulationResult object."""
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2])
        assert records[0].result is not records[1].result

    def test_config_not_mutated_by_repeated(self) -> None:
        """The base config must not be altered by run_repeated."""
        config = _minimal_config(seed=10)
        original_seed = config.parameters.random_seed
        run_repeated(config, seeds=[20, 30])
        assert config.parameters.random_seed == original_seed


# ---------------------------------------------------------------------------
# Test 5 — Sweep size: N × M = N×M runs
# ---------------------------------------------------------------------------


class TestSweepSize:
    """4 configs × 5 seeds = exactly 20 runs."""

    def _make_points(self, n: int) -> list[SweepPoint]:
        """Create n SweepPoints with 1 agent each (trivially fast)."""
        return [
            SweepPoint(
                label=f"pt{i}",
                config=_minimal_config(scenario_name=f"scenario_{i}"),
                params={"variant": i},
            )
            for i in range(n)
        ]

    def test_4x5_produces_20_runs(self) -> None:
        points = self._make_points(4)
        seeds = [1, 2, 3, 4, 5]
        result = run_sweep(points, seeds)
        assert result.run_count == 20

    def test_1x1_produces_1_run(self) -> None:
        points = self._make_points(1)
        result = run_sweep(points, [42])
        assert result.run_count == 1

    def test_3x3_produces_9_runs(self) -> None:
        points = self._make_points(3)
        result = run_sweep(points, [1, 2, 3])
        assert result.run_count == 9

    def test_returns_experiment_result(self) -> None:
        points = self._make_points(2)
        result = run_sweep(points, [1, 2])
        assert isinstance(result, ExperimentResult)

    def test_aggregate_populated(self) -> None:
        points = self._make_points(2)
        result = run_sweep(points, [1, 2])
        assert result.aggregate is not None
        assert result.aggregate.run_count == 4

    def test_empty_sweep_points_raises(self) -> None:
        with pytest.raises(ValueError, match="sweep_points must not be empty"):
            run_sweep([], [1, 2, 3])

    def test_empty_seeds_raises(self) -> None:
        points = self._make_points(1)
        with pytest.raises(ValueError, match="seeds must not be empty"):
            run_sweep(points, [])


# ---------------------------------------------------------------------------
# Test 6 — Parameter isolation: base config must not be mutated
# ---------------------------------------------------------------------------


class TestParameterIsolation:
    """Sweep execution must not mutate the SweepPoint configs."""

    def test_sweep_does_not_mutate_base_config(self) -> None:
        base_config = _minimal_config(seed=99)
        original_seed = base_config.parameters.random_seed
        point = SweepPoint(
            label="test",
            config=base_config,
            params={"capacity": 1},
        )
        run_sweep([point], seeds=[1, 2, 3])
        assert base_config.parameters.random_seed == original_seed

    def test_sweep_point_params_not_mutated(self) -> None:
        original_params = {"agent_count": 6}
        point = SweepPoint(
            label="test",
            config=_minimal_config(),
            params=original_params,
        )
        run_sweep([point], seeds=[1])
        assert point.params == {"agent_count": 6}

    def test_different_seeds_produce_independent_configs(self) -> None:
        """Verifying that each run gets its own config copy."""
        config = _minimal_config(seed=1)
        point = SweepPoint(label="pt", config=config, params={})
        result = run_sweep([point], seeds=[10, 20, 30])
        seeds_in_records = [r.seed for r in result.runs]
        assert seeds_in_records == [10, 20, 30]


# ---------------------------------------------------------------------------
# Test 7 — Aggregation: known synthetic data produces expected values
# ---------------------------------------------------------------------------


class TestAggregation:
    """Aggregate statistics are mathematically correct."""

    def _records_with_known_values(self) -> list[RunRecord]:
        """Three records with known mean_evacuation_time values: 10, 20, 30."""
        return [
            _synthetic_record(run_id="r0", mean_evacuation_time=10.0,
                               total_timesteps=10, total_waiting_steps=0,
                               congested_cell_steps=0, max_occupancy_ratio=0.5,
                               evacuated_count=5),
            _synthetic_record(run_id="r1", mean_evacuation_time=20.0,
                               total_timesteps=20, total_waiting_steps=2,
                               congested_cell_steps=1, max_occupancy_ratio=0.8,
                               evacuated_count=5),
            _synthetic_record(run_id="r2", mean_evacuation_time=30.0,
                               total_timesteps=30, total_waiting_steps=4,
                               congested_cell_steps=2, max_occupancy_ratio=1.0,
                               evacuated_count=5),
        ]

    def test_run_count(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        assert agg.run_count == 3

    def test_mean_evacuation_time_mean(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        assert agg.mean_evacuation_time_mean == pytest.approx(20.0)

    def test_mean_evacuation_time_min(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        assert agg.mean_evacuation_time_min == pytest.approx(10.0)

    def test_mean_evacuation_time_max(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        assert agg.mean_evacuation_time_max == pytest.approx(30.0)

    def test_mean_evacuation_time_std(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        expected_std = statistics.stdev([10.0, 20.0, 30.0])
        assert agg.mean_evacuation_time_std == pytest.approx(expected_std)

    def test_mean_evacuation_time_median(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        assert agg.mean_evacuation_time_median == pytest.approx(20.0)

    def test_total_timesteps_aggregates(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        assert agg.total_timesteps_mean == pytest.approx(20.0)
        assert agg.total_timesteps_min == 10
        assert agg.total_timesteps_max == 30

    def test_evacuation_rate_mean(self) -> None:
        records = self._records_with_known_values()
        agg = aggregate_results(records)
        # All runs have 5/5 agents evacuated → rate 1.0
        assert agg.evacuation_rate_mean == pytest.approx(1.0)

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_results([])

    def test_single_record(self) -> None:
        records = [_synthetic_record(run_id="only", mean_evacuation_time=15.0,
                                     total_timesteps=15, total_waiting_steps=1,
                                     congested_cell_steps=0, max_occupancy_ratio=0.5,
                                     evacuated_count=5)]
        agg = aggregate_results(records)
        assert agg.run_count == 1
        assert agg.mean_evacuation_time_mean == pytest.approx(15.0)
        assert agg.mean_evacuation_time_std == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 8 — None handling: aggregation must not crash on None metrics
# ---------------------------------------------------------------------------


class TestNoneHandling:
    """None metric values must be excluded from aggregates, not turned into 0."""

    def test_all_none_mean_evac_time_gives_none_aggregate(self) -> None:
        """If every run has mean_evacuation_time=None, aggregates are None."""
        records = [
            _synthetic_record(run_id=f"r{i}", mean_evacuation_time=None,
                               evacuated_count=0, total_timesteps=100,
                               total_waiting_steps=0, congested_cell_steps=0,
                               max_occupancy_ratio=0.0)
            for i in range(3)
        ]
        agg = aggregate_results(records)
        assert agg.mean_evacuation_time_mean is None
        assert agg.mean_evacuation_time_min is None
        assert agg.mean_evacuation_time_max is None
        assert agg.mean_evacuation_time_std is None
        assert agg.mean_evacuation_time_median is None

    def test_mixed_none_and_values_excludes_none(self) -> None:
        """None values are excluded; mean is over non-None values only."""
        records = [
            _synthetic_record(run_id="r0", mean_evacuation_time=None,
                               evacuated_count=0, total_timesteps=100,
                               total_waiting_steps=0, congested_cell_steps=0,
                               max_occupancy_ratio=0.0),
            _synthetic_record(run_id="r1", mean_evacuation_time=10.0,
                               evacuated_count=5, total_timesteps=10,
                               total_waiting_steps=0, congested_cell_steps=0,
                               max_occupancy_ratio=0.0),
            _synthetic_record(run_id="r2", mean_evacuation_time=20.0,
                               evacuated_count=5, total_timesteps=20,
                               total_waiting_steps=0, congested_cell_steps=0,
                               max_occupancy_ratio=0.0),
        ]
        agg = aggregate_results(records)
        # Mean of [10, 20] = 15; None is excluded, not treated as 0
        assert agg.mean_evacuation_time_mean == pytest.approx(15.0)
        assert agg.mean_evacuation_time_min == pytest.approx(10.0)
        assert agg.mean_evacuation_time_max == pytest.approx(20.0)

    def test_none_does_not_become_zero(self) -> None:
        """Ensure None is not silently zeroed (which would corrupt the mean)."""
        records = [
            _synthetic_record(run_id="r0", mean_evacuation_time=None,
                               evacuated_count=0, total_timesteps=100,
                               total_waiting_steps=0, congested_cell_steps=0,
                               max_occupancy_ratio=0.0),
            _synthetic_record(run_id="r1", mean_evacuation_time=30.0,
                               evacuated_count=5, total_timesteps=30,
                               total_waiting_steps=0, congested_cell_steps=0,
                               max_occupancy_ratio=0.0),
        ]
        agg = aggregate_results(records)
        # If None were treated as 0, mean would be 15; correct mean is 30
        assert agg.mean_evacuation_time_mean == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Test 9 — CSV export
# ---------------------------------------------------------------------------


class TestCSVExport:
    """CSV export produces a valid file with expected structure."""

    def test_csv_file_is_created(self, tmp_path: Path) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2, 3])
        out = tmp_path / "results.csv"
        export_csv(records, out)
        assert out.exists()

    def test_csv_row_count(self, tmp_path: Path) -> None:
        """CSV has exactly one header row + one data row per record."""
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2, 3, 4, 5])
        out = tmp_path / "results.csv"
        export_csv(records, out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 6  # 1 header + 5 data rows

    def test_csv_has_required_columns(self, tmp_path: Path) -> None:
        """CSV must contain the core identifier and metric columns."""
        required = {
            "run_id", "scenario_name", "seed",
            "total_agents", "evacuated_count", "evacuation_rate",
            "mean_evacuation_time", "total_timesteps",
        }
        config = _minimal_config()
        records = run_repeated(config, seeds=[1])
        out = tmp_path / "results.csv"
        export_csv(records, out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = set(reader.fieldnames or [])
        assert required.issubset(headers)

    def test_csv_seed_values(self, tmp_path: Path) -> None:
        """Seed column values match the seeds used."""
        config = _minimal_config(seed=None)
        seeds = [10, 20, 30]
        records = run_repeated(config, seeds=seeds)
        out = tmp_path / "seeds.csv"
        export_csv(records, out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            csv_seeds = [int(row["seed"]) for row in reader]
        assert csv_seeds == seeds

    def test_csv_sweep_includes_param_columns(self, tmp_path: Path) -> None:
        """Sweep param keys appear as CSV columns."""
        points = [
            SweepPoint(label="a6", config=_minimal_config(), params={"agent_count": 6}),
            SweepPoint(label="a12", config=_minimal_config(), params={"agent_count": 12}),
        ]
        result = run_sweep(points, seeds=[1, 2])
        out = tmp_path / "sweep.csv"
        export_csv(result.runs, out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = set(reader.fieldnames or [])
        assert "agent_count" in headers

    def test_csv_empty_records_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No records"):
            export_csv([], tmp_path / "empty.csv")

    def test_csv_parent_dirs_created(self, tmp_path: Path) -> None:
        """Export creates parent directories if they don't exist."""
        config = _minimal_config()
        records = run_repeated(config, seeds=[1])
        deep = tmp_path / "a" / "b" / "c" / "results.csv"
        export_csv(records, deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# Test 10 — JSON export
# ---------------------------------------------------------------------------


class TestJSONExport:
    """JSON export produces a valid, parseable file with expected structure."""

    def test_json_file_is_created(self, tmp_path: Path) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2])
        exp = ExperimentResult(
            experiment_id="test_exp",
            runs=records,
            aggregate=aggregate_results(records),
        )
        out = tmp_path / "experiment.json"
        export_json(exp, out)
        assert out.exists()

    def test_json_is_parseable(self, tmp_path: Path) -> None:
        """The exported JSON must be loadable with the standard json library."""
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2, 3])
        exp = ExperimentResult(
            experiment_id="parseable_test",
            runs=records,
            aggregate=aggregate_results(records),
        )
        out = tmp_path / "parseable.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_json_experiment_id_preserved(self, tmp_path: Path) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1])
        exp = ExperimentResult(experiment_id="my_unique_id", runs=records)
        out = tmp_path / "exp.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["experiment_id"] == "my_unique_id"

    def test_json_run_count(self, tmp_path: Path) -> None:
        config = _minimal_config()
        seeds = [1, 2, 3, 4]
        records = run_repeated(config, seeds=seeds)
        exp = ExperimentResult(experiment_id="count_test", runs=records)
        out = tmp_path / "count.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["run_count"] == 4
        assert len(data["runs"]) == 4

    def test_json_run_fields(self, tmp_path: Path) -> None:
        """Each run in JSON must contain expected keys."""
        required_keys = {"run_id", "scenario_name", "seed", "params", "result"}
        config = _minimal_config()
        records = run_repeated(config, seeds=[42])
        exp = ExperimentResult(experiment_id="fields_test", runs=records)
        out = tmp_path / "fields.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        run = data["runs"][0]
        assert required_keys.issubset(set(run.keys()))

    def test_json_result_fields(self, tmp_path: Path) -> None:
        """Result dict in JSON must contain the standard metric keys."""
        config = _minimal_config()
        records = run_repeated(config, seeds=[1])
        exp = ExperimentResult(experiment_id="metrics_test", runs=records)
        out = tmp_path / "metrics.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        result_keys = set(data["runs"][0]["result"].keys())
        assert "total_agents" in result_keys
        assert "evacuated_count" in result_keys
        assert "mean_evacuation_time" in result_keys
        assert "congested_cell_steps" in result_keys

    def test_json_aggregate_present_when_provided(self, tmp_path: Path) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1, 2])
        exp = ExperimentResult(
            experiment_id="agg_test",
            runs=records,
            aggregate=aggregate_results(records),
        )
        out = tmp_path / "with_agg.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["aggregate"] is not None
        assert "run_count" in data["aggregate"]

    def test_json_aggregate_null_when_absent(self, tmp_path: Path) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1])
        exp = ExperimentResult(experiment_id="no_agg", runs=records)
        # aggregate not set → should be null
        out = tmp_path / "no_agg.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["aggregate"] is None

    def test_json_seeds_correct(self, tmp_path: Path) -> None:
        config = _minimal_config(seed=None)
        seeds = [5, 10, 15]
        records = run_repeated(config, seeds=seeds)
        exp = ExperimentResult(experiment_id="seeds_test", runs=records)
        out = tmp_path / "seeds.json"
        export_json(exp, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        json_seeds = [run["seed"] for run in data["runs"]]
        assert json_seeds == seeds

    def test_json_parent_dirs_created(self, tmp_path: Path) -> None:
        config = _minimal_config()
        records = run_repeated(config, seeds=[1])
        exp = ExperimentResult(experiment_id="dirs_test", runs=records)
        deep = tmp_path / "x" / "y" / "experiment.json"
        export_json(exp, deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# Test 11 — run_sweep returns ExperimentResult with aggregate
# ---------------------------------------------------------------------------


class TestSweepResult:
    """run_sweep returns a fully populated ExperimentResult."""

    def test_experiment_id_is_string(self) -> None:
        points = [SweepPoint("p0", _minimal_config(), {})]
        result = run_sweep(points, [1])
        assert isinstance(result.experiment_id, str)
        assert result.experiment_id

    def test_custom_experiment_id(self) -> None:
        points = [SweepPoint("p0", _minimal_config(), {})]
        result = run_sweep(points, [1], experiment_id="my_exp_001")
        assert result.experiment_id == "my_exp_001"

    def test_run_ids_encode_label_and_seed(self) -> None:
        points = [SweepPoint("mypoint", _minimal_config(), {})]
        result = run_sweep(points, [42], experiment_id="expX")
        assert "mypoint" in result.runs[0].run_id
        assert "42" in result.runs[0].run_id

    def test_params_in_run_records(self) -> None:
        points = [
            SweepPoint("cap1", _minimal_config(), {"capacity": 1}),
            SweepPoint("cap2", _minimal_config(capacity=2), {"capacity": 2}),
        ]
        result = run_sweep(points, [1])
        capacities = [r.params["capacity"] for r in result.runs]
        assert set(capacities) == {1, 2}
