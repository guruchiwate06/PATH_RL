"""
experiment.py
-------------
Scenario runner and experimental framework for the 2D multi-agent
evacuation simulation.

Responsibilities
~~~~~~~~~~~~~~~~
- Wrap ``Simulation.run()`` for single, repeated-seed, and sweep execution.
- Collect ``RunRecord`` instances that pair metadata with ``SimulationResult``.
- Aggregate numeric metrics over collections of runs.
- Export results to CSV (one row per run) and JSON (full experiment).

Non-responsibilities (intentionally excluded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- This module does NOT modify any simulation behaviour.
- It does NOT implement new movement strategies, metrics, or environment models.
- The existing ``Simulation`` class remains the single source of truth for
  all simulation logic.
- This module is NOT imported by any core simulation module; the simulation
  remains fully usable without it.

Dependency direction (enforced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    experiment.py
         ↓
    simulation.py
         ↓
    metrics.py

Design notes
~~~~~~~~~~~~
- ``RunRecord`` is a plain dataclass; it carries a ``SimulationResult``
  directly so all existing metrics are preserved without re-calculation.
- ``SweepPoint`` is the caller's responsibility to construct with a valid
  ``SimulationConfig``.  The framework never mutates a shared config; each
  run receives its own seed-overridden copy via Pydantic ``model_copy``.
- Seed override is the only config mutation the framework performs, and it
  happens via ``SimulationParameters.model_copy``, which re-validates the
  model.
- Aggregation handles ``None`` metric values correctly: ``None`` values are
  excluded from numeric aggregates; if all values are ``None`` the
  aggregate is reported as ``None``, not zero.
- CSV export collects the union of all param keys across records so that
  mixed sweeps export cleanly.
"""

from __future__ import annotations

import csv
import datetime
import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.metrics import SimulationResult
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

#: Fixed ordered list of SimulationResult metric field names used in
#: CSV / JSON output.  Any new metrics added to SimulationResult must be
#: reflected here to appear in exports.
_METRIC_FIELDS: tuple[str, ...] = (
    "total_agents",
    "evacuated_count",
    "non_evacuated_count",
    "total_timesteps",
    "evacuation_rate",
    "mean_evacuation_time",
    "min_evacuation_time",
    "max_evacuation_time",
    "total_waiting_steps",
    "mean_waiting_steps",
    "max_waiting_steps",
    "max_cell_occupancy",
    "max_occupancy_ratio",
    "congested_cell_steps",
)


# ---------------------------------------------------------------------------
# Helper: seed application
# ---------------------------------------------------------------------------


def _apply_seed(config: SimulationConfig, seed: int) -> SimulationConfig:
    """
    Return a copy of *config* with ``random_seed`` set to *seed*.

    Uses Pydantic ``model_copy`` so that the original config is never
    mutated and the result is fully validated.

    Parameters
    ----------
    config:
        Source configuration.
    seed:
        New random seed value.

    Returns
    -------
    SimulationConfig
        An independent copy identical to *config* except for
        ``parameters.random_seed``.
    """
    new_params = config.parameters.model_copy(update={"random_seed": seed})
    return config.model_copy(update={"parameters": new_params})


# ---------------------------------------------------------------------------
# Helper: ID generation
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Return a compact UTC timestamp string suitable for IDs."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")


def _make_run_id(scenario_name: str, seed: Optional[int], index: Optional[int] = None) -> str:
    """
    Generate a human-readable run identifier.

    Parameters
    ----------
    scenario_name:
        Scenario label from the config.
    seed:
        Random seed used for this run, or ``None``.
    index:
        Optional sequential index to ensure uniqueness when multiple runs
        share the same scenario name and seed.
    """
    seed_part = f"seed{seed}" if seed is not None else "noseed"
    idx_part = f"_{index:04d}" if index is not None else ""
    return f"{scenario_name}_{seed_part}{idx_part}"


def _make_experiment_id() -> str:
    """Generate a timestamp-based experiment identifier."""
    return f"exp_{_timestamp()}"


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """
    Record of a single completed simulation run.

    Attributes
    ----------
    run_id:
        Unique string identifying this run within an experiment.
    scenario_name:
        Scenario label carried from the config.
    seed:
        Random seed used for this run (may be ``None`` for non-deterministic
        runs or runs whose config already specifies a seed).
    params:
        Dictionary of parameter values that distinguish this run within a
        sweep (e.g. ``{"agent_count": 12, "capacity": 2}``).  Empty for
        single runs and repeated-seed experiments.
    result:
        The ``SimulationResult`` returned by ``Simulation.run()``.
    """

    run_id: str
    scenario_name: str
    seed: Optional[int]
    params: dict[str, Any]
    result: SimulationResult

    def to_flat_dict(self, param_keys: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Flatten into a single ``dict`` suitable for a CSV row.

        The output column order is:

        1. Identifier columns: ``run_id``, ``scenario_name``, ``seed``.
        2. Sweep parameter columns (provided by *param_keys*; missing keys
           filled with ``None``).
        3. Metric columns (fixed order, see ``_METRIC_FIELDS``).

        Parameters
        ----------
        param_keys:
            Ordered list of parameter key names to include.  Keys absent
            from this record's ``params`` dict are written as ``None``.
            Pass ``None`` to use only the keys present in this record.
        """
        row: dict[str, Any] = {}

        # Identifier columns
        row["run_id"] = self.run_id
        row["scenario_name"] = self.scenario_name
        row["seed"] = self.seed

        # Sweep parameter columns
        keys = param_keys if param_keys is not None else sorted(self.params.keys())
        for k in keys:
            row[k] = self.params.get(k)

        # Metric columns
        for metric in _METRIC_FIELDS:
            row[metric] = getattr(self.result, metric, None)

        return row


# ---------------------------------------------------------------------------
# Aggregation result
# ---------------------------------------------------------------------------


@dataclass
class AggregateStats:
    """
    Aggregate statistics computed over a collection of ``RunRecord`` objects.

    Numeric aggregates that correspond to an ``Optional`` metric in
    ``SimulationResult`` (e.g. ``mean_evacuation_time``) are themselves
    ``Optional``: they are ``None`` when every run in the collection produced
    a ``None`` for that metric (e.g. no agents evacuated in any run).

    Attributes
    ----------
    run_count:
        Total number of runs included in the aggregate.
    mean_evacuation_time_mean:
        Mean of per-run ``mean_evacuation_time`` values (excluding ``None``).
    mean_evacuation_time_min:
        Minimum of per-run ``mean_evacuation_time`` values.
    mean_evacuation_time_max:
        Maximum of per-run ``mean_evacuation_time`` values.
    mean_evacuation_time_std:
        Standard deviation of per-run ``mean_evacuation_time`` values.
        ``0.0`` if only one non-``None`` value is present.
    mean_evacuation_time_median:
        Median of per-run ``mean_evacuation_time`` values.
    evacuation_rate_mean:
        Mean evacuation rate across runs.
    evacuation_rate_min:
        Minimum evacuation rate.
    evacuation_rate_max:
        Maximum evacuation rate.
    total_timesteps_mean:
        Mean total timesteps.
    total_timesteps_min:
        Minimum total timesteps.
    total_timesteps_max:
        Maximum total timesteps.
    total_waiting_steps_mean:
        Mean of total waiting steps.
    total_waiting_steps_min:
        Minimum total waiting steps.
    total_waiting_steps_max:
        Maximum total waiting steps.
    congested_cell_steps_mean:
        Mean of congested-cell-step counts.
    congested_cell_steps_min:
        Minimum congested-cell-step count.
    congested_cell_steps_max:
        Maximum congested-cell-step count.
    max_occupancy_ratio_mean:
        Mean of per-run maximum occupancy ratios.
    max_occupancy_ratio_min:
        Minimum per-run maximum occupancy ratio.
    max_occupancy_ratio_max:
        Maximum per-run maximum occupancy ratio.
    """

    run_count: int

    # Evacuation time (Optional — None if no run produced evacuations)
    mean_evacuation_time_mean: Optional[float]
    mean_evacuation_time_min: Optional[float]
    mean_evacuation_time_max: Optional[float]
    mean_evacuation_time_std: Optional[float]
    mean_evacuation_time_median: Optional[float]

    # Evacuation rate (always present — 0.0 if nobody evacuated)
    evacuation_rate_mean: float
    evacuation_rate_min: float
    evacuation_rate_max: float

    # Timesteps
    total_timesteps_mean: float
    total_timesteps_min: int
    total_timesteps_max: int

    # Waiting
    total_waiting_steps_mean: float
    total_waiting_steps_min: int
    total_waiting_steps_max: int

    # Congestion
    congested_cell_steps_mean: float
    congested_cell_steps_min: int
    congested_cell_steps_max: int
    max_occupancy_ratio_mean: float
    max_occupancy_ratio_min: float
    max_occupancy_ratio_max: float


# ---------------------------------------------------------------------------
# Sweep point
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    """
    A single named configuration point in a parameter sweep.

    The caller is responsible for constructing a valid ``SimulationConfig``
    for each point.  The framework never mutates a shared config object.

    Attributes
    ----------
    label:
        Human-readable name for this sweep point
        (e.g. ``"agents=12"``, ``"capacity=2"``).
    config:
        A fully-validated ``SimulationConfig`` for this point.
    params:
        Dictionary of parameter values that identify this point in exported
        results (e.g. ``{"agent_count": 12}``).  These values are stored in
        ``RunRecord.params`` and appear as columns in CSV exports.
    """

    label: str
    config: SimulationConfig
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experiment result container
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """
    Container for all results from a single experiment.

    Attributes
    ----------
    experiment_id:
        Unique identifier for this experiment run.
    runs:
        List of ``RunRecord`` objects, one per simulation execution.
    aggregate:
        Aggregate statistics over all ``runs``, or ``None`` if aggregation
        has not been computed.
    """

    experiment_id: str
    runs: list[RunRecord] = field(default_factory=list)
    aggregate: Optional[AggregateStats] = None

    @property
    def run_count(self) -> int:
        """Total number of completed runs."""
        return len(self.runs)


# ---------------------------------------------------------------------------
# Core execution functions
# ---------------------------------------------------------------------------


def run_scenario(
    config: SimulationConfig,
    *,
    seed: Optional[int] = None,
    params: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> RunRecord:
    """
    Run a single simulation scenario.

    Creates a fresh ``Simulation`` with ``ShortestPathStrategy``, runs it to
    completion, and returns the result wrapped in a ``RunRecord``.

    Parameters
    ----------
    config:
        A fully-validated ``SimulationConfig``.
    seed:
        If provided, overrides ``config.parameters.random_seed`` for this
        run only.  The original *config* is never mutated.
    params:
        Optional metadata dictionary for sweep identification (e.g.
        ``{"agent_count": 6}``).  Not used in simulation logic.
    run_id:
        Optional explicit run identifier.  Auto-generated from scenario name
        and seed if not provided.

    Returns
    -------
    RunRecord
        A record containing the run identifier, seed, params, and the
        ``SimulationResult``.
    """
    # Apply seed override without mutating the original config
    effective_config: SimulationConfig = (
        _apply_seed(config, seed) if seed is not None else config
    )

    # Each call creates an independent Simulation instance
    sim = Simulation(effective_config, ShortestPathStrategy())
    result = sim.run()

    effective_seed = effective_config.parameters.random_seed
    rid = run_id or _make_run_id(config.scenario_name, effective_seed)

    return RunRecord(
        run_id=rid,
        scenario_name=config.scenario_name,
        seed=effective_seed,
        params=dict(params) if params else {},
        result=result,
    )


def run_repeated(
    config: SimulationConfig,
    seeds: list[int],
) -> list[RunRecord]:
    """
    Run the same scenario independently for each seed in *seeds*.

    Each seed produces an independent simulation instance.  No state is
    shared between runs.

    Parameters
    ----------
    config:
        Base scenario configuration.  Not mutated.
    seeds:
        List of integer seeds.  Must not be empty.

    Returns
    -------
    list[RunRecord]
        One ``RunRecord`` per seed, in the same order as *seeds*.

    Raises
    ------
    ValueError
        If *seeds* is empty.
    """
    if not seeds:
        raise ValueError("seeds must not be empty.")

    return [
        run_scenario(
            config,
            seed=s,
            run_id=_make_run_id(config.scenario_name, s, index=i),
        )
        for i, s in enumerate(seeds)
    ]


def run_sweep(
    sweep_points: list[SweepPoint],
    seeds: list[int],
    *,
    experiment_id: Optional[str] = None,
) -> ExperimentResult:
    """
    Run a full parameter sweep: every ``SweepPoint`` × every seed.

    Total runs = ``len(sweep_points) × len(seeds)``.

    Each combination produces an independent simulation instance.

    Parameters
    ----------
    sweep_points:
        List of named configuration points.  Each must contain a fully-
        validated ``SimulationConfig`` and a ``params`` dict describing the
        parameter values that differ from the baseline.
    seeds:
        List of integer seeds applied to every sweep point.
    experiment_id:
        Optional explicit experiment identifier.  Auto-generated from the
        current UTC timestamp if not provided.

    Returns
    -------
    ExperimentResult
        Contains all ``len(sweep_points) × len(seeds)`` run records and
        aggregate statistics over all runs.

    Raises
    ------
    ValueError
        If *sweep_points* or *seeds* is empty.
    """
    if not sweep_points:
        raise ValueError("sweep_points must not be empty.")
    if not seeds:
        raise ValueError("seeds must not be empty.")

    exp_id = experiment_id or _make_experiment_id()
    runs: list[RunRecord] = []

    for pt_idx, point in enumerate(sweep_points):
        for s in seeds:
            rid = f"{exp_id}_pt{pt_idx:03d}_{point.label}_seed{s}"
            record = run_scenario(
                point.config,
                seed=s,
                params=dict(point.params),
                run_id=rid,
            )
            runs.append(record)

    experiment = ExperimentResult(experiment_id=exp_id, runs=runs)
    experiment.aggregate = aggregate_results(runs)
    return experiment


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg_optional_float(
    values: list[Optional[float]],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute (mean, min, max, std, median) over non-``None`` values.

    Returns a 5-tuple of ``None`` values if every element is ``None``.
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None, None, None, None, None
    mean = statistics.mean(non_null)
    mn = min(non_null)
    mx = max(non_null)
    std = statistics.stdev(non_null) if len(non_null) > 1 else 0.0
    med = statistics.median(non_null)
    return mean, mn, mx, std, med


def _agg_float(values: list[float]) -> tuple[float, float, float]:
    """Compute (mean, min, max) over a non-empty list of floats."""
    return statistics.mean(values), min(values), max(values)


def _agg_int(values: list[int]) -> tuple[float, int, int]:
    """Compute (mean, min, max) over a non-empty list of ints."""
    return statistics.mean(values), min(values), max(values)


def aggregate_results(records: list[RunRecord]) -> AggregateStats:
    """
    Compute aggregate statistics over a collection of ``RunRecord`` objects.

    ``None`` values in ``SimulationResult`` fields are excluded from numeric
    aggregates.  If all values for a field are ``None``, the corresponding
    aggregate fields are ``None`` (not zero).

    Parameters
    ----------
    records:
        Non-empty list of ``RunRecord`` objects.

    Returns
    -------
    AggregateStats

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("Cannot aggregate an empty list of RunRecords.")

    results = [r.result for r in records]

    # Optional float field: mean_evacuation_time
    met_mean, met_min, met_max, met_std, met_med = _agg_optional_float(
        [r.mean_evacuation_time for r in results]
    )

    rate_mean, rate_min, rate_max = _agg_float(
        [r.evacuation_rate for r in results]
    )
    ts_mean, ts_min, ts_max = _agg_int(
        [r.total_timesteps for r in results]
    )
    wait_mean, wait_min, wait_max = _agg_int(
        [r.total_waiting_steps for r in results]
    )
    cong_mean, cong_min, cong_max = _agg_int(
        [r.congested_cell_steps for r in results]
    )
    occ_mean, occ_min, occ_max = _agg_float(
        [r.max_occupancy_ratio for r in results]
    )

    return AggregateStats(
        run_count=len(records),
        mean_evacuation_time_mean=met_mean,
        mean_evacuation_time_min=met_min,
        mean_evacuation_time_max=met_max,
        mean_evacuation_time_std=met_std,
        mean_evacuation_time_median=met_med,
        evacuation_rate_mean=rate_mean,
        evacuation_rate_min=rate_min,
        evacuation_rate_max=rate_max,
        total_timesteps_mean=ts_mean,
        total_timesteps_min=ts_min,
        total_timesteps_max=ts_max,
        total_waiting_steps_mean=wait_mean,
        total_waiting_steps_min=wait_min,
        total_waiting_steps_max=wait_max,
        congested_cell_steps_mean=cong_mean,
        congested_cell_steps_min=cong_min,
        congested_cell_steps_max=cong_max,
        max_occupancy_ratio_mean=occ_mean,
        max_occupancy_ratio_min=occ_min,
        max_occupancy_ratio_max=occ_max,
    )


# ---------------------------------------------------------------------------
# Export: CSV
# ---------------------------------------------------------------------------


def export_csv(
    records: list[RunRecord],
    path: "Path | str",
) -> Path:
    """
    Export a list of ``RunRecord`` objects to a CSV file.

    One row per run.  Columns:

    1. ``run_id``, ``scenario_name``, ``seed``
    2. All unique sweep parameter keys (union across all records, sorted)
    3. All metric fields (fixed order per ``_METRIC_FIELDS``)

    Missing parameter values (a record that lacks a key present in another
    record) are written as an empty cell.

    Parameters
    ----------
    records:
        Non-empty list of ``RunRecord`` objects to export.
    path:
        Destination file path.  Parent directories are created if necessary.

    Returns
    -------
    Path
        The resolved path of the written file.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("No records to export.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect the union of all param keys across records (sorted for reproducibility)
    all_param_keys: list[str] = sorted(
        {k for r in records for k in r.params}
    )

    rows = [r.to_flat_dict(param_keys=all_param_keys) for r in records]
    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# ---------------------------------------------------------------------------
# Export: JSON
# ---------------------------------------------------------------------------


def _serialize_result(result: SimulationResult) -> dict[str, Any]:
    """Serialize a ``SimulationResult`` to a JSON-compatible dict."""
    return {metric: getattr(result, metric, None) for metric in _METRIC_FIELDS}


def _serialize_run_record(record: RunRecord) -> dict[str, Any]:
    """Serialize a ``RunRecord`` to a JSON-compatible dict."""
    return {
        "run_id": record.run_id,
        "scenario_name": record.scenario_name,
        "seed": record.seed,
        "params": record.params,
        "result": _serialize_result(record.result),
    }


def _serialize_aggregate(agg: AggregateStats) -> dict[str, Any]:
    """Serialize an ``AggregateStats`` to a JSON-compatible dict."""
    # asdict handles Optional fields (None stays None)
    return asdict(agg)


def export_json(
    experiment: ExperimentResult,
    path: "Path | str",
) -> Path:
    """
    Export an ``ExperimentResult`` to a JSON file.

    The output contains:

    - ``experiment_id``: str
    - ``run_count``: int
    - ``runs``: list of run records (one per simulation execution)
    - ``aggregate``: aggregate statistics, or ``null`` if not computed

    The output is plain JSON and can be read with the standard ``json``
    library without importing any simulation classes.

    Parameters
    ----------
    experiment:
        The ``ExperimentResult`` to serialize.
    path:
        Destination file path.  Parent directories are created if necessary.

    Returns
    -------
    Path
        The resolved path of the written file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "experiment_id": experiment.experiment_id,
        "run_count": experiment.run_count,
        "runs": [_serialize_run_record(r) for r in experiment.runs],
        "aggregate": (
            _serialize_aggregate(experiment.aggregate)
            if experiment.aggregate is not None
            else None
        ),
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    return out_path
