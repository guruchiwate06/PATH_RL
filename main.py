"""
main.py
-------
Entry point for the evacuation simulation.

Loads a scenario, runs the simulation with ``ShortestPathStrategy``, and
prints a detailed outcome report with per-agent evacuation times.

Usage
-----
    python main.py

    # Use a custom scenario file:
    python main.py --scenario scenarios/basic_building.json

    # Show a matplotlib grid snapshot after simulation:
    python main.py --visualize

    # Run a multi-seed experiment (Stage 5):
    python main.py --experiment --seeds 1,2,3,4,5

    # Run experiment and save results to a directory:
    python main.py --experiment --seeds 1,2,3,4,5 --output-dir results/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the package parent directory (RL_ENV/) is on sys.path so that
# `import evacuation_simulation` works regardless of the invocation CWD.
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D multi-agent evacuation simulation — Stage 5",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).parent / "scenarios" / "basic_building.json",
        help="Path to a scenario JSON file (default: scenarios/basic_building.json)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open interactive Simulation Visualizer & Inspector",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Alias for --visualize (opens interactive inspector)",
    )
    parser.add_argument(
        "--play", "--autoplay",
        action="store_true",
        dest="play",
        help="Automatically start playing animation upon launch",
    )
    parser.add_argument(
        "--strategy",
        choices=["shortest", "null"],
        default="shortest",
        help="Movement strategy to use (default: shortest)",
    )
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        default=None,
        help="Save a PNG snapshot of the initial simulation state to this path",
    )
    # --- Stage 5: experiment mode ---
    parser.add_argument(
        "--experiment",
        action="store_true",
        help=(
            "Run a multi-seed experiment instead of a single simulation. "
            "Use --seeds to specify seeds and --output-dir for CSV/JSON export."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated list of integer seeds for --experiment mode. "
            "Example: --seeds 1,2,3,4,5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for experiment output files (CSV and JSON). "
            "Used with --experiment. Default: ./experiment_output/"
        ),
    )
    return parser.parse_args()


def _run_single(args: argparse.Namespace) -> int:
    """Run the existing single-scenario simulation flow (unchanged from Stage 4)."""
    from evacuation_simulation.simulation.config import SimulationConfig
    from evacuation_simulation.simulation.simulation import (
        NullMovementStrategy,
        Simulation,
    )
    from evacuation_simulation.simulation.strategy import ShortestPathStrategy

    # --- Load scenario ---------------------------------------------------
    print(f"Loading scenario : {args.scenario}")
    try:
        config = SimulationConfig.from_json(args.scenario)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Scenario validation error: {exc}", file=sys.stderr)
        return 1

    print(f"Scenario         : {config.scenario_name!r}")
    print(f"Grid             : {config.grid.rows}×{config.grid.cols}")
    print(f"Agents           : {len(config.agents)}")
    print(f"Exits            : {config.exits}")
    print(f"Walls            : {len(config.walls)} cells")
    print(f"Strategy         : {args.strategy}")
    print(f"Random seed      : {config.parameters.random_seed}")
    print()

    # --- Build simulation ------------------------------------------------
    strategy_factory = (
        (lambda: ShortestPathStrategy())
        if args.strategy == "shortest"
        else (lambda: NullMovementStrategy())
    )
    sim = Simulation(config, movement_strategy=strategy_factory())

    print(f"Navigation graph : {sim.nav_graph}")
    print()

    # --- Optional: Save initial snapshot ---------------------------------
    if args.save_snapshot:
        try:
            from visualize import SimulationVisualizer
            viz = SimulationVisualizer(
                config=config,
                movement_strategy_factory=strategy_factory,
            )
            saved_path = viz.save_snapshot(args.save_snapshot)
            print(f"[visualize] Snapshot saved: {saved_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[visualize] Could not save snapshot: {exc}")

    # --- Interactive Visualization Mode ----------------------------------
    if args.visualize or args.inspect or args.play:
        print("Launching interactive Simulation Visualizer & Inspector…")
        try:
            from visualize import SimulationVisualizer
            viz = SimulationVisualizer(
                config=config,
                movement_strategy_factory=strategy_factory,
            )
            viz.show(auto_play=args.play)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[visualize] Error launching interactive GUI: {exc}")
            print("Falling back to batch simulation execution…\n")

    # --- Batch Simulation Mode -------------------------------------------
    print("Running simulation…")
    result = sim.run()
    print()

    # --- Print result ----------------------------------------------------
    print(result)
    print()
    print(f"Termination      : {sim.state.termination_reason}")
    print()

    # --- Per-agent breakdown --------------------------------------------
    if result.evacuation_times:
        print("Per-agent evacuation times:")
        for agent in sorted(
            sim.agents,
            key=lambda a: (a.evacuation_timestep or 999_999, a.agent_id),
        ):
            if agent.evacuation_timestep is not None:
                print(f"  {agent.agent_id:12s}  evacuated at step {agent.evacuation_timestep}")
            else:
                print(f"  {agent.agent_id:12s}  NOT evacuated")

    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    """
    Stage 5: multi-seed experiment mode.

    Loads the specified scenario, runs it once per seed, aggregates results,
    and optionally exports CSV and JSON.
    """
    from evacuation_simulation.simulation.config import SimulationConfig
    from evacuation_simulation.simulation.experiment import (
        aggregate_results,
        export_csv,
        export_json,
        run_repeated,
        ExperimentResult,
    )

    # --- Parse seeds -------------------------------------------------------
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        except ValueError:
            print(
                "Error: --seeds must be a comma-separated list of integers. "
                f"Got: {args.seeds!r}",
                file=sys.stderr,
            )
            return 1
    else:
        seeds = [1, 2, 3, 4, 5]
        print(f"No --seeds specified; using default seeds: {seeds}")

    # --- Load scenario -----------------------------------------------------
    print(f"Loading scenario : {args.scenario}")
    try:
        config = SimulationConfig.from_json(args.scenario)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Scenario validation error: {exc}", file=sys.stderr)
        return 1

    print(f"Scenario         : {config.scenario_name!r}")
    print(f"Grid             : {config.grid.rows}×{config.grid.cols}")
    print(f"Agents           : {len(config.agents)}")
    print(f"Seeds            : {seeds}")
    print(f"Total runs       : {len(seeds)}")
    print()

    # --- Execute experiment ------------------------------------------------
    t0 = time.perf_counter()
    records = run_repeated(config, seeds=seeds)
    elapsed = time.perf_counter() - t0

    # --- Aggregate ---------------------------------------------------------
    agg = aggregate_results(records)

    # --- Print per-run results ---------------------------------------------
    print("Per-run results:")
    print(f"  {'run_id':<40}  {'seed':>6}  {'evacuated':>10}  {'mean_evac_t':>12}  {'wait_steps':>11}")
    print("  " + "-" * 85)
    for rec in records:
        r = rec.result
        met = f"{r.mean_evacuation_time:.1f}" if r.mean_evacuation_time is not None else "N/A"
        print(
            f"  {rec.run_id:<40}  {str(rec.seed):>6}  "
            f"{r.evacuated_count:>9}/{r.total_agents:<1}  "
            f"{met:>12}  {r.total_waiting_steps:>11}"
        )

    # --- Print aggregate summary ------------------------------------------
    print()
    print("Aggregate summary:")
    print(f"  Runs                      : {agg.run_count}")
    print(f"  Evacuation rate  mean     : {agg.evacuation_rate_mean:.1%}")
    print(f"  Evacuation rate  min/max  : {agg.evacuation_rate_min:.1%} / {agg.evacuation_rate_max:.1%}")
    if agg.mean_evacuation_time_mean is not None:
        print(f"  Mean evac. time  mean     : {agg.mean_evacuation_time_mean:.2f}")
        print(f"  Mean evac. time  min/max  : {agg.mean_evacuation_time_min:.2f} / {agg.mean_evacuation_time_max:.2f}")
        if agg.mean_evacuation_time_std is not None:
            print(f"  Mean evac. time  std      : {agg.mean_evacuation_time_std:.4f}")
    else:
        print("  Mean evac. time           : N/A (no agents evacuated in any run)")
    print(f"  Total timesteps  mean     : {agg.total_timesteps_mean:.1f}")
    print(f"  Total wait steps mean     : {agg.total_waiting_steps_mean:.2f}")
    print(f"  Congested cell-steps mean : {agg.congested_cell_steps_mean:.2f}")
    print()
    print(f"  Total runtime   : {elapsed:.3f}s")
    print(f"  Avg per run     : {elapsed / len(records):.4f}s")

    # --- Export ------------------------------------------------------------
    output_dir = args.output_dir or Path("experiment_output")
    experiment = ExperimentResult(
        experiment_id=f"exp_{config.scenario_name}",
        runs=records,
        aggregate=agg,
    )

    csv_path = export_csv(records, output_dir / f"{config.scenario_name}_results.csv")
    json_path = export_json(experiment, output_dir / f"{config.scenario_name}_experiment.json")

    print()
    print(f"  CSV  saved to : {csv_path}")
    print(f"  JSON saved to : {json_path}")

    return 0


def main() -> int:
    args = _parse_args()

    if args.experiment:
        return _run_experiment(args)
    return _run_single(args)


if __name__ == "__main__":
    sys.exit(main())
