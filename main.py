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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package parent directory (RL_ENV/) is on sys.path so that
# `import evacuation_simulation` works regardless of the invocation CWD.
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D multi-agent evacuation simulation — Stage 2",
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
        help="Show a matplotlib grid snapshot of the initial and final states",
    )
    parser.add_argument(
        "--strategy",
        choices=["shortest", "null"],
        default="shortest",
        help="Movement strategy to use (default: shortest)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

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
    strategy = (
        ShortestPathStrategy()
        if args.strategy == "shortest"
        else NullMovementStrategy()
    )
    sim = Simulation(config, movement_strategy=strategy)

    print(f"Navigation graph : {sim.nav_graph}")
    print()

    # --- Optional: show initial state ------------------------------------
    if args.visualize:
        try:
            from visualize import render_grid
            render_grid(sim, title="Initial State", show=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[visualize] Could not render: {exc}")

    # --- Run simulation --------------------------------------------------
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

    # --- Optional: show final state --------------------------------------
    if args.visualize:
        try:
            from visualize import render_grid
            render_grid(sim, title="Final State", show=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[visualize] Could not render: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
