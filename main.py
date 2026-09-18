"""
main.py
-------
Entry point for the evacuation simulation.

Loads the basic_building scenario, runs the simulation, and prints the
resulting outcome report.

Usage
-----
    python main.py

    # Optionally override the scenario file:
    python main.py --scenario scenarios/basic_building.json
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
        description="2D multi-agent evacuation simulation",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).parent / "scenarios" / "basic_building.json",
        help="Path to a scenario JSON file (default: scenarios/basic_building.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Import here so that the entry point itself remains importable even if
    # optional dependencies are partially installed.
    from evacuation_simulation.simulation.config import SimulationConfig
    from evacuation_simulation.simulation.simulation import Simulation

    print(f"Loading scenario: {args.scenario}")
    try:
        config = SimulationConfig.from_json(args.scenario)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Scenario validation error: {exc}", file=sys.stderr)
        return 1

    print(f"Scenario: {config.scenario_name!r}")
    print(f"Grid    : {config.grid.rows}×{config.grid.cols}")
    print(f"Agents  : {len(config.agents)}")
    print(f"Exits   : {config.exits}")
    print(f"Walls   : {len(config.walls)} cells")
    print()

    sim = Simulation(config)
    print(f"Navigation graph: {sim.nav_graph}")
    print(f"Starting simulation …")
    print()

    result = sim.run()

    print(result)
    print()
    print(f"Termination reason: {sim.state.termination_reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
