"""
stage10_demo.py
---------------
Stage 10 Demonstration Script: Realistic Floor-Plan & Exit Representation.

Demonstrates:
1. FloorPlan with explicit exterior and interior DoorOpenings.
2. Derivation of CandidateExit objects strictly from exterior openings (excluding interior doors).
3. Preservation of physical opening properties (widths, coordinates).
4. Combinatorial generation of ExitConfiguration objects with widths.
5. Multi-configuration execution on the same FloorPlan without mutation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure RL_ENV is in sys.path
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.generation import (
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


def run_demo() -> None:
    print("=" * 80)
    print(" STAGE 10: REALISTIC FLOOR-PLAN & EXIT REPRESENTATION DEMO")
    print("=" * 80)
    print()

    scenarios_dir = Path("scenarios")
    scenario_files = ["basic_building.json", "bottleneck.json"]

    for filename in scenario_files:
        filepath = scenarios_dir / filename
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        config = SimulationConfig(**data)
        floor_plan = config.floor_plan

        print(f"Floor plan: {config.scenario_name} ({filepath.name})")
        print(f"  Grid size: {floor_plan.grid.rows}x{floor_plan.grid.cols}")
        print(f"  Walls: {len(floor_plan.walls)} solid wall cells")
        print(f"  Total DoorOpenings defined: {len(floor_plan.doors)}")

        # Distinguish exterior vs interior openings
        exterior_doors = [d for d in floor_plan.doors if d.exterior]
        interior_doors = [d for d in floor_plan.doors if not d.exterior]

        print(f"  Exterior openings ({len(exterior_doors)}):")
        for d in exterior_doors:
            label = f" [{d.door_id}]" if d.door_id else ""
            print(f"    - position={d.position}, width={d.width:.1f}m{label}")

        if interior_doors:
            print(f"  Interior openings ({len(interior_doors)}) [EXCLUDED from exit candidates]:")
            for d in interior_doors:
                label = f" [{d.door_id}]" if d.door_id else ""
                print(f"    - position={d.position}, width={d.width:.1f}m{label} (interior passage)")

        # 1. Candidate Exit Generation
        candidates = generate_candidate_exits(floor_plan)
        print(f"\n  Candidate exits derived: {len(candidates)}")
        for c in candidates:
            label = f" [{c.door_id}]" if c.door_id else ""
            print(f"    * candidate: position={c.position}, width={c.width:.1f}m{label}")

        if not candidates:
            print("  (No candidate exits available)")
            print("-" * 80)
            continue

        # 2. Exit Configuration Generation
        for requested_exits in [1, 2]:
            if requested_exits > len(candidates):
                continue

            configs = list(generate_exit_configurations(candidates, requested_exits))
            print(f"\n  Requested exits: {requested_exits}")
            print(f"  Generated configurations: {len(configs)}")

            for cfg in configs:
                exit_str = ", ".join(f"{pos} (w={w:.1f}m)" for pos, w in zip(cfg.exits, cfg.widths))
                print(f"    Config {cfg.configuration_id}: [{exit_str}]")

        # 3. Demonstration of Simulation Execution on Same FloorPlan with Different Configs
        print("\n  Multi-configuration Simulation Execution (Same FloorPlan, 0 mutation):")
        two_exit_configs = list(generate_exit_configurations(candidates, min(2, len(candidates))))
        for i, cfg in enumerate(two_exit_configs[:2]):
            eval_config = SimulationConfig(
                scenario_name=f"{config.scenario_name}_{cfg.configuration_id}",
                floor_plan=floor_plan,
                exit_configuration=cfg,
                occupants=config.occupants,
                parameters=config.parameters,
            )
            sim = Simulation(eval_config, movement_strategy=ShortestPathStrategy())
            result = sim.run()
            exits_summary = ", ".join(f"{pos} ({w}m)" for pos, w in zip(cfg.exits, cfg.widths))
            print(
                f"    - Simulation with {cfg.configuration_id} ({exits_summary}): "
                f"Evacuated {result.evacuated_count}/{len(config.agents)} agents in {result.total_timesteps} steps"
            )

        print("-" * 80)


if __name__ == "__main__":
    run_demo()
