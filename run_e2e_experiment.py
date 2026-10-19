"""
run_e2e_experiment.py
---------------------
End-to-end evacuation experiment runner for PATH_RL.

Pipeline
--------
1. Generate a synthetic commercial building using CommercialLayoutGenerator.
2. Validate the generated exit configuration against NBC 2016 Part 4 rules.
3. Seed occupants using OccupantSeeder.
4. Evaluate (simulate) the evacuation using ConfigurationEvaluator.
5. Save structured results to experiment_output/.

Usage
-----
    python evacuation_simulation/run_e2e_experiment.py
    python evacuation_simulation/run_e2e_experiment.py --seed 42 --occupants 30
    python evacuation_simulation/run_e2e_experiment.py --seed 7 --occupants 50 --exits 4 --rows 26 --cols 48
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure the project root (parent of evacuation_simulation/) is importable
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationParameters
from evacuation_simulation.simulation.evaluation import ConfigurationEvaluator
from evacuation_simulation.simulation.generators.commercial import (
    CommercialGeneratorConfig,
    CommercialLayoutGenerator,
)
from evacuation_simulation.simulation.occupancy_seeder import (
    OccupantSeeder,
    OccupantSeederConfig,
)
from evacuation_simulation.simulation.validator import ExitValidator

# ---------------------------------------------------------------------------
# Output directory — mirrors existing project convention
# ---------------------------------------------------------------------------
_OUTPUT_DIR = Path(__file__).parent / "experiment_output"


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    seed: int = 42,
    total_occupants: int = 20,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_exterior_exits: int = 2,
    max_timesteps: int = 500,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute one complete end-to-end synthetic building evacuation experiment.

    Default parameters produce a 20x30 grid (15m x 10m commercial floor)
    that satisfies all NBC 2016 Part 4 constraints (R1-R5) with 20 occupants
    and 2 exits.

    Returns
    -------
    dict
        Full experiment results ready for JSON serialisation.
    """
    sep = "=" * 70

    # -----------------------------------------------------------------------
    # Step 1: Generate synthetic commercial layout
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n{sep}")
        print("  PATH_RL — End-to-End Synthetic Building Evacuation Experiment")
        print(sep)
        print(f"\n[1/4] Generating synthetic commercial layout ...")
        print(f"      seed={seed}, grid={grid_rows}x{grid_cols}, "
              f"exits={num_exterior_exits}")

    gen_config = CommercialGeneratorConfig(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        num_exterior_exits=num_exterior_exits,
    )
    generator = CommercialLayoutGenerator()
    layout = generator.generate(gen_config)
    floor_plan = layout.floor_plan
    exit_config = layout.exit_configuration

    if verbose:
        print(f"      Layout generated: {floor_plan.grid.rows}x{floor_plan.grid.cols} grid, "
              f"{len(floor_plan.walls)} wall cells, "
              f"{len(floor_plan.doors)} doors, "
              f"{len(layout.rooms)} rooms")
        print(f"      Reference exit config: {exit_config.configuration_id} "
              f"-> exits at {exit_config.exits}")

    # -----------------------------------------------------------------------
    # Step 2: Validate exit configuration against NBC 2016 rules
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[2/4] Validating exit configuration ...")

    validator = ExitValidator()
    # Use an empty occupant scenario for the structure-level compliance check
    # (occupant load is also checked at evaluation time with real occupants)
    from evacuation_simulation.simulation.config import OccupantScenario
    validation_result = validator.check(floor_plan, exit_config, OccupantScenario(agents=[]))

    if verbose:
        status = "COMPLIANT [OK]" if validation_result.is_compliant else "NON-COMPLIANT [FAIL]"
        print(f"      NBC 2016 validation: {status}")
        if validation_result.violations:
            for v in validation_result.violations:
                print(f"      VIOLATION [{v.rule_id}]: {v.message}")

    if not validation_result.is_compliant:
        # Report without aborting — the experiment records the failure
        if verbose:
            print("      WARNING: exit configuration is non-compliant. "
                  "Evaluation will still be attempted and flagged.")

    # -----------------------------------------------------------------------
    # Step 3: Seed occupants
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[3/4] Seeding {total_occupants} occupants ...")

    seeder_config = OccupantSeederConfig(
        total_occupants=total_occupants,
        seed=seed,
        strategy="random_walkable",
        max_occupants_per_cell=1,
    )
    occupant_scenario = OccupantSeeder.seed(floor_plan, exit_config, seeder_config)

    if verbose:
        print(f"      Placed {len(occupant_scenario.agents)} occupants "
              f"on walkable, non-exit cells")

    # -----------------------------------------------------------------------
    # Step 4: Evaluate (simulate evacuation)
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[4/4] Running evacuation simulation ...")

    params = SimulationParameters(
        max_timesteps=max_timesteps,
        random_seed=seed,
        default_cell_capacity=1,
    )
    evaluator = ConfigurationEvaluator(validator=validator)
    eval_result = evaluator.evaluate(floor_plan, exit_config, occupant_scenario, params)

    if verbose:
        print(f"\n{sep}")
        print("  RESULTS")
        print(sep)
        m = eval_result.metrics
        if m is not None:
            print(f"  Occupants placed     : {m.total_agents}")
            print(f"  Evacuated            : {m.evacuated_count} / {m.total_agents}")
            print(f"  Remaining (trapped)  : {m.non_evacuated_count}")
            print(f"  Evacuation rate      : {m.evacuation_rate:.1%}")
            print(f"  Total timesteps run  : {eval_result.simulation_result.total_timesteps}")
            evac_status = "complete" if eval_result.evacuation_complete else "INCOMPLETE"
            print(f"  Total evacuation time: {m.total_evacuation_time} ({evac_status})")
            print(f"  Mean evac. time      : "
                  f"{m.mean_evacuation_time:.1f}" if m.mean_evacuation_time else "  Mean evac. time      : N/A")
            print(f"  Max evac. time       : {m.max_evacuation_time}")
            print(f"  Mean waiting steps   : {m.mean_waiting_steps:.2f}")
            print(f"  Congested cell-steps : {m.congested_cell_steps}")
            print(f"  Score (primary obj.) : {eval_result.score}")
            print(f"  Exit utilisation     : {m.exit_utilization}")
        else:
            print("  Simulation did not run (configuration was invalid).")
        print(f"\n  Feasible             : {eval_result.is_feasible}")
        print(f"  NBC compliant        : {validation_result.is_compliant}")
        if validation_result.violations:
            for v in validation_result.violations:
                print(f"    VIOLATION [{v.rule_id}]: {v.message}")
        print(sep)

    # -----------------------------------------------------------------------
    # Build serialisable result dict
    # -----------------------------------------------------------------------
    m = eval_result.metrics
    violations_data = [
        {"rule_id": v.rule_id, "message": v.message}
        for v in validation_result.violations
    ] if validation_result.violations else []

    exit_util_serializable = {}
    if m and m.exit_utilization:
        for k, v in m.exit_utilization.items():
            exit_util_serializable[str(k)] = v

    result: dict[str, Any] = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "generator": "CommercialLayoutGenerator",
            "generator_config": {
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "num_exterior_exits": num_exterior_exits,
                "cell_size_m": gen_config.cell_size_m,
            },
        },
        "layout": {
            "grid_rows": floor_plan.grid.rows,
            "grid_cols": floor_plan.grid.cols,
            "wall_cells": len(floor_plan.walls),
            "total_doors": len(floor_plan.doors),
            "exterior_doors": sum(1 for d in floor_plan.doors if d.exterior),
            "room_count": len(layout.rooms),
            "rooms": [
                {
                    "room_id": r.room_id,
                    "room_type": r.room_type,
                    "area_cells": len(r.cells),
                    "door_cell": list(r.door_cell),
                }
                for r in layout.rooms
            ],
        },
        "exit_configuration": {
            "configuration_id": exit_config.configuration_id,
            "exits": [list(e) for e in exit_config.exits],
            "widths_m": list(exit_config.widths),
        },
        "constraint_validation": {
            "is_compliant": validation_result.is_compliant,
            "violation_count": len(violations_data),
            "violations": violations_data,
        },
        "occupants": {
            "total_placed": len(occupant_scenario.agents),
            "seeder_seed": seed,
        },
        "simulation": {
            "max_timesteps": max_timesteps,
            "is_feasible": eval_result.is_feasible,
            "evacuation_complete": eval_result.evacuation_complete,
            "total_timesteps_run": (
                eval_result.simulation_result.total_timesteps
                if eval_result.simulation_result
                else None
            ),
        },
        "metrics": {
            "total_agents": m.total_agents if m else None,
            "evacuated_count": m.evacuated_count if m else None,
            "non_evacuated_count": m.non_evacuated_count if m else None,
            "evacuation_rate": m.evacuation_rate if m else None,
            "total_evacuation_time": m.total_evacuation_time if m else None,
            "mean_evacuation_time": m.mean_evacuation_time if m else None,
            "min_evacuation_time": m.min_evacuation_time if m else None,
            "max_evacuation_time": m.max_evacuation_time if m else None,
            "mean_waiting_steps": m.mean_waiting_steps if m else None,
            "max_waiting_steps": m.max_waiting_steps if m else None,
            "total_waiting_steps": m.total_waiting_steps if m else None,
            "congested_cell_steps": m.congested_cell_steps if m else None,
            "max_cell_occupancy": m.max_cell_occupancy if m else None,
            "max_occupancy_ratio": m.max_occupancy_ratio if m else None,
            "exit_utilization": exit_util_serializable,
            "score_primary_objective": eval_result.score,
        } if m else None,
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PATH_RL -- End-to-End Synthetic Evacuation Experiment"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--occupants", type=int, default=20, help="Number of occupants (default: 20)")
    parser.add_argument("--rows", type=int, default=20, help="Grid rows (default: 20, ~10m)")
    parser.add_argument("--cols", type=int, default=30, help="Grid cols (default: 30, ~15m)")
    parser.add_argument("--exits", type=int, default=2, choices=[2, 3, 4],
                        help="Number of exterior exits (default: 2)")
    parser.add_argument("--max-timesteps", type=int, default=500,
                        help="Simulation timestep limit (default: 500)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file path (auto-named if omitted)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    result = run_experiment(
        seed=args.seed,
        total_occupants=args.occupants,
        grid_rows=args.rows,
        grid_cols=args.cols,
        num_exterior_exits=args.exits,
        max_timesteps=args.max_timesteps,
        verbose=True,
    )

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _OUTPUT_DIR / f"synthetic_e2e_seed{args.seed}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"\nResults saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
