"""
run_baseline_experiment.py
--------------------------
Stage 17: Exhaustive Exit-Configuration Baseline Experiment Runner.

Connects:
1. Procedural commercial layout generator (CommercialLayoutGenerator)
2. Deterministic occupant seeder (OccupantSeeder)
3. Exhaustive combinatorial exit searcher (ExitConfigurationSearcher)
4. NBC 2016 Part 4 regulatory validator (ExitValidator)
5. Simulation evaluator & ranking engine (ConfigurationEvaluator, rank_evaluations)
6. Structured baseline benchmark export to experiment_output/baseline/

Usage:
    python evacuation_simulation/run_baseline_experiment.py
    python evacuation_simulation/run_baseline_experiment.py --seed 42 --occupants 20
    python evacuation_simulation/run_baseline_experiment.py --seeds 42,20,32,43
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure the project root (parent of evacuation_simulation/) is importable
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))
_SIM_DIR = Path(__file__).parent.resolve()
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from evacuation_simulation.simulation.config import (
    FloorPlan,
    OccupantScenario,
    SimulationParameters,
)
from evacuation_simulation.simulation.evaluation import ConfigurationEvaluator
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
)
from evacuation_simulation.simulation.generators.commercial import (
    CommercialGeneratorConfig,
    CommercialLayoutGenerator,
)
from evacuation_simulation.simulation.occupancy_seeder import (
    OccupantSeeder,
    OccupantSeederConfig,
)
from evacuation_simulation.simulation.search import (
    ExitConfigurationSearcher,
    SearchResult,
)
from evacuation_simulation.simulation.validator import ExitValidator

# ---------------------------------------------------------------------------
# Default directories
# ---------------------------------------------------------------------------
_BASELINE_OUTPUT_DIR = Path(__file__).parent / "experiment_output" / "baseline"


# ---------------------------------------------------------------------------
# Single Baseline Experiment
# ---------------------------------------------------------------------------


def run_baseline_experiment(
    seed: int = 42,
    total_occupants: int = 20,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_candidate_exits: int = 4,
    exit_count: int = 2,
    max_timesteps: int = 500,
    max_configurations: Optional[int] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute one complete reproducible exhaustive exit-configuration baseline experiment.

    1. Generates a synthetic building using CommercialLayoutGenerator.
    2. Creates one fixed occupant scenario for that building using OccupantSeeder.
    3. Enumerates all candidate exit combinations using Stage 13 searcher.
    4. Rejects configurations violating NBC 2016 safety constraints (R1-R5).
    5. Evaluates each feasible configuration using the discrete BFS simulation engine.
    6. Ranks all configurations by primary objective (minimize total evacuation time).
    7. Saves full results to experiment_output/baseline/.

    Parameters
    ----------
    seed:
        Random seed for building geometry, occupant seeding, and simulation.
    total_occupants:
        Number of agents to place in the building.
    grid_rows:
        Grid vertical dimension in cells (default 20 = 10m).
    grid_cols:
        Grid horizontal dimension in cells (default 30 = 15m).
    num_candidate_exits:
        Number of candidate exterior door openings to place (default 4).
    exit_count:
        Number of active exits per evaluated configuration (default 2).
    max_timesteps:
        Safety ceiling on simulation steps per episode.
    max_configurations:
        Optional safeguard ceiling on combinations. None = fully exhaustive.
    output_dir:
        Output folder for JSON files. Defaults to experiment_output/baseline/.
    verbose:
        Whether to print progress and formatted results to stdout.

    Returns
    -------
    dict[str, Any]
        Serialisable dictionary of baseline results.
    """
    sep = "=" * 72
    out_dir = output_dir or _BASELINE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{sep}")
        print("  PATH_RL — Stage 17: Exhaustive Exit-Configuration Baseline")
        print(sep)
        print(f"  Configuration : seed={seed}, grid={grid_rows}x{grid_cols}, "
              f"occupants={total_occupants}, candidates={num_candidate_exits}, "
              f"k_exits={exit_count}")

    # -----------------------------------------------------------------------
    # Step 1: Generate synthetic commercial layout
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[1/5] Generating synthetic commercial layout ...")

    gen_config = CommercialGeneratorConfig(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        num_exterior_exits=num_candidate_exits,
    )
    generator = CommercialLayoutGenerator()
    layout = generator.generate(gen_config)
    floor_plan: FloorPlan = layout.floor_plan

    candidates: list[CandidateExit] = generate_candidate_exits(floor_plan)
    if verbose:
        print(f"      Grid size: {floor_plan.grid.rows}x{floor_plan.grid.cols} cells, "
              f"{len(floor_plan.walls)} walls, {len(layout.rooms)} rooms")
        print(f"      Candidate exits pool: {len(candidates)} doors -> "
              f"{[c.door_id or str(c.position) for c in candidates]}")

    # -----------------------------------------------------------------------
    # Step 2: Create ONE fixed occupant scenario for fair comparison
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[2/5] Seeding fixed occupant scenario ({total_occupants} occupants) ...")

    seeder_config = OccupantSeederConfig(
        total_occupants=total_occupants,
        seed=seed,
        strategy="random_walkable",
        max_occupants_per_cell=1,
    )
    occupants: OccupantScenario = OccupantSeeder.seed(
        floor_plan, layout.exit_configuration, seeder_config
    )

    if verbose:
        print(f"      Seeded {len(occupants.agents)} agents. "
              "Identical occupant scenario will be used across all exit configurations.")

    # -----------------------------------------------------------------------
    # Step 3 & 4: Exhaustive Search, Constraint Validation, and Simulation
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[3/5] Enumerating and evaluating exit configurations (k={exit_count}) ...")

    sim_params = SimulationParameters(
        max_timesteps=max_timesteps,
        random_seed=seed,
        default_cell_capacity=1,
    )
    validator = ExitValidator()
    evaluator = ConfigurationEvaluator(validator=validator)
    searcher = ExitConfigurationSearcher(evaluator=evaluator, validator=validator)

    search_result: SearchResult = searcher.search(
        floor_plan=floor_plan,
        candidates=candidates,
        occupant_scenario=occupants,
        exit_count=exit_count,
        parameters=sim_params,
        top_k=len(candidates) * 10,  # capture all in top_k ranking
        max_configurations=max_configurations,
        scenario_name=f"synthetic_commercial_seed_{seed}",
    )

    # -----------------------------------------------------------------------
    # Step 5: Analyse, Rank, and Summarise Results
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4/5] Ranking and analysing configurations ...")

    is_exhaustive = (
        max_configurations is None
        or search_result.total_configurations <= max_configurations
    )

    feasible_evals = [e for e in search_result.all_evaluations if e.is_feasible]
    best_eval = search_result.best_evaluation
    best_cfg = search_result.best_configuration

    best_score: Optional[int] = best_eval.score if best_eval else None
    worst_feasible_score: Optional[int] = (
        max((e.score for e in feasible_evals if e.score is not None), default=None)
        if feasible_evals
        else None
    )

    tied_evals = (
        [e for e in feasible_evals if e.score == best_score]
        if best_score is not None
        else []
    )
    tied_for_best_count = len(tied_evals)
    tied_cfg_ids = [e.configuration_id for e in tied_evals]

    # Build ranked summary of all evaluations
    all_configs_ranked: list[dict[str, Any]] = []
    ranked_evals = search_result.top_k_evaluations

    for rank, er in enumerate(ranked_evals, start=1):
        m = er.metrics
        sim_res = er.simulation_result
        cfg_item: dict[str, Any] = {
            "rank": rank,
            "configuration_id": er.configuration_id,
            "is_feasible": er.is_feasible,
            "is_compliant": er.validation_result.is_compliant,
            "score": er.score,
            "total_evacuation_time": m.total_evacuation_time if m else None,
            "mean_evacuation_time": (
                round(m.mean_evacuation_time, 2) if m and m.mean_evacuation_time else None
            ),
            "evacuation_rate": m.evacuation_rate if m else 0.0,
            "evacuated_count": m.evacuated_count if m else 0,
            "non_evacuated_count": m.non_evacuated_count if m else (len(occupants.agents) if not er.is_feasible else 0),
            "total_timesteps": sim_res.total_timesteps if sim_res else None,
            "violations": [
                {"rule_id": v.rule_id, "message": v.message}
                for v in er.validation_result.violations
            ],
        }
        all_configs_ranked.append(cfg_item)

    # -----------------------------------------------------------------------
    # Step 6: Assemble output dictionary
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_cfg_dict = None
    if best_cfg and best_eval and best_eval.metrics:
        m = best_eval.metrics
        best_cfg_dict = {
            "configuration_id": best_cfg.configuration_id,
            "exits": [list(pos) for pos in best_cfg.exits],
            "widths": best_cfg.widths,
            "total_evacuation_time": m.total_evacuation_time,
            "mean_evacuation_time": round(m.mean_evacuation_time, 2) if m.mean_evacuation_time else None,
            "evacuation_rate": m.evacuation_rate,
            "evacuated_count": m.evacuated_count,
            "mean_waiting_steps": round(m.mean_waiting_steps, 2),
            "congested_cell_steps": m.congested_cell_steps,
            "max_cell_occupancy": m.max_cell_occupancy,
            "exit_utilization": {str(k): v for k, v in m.exit_utilization.items()},
        }

    output_data: dict[str, Any] = {
        "experiment_type": "exhaustive_exit_configuration_baseline",
        "stage": 17,
        "timestamp": timestamp,
        "environment": {
            "seed": seed,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "cell_size_m": layout.metadata.get("cell_size_m", 0.5),
            "total_occupants": total_occupants,
            "num_candidate_exits": len(candidates),
            "exit_count": exit_count,
            "max_timesteps": max_timesteps,
            "candidate_exits": [
                {
                    "door_id": c.door_id,
                    "position": list(c.position),
                    "width": c.width,
                }
                for c in candidates
            ],
        },
        "search_statistics": {
            "is_exhaustive": is_exhaustive,
            "total_generated": search_result.total_configurations,
            "rejected_by_constraints": search_result.invalid_configurations,
            "evaluated": search_result.evaluated_configurations,
            "feasible": search_result.valid_configurations,
            "infeasible": search_result.invalid_configurations,
            "runtime_seconds": round(search_result.search_time_seconds, 4),
            "evaluations_per_second": round(search_result.evaluations_per_second, 2),
        },
        "baseline_summary": {
            "has_feasible_solution": search_result.has_feasible_solution,
            "best_configuration_id": best_cfg.configuration_id if best_cfg else None,
            "best_exit_positions": (
                [list(pos) for pos in best_cfg.exits] if best_cfg else None
            ),
            "best_score": best_score,
            "worst_feasible_score": worst_feasible_score,
            "tied_for_best_count": tied_for_best_count,
            "tied_configuration_ids": tied_cfg_ids,
            "best_configuration": best_cfg_dict,
        },
        "all_configurations_ranked": all_configs_ranked,
    }

    # Save to JSON
    json_path = out_dir / f"baseline_seed{seed}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\n[5/5] Saved baseline results to:\n      {json_path}")
        print(f"\n{sep}")
        print("  STAGE 17 BASELINE SUMMARY REPORT")
        print(sep)
        print(f"  Layout Seed              : {seed} ({grid_rows}x{grid_cols} grid)")
        print(f"  Fixed Occupant Count     : {total_occupants}")
        print(f"  Candidate Exits Pool     : {len(candidates)}")
        print(f"  Exits to Place (k)       : {exit_count}")
        print(f"  Exhaustive Search        : {'YES (full space evaluated)' if is_exhaustive else 'LIMITED/TRUNCATED'}")
        print(f"  Total Combinations Gen   : {search_result.total_configurations}")
        print(f"  Rejected by Constraints  : {search_result.invalid_configurations}")
        print(f"  Feasible Configurations  : {len(feasible_evals)}")
        print(f"  Simulations Evaluated    : {search_result.evaluated_configurations}")
        print(f"  Search Runtime           : {search_result.search_time_seconds:.4f}s "
              f"({search_result.evaluations_per_second:.1f} evals/sec)")
        print(sep)
        if best_cfg and best_eval and best_eval.metrics:
            m = best_eval.metrics
            print(f"  WINNING CONFIGURATION    : {best_cfg.configuration_id}")
            print(f"  Exit Positions           : {best_cfg.exits}")
            print(f"  Best Evacuation Time     : {best_score} timesteps (MINIMIZED)")
            print(f"  Worst Feasible Time      : {worst_feasible_score} timesteps")
            print(f"  Configurations Tied      : {tied_for_best_count} tied for {best_score} steps "
                  f"({', '.join(tied_cfg_ids)})")
            print(f"  Evacuation Rate          : {m.evacuation_rate:.1%} ({m.evacuated_count}/{m.total_agents})")
            print(f"  Mean Evacuation Time     : {m.mean_evacuation_time:.2f} steps")
            print(f"  Mean Waiting Steps       : {m.mean_waiting_steps:.2f} steps/agent")
            print(f"  Congested Cell-Steps     : {m.congested_cell_steps}")
            print(f"  Exit Utilisation         : {m.exit_utilization}")
        else:
            print("  NO FEASIBLE CONFIGURATION FOUND.")
            print(f"  All {search_result.total_configurations} configurations violated hard constraints.")
        print(sep)

    return output_data


# ---------------------------------------------------------------------------
# Multi-Seed Baseline Suite Runner
# ---------------------------------------------------------------------------


def run_baseline_suite(
    seeds: list[int],
    total_occupants: int = 20,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_candidate_exits: int = 4,
    exit_count: int = 2,
    max_timesteps: int = 500,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute exhaustive baseline experiments across multiple seeds.
    """
    out_dir = output_dir or _BASELINE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_results: list[dict[str, Any]] = []
    for s in seeds:
        res = run_baseline_experiment(
            seed=s,
            total_occupants=total_occupants,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            num_candidate_exits=num_candidate_exits,
            exit_count=exit_count,
            max_timesteps=max_timesteps,
            output_dir=out_dir,
            verbose=verbose,
        )
        suite_results.append(res)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "suite_type": "exhaustive_baseline_multiseed_suite",
        "stage": 17,
        "timestamp": timestamp,
        "seeds": seeds,
        "total_seeds": len(seeds),
        "total_occupants": total_occupants,
        "grid": [grid_rows, grid_cols],
        "exit_count": exit_count,
        "runs": suite_results,
    }

    suite_json_path = out_dir / f"baseline_suite_{timestamp}.json"
    with open(suite_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nSaved Multi-Seed Baseline Suite Summary to:\n  {suite_json_path}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PATH_RL Stage 17 — Exhaustive Exit-Configuration Baseline Experiment"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for single run (default: 42)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed baseline suite (e.g. 42,20,32,43)",
    )
    parser.add_argument("--occupants", type=int, default=20, help="Total occupants (default: 20)")
    parser.add_argument("--rows", type=int, default=20, help="Grid rows (default: 20)")
    parser.add_argument("--cols", type=int, default=30, help="Grid cols (default: 30)")
    parser.add_argument(
        "--candidates", type=int, default=4, help="Number of candidate exterior exits (default: 4)"
    )
    parser.add_argument(
        "--exit-count", type=int, default=2, help="Number of exits per configuration (default: 2)"
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=500, help="Max simulation timesteps (default: 500)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save JSON results"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress detailed console logging"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    verbose = not args.quiet

    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        run_baseline_suite(
            seeds=seed_list,
            total_occupants=args.occupants,
            grid_rows=args.rows,
            grid_cols=args.cols,
            num_candidate_exits=args.candidates,
            exit_count=args.exit_count,
            max_timesteps=args.max_timesteps,
            output_dir=out_dir,
            verbose=verbose,
        )
    else:
        run_baseline_experiment(
            seed=args.seed,
            total_occupants=args.occupants,
            grid_rows=args.rows,
            grid_cols=args.cols,
            num_candidate_exits=args.candidates,
            exit_count=args.exit_count,
            max_timesteps=args.max_timesteps,
            output_dir=out_dir,
            verbose=verbose,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
