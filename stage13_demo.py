"""
stage13_demo.py
---------------
Stage 13 Demo: Non-RL Exit Configuration Search Baseline.

Demonstrates the complete Stage 13 pipeline:

1. Load an existing scenario (scenarios/bottleneck.json).
2. Generate candidate exits from exterior door openings.
3. Run the Non-RL Exhaustive Exit Configuration Searcher (ExitConfigurationSearcher).
4. Display search statistics (total, valid, invalid, evaluated, search time, throughput).
5. Display Top 3 ranked configurations.
6. Display the Best Configuration and its primary objective (minimize total evacuation time).
7. Display key secondary metrics for the winning configuration.

PRIMARY OBJECTIVE: MINIMIZE TOTAL EVACUATION TIME
Lower score = better configuration.

SCIENTIFIC FAIR COMPARISON:
All configurations are evaluated against the EXACT SAME FloorPlan,
OccupantScenario, and SimulationParameters (including random seed).
Only the ExitConfiguration changes.

Usage:
    python stage13_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root (RL_ENV) is on sys.path
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig, SimulationParameters
from evacuation_simulation.simulation.generation import generate_candidate_exits
from evacuation_simulation.simulation.search import ExitConfigurationSearcher


def print_separator(char="=", width=64):
    print(char * width)


def print_header(title: str):
    print_separator("=")
    print(f"  {title}")
    print_separator("=")


def run_demo():
    print_header("STAGE 13 -- NON-RL SEARCH BASELINE")
    print("Primary objective : MINIMIZE TOTAL EVACUATION TIME")
    print("Strategy          : Exhaustive Combinatorial Search (Baseline)")
    print()

    # ------------------------------------------------------------------
    # 1. Load scenario
    # ------------------------------------------------------------------
    scenario_path = Path("scenarios/bottleneck.json")
    if not scenario_path.exists():
        scenario_path = Path(__file__).parent / "scenarios" / "bottleneck.json"
    sc = SimulationConfig.from_json(scenario_path)

    floor_plan = sc.floor_plan
    occupant_scenario = sc.occupants
    params = SimulationParameters(max_timesteps=200, random_seed=42)

    print(f"Scenario          : {sc.scenario_name or 'bottleneck'}")
    print(f"Floor plan        : {floor_plan.grid.rows}x{floor_plan.grid.cols} grid")
    print(f"Agents            : {len(occupant_scenario.agents)}")
    print(f"Random seed       : {params.random_seed}")
    print()

    # ------------------------------------------------------------------
    # 2. Candidate Exits
    # ------------------------------------------------------------------
    candidates = generate_candidate_exits(floor_plan)
    print(f"Candidate exits   : {len(candidates)}")
    for c in candidates:
        print(f"  - Door ID: {c.door_id or 'legacy':<12} Position: {str(c.position):<8} Width: {c.width}m")
    print()

    # ------------------------------------------------------------------
    # 3. Run Non-RL Search for 2-exit configurations
    # ------------------------------------------------------------------
    exit_count = 2
    print_separator("-")
    print(f"Searching all 2-exit combinations (choose({len(candidates)}, {exit_count}))...")
    print_separator("-")

    searcher = ExitConfigurationSearcher()
    result = searcher.search(
        floor_plan=floor_plan,
        candidates=candidates,
        occupant_scenario=occupant_scenario,
        exit_count=exit_count,
        parameters=params,
        top_k=3,
        scenario_name="bottleneck",
    )

    # ------------------------------------------------------------------
    # 4. Display Search Statistics
    # ------------------------------------------------------------------
    print(f"\nSearch Space & Validation Statistics:")
    print(f"  Total generated configurations : {result.total_configurations}")
    print(f"  Valid configurations (passed)  : {result.valid_configurations}")
    print(f"  Invalid configurations (R1-R5) : {result.invalid_configurations}")
    print(f"  Evaluated (simulations run)    : {result.evaluated_configurations}")
    print(f"  Search runtime                 : {result.search_time_seconds:.4f} seconds")
    print(f"  Evaluation throughput          : {result.evaluations_per_second:.1f} evaluations/sec")
    print()

    # ------------------------------------------------------------------
    # 5. Display Top-K Rankings
    # ------------------------------------------------------------------
    print_separator("-")
    print(f"TOP {len(result.top_k_evaluations)} RANKED CONFIGURATIONS")
    print_separator("-")

    for rank, er in enumerate(result.top_k_evaluations, start=1):
        status = "FEASIBLE" if er.is_feasible else "INFEASIBLE"
        score_str = f"{er.score} steps" if er.score is not None else "N/A"
        mean_str = f"{er.metrics.mean_evacuation_time:.2f}" if er.metrics and er.metrics.mean_evacuation_time else "N/A"
        rate_str = f"{er.metrics.evacuation_rate:.0%}" if er.metrics else "N/A"
        
        print(f"Rank {rank}: [{er.configuration_id}] Status: {status}")
        print(f"  Total evacuation time (Score) : {score_str}")
        print(f"  Mean evacuation time          : {mean_str} steps")
        print(f"  Evacuation rate               : {rate_str}")
        if not er.validation_result.is_compliant:
            for v in er.validation_result.violations:
                print(f"  Violation: [{v.rule_id}] {v.message}")
        print()

    # ------------------------------------------------------------------
    # 6. Display Best Configuration
    # ------------------------------------------------------------------
    print_separator("=")
    if result.best_configuration and result.best_evaluation:
        best_cfg = result.best_configuration
        best_eval = result.best_evaluation
        m = best_eval.metrics

        print("BEST CONFIGURATION SUMMARY")
        print_separator("=")
        print(f"  Configuration ID         : {best_cfg.configuration_id}")
        print(f"  Exit positions           : {best_cfg.exits}")
        print(f"  Opening widths           : {best_cfg.widths}")
        print(f"  Regulatory Compliance    : {'COMPLIANT' if best_eval.validation_result.is_compliant else 'NON-COMPLIANT'}")
        print()
        print("PRIMARY OBJECTIVE:")
        print(f"  Total Evacuation Time    : {best_eval.score} timesteps (MINIMIZED)")
        print()
        print("SECONDARY METRICS:")
        if m:
            print(f"  Evacuation Rate          : {m.evacuation_rate:.0%} ({m.evacuated_count}/{m.total_agents} agents)")
            print(f"  Mean Evacuation Time     : {m.mean_evacuation_time:.2f} timesteps")
            print(f"  Min / Max Evac Time      : {m.min_evacuation_time} / {m.max_evacuation_time} timesteps")
            print(f"  Mean Waiting Steps       : {m.mean_waiting_steps:.2f} steps/agent")
            print(f"  Total Waiting Steps      : {m.total_waiting_steps} steps across all agents")
            print(f"  Congested Cell Steps     : {m.congested_cell_steps}")
            print(f"  Max Cell Occupancy       : {m.max_cell_occupancy} agents (ratio: {m.max_occupancy_ratio:.1f}x)")
            print(f"  Exit Utilization         : {m.exit_utilization}")
    else:
        print("NO FEASIBLE CONFIGURATION FOUND.")
    print_separator("=")
    print("Baseline execution complete. Benchmark established for future RL comparison.")


if __name__ == "__main__":
    run_demo()
