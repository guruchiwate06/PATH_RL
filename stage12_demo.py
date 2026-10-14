"""
stage12_demo.py
---------------
Stage 12 Demo: Exit Configuration Evaluation and Ranking.

Demonstrates the complete Stage 12 pipeline:

1. Load an existing scenario (bottleneck).
2. Generate candidate exit configurations.
3. Validate them through Stage 11 (ExitValidator).
4. Evaluate valid configurations (evacuation simulation).
5. Rank them by primary objective (minimize total evacuation time).
6. Print a concise comparison table.

PRIMARY OBJECTIVE: MINIMIZE TOTAL EVACUATION TIME
Lower score = better configuration.

SCIENTIFIC NOTE:
All configurations are evaluated against the SAME FloorPlan,
OccupantScenario, and simulation parameters (including random_seed).
Only the ExitConfiguration changes, ensuring a fair comparison.

Usage:
    python stage12_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the package root (RL_ENV) is on the path, matching other demo scripts
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig, SimulationParameters
from evacuation_simulation.simulation.evaluation import (
    ConfigurationEvaluator,
    rank_evaluations,
)
from evacuation_simulation.simulation.generation import (
    generate_candidate_exits,
    generate_exit_configurations,
)


def print_separator(char="=", width=62):
    print(char * width)


def print_header(title: str):
    print_separator()
    print(f"  {title}")
    print_separator()


def run_demo():
    print_header("STAGE 12 EVALUATION DEMO")
    print("Primary objective : MINIMIZE TOTAL EVACUATION TIME")
    print("Scenario          : bottleneck")
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

    print(f"Floor plan        : {floor_plan.grid.rows}x{floor_plan.grid.cols} grid")
    print(f"Agents            : {len(occupant_scenario.agents)}")
    print(f"Exterior doors    : {sum(1 for d in floor_plan.doors if d.exterior)}")
    print(f"Random seed       : {params.random_seed}")
    print()

    # ------------------------------------------------------------------
    # 2. Generate candidate exits
    # ------------------------------------------------------------------
    candidates = generate_candidate_exits(floor_plan)
    print(f"Candidate exits   : {len(candidates)}")
    for c in candidates:
        print(f"  {c.door_id or 'legacy'} @ {c.position}  width={c.width}m")
    print()

    # ------------------------------------------------------------------
    # 3. Generate exit configurations (1-exit and 2-exit combos)
    # ------------------------------------------------------------------
    configurations = []

    if len(candidates) >= 1:
        for cfg in generate_exit_configurations(candidates, exit_count=1):
            configurations.append(cfg)
    if len(candidates) >= 2:
        for cfg in generate_exit_configurations(candidates, exit_count=2):
            configurations.append(cfg)

    print(f"Configurations    : {len(configurations)} total")
    print()

    # ------------------------------------------------------------------
    # 4. Evaluate all configurations
    # ------------------------------------------------------------------
    evaluator = ConfigurationEvaluator()
    print("Evaluating configurations...")
    results = evaluator.evaluate_configurations(
        floor_plan, configurations, occupant_scenario, parameters=params
    )
    print(f"Evaluations done  : {len(results)}")
    print()

    # ------------------------------------------------------------------
    # 5. Rank
    # ------------------------------------------------------------------
    ranked = rank_evaluations(results)

    # ------------------------------------------------------------------
    # 6. Print detailed results for each configuration
    # ------------------------------------------------------------------
    for i, r in enumerate(results):
        print_separator("-", 62)
        cfg_exits = configurations[i].exits
        print(f"Configuration     : {r.configuration_id}")
        print(f"Exits selected    : {[list(e) for e in cfg_exits]}")
        if not r.validation_result.is_compliant:
            print("Compliance        : FAIL")
            for v in r.validation_result.violations:
                print(f"  Violation: [{v.rule_id}] {v.message}")
            print("Score             : (not evaluated — invalid)")
        else:
            print("Compliance        : PASS")
            if r.evacuation_complete:
                m = r.metrics
                print(f"Total evac time   : {m.total_evacuation_time} timesteps")
                print(f"Mean  evac time   : {m.mean_evacuation_time:.2f}" if m.mean_evacuation_time else "Mean  evac time   : N/A")
                print(f"Max   evac time   : {m.max_evacuation_time}")
                print(f"Evacuation rate   : {m.evacuation_rate:.0%}")
                print(f"Mean waiting      : {m.mean_waiting_steps:.2f} steps/agent")
                print(f"Congested steps   : {m.congested_cell_steps}")
                print(f"Score             : {r.score}  (primary objective)")
                if m.exit_utilization:
                    util_str = ", ".join(
                        f"{cell}: {cnt}" for cell, cnt in sorted(m.exit_utilization.items())
                    )
                    print(f"Exit utilization  : {util_str}")
            else:
                print(f"Evacuation rate   : {r.metrics.evacuation_rate:.0%} (INCOMPLETE)")
                print(f"Non-evacuated     : {r.metrics.non_evacuated_count}")
                print("Score             : (incomplete evacuation)")
    print()

    # ------------------------------------------------------------------
    # 7. Summary ranking table
    # ------------------------------------------------------------------
    print_header("RANKING SUMMARY")
    print(
        f"{'Rank':<5} {'Config':<30} {'Valid':<7} "
        f"{'TotalTime':<11} {'MeanTime':<10} {'Waiting':<9} "
        f"{'Congestion':<12} {'Score'}"
    )
    print("-" * 90)

    cfg_by_id = {cfg.configuration_id: cfg for cfg in configurations}

    for rank_num, r in enumerate(ranked, 1):
        valid_str = "YES" if r.validation_result.is_compliant else "NO"
        if r.metrics and r.evacuation_complete:
            m = r.metrics
            time_str = str(m.total_evacuation_time)
            mean_str = f"{m.mean_evacuation_time:.1f}" if m.mean_evacuation_time else "-"
            wait_str = f"{m.mean_waiting_steps:.2f}"
            cong_str = str(m.congested_cell_steps)
            score_str = str(r.score)
        elif r.metrics and not r.evacuation_complete:
            time_str = "INCOMPLETE"
            mean_str = "-"
            wait_str = "-"
            cong_str = "-"
            score_str = "-"
        else:
            time_str = "-"
            mean_str = "-"
            wait_str = "-"
            cong_str = "-"
            score_str = "-"

        print(
            f"{rank_num:<5} {r.configuration_id:<30} {valid_str:<7} "
            f"{time_str:<11} {mean_str:<10} {wait_str:<9} "
            f"{cong_str:<12} {score_str}"
        )

    print()
    print_separator()
    print("Primary objective : MINIMIZE TOTAL EVACUATION TIME")
    if ranked and ranked[0].is_feasible:
        print(f"Best configuration: {ranked[0].configuration_id}")
        print(f"Best score        : {ranked[0].score} timesteps")
    else:
        print("No feasible configuration found.")

    n_valid = sum(1 for r in results if r.validation_result.is_compliant)
    n_feasible = sum(1 for r in results if r.is_feasible)
    n_invalid = len(results) - n_valid
    print()
    print(f"Total evaluated   : {len(results)}")
    print(f"Passed regulation : {n_valid}")
    print(f"Fully feasible    : {n_feasible}  (compliant + complete evacuation)")
    print(f"Rejected          : {n_invalid}  (regulation failure)")
    print_separator()


if __name__ == "__main__":
    run_demo()
