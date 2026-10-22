"""
evaluate_rl.py
--------------
Stage 18: Evaluation of Trained RL Agent vs Baselines.

Compares:
1. Trained Masked Actor-Critic RL Policy (deterministic greedy evaluation).
2. Random Valid-Action Baseline (samples uniformly among unmasked legal exits).
3. Stage 17 Exhaustive Search Baseline (combinatorial ground-truth optimum).

Evaluates across:
- Held-out unseen synthetic seeds (e.g. 2, 5, 43, 49)
- Training seen synthetic seeds (e.g. 42, 20, 32, 24)

Outputs structured comparison JSON to experiment_output/rl/evaluation_results.json.

Usage:
    python evacuation_simulation/evaluate_rl.py
    python evacuation_simulation/evaluate_rl.py --model experiment_output/rl/model_stage18.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure import paths
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))
_SIM_DIR = Path(__file__).parent.resolve()
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from evacuation_simulation.simulation.config import SimulationConfig, SimulationParameters
from evacuation_simulation.simulation.generators.commercial import (
    CommercialGeneratorConfig,
    CommercialLayoutGenerator,
)
from evacuation_simulation.simulation.occupancy_seeder import (
    OccupantSeeder,
    OccupantSeederConfig,
)
from evacuation_simulation.simulation.rl_environment import ExitPlacementEnv
from evacuation_simulation.simulation.rl_policy import MaskedActorCritic
from evacuation_simulation.simulation.search import search_exit_configurations

_RL_OUTPUT_DIR = Path(__file__).parent / "experiment_output" / "rl"
DEFAULT_EVAL_SEEDS = [2, 5, 43, 49]
DEFAULT_TRAIN_SEEDS_SUBSET = [42, 20, 32, 24]


def create_scenario(
    seed: int,
    total_occupants: int = 20,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_candidate_exits: int = 4,
    max_timesteps: int = 500,
) -> SimulationConfig:
    """Generate a reproducible scenario with synthetic layout and fixed occupants."""
    gen_config = CommercialGeneratorConfig(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        num_exterior_exits=num_candidate_exits,
    )
    generator = CommercialLayoutGenerator()
    layout = generator.generate(gen_config)

    seeder_config = OccupantSeederConfig(
        total_occupants=total_occupants,
        seed=seed,
        strategy="random_walkable",
        max_occupants_per_cell=1,
    )
    occupants = OccupantSeeder.seed(
        layout.floor_plan, layout.exit_configuration, seeder_config
    )

    return SimulationConfig(
        scenario_name=f"eval_commercial_seed_{seed}",
        floor_plan=layout.floor_plan,
        exit_configuration=layout.exit_configuration,
        occupants=occupants,
        parameters=SimulationParameters(max_timesteps=max_timesteps, random_seed=seed),
    )


def evaluate_policy_on_scenario(
    model: MaskedActorCritic,
    scenario: SimulationConfig,
    exit_count: int = 2,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate the trained RL policy on a single scenario."""
    env = ExitPlacementEnv()
    obs = env.reset(scenario, exit_count=exit_count)

    start_time = time.perf_counter()
    terminated = False
    info: dict[str, Any] = {}

    while not terminated:
        action, _, _ = model.get_action(obs, deterministic=deterministic)
        obs, reward, terminated, info = env.step(action)

    decision_time = time.perf_counter() - start_time

    # Retrieve selected exit coordinates
    selected_exits = [
        env.candidates[idx].position for idx in env.selected_indices
    ]

    return {
        "method": "Trained_RL_Agent",
        "is_feasible": bool(info.get("is_feasible", False)),
        "is_compliant": bool(info.get("is_compliant", False)),
        "total_evacuation_time": info.get("total_evacuation_time"),
        "reward": reward,
        "selected_indices": list(env.selected_indices),
        "selected_exits": [list(pos) for pos in selected_exits],
        "decision_time_seconds": round(decision_time, 5),
        "violations": info.get("violations", []),
    }


def evaluate_random_baseline_on_scenario(
    scenario: SimulationConfig,
    exit_count: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a random legal action baseline on a single scenario."""
    rng = random.Random(seed)
    env = ExitPlacementEnv()
    obs = env.reset(scenario, exit_count=exit_count)

    start_time = time.perf_counter()
    terminated = False
    info: dict[str, Any] = {}

    while not terminated:
        valid_actions = [i for i, m in enumerate(obs.action_mask) if m]
        if not valid_actions:
            action = 0
        else:
            action = rng.choice(valid_actions)
        obs, reward, terminated, info = env.step(action)

    decision_time = time.perf_counter() - start_time
    selected_exits = [
        env.candidates[idx].position for idx in env.selected_indices
    ]

    return {
        "method": "Random_Valid_Baseline",
        "is_feasible": bool(info.get("is_feasible", False)),
        "is_compliant": bool(info.get("is_compliant", False)),
        "total_evacuation_time": info.get("total_evacuation_time"),
        "reward": reward,
        "selected_indices": list(env.selected_indices),
        "selected_exits": [list(pos) for pos in selected_exits],
        "decision_time_seconds": round(decision_time, 5),
        "violations": info.get("violations", []),
    }


def evaluate_exhaustive_baseline_on_scenario(
    scenario: SimulationConfig,
    exit_count: int = 2,
) -> dict[str, Any]:
    """Run Stage 17 exhaustive search to find the ground truth optimal configuration."""
    start_time = time.perf_counter()
    res = search_exit_configurations(
        floor_plan=scenario.floor_plan,
        occupant_scenario=scenario.occupants,
        exit_count=exit_count,
        parameters=scenario.parameters,
    )
    search_time = time.perf_counter() - start_time

    best_cfg = res.best_configuration
    best_eval = res.best_evaluation

    return {
        "method": "Stage17_Exhaustive_Search",
        "is_feasible": res.has_feasible_solution,
        "is_compliant": bool(best_eval.validation_result.is_compliant) if best_eval else False,
        "total_evacuation_time": (
            best_eval.metrics.total_evacuation_time
            if (best_eval and best_eval.metrics)
            else None
        ),
        "selected_exits": [list(pos) for pos in best_cfg.exits] if best_cfg else [],
        "decision_time_seconds": round(search_time, 5),
        "total_evaluated_combinations": res.evaluated_configurations,
        "total_combinations": res.total_configurations,
    }


def run_evaluation_suite(
    model_path: Optional[Path] = None,
    eval_seeds: Optional[list[int]] = None,
    train_seeds_subset: Optional[list[int]] = None,
    total_occupants: int = 20,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_candidate_exits: int = 4,
    exit_count: int = 2,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run comprehensive evaluation suite comparing RL, Random, and Exhaustive search.
    """
    out_dir = output_dir or _RL_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    m_path = model_path or (out_dir / "model_stage18.pt")

    if not m_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {m_path}. Train the model first.")

    model, metadata = MaskedActorCritic.load(m_path)

    held_out_seeds = eval_seeds or DEFAULT_EVAL_SEEDS
    training_subset = train_seeds_subset or DEFAULT_TRAIN_SEEDS_SUBSET

    sep = "=" * 76

    if verbose:
        print(f"\n{sep}")
        print("  PATH_RL — Stage 18: RL Policy Evaluation vs Baselines")
        print(sep)
        print(f"  Model Path        : {m_path}")
        print(f"  Model Metadata    : {metadata}")
        print(f"  Held-Out Seeds    : {held_out_seeds} (UNSEEN)")
        print(f"  Training Seeds    : {training_subset} (SEEN)")
        print(f"  Grid Size         : {grid_rows}x{grid_cols}, Occupants: {total_occupants}")

    def evaluate_group(group_name: str, seeds_list: list[int]) -> list[dict[str, Any]]:
        results = []
        for s in seeds_list:
            sc = create_scenario(
                seed=s,
                total_occupants=total_occupants,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                num_candidate_exits=num_candidate_exits,
            )

            rl_res = evaluate_policy_on_scenario(model, sc, exit_count=exit_count, deterministic=True)
            rand_res = evaluate_random_baseline_on_scenario(sc, exit_count=exit_count, seed=s)
            exh_res = evaluate_exhaustive_baseline_on_scenario(sc, exit_count=exit_count)

            # Compute optimality gap if exhaustive optimum exists
            opt_time = exh_res.get("total_evacuation_time")
            rl_time = rl_res.get("total_evacuation_time")
            gap = None
            if opt_time is not None and rl_time is not None:
                gap = round(((rl_time - opt_time) / opt_time) * 100.0, 1)

            entry = {
                "seed": s,
                "group": group_name,
                "rl_agent": rl_res,
                "random_baseline": rand_res,
                "exhaustive_baseline": exh_res,
                "rl_optimality_gap_percent": gap,
            }
            results.append(entry)
        return results

    held_out_results = evaluate_group("held_out_unseen", held_out_seeds)
    training_results = evaluate_group("training_seen", training_subset)

    # Aggregates helper
    def compute_aggregates(entries: list[dict[str, Any]]) -> dict[str, Any]:
        rl_feas = [e["rl_agent"]["is_feasible"] for e in entries]
        rand_feas = [e["random_baseline"]["is_feasible"] for e in entries]
        exh_feas = [e["exhaustive_baseline"]["is_feasible"] for e in entries]

        rl_times = [e["rl_agent"]["total_evacuation_time"] for e in entries if e["rl_agent"]["is_feasible"]]
        rand_times = [e["random_baseline"]["total_evacuation_time"] for e in entries if e["random_baseline"]["is_feasible"]]
        exh_times = [e["exhaustive_baseline"]["total_evacuation_time"] for e in entries if e["exhaustive_baseline"]["is_feasible"]]

        rl_latencies = [e["rl_agent"]["decision_time_seconds"] for e in entries]
        exh_latencies = [e["exhaustive_baseline"]["decision_time_seconds"] for e in entries]

        return {
            "total_scenarios": len(entries),
            "rl_agent": {
                "feasibility_rate": sum(rl_feas) / len(rl_feas) if rl_feas else 0.0,
                "mean_evacuation_time": round(sum(rl_times) / len(rl_times), 2) if rl_times else None,
                "mean_latency_ms": round(sum(rl_latencies) / len(rl_latencies) * 1000.0, 2),
            },
            "random_baseline": {
                "feasibility_rate": sum(rand_feas) / len(rand_feas) if rand_feas else 0.0,
                "mean_evacuation_time": round(sum(rand_times) / len(rand_times), 2) if rand_times else None,
            },
            "exhaustive_search": {
                "feasibility_rate": sum(exh_feas) / len(exh_feas) if exh_feas else 0.0,
                "mean_evacuation_time": round(sum(exh_times) / len(exh_times), 2) if exh_times else None,
                "mean_latency_ms": round(sum(exh_latencies) / len(exh_latencies) * 1000.0, 2),
            },
        }

    held_out_agg = compute_aggregates(held_out_results)
    train_agg = compute_aggregates(training_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "evaluation_type": "rl_vs_baselines_benchmark",
        "stage": 18,
        "timestamp": timestamp,
        "model_path": str(m_path),
        "held_out_summary": held_out_agg,
        "training_seen_summary": train_agg,
        "held_out_scenarios": held_out_results,
        "training_scenarios": training_results,
    }

    eval_json_path = out_dir / f"evaluation_results_{timestamp}.json"
    latest_eval_json = out_dir / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(latest_eval_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{sep}")
        print("  STAGE 18 BENCHMARK RESULTS")
        print(sep)
        print("  1. HELD-OUT SCENARIOS (Unseen Layouts & Seeds: " + str(held_out_seeds) + ")")
        print(f"     RL Agent Feasibility Rate      : {held_out_agg['rl_agent']['feasibility_rate']:.1%}")
        print(f"     Random Baseline Feasibility    : {held_out_agg['random_baseline']['feasibility_rate']:.1%}")
        print(f"     Exhaustive Feasibility         : {held_out_agg['exhaustive_search']['feasibility_rate']:.1%}")
        rl_t = held_out_agg['rl_agent']['mean_evacuation_time']
        exh_t = held_out_agg['exhaustive_search']['mean_evacuation_time']
        print(f"     RL Mean Evacuation Time        : {rl_t} steps" if rl_t else "     RL Mean Evacuation Time        : N/A")
        print(f"     Exhaustive Mean Evacuation Time: {exh_t} steps" if exh_t else "     Exhaustive Mean Evacuation Time: N/A")
        print(f"     RL Mean Decision Latency       : {held_out_agg['rl_agent']['mean_latency_ms']} ms")
        print(f"     Exhaustive Search Latency      : {held_out_agg['exhaustive_search']['mean_latency_ms']} ms")
        print("-" * 76)
        print("  2. TRAINING SCENARIOS (Seen Layouts & Seeds: " + str(training_subset) + ")")
        print(f"     RL Agent Feasibility Rate      : {train_agg['rl_agent']['feasibility_rate']:.1%}")
        print(f"     Random Baseline Feasibility    : {train_agg['random_baseline']['feasibility_rate']:.1%}")
        print(f"     Exhaustive Feasibility         : {train_agg['exhaustive_search']['feasibility_rate']:.1%}")
        print(f"     RL Mean Evacuation Time        : {train_agg['rl_agent']['mean_evacuation_time']} steps")
        print(f"     Exhaustive Mean Evacuation Time: {train_agg['exhaustive_search']['mean_evacuation_time']} steps")
        print(sep)
        print("\n  SCENARIO BREAKDOWN:")
        for res in held_out_results + training_results:
            s = res["seed"]
            grp = res["group"]
            rl_st = "FEASIBLE (" + str(res["rl_agent"]["total_evacuation_time"]) + "s)" if res["rl_agent"]["is_feasible"] else "NON-COMPLIANT"
            rnd_st = "FEASIBLE (" + str(res["random_baseline"]["total_evacuation_time"]) + "s)" if res["random_baseline"]["is_feasible"] else "NON-COMPLIANT"
            exh_st = "FEASIBLE (" + str(res["exhaustive_baseline"]["total_evacuation_time"]) + "s)" if res["exhaustive_baseline"]["is_feasible"] else "NON-COMPLIANT"
            gap_str = f"{res['rl_optimality_gap_percent']:+.1f}%" if res["rl_optimality_gap_percent"] is not None else "N/A"
            print(f"  Seed {s:2d} [{grp:<15}]: RL={rl_st:<20} | Rnd={rnd_st:<20} | Opt={exh_st:<20} | Gap={gap_str}")
        print(sep)
        print(f"  Saved benchmark to: {latest_eval_json}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PATH_RL Stage 18 — RL Evaluation vs Baselines"
    )
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .pt checkpoint")
    parser.add_argument("--eval-seeds", type=str, default=None, help="Comma-separated held-out evaluation seeds")
    parser.add_argument("--train-seeds", type=str, default=None, help="Comma-separated training seen seeds")
    parser.add_argument("--occupants", type=int, default=20, help="Occupants per building")
    parser.add_argument("--rows", type=int, default=20, help="Grid rows")
    parser.add_argument("--cols", type=int, default=30, help="Grid cols")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    m_path = Path(args.model) if args.model else None
    out_dir = Path(args.output_dir) if args.output_dir else None
    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()] if args.eval_seeds else None
    train_seeds = [int(s.strip()) for s in args.train_seeds.split(",") if s.strip()] if args.train_seeds else None

    run_evaluation_suite(
        model_path=m_path,
        eval_seeds=eval_seeds,
        train_seeds_subset=train_seeds,
        total_occupants=args.occupants,
        grid_rows=args.rows,
        grid_cols=args.cols,
        output_dir=out_dir,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
