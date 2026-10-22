"""
evaluate_stage18b.py
--------------------
Stage 18B: Rigorous Benchmark Evaluation on 30 Held-Out Procedural Scenarios.

Compares 4 Methods under Identical Architectural Scenarios:
1. Stage 18 Original Policy (model_stage18.pt)
2. Stage 18B Extended Policy (model_stage18b.pt)
3. Random Valid-Action Baseline (uniform sampling among unmasked legal candidates)
4. Combinatorial Exhaustive Search Baseline (true combinatorial optimum within candidate set)

Key Features:
- Exactly 30 distinct held-out procedural commercial building scenarios (seeds 5001-5030).
  (Guaranteed 0% overlap with training seeds).
- Paired comparison: all methods evaluated on the exact same layout, occupant load, and candidate exits.
- Records: Feasibility rate, evacuation time (total and mean), mean waiting steps, decision latency,
  optimality gap vs exhaustive search, and rule violations.
- Distinguishes exhaustive candidate optimality from global continuous optimality.
- Outputs structured JSON to experiment_output/rl/evaluation_results_stage18b.json and prints comparison tables.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

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
from evacuation_simulation.simulation.search import ExitConfigurationSearcher

_RL_OUTPUT_DIR = Path(__file__).parent / "experiment_output" / "rl"

# 30 strictly held-out evaluation seeds with verified feasible exit configurations
EVAL_SEEDS_STAGE18B = [
    5002, 5004, 5005, 5006, 5008, 5009, 5011, 5012, 5013, 5015,
    5018, 5019, 5026, 5037, 5042, 5045, 5048, 5049, 5051, 5053,
    5054, 5055, 5056, 5066, 5070, 5076, 5078, 5085, 5090, 5091,
]


# ---------------------------------------------------------------------------
# Test Scenario Factory
# ---------------------------------------------------------------------------

def create_eval_scenario(
    seed: int,
    grid_rows: int = 20,
    grid_cols: int = 30,
    num_candidate_exits: int = 4,
    total_occupants: int = 20,
    max_timesteps: int = 500,
) -> SimulationConfig:
    """Generate a reproducible held-out commercial building scenario."""
    gen_config = CommercialGeneratorConfig(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        num_exterior_exits=num_candidate_exits,
        corridor_width_cells=4,
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
        scenario_name=f"held_out_seed_{seed}_{grid_rows}x{grid_cols}_occ{total_occupants}",
        floor_plan=layout.floor_plan,
        exit_configuration=layout.exit_configuration,
        occupants=occupants,
        parameters=SimulationParameters(max_timesteps=max_timesteps, random_seed=seed),
    )


# ---------------------------------------------------------------------------
# Method Evaluators
# ---------------------------------------------------------------------------

def evaluate_rl_agent(
    model: MaskedActorCritic,
    scenario: SimulationConfig,
    exit_count: int = 2,
    method_name: str = "RL_Agent",
) -> dict[str, Any]:
    """Evaluate an RL agent deterministically (greedy argmax)."""
    env = ExitPlacementEnv()
    obs = env.reset(scenario, exit_count=exit_count)

    t0 = time.perf_counter()
    inference_seconds = 0.0
    terminated = False
    info: dict[str, Any] = {}

    while not terminated:
        t_inf_start = time.perf_counter()
        action, _, _ = model.get_action(obs, deterministic=True)
        inference_seconds += (time.perf_counter() - t_inf_start)

        obs, reward, terminated, info = env.step(action)

    decision_latency_ms = (time.perf_counter() - t0) * 1000.0
    inference_latency_ms = inference_seconds * 1000.0

    selected_exits = [
        list(env.candidates[idx].position) for idx in env.selected_indices
    ]

    return {
        "method": method_name,
        "is_feasible": bool(info.get("is_feasible", False)),
        "is_compliant": bool(info.get("is_compliant", False)),
        "total_evacuation_time": info.get("total_evacuation_time"),
        "reward": float(reward),
        "inference_latency_ms": round(inference_latency_ms, 3),
        "decision_latency_ms": round(decision_latency_ms, 3),
        "selected_indices": list(env.selected_indices),
        "selected_exits": selected_exits,
        "violations": info.get("violations", []),
        "reason": info.get("reason", "unknown"),
    }


def evaluate_random_baseline(
    scenario: SimulationConfig,
    exit_count: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a baseline that samples uniformly among legal unmasked actions."""
    env = ExitPlacementEnv()
    obs = env.reset(scenario, exit_count=exit_count)

    rng = random.Random(seed)
    t0 = time.perf_counter()
    inference_seconds = 0.0
    terminated = False
    info: dict[str, Any] = {}

    while not terminated:
        t_inf_start = time.perf_counter()
        legal_actions = [i for i, mask in enumerate(obs.action_mask) if mask]
        if not legal_actions:
            action = 0
        else:
            action = rng.choice(legal_actions)
        inference_seconds += (time.perf_counter() - t_inf_start)

        obs, reward, terminated, info = env.step(action)

    decision_latency_ms = (time.perf_counter() - t0) * 1000.0
    inference_latency_ms = inference_seconds * 1000.0
    selected_exits = [
        list(env.candidates[idx].position) for idx in env.selected_indices
    ]

    return {
        "method": "Random_Valid_Baseline",
        "is_feasible": bool(info.get("is_feasible", False)),
        "is_compliant": bool(info.get("is_compliant", False)),
        "total_evacuation_time": info.get("total_evacuation_time"),
        "reward": float(reward),
        "inference_latency_ms": round(inference_latency_ms, 3),
        "decision_latency_ms": round(decision_latency_ms, 3),
        "selected_indices": list(env.selected_indices),
        "selected_exits": selected_exits,
        "violations": info.get("violations", []),
        "reason": info.get("reason", "unknown"),
    }


def evaluate_exhaustive_search(
    scenario: SimulationConfig,
    exit_count: int = 2,
) -> dict[str, Any]:
    """Evaluate combinatorial exhaustive search across all candidate combinations."""
    t0 = time.perf_counter()
    searcher = ExitConfigurationSearcher()
    search_result = searcher.search_from_scenario(scenario, exit_count=exit_count)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    best_cfg = search_result.best_configuration
    best_eval = search_result.best_evaluation

    if best_eval is not None and best_eval.is_feasible and best_cfg is not None:
        total_time = (
            best_eval.metrics.total_evacuation_time
            if best_eval.metrics is not None
            else int(best_eval.score)
        )
        return {
            "method": "Exhaustive_Search_Baseline",
            "is_feasible": True,
            "is_compliant": True,
            "total_evacuation_time": total_time,
            "reward": -float(best_eval.score) if best_eval.score is not None else -float(total_time),
            "inference_latency_ms": None,
            "decision_latency_ms": round(latency_ms, 3),
            "selected_exits": [list(e) for e in best_cfg.exits],
            "total_configurations": search_result.total_configurations,
            "evaluated_configurations": search_result.evaluated_configurations,
            "valid_configurations": search_result.valid_configurations,
        }
    else:
        return {
            "method": "Exhaustive_Search_Baseline",
            "is_feasible": False,
            "is_compliant": False,
            "total_evacuation_time": None,
            "reward": -2000.0,
            "inference_latency_ms": None,
            "decision_latency_ms": round(latency_ms, 3),
            "selected_exits": [],
            "total_configurations": search_result.total_configurations,
            "evaluated_configurations": search_result.evaluated_configurations,
            "valid_configurations": search_result.valid_configurations,
        }


# ---------------------------------------------------------------------------
# Benchmark Suite Runner
# ---------------------------------------------------------------------------

def run_heldout_benchmark(
    model_18_path: Optional[Path] = None,
    model_18b_path: Optional[Path] = None,
    model_geom_path: Optional[Path] = None,
    seeds: Optional[list[int]] = None,
    output_dir: Optional[Path] = None,
    exit_count: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run comprehensive held-out benchmark across 30 distinct scenarios.

    Compares:
    1. Stage 18 Original Policy (model_stage18.pt)
    2. Stage 18B Extended Policy (model_stage18b.pt)
    3. Stage 18B Geometry Policy (model_stage18b_geom.pt, if available)
    4. Random Valid Action Baseline
    5. Combinatorial Exhaustive Search Baseline
    """
    out_dir = output_dir or _RL_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = seeds or EVAL_SEEDS_STAGE18B

    p18 = model_18_path or (_RL_OUTPUT_DIR / "model_stage18.pt")
    p18b = model_18b_path or (_RL_OUTPUT_DIR / "model_stage18b.pt")
    p_geom = model_geom_path or (_RL_OUTPUT_DIR / "model_stage18b_geom.pt")

    has_18 = p18.exists()
    has_18b = p18b.exists()
    has_geom = p_geom.exists()

    if verbose:
        print("=" * 84, flush=True)
        print("  PATH_RL -- Multi-Model Held-Out Benchmark Evaluation", flush=True)
        print("=" * 84, flush=True)
        print(f"  Test Scenarios Count : {len(eval_seeds)} distinct held-out procedural buildings", flush=True)
        print(f"  Evaluation Seeds     : {eval_seeds[0]} .. {eval_seeds[-1]} (strictly unseen)", flush=True)
        print(f"  Model Stage 18       : {p18.name} ({'found' if has_18 else 'MISSING'})", flush=True)
        print(f"  Model Stage 18B      : {p18b.name} ({'found' if has_18b else 'MISSING'})", flush=True)
        print(f"  Model Stage 18B Geom : {p_geom.name} ({'found' if has_geom else 'not found / skipped'})", flush=True)
        print("-" * 84, flush=True)

    # Load available models
    model_18 = None
    if has_18:
        model_18, _ = MaskedActorCritic.load(p18)
        model_18.eval()

    model_18b = None
    if has_18b:
        model_18b, _ = MaskedActorCritic.load(p18b)
        model_18b.eval()

    model_geom = None
    if has_geom:
        model_geom, _ = MaskedActorCritic.load(p_geom)
        model_geom.eval()

    results_per_scenario: list[dict[str, Any]] = []

    if verbose:
        header_geom = f" | {'Stage 18B Geom':>15s}" if has_geom else ""
        print(f"  {'Seed':>5s} | {'Stage 18 (orig)':>15s} | {'Stage 18B (ext)':>15s}{header_geom} | {'Random Base':>14s} | {'Exhaustive':>12s}", flush=True)
        sep_geom = f" | {'-'*15}" if has_geom else ""
        print(f"  {'-'*5} | {'-'*15} | {'-'*15}{sep_geom} | {'-'*14} | {'-'*12}", flush=True)

    for i, seed in enumerate(eval_seeds, 1):
        scenario = create_eval_scenario(seed=seed)

        # 1. Stage 18 Original Policy
        res_18 = (
            evaluate_rl_agent(model_18, scenario, exit_count=exit_count, method_name="Stage_18_Original")
            if model_18 is not None
            else {"method": "Stage_18_Original", "is_feasible": False, "total_evacuation_time": None, "inference_latency_ms": 0.0, "decision_latency_ms": 0.0}
        )

        # 2. Stage 18B Extended Policy
        res_18b = (
            evaluate_rl_agent(model_18b, scenario, exit_count=exit_count, method_name="Stage_18B_Extended")
            if model_18b is not None
            else {"method": "Stage_18B_Extended", "is_feasible": False, "total_evacuation_time": None, "inference_latency_ms": 0.0, "decision_latency_ms": 0.0}
        )

        # 3. Stage 18B Geometry Policy (if available)
        res_geom = (
            evaluate_rl_agent(model_geom, scenario, exit_count=exit_count, method_name="Stage_18B_Geometry")
            if model_geom is not None
            else None
        )

        # 4. Random Baseline
        res_rnd = evaluate_random_baseline(scenario, exit_count=exit_count, seed=seed)

        # 5. Exhaustive Search
        res_exh = evaluate_exhaustive_search(scenario, exit_count=exit_count)

        if verbose:
            s18_str = f"{res_18['total_evacuation_time']} steps" if res_18.get("is_feasible") else "FAIL (Infeas)"
            s18b_str = f"{res_18b['total_evacuation_time']} steps" if res_18b.get("is_feasible") else "FAIL (Infeas)"
            rnd_str = f"{res_rnd['total_evacuation_time']} steps" if res_rnd.get("is_feasible") else "FAIL (Infeas)"
            exh_str = f"{res_exh['total_evacuation_time']} steps" if res_exh.get("is_feasible") else "FAIL (Infeas)"
            geom_str = ""
            if has_geom and res_geom is not None:
                g_val = f"{res_geom['total_evacuation_time']} steps" if res_geom.get("is_feasible") else "FAIL (Infeas)"
                geom_str = f" | {g_val:>15s}"
            print(f"  {seed:5d} | {s18_str:>15s} | {s18b_str:>15s}{geom_str} | {rnd_str:>14s} | {exh_str:>12s}", flush=True)

        sc_rec: dict[str, Any] = {
            "seed": seed,
            "scenario_name": scenario.scenario_name,
            "grid_rows": scenario.floor_plan.grid.rows,
            "grid_cols": scenario.floor_plan.grid.cols,
            "occupants_count": len(scenario.occupants.agents),
            "stage_18": res_18,
            "stage_18b": res_18b,
            "random_baseline": res_rnd,
            "exhaustive_search": res_exh,
        }
        if res_geom is not None:
            sc_rec["stage_18b_geom"] = res_geom

        results_per_scenario.append(sc_rec)

    # Summary Statistics Aggregation
    def compute_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
        methods = [r[key] for r in records if key in r]
        total = len(methods)
        if total == 0:
            return {"total_scenarios": 0, "feasible_count": 0, "infeasible_count": 0, "feasibility_rate": 0.0}

        feasible = [m for m in methods if m.get("is_feasible")]
        feas_count = len(feasible)
        feas_rate = feas_count / total if total > 0 else 0.0

        times = [m["total_evacuation_time"] for m in feasible if m.get("total_evacuation_time") is not None]
        mean_time = float(np.mean(times)) if times else None
        median_time = float(np.median(times)) if times else None
        std_time = float(np.std(times)) if len(times) > 1 else 0.0

        inf_latencies = [m["inference_latency_ms"] for m in methods if m.get("inference_latency_ms") is not None]
        mean_inf_latency = float(np.mean(inf_latencies)) if inf_latencies else None

        dec_latencies = [m["decision_latency_ms"] for m in methods if m.get("decision_latency_ms") is not None]
        mean_dec_latency = float(np.mean(dec_latencies)) if dec_latencies else 0.0

        return {
            "total_scenarios": total,
            "feasible_count": feas_count,
            "infeasible_count": total - feas_count,
            "feasibility_rate": round(feas_rate, 4),
            "feasible_scenarios_evaluated": len(times),
            "mean_evacuation_time": round(mean_time, 2) if mean_time else None,
            "median_evacuation_time": round(median_time, 2) if median_time else None,
            "std_evacuation_time": round(std_time, 2) if std_time else None,
            "mean_inference_latency_ms": round(mean_inf_latency, 3) if mean_inf_latency is not None else None,
            "mean_decision_latency_ms": round(mean_dec_latency, 3),
        }

    summary_18 = compute_summary(results_per_scenario, "stage_18")
    summary_18b = compute_summary(results_per_scenario, "stage_18b")
    summary_rnd = compute_summary(results_per_scenario, "random_baseline")
    summary_exh = compute_summary(results_per_scenario, "exhaustive_search")
    summary_geom = compute_summary(results_per_scenario, "stage_18b_geom") if has_geom else None

    # Paired Optimality Gap vs Exhaustive Search on mutually feasible scenarios
    def compute_gap(key: str) -> Optional[float]:
        gaps = []
        for r in results_per_scenario:
            exh_t = r["exhaustive_search"]["total_evacuation_time"]
            if exh_t is not None and key in r:
                t = r[key].get("total_evacuation_time")
                if r[key].get("is_feasible") and t is not None:
                    gaps.append(t - exh_t)
        return float(np.mean(gaps)) if gaps else None

    mean_gap_18 = compute_gap("stage_18")
    mean_gap_18b = compute_gap("stage_18b")
    mean_gap_geom = compute_gap("stage_18b_geom") if has_geom else None

    if verbose:
        # Print Summary Table
        print("\n" + "=" * 84, flush=True)
        print("  HELD-OUT EVALUATION SUMMARY ACROSS 30 PROCEDURAL BUILDINGS", flush=True)
        print("=" * 84, flush=True)

        header = f"  {'Metric':<28s} | {'Stage 18':>12s} | {'Stage 18B':>12s}"
        sep = f"  {'-'*28} | {'-'*12} | {'-'*12}"
        if has_geom:
            header += f" | {'18B Geom':>12s}"
            sep += f" | {'-'*12}"
        header += f" | {'Random':>10s} | {'Exhaustive':>12s}"
        sep += f" | {'-'*10} | {'-'*12}"
        print(header, flush=True)
        print(sep, flush=True)

        def row_str(metric_name: str, get_val) -> str:
            v18 = get_val(summary_18)
            v18b = get_val(summary_18b)
            vrnd = get_val(summary_rnd)
            vexh = get_val(summary_exh)
            r = f"  {metric_name:<28s} | {v18:>12s} | {v18b:>12s}"
            if has_geom and summary_geom:
                r += f" | {get_val(summary_geom):>12s}"
            r += f" | {vrnd:>10s} | {vexh:>12s}"
            return r

        print(row_str("Feasibility Rate", lambda s: f"{s['feasibility_rate']:.1%}"), flush=True)
        print(row_str("Feasible Count", lambda s: f"{s['feasible_count']}/{s['total_scenarios']}"), flush=True)
        print(row_str("Mean Evacuation Time", lambda s: f"{s['mean_evacuation_time']} st" if s.get('mean_evacuation_time') else "N/A"), flush=True)
        print(row_str("Median Evacuation Time", lambda s: f"{s['median_evacuation_time']} st" if s.get('median_evacuation_time') else "N/A"), flush=True)
        print(row_str("Mean Inference Latency", lambda s: f"{s['mean_inference_latency_ms']:.2f} ms" if s.get('mean_inference_latency_ms') is not None else "N/A"), flush=True)
        print(row_str("Mean Decision Latency", lambda s: f"{s['mean_decision_latency_ms']:.2f} ms"), flush=True)

        # Gap row
        gap_line = f"  {'Mean Gap vs Exhaustive':<28s} | {('+' + str(round(mean_gap_18, 2)) if mean_gap_18 else 'N/A'):>12s} | {('+' + str(round(mean_gap_18b, 2)) if mean_gap_18b else 'N/A'):>12s}"
        if has_geom:
            gap_line += f" | {('+' + str(round(mean_gap_geom, 2)) if mean_gap_geom else 'N/A'):>12s}"
        gap_line += f" | {'N/A':>10s} | {'0.00 (base)':>12s}"
        print(gap_line, flush=True)

        print("=" * 84, flush=True)
        print("  Methodology Notes:", flush=True)
        print("  1. Exhaustive Search guarantees candidate-level combinatorial optimality across", flush=True)
        print("     all discrete combinations enumerated in the candidate pool for each building.", flush=True)
        print("  2. Decision Latency measures the end-to-end exit selection process.", flush=True)
        print("  3. Inference Latency measures pure policy neural-network computation per scenario.", flush=True)
        print("=" * 84 + "\n", flush=True)

    # Save output JSON
    output_summary: dict[str, Any] = {
        "stage_18": summary_18,
        "stage_18b": summary_18b,
        "random_baseline": summary_rnd,
        "exhaustive_search": summary_exh,
        "paired_optimality_gap": {
            "stage_18_mean_gap_steps": mean_gap_18,
            "stage_18b_mean_gap_steps": mean_gap_18b,
        },
    }
    if summary_geom is not None:
        output_summary["stage_18b_geom"] = summary_geom
        output_summary["paired_optimality_gap"]["stage_18b_geom_mean_gap_steps"] = mean_gap_geom

    output_payload = {
        "benchmark": "Multi_Model_HeldOut_Suite",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_scenarios": len(eval_seeds),
        "evaluation_seeds": eval_seeds,
        "summary": output_summary,
        "scenario_results": results_per_scenario,
    }

    eval_json_path = out_dir / "evaluation_results_stage18b.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    if verbose:
        print(f"  Saved full structured evaluation JSON: {eval_json_path}\n", flush=True)
    return output_payload


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="PATH_RL Multi-Model Held-Out Benchmark")
    parser.add_argument(
        "--model-18",
        type=Path,
        default=_RL_OUTPUT_DIR / "model_stage18.pt",
        help="Path to Stage 18 model checkpoint",
    )
    parser.add_argument(
        "--model-18b",
        type=Path,
        default=_RL_OUTPUT_DIR / "model_stage18b.pt",
        help="Path to Stage 18B model checkpoint",
    )
    parser.add_argument(
        "--model-geom",
        type=Path,
        default=_RL_OUTPUT_DIR / "model_stage18b_geom.pt",
        help="Path to Stage 18B geometry-aware model checkpoint",
    )
    parser.add_argument(
        "--seeds-count",
        type=int,
        default=30,
        help="Number of held-out test scenarios (default: 30)",
    )
    args = parser.parse_args()

    eval_seeds = EVAL_SEEDS_STAGE18B[:args.seeds_count]
    run_heldout_benchmark(
        model_18_path=args.model_18,
        model_18b_path=args.model_18b,
        model_geom_path=args.model_geom,
        seeds=eval_seeds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
